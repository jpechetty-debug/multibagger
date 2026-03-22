"""Sovereign engine metrics exporter."""

from __future__ import annotations

# ===========================================================================

import threading
import time
from collections import defaultdict
from pathlib import Path


class Gauge:
    """Single metric gauge — thread-safe float."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()
        self._updated_at: float = 0.0

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value
            self._updated_at = time.time()

    def get(self) -> float:
        with self._lock:
            return self._value

    def to_prometheus(self) -> str:
        return (
            f"# HELP sovereign_{self.name} {self.description}\n"
            f"# TYPE sovereign_{self.name} gauge\n"
            f"sovereign_{self.name} {self._value}\n"
        )


class Counter:
    """Monotonically increasing counter."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def get(self) -> float:
        with self._lock:
            return self._value

    def to_prometheus(self) -> str:
        return (
            f"# HELP sovereign_{self.name} {self.description}\n"
            f"# TYPE sovereign_{self.name} counter\n"
            f"sovereign_{self.name}_total {self._value}\n"
        )


class SovereignMetrics:
    """
    Prometheus-compatible metrics registry for the Sovereign engine.

    Exports all metrics as text/plain at /metrics.
    Wire into the Streamlit app or run as a standalone HTTP server.

    Usage
    -----
        metrics = SovereignMetrics()

        # Update from pipeline
        metrics.var_exceedance_count.inc()
        metrics.drawdown_current_pct.set(0.084)
        metrics.live_sharpe.set(1.42)

        # Export
        print(metrics.export())

        # Start HTTP server on port 9090
        metrics.start_server(port=9090)

    Grafana datasource: Prometheus → http://localhost:9090
    """

    def __init__(self):
        # ── Risk metrics ──────────────────────────────────────────────────
        self.var_exceedance_count = Counter(
            "var_exceedance_count",
            "Number of times portfolio VaR exceeded the configured limit"
        )
        self.drawdown_current_pct = Gauge(
            "drawdown_current_pct",
            "Current peak-to-trough portfolio drawdown as a fraction"
        )
        self.drawdown_max_30d_pct = Gauge(
            "drawdown_max_30d_pct",
            "Maximum drawdown over rolling 30-day window"
        )
        self.exposure_multiplier = Gauge(
            "exposure_multiplier",
            "Current Kelly × regime exposure multiplier (1.0 = full)"
        )
        self.circuit_breaker_active = Gauge(
            "circuit_breaker_active",
            "1 if drawdown circuit breaker is active, 0 otherwise"
        )

        # ── ML health ─────────────────────────────────────────────────────
        self.live_sharpe = Gauge(
            "live_sharpe",
            "Live Sharpe ratio from PerformanceTracker (annualised)"
        )
        self.oos_sharpe = Gauge(
            "oos_sharpe",
            "Out-of-sample Sharpe ratio from walk-forward validation"
        )
        self.sharpe_decay_pct = Gauge(
            "sharpe_decay_pct",
            "Percentage Sharpe decay (live vs OOS). Alert if > 20%"
        )
        self.model_return_correlation = Gauge(
            "model_return_correlation",
            "Correlation between predicted and actual returns. Alert if < 0.30"
        )
        self.retrain_required = Gauge(
            "retrain_required",
            "1 if model decay triggers retrain, 0 otherwise"
        )

        # ── Data freshness ─────────────────────────────────────────────────
        self.feature_store_hash_check_age_s = Gauge(
            "feature_store_hash_check_age_s",
            "Seconds since last feature store hash integrity check"
        )
        self.last_scan_age_s = Gauge(
            "last_scan_age_s",
            "Seconds since last full pipeline scan completed"
        )
        self.data_stale_ticker_count = Gauge(
            "data_stale_ticker_count",
            "Number of tickers with data older than WARN_STALE_DAYS"
        )

        # ── API health ─────────────────────────────────────────────────────
        self.api_latency_p95_ms = Gauge(
            "api_latency_p95_ms",
            "95th percentile API response latency in milliseconds"
        )
        self.api_error_rate = Gauge(
            "api_error_rate",
            "Fraction of API calls returning errors over the last hour"
        )

        # ── Portfolio ──────────────────────────────────────────────────────
        self.portfolio_positions = Gauge(
            "portfolio_positions",
            "Number of open positions in the portfolio"
        )
        self.kelly_implied_cash_pct = Gauge(
            "kelly_implied_cash_pct",
            "Fraction of capital kept as cash by Kelly sizer"
        )
        self.portfolio_vix_current = Gauge(
            "portfolio_vix_current",
            "Current India VIX value"
        )

        # ── Regime ────────────────────────────────────────────────────────
        self.regime_composite_score = Gauge(
            "regime_composite_score",
            "Current 8-state regime composite score (0-100)"
        )

        self._all: list[Gauge | Counter] = [
            self.var_exceedance_count,
            self.drawdown_current_pct,
            self.drawdown_max_30d_pct,
            self.exposure_multiplier,
            self.circuit_breaker_active,
            self.live_sharpe,
            self.oos_sharpe,
            self.sharpe_decay_pct,
            self.model_return_correlation,
            self.retrain_required,
            self.feature_store_hash_check_age_s,
            self.last_scan_age_s,
            self.data_stale_ticker_count,
            self.api_latency_p95_ms,
            self.api_error_rate,
            self.portfolio_positions,
            self.kelly_implied_cash_pct,
            self.portfolio_vix_current,
            self.regime_composite_score,
        ]

    def export(self) -> str:
        """Return all metrics in Prometheus text format."""
        return "\n".join(m.to_prometheus() for m in self._all)

    def update_from_db(self) -> None:
        """Pull latest values from the ops database and update gauges."""
        try:
            from data.db import db
            # Last scan age
            history = db.list_run_history(limit=1)
            if history:
                finished = history[0].get("finished_at") or 0
                self.last_scan_age_s.set(time.time() - float(finished))

            # VIX from latest snapshot
            vix_snap = db.get_latest_market_snapshot("india_vix")
            if vix_snap:
                vix_val = vix_snap.get("payload", {}).get("vix_value") or 0
                self.portfolio_vix_current.set(float(vix_val))

            # Stale tickers
            audits = db.list_latest_audit_rows()
            import config
            stale = sum(
                1 for a in audits
                if isinstance(a, dict) and a.get("score", 100) < config.MIN_DATA_QUALITY
            )
            self.data_stale_ticker_count.set(float(stale))

        except Exception:
            pass  # Metrics should never crash the engine

    def start_server(self, port: int = 9090, host: str = "0.0.0.0") -> threading.Thread:
        """
        Start a background HTTP server serving /metrics on the given port.
        Non-blocking — returns the thread.
        """
        import http.server

        metrics_instance = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    metrics_instance.update_from_db()
                    body = metrics_instance.export().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", len(body))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):
                pass  # suppress access log noise

        server = http.server.HTTPServer((host, port), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return t


# Singleton for import convenience
sovereign_metrics = SovereignMetrics()
