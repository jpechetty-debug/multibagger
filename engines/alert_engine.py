"""Telegram-backed alerting for pipeline and portfolio events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time

import requests

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.risk.vix_filter import VixFilter
from models.schemas import PipelineTickerResult


class AlertEngine:
    """Emits operational and trading alerts."""

    def __init__(self, fetcher: DataFetcher | None = None, vix_filter: VixFilter | None = None) -> None:
        """Initialize the alert engine."""

        self.fetcher = fetcher or DataFetcher()
        self.vix_filter = vix_filter or VixFilter()

    def process_run(self, results: list[PipelineTickerResult]) -> list[dict[str, object]]:
        """Emit all alert types relevant to a completed pipeline run."""

        alerts: list[dict[str, object]] = []
        alerts.extend(self._buy_signal_alerts(results))
        alerts.extend(self._swing_entry_alerts(results))
        alerts.extend(self._multibagger_candidate_alerts(results))
        alerts.extend(self._fair_value_alerts(results))
        alerts.extend(self._score_change_alerts(results))
        alerts.extend(self._stop_loss_alerts())
        vix_alert = self._vix_spike_alert()
        if vix_alert:
            alerts.append(vix_alert)
        return alerts

    def send_daily_report(self) -> dict[str, object]:
        """Send an end-of-day summary report."""

        signals = db.list_signals()
        positions = db.list_portfolio_positions()
        buy_count = sum(signal["action"] == "BUY" for signal in signals)
        watch_count = sum(signal["action"] == "WATCH" for signal in signals)
        reject_count = sum(signal["action"] == "REJECT" for signal in signals)
        total_value = sum(position.market_value for position in positions)
        body = (
            f"Signals: BUY={buy_count}, WATCH={watch_count}, REJECT={reject_count}\n"
            f"Open positions: {len(positions)}\n"
            f"Invested value: {total_value:,.2f}"
        )
        return self._send_alert("daily_report", "Daily Sovereign Report", body, {"buy_count": buy_count, "watch_count": watch_count, "reject_count": reject_count})

    def send_weekly_report(self) -> dict[str, object]:
        """Send a weekly model and operations report."""

        history = db.list_run_history(limit=20)
        recent_runs = [row for row in history if row["finished_at"] >= int(time.time()) - 7 * 86400]
        active_model = db.get_active_model_version(config.DEFAULT_MODEL_NAME)
        body = (
            f"Pipeline/CLI runs this week: {len(recent_runs)}\n"
            f"Active model: {active_model.version if active_model else 'none'}\n"
            f"Model source: {active_model.metadata.get('source') if active_model else 'n/a'}"
        )
        metadata = {"weekly_run_count": len(recent_runs), "active_model": active_model.version if active_model else None}
        return self._send_alert("weekly_report", "Weekly Sovereign Report", body, metadata)

    def _buy_signal_alerts(self, results: list[PipelineTickerResult]) -> list[dict[str, object]]:
        """Alert on BUY signals produced by the latest pipeline run."""

        alerts: list[dict[str, object]] = []
        for result in results:
            if result.action != "BUY":
                continue
            dedupe_key = f"buy:{result.ticker}:{datetime.now(timezone.utc).date().isoformat()}"
            if self._recent_alert_exists(dedupe_key, lookback_seconds=86400):
                continue
            strategy = getattr(result, "strategy_tag", "POSITIONAL")
            body = f"[{strategy}] {result.ticker} generated a BUY signal at score {result.score or 0:.1f}"
            if result.upside_pct is not None:
                body += f" with upside {result.upside_pct * 100:.1f}%"
            alerts.append(self._send_alert("buy_signal", f"BUY signal: {result.ticker}", body, {"ticker": result.ticker, "strategy": str(strategy), "dedupe_key": dedupe_key}))
        return alerts

    def _swing_entry_alerts(self, results: list[PipelineTickerResult]) -> list[dict[str, object]]:
        """Alert on ENTRY signals from the swing engine."""

        alerts: list[dict[str, object]] = []
        for result in results:
            if result.action != "ENTRY":
                continue
            dedupe_key = f"swing_entry:{result.ticker}:{datetime.now(timezone.utc).date().isoformat()}"
            if self._recent_alert_exists(dedupe_key, lookback_seconds=86400):
                continue
            body = f"[SWING] {result.ticker} ENTRY signal (confidence {result.score or 0:.1f})"
            alerts.append(self._send_alert("swing_entry", f"Swing ENTRY: {result.ticker}", body, {"ticker": result.ticker, "strategy": "SWING", "dedupe_key": dedupe_key}))
        return alerts

    def _multibagger_candidate_alerts(self, results: list[PipelineTickerResult]) -> list[dict[str, object]]:
        """Alert on high-conviction multibagger candidates."""

        alerts: list[dict[str, object]] = []
        for result in results:
            strategy = getattr(result, "strategy_tag", None)
            if strategy is None or str(strategy) != "MULTIBAGGER":
                continue
            if result.action != "BUY":
                continue
            dedupe_key = f"mb_candidate:{result.ticker}:{datetime.now(timezone.utc).date().isoformat()}"
            if self._recent_alert_exists(dedupe_key, lookback_seconds=86400 * 7):  # weekly dedup
                continue
            body = f"[MULTIBAGGER] {result.ticker} passed quality filter with conviction {result.score or 0:.1f}"
            alerts.append(self._send_alert("multibagger_candidate", f"Multibagger: {result.ticker}", body, {"ticker": result.ticker, "strategy": "MULTIBAGGER", "dedupe_key": dedupe_key}))
        return alerts

    def _fair_value_alerts(self, results: list[PipelineTickerResult]) -> list[dict[str, object]]:
        """Alert when the current price has reached the latest fair value."""

        alerts: list[dict[str, object]] = []
        for result in results:
            if result.fair_value in (None, 0):
                continue
            record = db.get_fundamental(result.ticker, effective=True)
            if record is None or record.price in (None, 0):
                continue
            price_gap_pct = abs(record.price / result.fair_value - 1.0)
            if price_gap_pct > config.ALERT_FAIR_VALUE_TOLERANCE_PCT:
                continue
            dedupe_key = f"fair_value:{result.ticker}:{datetime.now(timezone.utc).date().isoformat()}"
            if self._recent_alert_exists(dedupe_key, lookback_seconds=86400):
                continue
            body = f"{result.ticker} price {record.price:.2f} is within {price_gap_pct * 100:.1f}% of fair value {result.fair_value:.2f}"
            alerts.append(self._send_alert("fair_value_hit", f"Fair value hit: {result.ticker}", body, {"ticker": result.ticker, "dedupe_key": dedupe_key}))
        return alerts

    def _score_change_alerts(self, results: list[PipelineTickerResult]) -> list[dict[str, object]]:
        """Alert when the score changes materially versus the previous run."""

        alerts: list[dict[str, object]] = []
        for result in results:
            history = db.list_score_history(result.ticker, limit=2)
            if len(history) < 2:
                continue
            latest = float(history[0]["total_score"])
            previous = float(history[1]["total_score"])
            delta = latest - previous
            if abs(delta) < config.ALERT_SCORE_CHANGE_THRESHOLD:
                continue
            dedupe_key = f"score_change:{result.ticker}:{history[0]['generated_at']}"
            if self._recent_alert_exists(dedupe_key, lookback_seconds=86400):
                continue
            direction = "up" if delta > 0 else "down"
            body = f"{result.ticker} score moved {direction} by {delta:+.1f} points to {latest:.1f}"
            alerts.append(self._send_alert("score_change", f"Score change: {result.ticker}", body, {"ticker": result.ticker, "delta": delta, "dedupe_key": dedupe_key}))
        return alerts

    def _stop_loss_alerts(self) -> list[dict[str, object]]:
        """Alert when a portfolio position has breached its stop loss."""

        alerts: list[dict[str, object]] = []
        for position in db.list_portfolio_positions():
            price = position.last_price
            record = db.get_fundamental(position.ticker, effective=True)
            if record is not None and record.price is not None:
                price = record.price
            if price > position.stop_loss:
                continue
            dedupe_key = f"stop_loss:{position.ticker}:{datetime.now(timezone.utc).date().isoformat()}"
            if self._recent_alert_exists(dedupe_key, lookback_seconds=86400):
                continue
            body = f"{position.ticker} price {price:.2f} is below stop loss {position.stop_loss:.2f}"
            alerts.append(self._send_alert("stop_loss_breach", f"Stop loss breach: {position.ticker}", body, {"ticker": position.ticker, "dedupe_key": dedupe_key}))
        return alerts

    def _vix_spike_alert(self) -> dict[str, object] | None:
        """Alert when India VIX spikes above the configured threshold."""

        market_snapshot = db.get_latest_market_snapshot("regime")
        vix_value = None
        if market_snapshot:
            vix_value = market_snapshot["payload"].get("india_vix")
        if vix_value is None:
            try:
                vix_value = self.vix_filter.evaluate().vix_value
            except Exception:
                return None
        if vix_value is None or vix_value < config.ALERT_VIX_SPIKE_THRESHOLD:
            return None
        dedupe_key = f"vix_spike:{datetime.now(timezone.utc).date().isoformat()}"
        if self._recent_alert_exists(dedupe_key, lookback_seconds=86400):
            return None
        body = f"India VIX is elevated at {vix_value:.2f}"
        return self._send_alert("vix_spike", "VIX spike", body, {"vix_value": vix_value, "dedupe_key": dedupe_key})

    def _send_alert(self, category: str, title: str, body: str, context: dict[str, object] | None = None) -> dict[str, object]:
        """Send an alert to Telegram when configured and always log it to ops."""

        now_ts = int(time.time())
        context = dict(context or {})
        context.update({"category": category, "body": body, "sent_at": now_ts})
        sent = False
        error: str | None = None
        if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID:
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={"chat_id": config.TELEGRAM_CHAT_ID, "text": f"{title}\n{body}"},
                    timeout=config.TELEGRAM_API_TIMEOUT,
                )
                response.raise_for_status()
                sent = True
            except Exception as exc:
                error = str(exc)
        else:
            error = "telegram_not_configured"
        db.log_engine_event("INFO" if sent else "WARN", "engines.alert_engine", title, {**context, "sent": sent, "error": error})
        return {"category": category, "title": title, "body": body, "sent": sent, "error": error, "created_at": now_ts}

    def _recent_alert_exists(self, dedupe_key: str, lookback_seconds: int) -> bool:
        """Return whether an alert with the same dedupe key was logged recently."""

        cutoff = int(time.time()) - lookback_seconds
        for row in db.list_logs(limit=200, component_prefix="engines.alert_engine"):
            if row["created_at"] < cutoff:
                continue
            if row["context"].get("dedupe_key") == dedupe_key:
                return True
        return False


if __name__ == "__main__":
    sample = PipelineTickerResult(ticker="RELIANCE", action="BUY", score=82.0, fair_value=3200.0, upside_pct=0.08, generated_at=int(time.time()))
    print(AlertEngine().process_run([sample]))
