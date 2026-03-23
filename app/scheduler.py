"""APScheduler jobs for scans, alerts, cache maintenance, and retraining."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
import time  # noqa: E402  (needed for job timestamps)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:  # pragma: no cover - import fallback for environments without APScheduler
    BackgroundScheduler = None
    CronTrigger = None

import config
from data.cache import cache_manager
from data.db import db
from engines.alert_engine import AlertEngine
from engines.pipeline import PipelineOrchestrator
from engines.ml.trainer import SovereignTrainer
from engines.ml.model_guard import ModelGuard
from ticker_list import get_universe
from engines.monitoring.metrics import sovereign_metrics
from engines.analysis.cycle_detector import CycleDetector
from engines.ml.model_guard import ModelGuard

logger = logging.getLogger(__name__)


class AppScheduler:
    """Registers and runs background jobs for the trading engine."""

    def __init__(
        self,
        pipeline: PipelineOrchestrator | None = None,
        trainer: SovereignTrainer | None = None,
        alert_engine: AlertEngine | None = None,
        scheduler: BackgroundScheduler | None = None,
    ) -> None:
        """Initialize the scheduler and its jobs."""

        self.pipeline = pipeline or PipelineOrchestrator()
        self.trainer = trainer or SovereignTrainer()
        self.alert_engine = alert_engine or AlertEngine()
        self.guard = ModelGuard()
        self.scheduler = scheduler
        if self.scheduler is None and BackgroundScheduler is not None:
            self.scheduler = BackgroundScheduler(timezone=config.SCHEDULER_TIMEZONE)
        if self.scheduler is not None:
            self._register_jobs()

    def _register_jobs(self) -> None:
        """Add all recurring jobs to the underlying scheduler."""

        assert self.scheduler is not None
        assert CronTrigger is not None
        self.scheduler.add_job(
            self.run_daily_scan,
            CronTrigger(hour=config.DAILY_SCAN_HOUR, minute=config.DAILY_SCAN_MINUTE, timezone=config.SCHEDULER_TIMEZONE),
            id="daily_scan",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.check_model_health,
            CronTrigger(hour=8, minute=30, day_of_week="mon-fri", timezone=config.SCHEDULER_TIMEZONE),
            id="model_health_check",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_swing_scan,
            CronTrigger(
                minute=f"*/{config.SWING_SCAN_INTERVAL_MIN}",
                hour=f"{config.MARKET_OPEN_HOUR}-{config.MARKET_CLOSE_HOUR}",
                day_of_week="mon-fri",
                timezone=config.SCHEDULER_TIMEZONE,
            ),
            id="swing_scan",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_multibagger_scan,
            CronTrigger(
                day_of_week=config.MB_SCAN_DAY,
                hour=config.DAILY_SCAN_HOUR,
                minute=0,
                timezone=config.SCHEDULER_TIMEZONE,
            ),
            id="multibagger_scan",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_weekly_retrain,
            CronTrigger(day_of_week=config.WEEKLY_RETRAIN_DAY, hour=config.WEEKLY_RETRAIN_HOUR, minute=config.WEEKLY_RETRAIN_MINUTE, timezone=config.SCHEDULER_TIMEZONE),
            id="weekly_retrain",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_daily_report,
            CronTrigger(hour=config.DAILY_REPORT_HOUR, minute=config.DAILY_REPORT_MINUTE, timezone=config.SCHEDULER_TIMEZONE),
            id="daily_report",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_weekly_report,
            CronTrigger(day_of_week=config.WEEKLY_RETRAIN_DAY, hour=config.DAILY_REPORT_HOUR, minute=config.DAILY_REPORT_MINUTE, timezone=config.SCHEDULER_TIMEZONE),
            id="weekly_report",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.evict_cache,
            CronTrigger(hour=config.CACHE_EVICT_HOUR, minute=config.CACHE_EVICT_MINUTE, timezone=config.SCHEDULER_TIMEZONE),
            id="cache_evict",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.optimize_databases,
            CronTrigger(hour=config.DB_OPTIMIZE_HOUR, minute=config.DB_OPTIMIZE_MINUTE, timezone=config.SCHEDULER_TIMEZONE),
            id="db_optimize",
            replace_existing=True,
        )
        # PIT snapshot — daily at 06:30 IST (after data fetch, before main scan)
        self.scheduler.add_job(
            self.run_pit_snapshot,
            CronTrigger(hour=6, minute=30, day_of_week="mon-fri",
                        timezone=config.SCHEDULER_TIMEZONE),
            id="pit_snapshot",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.run_daily_backup,
            CronTrigger(hour=config.DAILY_SCAN_HOUR - 1, minute=0, timezone=config.SCHEDULER_TIMEZONE),
            id="daily_backup",
            replace_existing=True,
        )
        # Metrics refresh — every 60 seconds
        self.scheduler.add_job(
            self.refresh_metrics,
            "interval",
            seconds=60,
            id="metrics_refresh",
            replace_existing=True,
        )
        # Fyers token refresh — daily at 07:45 IST (before market open)
        self.scheduler.add_job(
            self.refresh_fyers_token,
            CronTrigger(hour=7, minute=45, day_of_week="mon-fri", timezone=config.SCHEDULER_TIMEZONE),
            id="fyers_token_refresh",
            replace_existing=True,
        )
        # Cycle detection — daily at 07:00 IST
        self.scheduler.add_job(
            self.run_cycle_detection,
            CronTrigger(hour=7, minute=0, day_of_week="mon-fri", timezone=config.SCHEDULER_TIMEZONE),
            id="cycle_detection",
            replace_existing=True,
        )
        # OOS vs live comparison — daily at 17:00 IST (after close)
        self.scheduler.add_job(
            self.run_oos_comparison,
            CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone=config.SCHEDULER_TIMEZONE),
            id="oos_comparison",
            replace_existing=True,
        )

    def start(self) -> None:
        """Start the background scheduler."""

        if self.scheduler is None:
            raise RuntimeError("APScheduler is not installed; install APScheduler to use AppScheduler")
        self.scheduler.start()
        db.log_engine_event("INFO", "app.scheduler", "scheduler started", {"jobs": self.describe_jobs()})

    def stop(self) -> None:
        """Stop the background scheduler."""

        if self.scheduler is None:
            return
        self.scheduler.shutdown(wait=False)
        db.log_engine_event("INFO", "app.scheduler", "scheduler stopped", {})

    def describe_jobs(self) -> list[str]:
        """Return a textual description of registered jobs."""

        if self.scheduler is None:
            return []
        descriptions: list[str] = []
        for job in self.scheduler.get_jobs():
            next_run_time = getattr(job, "next_run_time", None)
            descriptions.append(f"{job.id}:{next_run_time}")
        return descriptions

    def run_daily_scan(self) -> dict[str, object]:
        """Run the default daily universe scan (positional strategy)."""

        started_at = int(time.time())
        try:
            result = asyncio.run(self.pipeline.run(get_universe(config.DEFAULT_SCAN_UNIVERSE), triggered_by="scheduler_daily_scan"))
            db.log_engine_event("INFO", "app.scheduler", "daily scan completed", result.model_dump())
            db.log_job_run("daily_scan", started_at, int(time.time()), "SUCCESS")
            return {"summary": result.summary, "regime": result.regime}
        except Exception as exc:
            db.log_job_run("daily_scan", started_at, int(time.time()), "FAIL", str(exc))
            raise

    def run_swing_scan(self) -> dict[str, object]:
        """Run the swing trade scanner on the liquid universe (every 30 min during market hours)."""

        started_at = int(time.time())
        try:
            from engines.swing.swing_signal_engine import SwingSignalEngine

            engine = SwingSignalEngine()
            tickers = get_universe("QUICK")  # liquid Nifty 500 subset for speed
            results = []
            for ticker in tickers:
                try:
                    signal = engine.evaluate(ticker)
                    result_dict = {
                        "ticker": signal.ticker, 
                        "action": signal.action.value if hasattr(signal.action, "value") else str(signal.action), 
                        "confidence": signal.confidence
                    }
                    
                    # Extract optional fields if available
                    metadata = getattr(signal, "metadata", {}) or {}
                    result_dict["price"] = getattr(signal, "price", None) or metadata.get("price")
                    result_dict["rsi"] = metadata.get("rsi")
                    result_dict["trend"] = metadata.get("trend")
                    result_dict["volume_surge"] = metadata.get("volume_surge")
                    result_dict["target"] = metadata.get("target")
                    result_dict["stop_loss"] = metadata.get("stop_loss")
                    
                    db.save_swing_signal(result_dict)
                    results.append(result_dict)
                except Exception as exc:
                    db.log_engine_event("WARN", "app.scheduler", "swing scan ticker failed", {"ticker": ticker, "error": str(exc)})
            entry_count = sum(1 for r in results if r["action"] == "ENTRY")
            db.log_engine_event("INFO", "app.scheduler", "swing scan completed", {"scanned": len(results), "entries": entry_count})
            db.log_job_run("swing_scan", started_at, int(time.time()), "SUCCESS")
            return {"scanned": len(results), "entries": entry_count, "results": results[:10]}
        except Exception as exc:
            db.log_job_run("swing_scan", started_at, int(time.time()), "FAIL", str(exc))
            raise

    def run_multibagger_scan(self) -> dict[str, object]:
        """Run the multibagger scanner on the full universe (weekly)."""

        started_at = int(time.time())
        try:
            from engines.multibagger.conviction_scorer import ConvictionScorer

            scorer = ConvictionScorer()
            tickers = get_universe(config.DEFAULT_SCAN_UNIVERSE)
            candidates = []
            for ticker in tickers:
                try:
                    candidate = scorer.score_ticker(ticker)
                    if candidate.action in ("BUY", "WATCH"):
                        cand_dict = candidate.model_dump()
                        db.save_multibagger_candidate(cand_dict)
                        candidates.append(cand_dict)
                except Exception as exc:
                    db.log_engine_event("WARN", "app.scheduler", "mb scan ticker failed", {"ticker": ticker, "error": str(exc)})
            db.log_engine_event("INFO", "app.scheduler", "multibagger scan completed", {"scanned": len(tickers), "candidates": len(candidates)})
            db.log_job_run("multibagger_scan", started_at, int(time.time()), "SUCCESS")
            return {"scanned": len(tickers), "candidates": len(candidates), "top": candidates[:5]}
        except Exception as exc:
            db.log_job_run("multibagger_scan", started_at, int(time.time()), "FAIL", str(exc))
            raise

    def run_weekly_retrain(self) -> dict[str, object]:
        """Train and activate the weekly model refresh."""

        started_at = int(time.time())
        try:
            version = self.trainer.run_training_pipeline()
            db.log_engine_event("INFO", "app.scheduler", "weekly retrain completed", version.model_dump())
            db.log_job_run("weekly_retrain", started_at, int(time.time()), "SUCCESS")
            return version.model_dump()
        except Exception as exc:
            db.log_job_run("weekly_retrain", started_at, int(time.time()), "FAIL", str(exc))
            raise

    def check_model_health(self) -> dict[str, Any]:
        """Check for model decay and trigger emergency retrain if needed."""

        started_at = int(time.time())
        try:
            status = self.guard.status()
            
            if status["fallback_active"]:
                db.log_engine_event("CRITICAL", "app.scheduler", "Model decay detected! Triggering emergency retrain.", status)
                self.alert_engine.send_custom_alert("🚨 EMERGENCY RETRAIN TRIGGERED", f"Model guard activated fallback. Reason: {status.get('fallback_reason', 'unknown')}")
                self.run_weekly_retrain()
            
            db.log_job_run("model_health_check", started_at, int(time.time()), "SUCCESS")
            return status
        except Exception as exc:
            db.log_job_run("model_health_check", started_at, int(time.time()), "FAIL", str(exc))
            return {"error": str(exc), "should_retrain": False}

    def run_daily_report(self) -> dict[str, object]:
        """Send the daily report alert."""

        result = self.alert_engine.send_daily_report()
        db.log_engine_event("INFO", "app.scheduler", "daily report sent", result)
        return result

    def run_weekly_report(self) -> dict[str, object]:
        """Send the weekly report alert."""

        result = self.alert_engine.send_weekly_report()
        db.log_engine_event("INFO", "app.scheduler", "weekly report sent", result)
        return result

    def evict_cache(self) -> dict[str, int]:
        """Evict expired cache entries."""

        removed = cache_manager.evict_expired()
        stats = cache_manager.stats()
        db.log_engine_event("INFO", "app.scheduler", "cache eviction complete", {"removed": removed, "stats": stats})
        return {"removed": removed, **stats}

    def optimize_databases(self) -> dict[str, str]:
        """Run SQLite PRAGMA optimize across databases."""

        result = db.optimize_databases()
        db.log_engine_event("INFO", "app.scheduler", "database optimize complete", result)
        return result

    def run_daily_backup(self) -> dict[str, str]:
        """Run daily database and model backups."""

        started_at = int(time.time())
        try:
            archive_path = db.backup_databases(backup_tag="daily")
            verified = db.verify_backup(archive_path)
            db.log_backup(str(archive_path), "SUCCESS", verified)
            db.log_job_run("daily_backup", started_at, int(time.time()), "SUCCESS" if verified else "WARN")
            db.log_engine_event("INFO", "app.scheduler", "daily backup completed", {"archive": archive_path.name, "verified": verified})
            return {"archive": archive_path.name, "verified": str(verified)}
        except Exception as exc:
            db.log_job_run("daily_backup", started_at, int(time.time()), "FAIL", str(exc))
            db.log_backup("N/A", f"FAIL: {str(exc)}", False)
            raise


    def refresh_metrics(self) -> dict:
        """Update all Prometheus metrics from the ops database."""
        try:
            sovereign_metrics.update_from_db()
            return {"status": "ok", "metrics": len(sovereign_metrics._all)}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def refresh_fyers_token(self) -> dict:
        """
        Check Fyers token age and warn if expiry is near.
        Auto-refresh requires a stored refresh_token — if absent, logs a warning
        so the operator knows to re-authenticate before market open.
        """
        import json
        from pathlib import Path
        token_path = Path("runtime/fyers_token.json")
        if not token_path.exists():
            db.log_engine_event("WARN", "app.scheduler", "fyers_token_refresh",
                                {"status": "token_file_missing"})
            return {"status": "token_file_missing"}
        try:
            data     = json.loads(token_path.read_text())
            age_h    = (time.time() - data.get("created_at", 0)) / 3600
            if age_h > 20:
                db.log_engine_event("WARN", "app.scheduler", "fyers_token_refresh",
                                    {"status": "token_stale", "age_hours": round(age_h, 1)})
                try:
                    self.alert_engine.send_custom_alert(
                        "⚠️ Fyers Token Stale",
                        f"Token is {age_h:.1f}h old. Re-authenticate before market open."
                    )
                except Exception:
                    pass
                return {"status": "stale", "age_hours": round(age_h, 1)}
            return {"status": "ok", "age_hours": round(age_h, 1)}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def run_pit_snapshot(self) -> dict:
        """Snapshot all fundamentals into the point-in-time store.

        FIX-5: The ML labeler requires fundamentals_pit to be populated.
        This job runs once per trading day (06:30 IST) and writes the current
        fundamental record for every ticker that has data in the main DB.
        Over time this accumulates a historical dataset usable for training.
        """
        started_at = int(time.time())
        saved      = 0
        failed     = 0

        try:
            tickers = get_universe(config.DEFAULT_SCAN_UNIVERSE)
            for ticker in tickers:
                try:
                    record = db.get_fundamental(ticker, effective=True)
                    if record is None:
                        continue

                    # Build a JSON-serialisable dict from the Pydantic model
                    fund_dict = record.model_dump()
                    # Remove non-serialisable or internal fields
                    for drop_key in ("ingestion_issues", "feature_vector"):
                        fund_dict.pop(drop_key, None)

                    source_meta = fund_dict.pop("source_metadata", {}) or {}

                    with db.connection("pit") as conn:
                        conn.execute(
                            """
                            INSERT INTO fundamentals_pit
                                (ticker, captured_at, fundamentals_json, source_metadata_json)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                ticker,
                                started_at,
                                json.dumps(fund_dict, default=str),
                                json.dumps(source_meta, default=str),
                            ),
                        )
                    saved += 1
                except Exception as exc:
                    logger.debug("PIT snapshot failed for %s: %s", ticker, exc)
                    failed += 1

            db.log_engine_event(
                "INFO", "app.scheduler", "pit_snapshot",
                {"saved": saved, "failed": failed, "total": len(tickers)},
            )
            db.log_job_run("pit_snapshot", started_at, int(time.time()), "SUCCESS")
            logger.info("PIT snapshot: %d saved, %d failed", saved, failed)
            return {"saved": saved, "failed": failed}

        except Exception as exc:
            db.log_job_run("pit_snapshot", started_at, int(time.time()), "FAIL", str(exc))
            logger.exception("PIT snapshot job failed")
            return {"error": str(exc)}

    def run_cycle_detection(self) -> dict:
        """Run market cycle detection and persist snapshot.

        FIX-5b: Market data is now sourced from real DB snapshots and live VIX
        rather than being derived from the regime engine's own composite score.
        This breaks the circular dependency:
            regime score → cycle input → sector weights → score → regime score

        Inputs used:
        - breadth_ratio:        latest value from the ops market_snapshot table
        - india_vix:            latest VIX from market_snapshot
        - gsec_10y / repo_rate: documented constants (update monthly from RBI)
        - fii_flow:             latest 20-day FII flow from market_snapshot

        All values fall back to neutral defaults if snapshots are absent so the
        job never crashes during cold-start.
        """
        try:
            # ── Fetch real market inputs ───────────────────────────────────────
            breadth_ratio = 1.0   # neutral default
            india_vix     = 15.0  # neutral default
            fii_20d_cr    = 0.0   # neutral default

            try:
                breadth_snap = db.get_latest_market_snapshot("breadth")
                if breadth_snap:
                    breadth_ratio = float(breadth_snap.get("payload", {}).get("breadth_ratio", 1.0))
            except Exception:
                pass

            try:
                vix_snap = db.get_latest_market_snapshot("india_vix")
                if vix_snap:
                    india_vix = float(vix_snap.get("payload", {}).get("vix_value", 15.0))
            except Exception:
                pass

            try:
                fii_snap = db.get_latest_market_snapshot("fii_flow")
                if fii_snap:
                    fii_20d_cr = float(fii_snap.get("payload", {}).get("net_fii_20d_cr", 0.0))
            except Exception:
                pass

            # Breadth slope: positive when breadth_ratio > 1 (more advancers)
            breadth_slope = (breadth_ratio - 1.0) * 2.0

            # VIX → cyclical vs defensive relative strength proxy
            # Low VIX = risk-on = cyclicals outperform
            cyc_def = max(-10.0, min(10.0, (20.0 - india_vix) * 0.5))

            market_data = {
                # ── Real inputs ──────────────────────────────────────────────
                "breadth_slope_20d":      breadth_slope,
                "cyclical_vs_defensive":  cyc_def,
                "net_fii_20d_cr":         fii_20d_cr,
                "fii_cyclical_pct":       60.0 if fii_20d_cr > 2000 else 40.0 if fii_20d_cr < -1000 else 50.0,

                # ── Static / slowly-changing — update monthly from RBI ───────
                # Last updated: 2026-03 (RBI MPC: repo 6.50%, 10Y Gsec ~6.85%)
                "gsec_10y":   6.85,
                "repo_rate":  6.50,

                # ── EPS revision differential (not yet wired — neutral) ──────
                "eps_revision_cyclical":   0.0,
                "eps_revision_defensive":  0.0,
            }

            detector = CycleDetector()
            result   = detector.detect(market_data)
            detector.save_snapshot(result)

            sovereign_metrics.regime_composite_score.set(
                50.0 + (breadth_slope * 10)   # lightweight proxy for the gauge
            )
            db.log_engine_event(
                "INFO", "app.scheduler", "cycle_detection",
                {"phase": result.phase, "confidence": result.confidence,
                 "breadth_slope": breadth_slope, "vix": india_vix},
            )
            return result.to_dict()
        except Exception as exc:
            db.log_engine_event("WARN", "app.scheduler", "cycle_detection_failed", {"error": str(exc)})
            logger.exception("Cycle detection job failed")
            return {"error": str(exc)}

    def run_oos_comparison(self) -> dict:
        """
        Compare live vs OOS performance and activate model guard if decayed.
        Runs after market close — uses today's outcomes to update tracker.
        """
        try:
            from engines.ml.performance_tracker import PerformanceTracker
            from data.db import db as _db
            tracker = PerformanceTracker()
            # Get active model metadata for OOS baseline
            model_version = _db.get_active_model_version("ensemble_forward_3m_ret")
            if not model_version:
                return {"status": "no_active_model"}
            meta = model_version.metadata if hasattr(model_version, "metadata") else {}
            oos_sharpe   = (meta.get("metrics", {}) or {}).get("wf_auc", 1.0)
            oos_hit_rate = (meta.get("metrics", {}) or {}).get("wf_precision", 0.54)
            report = tracker.compare(
                model_id=model_version.version,
                oos_sharpe=oos_sharpe,
                oos_hit_rate=oos_hit_rate,
                window_days=60,
            )
            # Update metrics
            sovereign_metrics.live_sharpe.set(report.live_sharpe)
            sovereign_metrics.oos_sharpe.set(report.oos_sharpe)
            sovereign_metrics.sharpe_decay_pct.set(abs(report.sharpe_decay) * 100)
            sovereign_metrics.model_return_correlation.set(report.return_corr)
            # Activate guard if needed
            guard = ModelGuard()
            if report.action == "RETRAIN_NOW":
                guard.force_fallback(f"OOS decay: {report.summary}")
                sovereign_metrics.retrain_required.set(1.0)
                db.log_engine_event("CRITICAL", "app.scheduler", "model_decay_fallback_activated",
                                    {"summary": report.summary})
            else:
                sovereign_metrics.retrain_required.set(0.0)
            db.log_engine_event("INFO", "app.scheduler", "oos_comparison", report.to_dict())
            return report.to_dict()
        except Exception as exc:
            db.log_engine_event("WARN", "app.scheduler", "oos_comparison_failed", {"error": str(exc)})
            return {"error": str(exc)}


if __name__ == "__main__":
    scheduler = AppScheduler()
    print(scheduler.describe_jobs())
