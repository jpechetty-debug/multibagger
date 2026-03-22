"""Pipeline orchestration for end-to-end ticker scans."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass

import config
from data.db import db
from data.fetcher import DataFetcher, DataQualitySkip
from engines.alert_engine import AlertEngine
from engines.analysis.earnings_revision import EarningsRevisionAnalyzer
from engines.analysis.fundamentals import FundamentalAnalyzer
from engines.analysis.liquidity import LiquidityAnalyzer
from engines.analysis.momentum import MomentumAnalyzer
from engines.analysis.ownership import OwnershipAnalyzer
from engines.analysis.risk_metrics import RiskMetricsAnalyzer
from engines.analysis.sector_rank import SectorRankAnalyzer
from engines.audit.data_auditor import DataAuditor
from engines.audit.pre_scan_gate import PreScanGate
from engines.risk.correlation import CorrelationAnalyzer
from engines.risk.vix_filter import VixFilter
from engines.score_engine.model import ScoreDependencies, ScoreEngine
from engines.score_engine.regime import RegimeDetector
from engines.score_engine.features import FeatureBuilder
from engines.signal_engine import SignalDependencies, SignalEngine
from engines.valuation_engine import ValuationEngine
from models.schemas import PipelineRunResult, PipelineTickerResult


@dataclass
class PipelineDependencies:
    """Dependency bundle for the pipeline."""

    fetcher: DataFetcher
    auditor: DataAuditor
    pre_scan_gate: PreScanGate
    fundamentals: FundamentalAnalyzer
    earnings_revision: EarningsRevisionAnalyzer
    momentum: MomentumAnalyzer
    ownership: OwnershipAnalyzer
    sector_rank: SectorRankAnalyzer
    liquidity: LiquidityAnalyzer
    risk_metrics: RiskMetricsAnalyzer
    regime: RegimeDetector
    vix_filter: VixFilter
    correlation: CorrelationAnalyzer
    score_engine: ScoreEngine
    valuation_engine: ValuationEngine
    signal_engine: SignalEngine
    alert_engine: AlertEngine


class PipelineOrchestrator:
    """Runs the full fetch -> audit gate -> analysis -> score -> signal pipeline."""

    def __init__(self, dependencies: PipelineDependencies | None = None) -> None:
        """Initialize the orchestrator and its shared dependencies."""

        if dependencies is None:
            fetcher = DataFetcher()
            auditor = DataAuditor(fetcher=fetcher)
            pre_scan_gate = PreScanGate()
            fundamentals = FundamentalAnalyzer(fetcher)
            earnings_revision = EarningsRevisionAnalyzer(fetcher)
            momentum = MomentumAnalyzer(fetcher)
            ownership = OwnershipAnalyzer(fetcher)
            sector_rank = SectorRankAnalyzer(fetcher)
            liquidity = LiquidityAnalyzer(fetcher)
            risk_metrics = RiskMetricsAnalyzer(fetcher)
            regime = RegimeDetector()
            vix_filter = VixFilter()
            score_engine = ScoreEngine(
                dependencies=ScoreDependencies(
                    fetcher=fetcher,
                    fundamentals=fundamentals,
                    earnings_revision=earnings_revision,
                    momentum=momentum,
                    ownership=ownership,
                    sector_rank=sector_rank,
                    risk_metrics=risk_metrics,
                    regime=regime,
                    features=FeatureBuilder(),
                )
            )
            valuation_engine = ValuationEngine(fetcher)
            signal_engine = SignalEngine(
                dependencies=SignalDependencies(
                    fetcher=fetcher,
                    score_engine=score_engine,
                    valuation_engine=valuation_engine,
                    momentum=momentum,
                    sector_rank=sector_rank,
                    ownership=ownership,
                    vix_filter=vix_filter,
                    pre_scan_gate=pre_scan_gate,
                )
            )
            dependencies = PipelineDependencies(
                fetcher=fetcher,
                auditor=auditor,
                pre_scan_gate=pre_scan_gate,
                fundamentals=fundamentals,
                earnings_revision=earnings_revision,
                momentum=momentum,
                ownership=ownership,
                sector_rank=sector_rank,
                liquidity=liquidity,
                risk_metrics=risk_metrics,
                regime=regime,
                vix_filter=vix_filter,
                correlation=CorrelationAnalyzer(),
                score_engine=score_engine,
                valuation_engine=valuation_engine,
                signal_engine=signal_engine,
                alert_engine=AlertEngine(fetcher=fetcher, vix_filter=vix_filter),
            )
        self.dependencies = dependencies

    async def run(self, tickers: list[str], triggered_by: str = "pipeline") -> PipelineRunResult:
        """Run the full pipeline for many tickers."""

        started_at = int(time.time())
        normalized_tickers = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
        if not normalized_tickers:
            raise ValueError("No tickers provided to the pipeline")

        regime_name: str | None = None
        try:
            regime_result = self.dependencies.regime.detect()
            regime_name = regime_result.regime.value
            db.save_market_snapshot("regime", regime_result.model_dump(), regime_result.as_of)
        except Exception as exc:
            db.log_engine_event("WARN", "engines.pipeline", "regime snapshot failed", {"error": str(exc)})

        try:
            vix_result = self.dependencies.vix_filter.evaluate()
            db.save_market_snapshot("vix_filter", vix_result.model_dump(), vix_result.as_of)
        except Exception as exc:
            db.log_engine_event("WARN", "engines.pipeline", "vix snapshot failed", {"error": str(exc)})

        semaphore = asyncio.Semaphore(config.PIPELINE_MAX_CONCURRENCY)

        async def _bound_process(ticker: str) -> PipelineTickerResult:
            async with semaphore:
                return await asyncio.to_thread(self._process_ticker, ticker, triggered_by)

        gathered = await asyncio.gather(*[_bound_process(ticker) for ticker in normalized_tickers], return_exceptions=True)
        results: list[PipelineTickerResult] = []
        for ticker, item in zip(normalized_tickers, gathered):
            if isinstance(item, Exception):
                db.log_engine_event("ERROR", "engines.pipeline", "async ticker processing failed", {"ticker": ticker, "error": str(item)})
                results.append(PipelineTickerResult(ticker=ticker, action="ERROR", error=str(item), generated_at=int(time.time())))
            else:
                results.append(item)
        self._persist_correlation_snapshot(results)
        try:
            self.dependencies.alert_engine.process_run(results)
        except Exception as exc:
            db.log_engine_event("WARN", "engines.pipeline", "alert processing failed", {"error": str(exc)})

        universe_audit_summary = None
        if hasattr(self.dependencies.auditor, "audit_universe"):
            try:
                universe_audit_summary = self.dependencies.auditor.audit_universe(
                    normalized_tickers,
                    triggered_by=f"{triggered_by}_post_scan",
                    refresh_live=False,
                )
            except TypeError:
                universe_audit_summary = self.dependencies.auditor.audit_universe(
                    normalized_tickers,
                    triggered_by=f"{triggered_by}_post_scan",
                )
            except Exception as exc:
                db.log_engine_event(
                    "WARN",
                    "engines.pipeline",
                    "post-scan audit failed",
                    {"error": str(exc), "triggered_by": triggered_by},
                )
        if universe_audit_summary is not None:
            db.log_engine_event(
                "INFO",
                "engines.pipeline",
                "post-scan universe audit summary",
                {
                    "triggered_by": triggered_by,
                    "tickers_audited": universe_audit_summary.tickers_audited,
                    "pass_count": universe_audit_summary.pass_count,
                    "warn_count": universe_audit_summary.warn_count,
                    "fail_count": universe_audit_summary.fail_count,
                    "incomplete_count": universe_audit_summary.incomplete_count,
                    "average_score": universe_audit_summary.average_score,
                    "median_score": universe_audit_summary.median_score,
                    "score_distribution": universe_audit_summary.score_distribution,
                    "source_health_alerts": universe_audit_summary.source_health_alerts,
                },
            )

        action_counts = Counter(result.action for result in results)
        processed_count = sum(result.action not in {"SKIP", "ERROR"} for result in results)
        skipped_count = action_counts.get("SKIP", 0)
        error_count = action_counts.get("ERROR", 0)
        finished_at = int(time.time())
        summary = (
            f"Processed {processed_count}/{len(normalized_tickers)} tickers; "
            f"{action_counts.get('BUY', 0)} BUY, {action_counts.get('WATCH', 0)} WATCH, "
            f"{action_counts.get('WEAK', 0)} WEAK, {action_counts.get('REJECT', 0)} REJECT, "
            f"{skipped_count} SKIP, {error_count} ERROR"
        )
        status = "PASS" if error_count == 0 and skipped_count == 0 else "WARN" if error_count == 0 else "FAIL"
        if not triggered_by.startswith(("cli_", "scheduler_")):
            db.log_run_history(
                "pipeline.run",
                {"tickers": normalized_tickers, "triggered_by": triggered_by},
                status,
                summary,
                started_at,
                finished_at,
            )

        # --- Phase 1: Post-scan data quality summary ---
        quality_pass = sum(1 for r in results if r.action not in {"SKIP", "ERROR"})
        quality_skip = skipped_count
        quality_error = error_count
        db.log_engine_event(
            "INFO",
            "engines.pipeline",
            "scan quality summary",
            {
                "total": len(normalized_tickers),
                "pass": quality_pass,
                "skip": quality_skip,
                "error": quality_error,
                "pass_rate_pct": round(quality_pass / max(1, len(normalized_tickers)) * 100, 1),
                "regime": regime_name,
                "triggered_by": triggered_by,
            },
        )

        return PipelineRunResult(
            tickers_requested=len(normalized_tickers),
            processed_count=processed_count,
            skipped_count=skipped_count,
            error_count=error_count,
            action_counts=dict(action_counts),
            regime=regime_name,
            results=results,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
        )

    def _process_ticker(self, ticker: str, triggered_by: str) -> PipelineTickerResult:
        """Process a single ticker through the pipeline."""

        normalized_ticker = ticker.strip().upper()
        generated_at = int(time.time())
        try:
            data = self.dependencies.fetcher.fetch(normalized_ticker)
            latest_audit = db.get_latest_audit(normalized_ticker)
            audit_stale = latest_audit is None or generated_at - latest_audit.timestamp > config.FAIL_STALE_DAYS * 86400
            if audit_stale:
                latest_audit = self.dependencies.auditor.audit_ticker(normalized_ticker, triggered_by=triggered_by)

            gate_result = self.dependencies.pre_scan_gate.check(normalized_ticker, data, latest_audit)
            if not gate_result.passed:
                db.log_engine_event(
                    "INFO",
                    "engines.pipeline",
                    "ticker skipped by pre-scan gate",
                    {"ticker": normalized_ticker, "skip_reason": gate_result.skip_reason, "warnings": gate_result.warnings},
                )
                return PipelineTickerResult(
                    ticker=normalized_ticker,
                    action="SKIP",
                    data_warnings=gate_result.warnings,
                    skip_reason=gate_result.skip_reason,
                    generated_at=generated_at,
                )

            analysis_results = {
                "fundamentals": self.dependencies.fundamentals.analyze(normalized_ticker, data),
                "earnings_revision": self.dependencies.earnings_revision.analyze(normalized_ticker, data),
                "momentum": self.dependencies.momentum.analyze(normalized_ticker, data),
                "ownership": self.dependencies.ownership.analyze(normalized_ticker, data),
                "sector_rank": self.dependencies.sector_rank.analyze(normalized_ticker, data),
                "liquidity": self.dependencies.liquidity.analyze(normalized_ticker, data),
                "risk_metrics": self.dependencies.risk_metrics.analyze(normalized_ticker, data),
            }
            for analysis_type, model in analysis_results.items():
                db.save_analysis_snapshot(normalized_ticker, analysis_type, model.model_dump(), model.as_of)

            score_result = self.dependencies.score_engine.score_ticker(normalized_ticker, data)
            valuation_result = self.dependencies.valuation_engine.value_ticker(normalized_ticker, data)
            signal_result = self.dependencies.signal_engine.evaluate(normalized_ticker, data)
            upside_pct = (
                (valuation_result.fair_value / data.price - 1.0)
                if valuation_result.fair_value not in (None, 0) and data.price not in (None, 0)
                else None
            )
            return PipelineTickerResult(
                ticker=normalized_ticker,
                action=signal_result.action,
                score=score_result.total_score,
                fair_value=valuation_result.fair_value,
                upside_pct=upside_pct,
                data_warnings=sorted(set(gate_result.warnings + signal_result.data_warnings)),
                generated_at=generated_at,
            )
        except DataQualitySkip as exc:
            db.log_engine_event("INFO", "engines.pipeline", "ticker skipped by fetcher quality gate", {"ticker": normalized_ticker, "reason": str(exc)})
            return PipelineTickerResult(
                ticker=normalized_ticker,
                action="SKIP",
                skip_reason="LOW_DATA_QUALITY",
                error=str(exc),
                generated_at=generated_at,
            )
        except Exception as exc:
            db.log_engine_event("ERROR", "engines.pipeline", "ticker processing failed", {"ticker": normalized_ticker, "error": str(exc)})
            return PipelineTickerResult(
                ticker=normalized_ticker,
                action="ERROR",
                error=str(exc),
                generated_at=generated_at,
            )

    def _persist_correlation_snapshot(self, results: list[PipelineTickerResult]) -> None:
        """Persist the latest correlation snapshot for actionable signals."""

        correlation_tickers = [result.ticker for result in results if result.action in {"BUY", "WATCH"}]
        if len(correlation_tickers) < 2:
            return
        try:
            correlation_result = self.dependencies.correlation.correlation_matrix(correlation_tickers)
        except Exception as exc:
            db.log_engine_event("WARN", "engines.pipeline", "correlation snapshot failed", {"error": str(exc), "tickers": correlation_tickers})
            return
        db.save_market_snapshot("signal_correlation", correlation_result.model_dump(), correlation_result.as_of)


class MultiStrategyOrchestrator:
    """Run all three strategy pipelines and produce unified, deduplicated output.

    Strategies:
    1. **Positional** – existing pipeline (fundamental + momentum)
    2. **Swing** – technical indicators + breakout patterns
    3. **Multibagger** – quality filter + conviction scoring
    """

    def __init__(
        self,
        positional_pipeline: PipelineOrchestrator | None = None,
    ) -> None:
        self.positional_pipeline = positional_pipeline or PipelineOrchestrator()

    async def run(
        self,
        positional_tickers: list[str],
        swing_tickers: list[str] | None = None,
        multibagger_tickers: list[str] | None = None,
        triggered_by: str = "multi_strategy",
    ) -> PipelineRunResult:
        """Run all strategy pipelines and return a unified result.

        Parameters
        ----------
        positional_tickers:
            Tickers to run through the positional pipeline.
        swing_tickers:
            Tickers to evaluate for swing trades.  Defaults to positional_tickers.
        multibagger_tickers:
            Tickers to evaluate as multibagger candidates. Defaults to positional_tickers.
        """

        from models.schemas import StrategyTag

        started_at = int(time.time())
        all_results: list[PipelineTickerResult] = []

        # ---- 1. Positional pipeline (existing) ----
        positional_result = await self.positional_pipeline.run(
            positional_tickers, triggered_by=f"{triggered_by}_positional"
        )
        for r in positional_result.results:
            r.strategy_tag = StrategyTag.POSITIONAL
        all_results.extend(positional_result.results)

        # ---- 2. Swing pipeline ----
        swing_tickers = swing_tickers or positional_tickers
        swing_results = self._run_swing(swing_tickers)
        all_results.extend(swing_results)

        # ---- 3. Multibagger pipeline ----
        mb_tickers = multibagger_tickers or positional_tickers
        mb_results = self._run_multibagger(mb_tickers)
        all_results.extend(mb_results)

        # ---- Deduplicate: if same ticker appears in multiple strategies, keep highest-priority ----
        deduped = self._deduplicate(all_results)

        action_counts = Counter(r.action for r in deduped)
        processed = sum(r.action not in {"SKIP", "ERROR", "HOLD"} for r in deduped)
        finished_at = int(time.time())
        summary = (
            f"Multi-strategy: {processed}/{len(deduped)} actionable; "
            f"BUY={action_counts.get('BUY', 0)}, ENTRY={action_counts.get('ENTRY', 0)}, "
            f"WATCH={action_counts.get('WATCH', 0)}"
        )

        return PipelineRunResult(
            tickers_requested=len(set(positional_tickers + swing_tickers + mb_tickers)),
            processed_count=processed,
            skipped_count=action_counts.get("SKIP", 0),
            error_count=action_counts.get("ERROR", 0),
            action_counts=dict(action_counts),
            regime=positional_result.regime,
            results=deduped,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
        )

    def _run_swing(self, tickers: list[str]) -> list[PipelineTickerResult]:
        """Run the swing engine across tickers."""

        from engines.swing.swing_signal_engine import SwingSignalEngine
        from models.schemas import StrategyTag

        results: list[PipelineTickerResult] = []
        engine = SwingSignalEngine()
        for ticker in tickers:
            try:
                signal = engine.evaluate(ticker)
                db.save_swing_signal(signal.model_dump())
                results.append(PipelineTickerResult(
                    ticker=signal.ticker,
                    action=signal.action.value,
                    score=signal.confidence,
                    strategy_tag=StrategyTag.SWING,
                    generated_at=signal.generated_at,
                ))
            except Exception as exc:
                db.log_engine_event("ERROR", "engines.pipeline", "swing evaluation failed", {"ticker": ticker, "error": str(exc)})
                results.append(PipelineTickerResult(
                    ticker=ticker.strip().upper(),
                    action="ERROR",
                    strategy_tag=StrategyTag.SWING,
                    error=str(exc),
                    generated_at=int(time.time()),
                ))
        return results

    def _run_multibagger(self, tickers: list[str]) -> list[PipelineTickerResult]:
        """Run the multibagger engine across tickers."""

        from engines.multibagger.conviction_scorer import ConvictionScorer
        from models.schemas import StrategyTag

        results: list[PipelineTickerResult] = []
        scorer = ConvictionScorer()
        for ticker in tickers:
            try:
                candidate = scorer.score_ticker(ticker)
                db.save_multibagger_candidate(candidate.model_dump())
                results.append(PipelineTickerResult(
                    ticker=candidate.ticker,
                    action=candidate.action,
                    score=candidate.conviction_score,
                    strategy_tag=StrategyTag.MULTIBAGGER,
                    generated_at=candidate.generated_at,
                ))
            except Exception as exc:
                db.log_engine_event("ERROR", "engines.pipeline", "multibagger evaluation failed", {"ticker": ticker, "error": str(exc)})
                results.append(PipelineTickerResult(
                    ticker=ticker.strip().upper(),
                    action="ERROR",
                    strategy_tag=StrategyTag.MULTIBAGGER,
                    error=str(exc),
                    generated_at=int(time.time()),
                ))
        return results

    @staticmethod
    def _deduplicate(results: list[PipelineTickerResult]) -> list[PipelineTickerResult]:
        """Deduplicate across strategies: for each ticker, keep the result with
        the highest action priority (BUY > ENTRY > WATCH > WEAK > HOLD > REJECT > SKIP > ERROR)."""

        from models.schemas import StrategyTag

        ACTION_PRIORITY = {"BUY": 8, "ENTRY": 7, "WATCH": 6, "WEAK": 5, "HOLD": 4, "REJECT": 3, "SKIP": 2, "ERROR": 1, "EXIT": 0}
        best: dict[str, PipelineTickerResult] = {}
        for r in results:
            existing = best.get(r.ticker)
            if existing is None or ACTION_PRIORITY.get(r.action, 0) > ACTION_PRIORITY.get(existing.action, 0):
                best[r.ticker] = r
        return list(best.values())


if __name__ == "__main__":
    from ticker_list import get_universe

    result = asyncio.run(PipelineOrchestrator().run(get_universe(config.DEFAULT_SCAN_UNIVERSE), triggered_by="__main__"))
    print(result.model_dump())
