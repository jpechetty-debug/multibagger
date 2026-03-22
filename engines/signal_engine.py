"""Signal generation engine."""

from __future__ import annotations

from dataclasses import dataclass

from data.db import db
from data.fetcher import DataFetcher
from engines.analysis.momentum import MomentumAnalyzer
from engines.analysis.ownership import OwnershipAnalyzer
from engines.analysis.sector_rank import SectorRankAnalyzer
from engines.audit.pre_scan_gate import PreScanGate
from engines.risk.vix_filter import VixFilter
from engines.score_engine.model import ScoreEngine
from engines.valuation_engine import ValuationEngine
from models.schemas import FundamentalData, SignalResult, VixState


@dataclass
class SignalDependencies:
    """Dependency bundle for the signal engine."""

    fetcher: DataFetcher
    score_engine: ScoreEngine
    valuation_engine: ValuationEngine
    momentum: MomentumAnalyzer
    sector_rank: SectorRankAnalyzer
    ownership: OwnershipAnalyzer
    vix_filter: VixFilter
    pre_scan_gate: PreScanGate


class SignalEngine:
    """Evaluates buy, watch, weak, and reject signals."""

    def __init__(self, dependencies: SignalDependencies | None = None) -> None:
        """Initialize the signal engine."""

        if dependencies is None:
            fetcher = DataFetcher()
            dependencies = SignalDependencies(
                fetcher=fetcher,
                score_engine=ScoreEngine(),
                valuation_engine=ValuationEngine(fetcher),
                momentum=MomentumAnalyzer(fetcher),
                sector_rank=SectorRankAnalyzer(fetcher),
                ownership=OwnershipAnalyzer(fetcher),
                vix_filter=VixFilter(),
                pre_scan_gate=PreScanGate(),
            )
        self.dependencies = dependencies

    def evaluate(self, ticker: str, data: FundamentalData | None = None) -> SignalResult:
        """Evaluate a trading signal for a ticker."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.dependencies.fetcher.fetch(normalized_ticker)
        latest_audit = db.get_latest_audit(normalized_ticker)
        gate_result = self.dependencies.pre_scan_gate.check(normalized_ticker, data, latest_audit)
        if not gate_result.passed:
            result = SignalResult(
                ticker=normalized_ticker,
                action="REJECT",
                confidence_score=0.0,
                reason_code=gate_result.skip_reason or "PRE_SCAN_FAIL",
                satisfied_conditions=[],
                failed_conditions=[gate_result.skip_reason or "PRE_SCAN_FAIL"],
                data_warnings=gate_result.warnings,
                generated_at=int(__import__("time").time()),
            )
            db.save_signal_result(result)
            return result

        score_result = self.dependencies.score_engine.score_ticker(normalized_ticker, data)
        valuation_result = self.dependencies.valuation_engine.value_ticker(normalized_ticker, data)
        momentum_result = self.dependencies.momentum.analyze(normalized_ticker, data)
        sector_rank_result = self.dependencies.sector_rank.analyze(normalized_ticker, data)
        ownership_result = self.dependencies.ownership.analyze(normalized_ticker, data)
        vix_result = self.dependencies.vix_filter.evaluate()

        conditions = {
            "score_gt_75": score_result.total_score > 75.0,
            "price_below_fair": bool(valuation_result.undervalued),
            "above_50dma": momentum_result.above_50dma,
            "sector_top_3": sector_rank_result.top_3,
            "vix_safe": vix_result.state is VixState.NORMAL,
            "ownership_clean": ownership_result.ownership_clean,
        }
        satisfied = [name for name, passed in conditions.items() if passed]
        failed = [name for name, passed in conditions.items() if not passed]

        confidence_score = min(
            100.0,
            score_result.total_score * 0.6
            + (len(satisfied) / len(conditions)) * 40.0,
        )

        import config
        if valuation_result.valuation_confidence is not None:
            if valuation_result.valuation_confidence < config.VALUATION_CONFIDENCE_LOW_MAX:
                confidence_score *= max(0.2, (valuation_result.valuation_confidence / 100.0))
                gate_result.warnings.append("LOW_VALUATION_CONFIDENCE")

        if len(satisfied) == len(conditions):
            action = "BUY"
        elif score_result.total_score >= 70.0 or len(satisfied) >= 4:
            action = "WATCH"
        elif score_result.total_score >= 50.0 or len(satisfied) >= 2:
            action = "WEAK"
        else:
            action = "REJECT"

        if action == "BUY" and valuation_result.valuation_confidence is not None and valuation_result.valuation_confidence < config.VALUATION_CONFIDENCE_ACTIONABLE_MIN:
            action = "WATCH"
            if "LOW_VALUATION_CONFIDENCE" not in gate_result.warnings:
                gate_result.warnings.append("VALUATION_QUARANTINE_DOWNGRADE")
        reason_code = "|".join(satisfied if action == "BUY" else failed[:3] if failed else ["MIXED_SIGNAL"])
        result = SignalResult(
            ticker=normalized_ticker,
            action=action,
            confidence_score=confidence_score,
            reason_code=reason_code,
            satisfied_conditions=satisfied,
            failed_conditions=failed,
            data_warnings=gate_result.warnings,
            generated_at=int(__import__("time").time()),
        )
        db.save_signal_result(result)
        db.log_engine_event("INFO", "engines.signal_engine", "signal evaluated", result.model_dump())
        return result


if __name__ == "__main__":
    print(SignalEngine().evaluate("RELIANCE").model_dump())
