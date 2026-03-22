"""Pre-scan data quality gate for future pipeline integration."""

from __future__ import annotations

import time

import config
from models.schemas import AuditReport, FundamentalData, GateResult


class PreScanGate:
    """Evaluates whether a ticker should proceed into analysis."""

    def check(
        self,
        ticker: str,
        data: FundamentalData,
        latest_audit: AuditReport | None = None,
    ) -> GateResult:
        """Return a hard-fail or soft-warn gate result for a ticker."""

        effective_quality_score = data.ingestion_quality_score
        now_ts = int(time.time())
        if latest_audit and now_ts - latest_audit.timestamp <= config.FAIL_STALE_DAYS * 86400:
            effective_quality_score = latest_audit.audit_quality_score

        if data.price is None or data.price <= 0:
            return GateResult(passed=False, effective_quality_score=effective_quality_score, skip_reason="INVALID_PRICE")
        if data.market_cap is None or data.market_cap < config.FIELD_AUDIT_RULES["market_cap"]["min_exclusive"]:
            return GateResult(passed=False, effective_quality_score=effective_quality_score, skip_reason="BELOW_MIN_MARKETCAP")
        if effective_quality_score < config.MIN_DATA_QUALITY:
            reasons = []
            for issue in data.ingestion_issues:
                if issue.startswith("CONFLICT_"):
                    reasons.append("CONFLICTING_SOURCES")
                elif issue.startswith("STALE_") and issue != "STALE_SOURCE_DATA":
                    reasons.append("STALE_FINANCIALS")
            skip_reason = reasons[0] if reasons else "LOW_DATA_QUALITY"
            return GateResult(passed=False, effective_quality_score=effective_quality_score, skip_reason=skip_reason)
        if data.roe_5y is None and data.roe_ttm is None:
            return GateResult(passed=False, effective_quality_score=effective_quality_score, skip_reason="NO_ROE_DATA")
        if data.sales_growth_5y is None:
            return GateResult(passed=False, effective_quality_score=effective_quality_score, skip_reason="NO_REVENUE_DATA")
        if data.pledge_pct is not None and data.pledge_pct > config.FIELD_AUDIT_RULES["pledge_pct"]["fail_high"]:
            return GateResult(passed=False, effective_quality_score=effective_quality_score, skip_reason="EXTREME_PLEDGE")
        if data.avg_volume is not None and data.avg_volume < config.FIELD_AUDIT_RULES["avg_volume"]["min_exclusive"]:
            return GateResult(passed=False, effective_quality_score=effective_quality_score, skip_reason="ILLIQUID")
        if now_ts - data.updated_at > config.FAIL_STALE_DAYS * 86400:
            return GateResult(passed=False, effective_quality_score=effective_quality_score, skip_reason="DATA_TOO_OLD")

        warnings: list[str] = []
        if effective_quality_score < 70:
            warnings.append("LOW_DATA_QUALITY_WARNING")
        if data.roe_5y is not None and data.roe_5y > config.FIELD_AUDIT_RULES["roe_5y"]["warn_high"]:
            warnings.append("UNUSUALLY_HIGH_ROE")
        if data.eps_growth_ttm is not None and data.eps_growth_ttm > config.FIELD_AUDIT_RULES["eps_growth_ttm"]["warn_high"]:
            warnings.append("EPS_BASE_EFFECT_RISK")
        if data.cfo_to_pat is not None and data.cfo_to_pat < config.FIELD_AUDIT_RULES["cfo_to_pat"]["warn_low"]:
            warnings.append("POOR_EARNINGS_QUALITY")
        if data.pledge_pct is not None and data.pledge_pct > config.FIELD_AUDIT_RULES["pledge_pct"]["warn_high"]:
            warnings.append("ELEVATED_PLEDGE")
        if data.roe_5y is not None and data.roe_ttm is not None and abs(data.roe_5y - data.roe_ttm) > 0.20:
            warnings.append("ROE_INCONSISTENCY")
        if now_ts - data.updated_at > config.WARN_STALE_DAYS * 86400:
            warnings.append("STALE_DATA")
        if data.peg_ratio is not None and data.peg_ratio < 0:
            warnings.append("INVALID_NEGATIVE_PEG")

        return GateResult(
            passed=True,
            effective_quality_score=effective_quality_score,
            skip_reason=None,
            warnings=warnings,
        )


if __name__ == "__main__":
    from data.db import db

    ticker = "RELIANCE"
    record = db.get_fundamental(ticker, effective=True)
    if record:
        print(PreScanGate().check(ticker, record).model_dump())
    else:
        print({"error": f"No stored data found for {ticker}"})
