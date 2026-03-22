"""Quality filter for multibagger candidates: ROE, CAGR, D/E, CFO/PAT, mcap, Piotroski, promoter, pledge."""

from __future__ import annotations

import time

import config
from data.db import db
from models.schemas import FundamentalData, QualityResult


class QualityFilter:
    """Screen stocks against hard multibagger quality thresholds."""

    def filter(self, ticker: str, data: FundamentalData) -> QualityResult:
        """Return a ``QualityResult`` indicating whether *ticker* passes all quality gates.

        Each gate is evaluated independently so that callers can see which
        specific criteria failed.
        """

        normalized = ticker.strip().upper()
        now_ts = int(time.time())
        fail_reasons: list[str] = []

        # --- ROE 5-year average ---
        roe_ok = (data.roe_5y or 0) >= config.MB_MIN_ROE_5Y
        if not roe_ok:
            fail_reasons.append(f"ROE 5Y {data.roe_5y} < {config.MB_MIN_ROE_5Y}")

        # --- Sales CAGR 5-year ---
        cagr_ok = (data.sales_growth_5y or 0) >= config.MB_MIN_SALES_CAGR_5Y
        if not cagr_ok:
            fail_reasons.append(f"Sales CAGR {data.sales_growth_5y} < {config.MB_MIN_SALES_CAGR_5Y}")

        # --- Debt / Equity ---
        debt_ok = (data.debt_equity or 0) <= config.MB_MAX_DEBT_EQUITY
        if not debt_ok:
            fail_reasons.append(f"D/E {data.debt_equity} > {config.MB_MAX_DEBT_EQUITY}")

        # --- CFO to PAT ---
        cfo_ok = (data.cfo_to_pat or 0) >= config.MB_MIN_CFO_TO_PAT
        if not cfo_ok:
            fail_reasons.append(f"CFO/PAT {data.cfo_to_pat} < {config.MB_MIN_CFO_TO_PAT}")

        # --- Market cap ceiling ---
        mcap_ok = data.market_cap is not None and data.market_cap <= config.MB_MAX_MARKET_CAP
        if not mcap_ok:
            fail_reasons.append(f"Market cap {data.market_cap} > {config.MB_MAX_MARKET_CAP}")

        # --- Piotroski score ---
        piotroski_ok = (data.piotroski_score or 0) >= config.MB_MIN_PIOTROSKI
        if not piotroski_ok:
            fail_reasons.append(f"Piotroski {data.piotroski_score} < {config.MB_MIN_PIOTROSKI}")

        # --- Promoter holding ---
        promoter_ok = (data.promoter_pct or 0) >= config.MB_MIN_PROMOTER_PCT
        if not promoter_ok:
            fail_reasons.append(f"Promoter {data.promoter_pct}% < {config.MB_MIN_PROMOTER_PCT}%")

        # --- Pledge percentage ---
        pledge_ok = (data.pledge_pct or 0) <= config.MB_MAX_PLEDGE_PCT
        if not pledge_ok:
            fail_reasons.append(f"Pledge {data.pledge_pct}% > {config.MB_MAX_PLEDGE_PCT}%")

        checks = [roe_ok, cagr_ok, debt_ok, cfo_ok, mcap_ok, piotroski_ok, promoter_ok, pledge_ok]
        passed = all(checks)
        quality_score = (sum(checks) / len(checks)) * 100.0

        result = QualityResult(
            ticker=normalized,
            passed=passed,
            roe_ok=roe_ok,
            cagr_ok=cagr_ok,
            debt_ok=debt_ok,
            cfo_ok=cfo_ok,
            mcap_ok=mcap_ok,
            piotroski_ok=piotroski_ok,
            promoter_ok=promoter_ok,
            pledge_ok=pledge_ok,
            quality_score=quality_score,
            fail_reasons=fail_reasons,
            as_of=now_ts,
        )
        db.log_engine_event("INFO", "engines.multibagger.quality_filter", "quality filter applied", result.model_dump())
        return result


if __name__ == "__main__":
    print("QualityFilter ready")
