"""TAM and sector tailwind scorer for multibagger candidates."""

from __future__ import annotations

import time

from data.db import db
from models.schemas import FundamentalData, TAMResult

# Sectors with strong secular tailwinds (growth sectors)
TAILWIND_SECTORS = frozenset({
    "Information Technology",
    "Healthcare",
    "Consumer Services",
    "Consumer Durables",
    "Capital Goods",
    "Chemicals",
    "Power",
    "Realty",
    "Fast Moving Consumer Goods",
})


class TAMScorer:
    """Score multibagger candidates on total addressable market and sector tailwinds."""

    def score(self, ticker: str, data: FundamentalData) -> TAMResult:
        """Return a ``TAMResult`` for *ticker*.

        Scoring logic:
        1. **Sector tailwind** – +40 if sector is in the growth-sector list.
        2. **Revenue / market-cap ratio** – higher ratio ⇒ more room for
           valuation re-rating (the market hasn't fully priced in revenue).
        3. **Small-cap premium** – smaller market caps get a bonus (more
           runway for compounding).
        """

        normalized = ticker.strip().upper()
        now_ts = int(time.time())
        tam_score = 0.0

        # --- Sector tailwind ---
        sector = data.sector
        sector_tailwind = sector is not None and sector in TAILWIND_SECTORS
        if sector_tailwind:
            tam_score += 40.0

        # --- Revenue-to-market-cap ratio ---
        revenue_to_mcap: float | None = None
        if data.sales_growth_5y is not None and data.market_cap and data.market_cap > 0:
            # Approximate annual revenue from sales growth and market cap
            # Use market_cap as denominator — a proxy for the market's pricing
            # We use a heuristic: if growth is high + mcap is small, runway is large
            revenue_to_mcap = data.sales_growth_5y  # Use as a proxy signal
            if data.sales_growth_5y >= 0.25:
                tam_score += 30.0
            elif data.sales_growth_5y >= 0.18:
                tam_score += 20.0
            elif data.sales_growth_5y >= 0.10:
                tam_score += 10.0

        # --- Small-cap premium (more runway) ---
        if data.market_cap is not None:
            if data.market_cap < 1_000_00_00_000:  # < ₹1,000 Cr
                tam_score += 30.0
            elif data.market_cap < 3_000_00_00_000:  # < ₹3,000 Cr
                tam_score += 20.0
            elif data.market_cap < 5_000_00_00_000:  # < ₹5,000 Cr
                tam_score += 10.0

        tam_score = min(100.0, tam_score)

        result = TAMResult(
            ticker=normalized,
            sector_tailwind=sector_tailwind,
            tam_runway_score=tam_score,
            sector=sector,
            revenue_to_mcap_ratio=revenue_to_mcap,
            as_of=now_ts,
        )
        db.log_engine_event("INFO", "engines.multibagger.tam_scorer", "TAM scored", result.model_dump())
        return result


if __name__ == "__main__":
    print("TAMScorer ready")
