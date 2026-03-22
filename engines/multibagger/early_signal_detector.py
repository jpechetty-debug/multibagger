"""Early signal detector for multibagger candidates: promoter buying, FII entry, earnings beats."""

from __future__ import annotations

import time

from data.db import db
from models.schemas import EarlySignalResult, FundamentalData


class EarlySignalDetector:
    """Detect early institutional and insider accumulation signals."""

    def detect(self, ticker: str, data: FundamentalData) -> EarlySignalResult:
        """Return an ``EarlySignalResult`` for *ticker*.

        Checks:
        1. **Promoter buying** – promoter_pct >= 50% suggests strong insider conviction.
        2. **FII entry** – positive fii_delta indicates institutional interest.
        3. **Earnings beats** – piotroski_score >= 7 (as a proxy for strong recent fundamentals).
        """

        normalized = ticker.strip().upper()
        now_ts = int(time.time())
        signals: list[str] = []
        score = 0.0

        # --- Promoter buying trend ---
        promoter_buying = False
        if data.promoter_pct is not None and data.promoter_pct >= 50.0:
            promoter_buying = True
            signals.append(f"Promoter holding strong at {data.promoter_pct:.1f}%")
            score += 30.0

        # --- FII entry (positive delta) ---
        fii_entry = False
        if data.fii_delta is not None and data.fii_delta > 0:
            fii_entry = True
            signals.append(f"FII accumulating (+{data.fii_delta:.2f}%)")
            score += 35.0

        # --- DII support ---
        if data.dii_delta is not None and data.dii_delta > 0:
            signals.append(f"DII support (+{data.dii_delta:.2f}%)")
            score += 10.0

        # --- Earnings beat proxy (strong Piotroski) ---
        earnings_beats = 0
        if data.piotroski_score is not None:
            if data.piotroski_score >= 8:
                earnings_beats = 4
                signals.append(f"Piotroski {data.piotroski_score}/9 — exceptional")
                score += 25.0
            elif data.piotroski_score >= 7:
                earnings_beats = 3
                signals.append(f"Piotroski {data.piotroski_score}/9 — strong")
                score += 15.0
            elif data.piotroski_score >= 6:
                earnings_beats = 2
                signals.append(f"Piotroski {data.piotroski_score}/9 — decent")
                score += 5.0

        # --- EPS growth as a supplemental beat signal ---
        if data.eps_growth_ttm is not None and data.eps_growth_ttm > 0.20:
            earnings_beats += 1
            signals.append(f"EPS growth {data.eps_growth_ttm:.0%} — strong")
            score += 10.0

        score = min(100.0, score)

        result = EarlySignalResult(
            ticker=normalized,
            promoter_buying=promoter_buying,
            fii_entry=fii_entry,
            earnings_beats=earnings_beats,
            early_signal_score=score,
            signals=signals,
            as_of=now_ts,
        )
        db.log_engine_event("INFO", "engines.multibagger.early_signal_detector", "early signals detected", result.model_dump())
        return result


if __name__ == "__main__":
    print("EarlySignalDetector ready")
