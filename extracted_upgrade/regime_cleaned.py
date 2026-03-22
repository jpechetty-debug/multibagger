"""Market regime detection."""

from __future__ import annotations

from typing import Any

from nsepython import indiavix, nse_get_advances_declines

import config
from data.db import db
from models.schemas import MarketRegime, RegimeResult


class RegimeDetector:
    """Detects the market regime from VIX and breadth."""

    def detect(self) -> RegimeResult:
        """Return the current regime."""

        vix_value = self._safe_float(indiavix())
        breadth_frame = nse_get_advances_declines(mode="pandas")
        advance_count = int((breadth_frame["pChange"].astype(float) > 0).sum()) if not breadth_frame.empty else 0
        decline_count = int((breadth_frame["pChange"].astype(float) < 0).sum()) if not breadth_frame.empty else 0
        breadth_ratio = (advance_count / decline_count) if decline_count else float(advance_count) if advance_count else None

        if vix_value is not None and breadth_ratio is not None and vix_value <= config.VIX_BULL_MAX and breadth_ratio >= config.BREADTH_BULL_MIN:
            regime = MarketRegime.BULL
            reason = "Low VIX with strong breadth"
        elif vix_value is not None and (vix_value >= config.VIX_BEAR_MIN or (breadth_ratio is not None and breadth_ratio <= config.BREADTH_BEAR_MAX)):
            regime = MarketRegime.BEAR
            reason = "Elevated VIX or weak breadth"
        elif (
            vix_value is not None
            and breadth_ratio is not None
            and config.VIX_BULL_MAX < vix_value <= config.VIX_SIDEWAYS_MAX
            and config.BREADTH_QUALITY_MIN <= breadth_ratio <= config.BREADTH_QUALITY_MAX
        ):
            regime = MarketRegime.QUALITY
            reason = "Balanced breadth with moderate VIX"
        else:
            regime = MarketRegime.SIDEWAYS
            reason = "Mixed breadth and volatility regime"

        # Safety: Ensure regime is a valid MarketRegime member
        if isinstance(regime, str):
            try:
                regime = MarketRegime(regime)
            except ValueError:
                regime = MarketRegime.QUALITY

        result = RegimeResult(
            regime=regime,
            india_vix=vix_value,
            breadth_ratio=breadth_ratio,
            advance_count=advance_count,
            decline_count=decline_count,
            reason=reason,
            as_of=int(__import__("time").time()),
        )
        db.log_engine_event("INFO", "engines.score_engine.regime", "regime detected", result.model_dump())
        return result

    def _safe_float(self, value: Any) -> float | None:
        """Convert a value to float when possible."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return None


if __name__ == "__main__":
    print(RegimeDetector().detect().model_dump())
