"""Breakout scanner: volume surge, 52-week high proximity, consolidation breakout."""

from __future__ import annotations

import time

import pandas as pd

import config
from data.db import db
from models.schemas import BreakoutResult


class BreakoutScanner:
    """Detect swing-relevant breakout patterns from price history."""

    def scan(self, ticker: str, price_df: pd.DataFrame) -> BreakoutResult:
        """Scan for volume surge, 52W high proximity, and consolidation breakout.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        price_df:
            DataFrame with ``Close`` and ``Volume`` columns (DatetimeIndex).
        """

        close = price_df["Close"].astype(float)
        volume = price_df["Volume"].astype(float)
        now_ts = int(time.time())

        # --- Volume surge: latest volume > 2× 50-day average ---
        volume_ratio: float | None = None
        volume_surge = False
        if len(volume) >= config.VOLUME_LONG_WINDOW:
            avg_vol = volume.rolling(config.VOLUME_LONG_WINDOW).mean().iloc[-1]
            if avg_vol and avg_vol > 0:
                volume_ratio = float(volume.iloc[-1] / avg_vol)
                volume_surge = volume_ratio >= config.SWING_VOLUME_SURGE_MULT

        # --- 52-week high proximity: price within 5% of 52W high ---
        pct_from_52w_high: float | None = None
        near_52w_high = False
        if len(close) >= 252:
            high_52w = close.iloc[-252:].max()
        elif len(close) >= 50:
            high_52w = close.max()
        else:
            high_52w = None
        if high_52w and high_52w > 0:
            pct_from_52w_high = float(1.0 - close.iloc[-1] / high_52w)
            near_52w_high = pct_from_52w_high <= config.SWING_52W_HIGH_PROXIMITY_PCT

        # --- Consolidation breakout: price breaks above recent range ---
        consolidation_breakout = False
        window = config.SWING_CONSOLIDATION_DAYS
        if len(close) > window + 1:
            recent_range = close.iloc[-(window + 1):-1]
            range_high = recent_range.max()
            range_low = recent_range.min()
            range_pct = (range_high - range_low) / range_low if range_low > 0 else 1.0
            if range_pct <= config.SWING_CONSOLIDATION_RANGE_PCT and close.iloc[-1] > range_high:
                consolidation_breakout = True

        # --- Composite breakout score (0-100) ---
        breakout_score = 0.0
        if volume_surge:
            breakout_score += 40.0
        if near_52w_high:
            breakout_score += 35.0
        if consolidation_breakout:
            breakout_score += 25.0

        result = BreakoutResult(
            ticker=ticker.strip().upper(),
            volume_surge=volume_surge,
            near_52w_high=near_52w_high,
            consolidation_breakout=consolidation_breakout,
            volume_ratio=volume_ratio,
            pct_from_52w_high=pct_from_52w_high,
            breakout_score=breakout_score,
            as_of=now_ts,
        )
        db.log_engine_event("INFO", "engines.swing.breakout_scanner", "breakout scan complete", result.model_dump())
        return result


if __name__ == "__main__":
    print("BreakoutScanner ready")
