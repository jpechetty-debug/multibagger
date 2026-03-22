"""
engines/regime/regime_engine.py
--------------------------------
Multi-signal market regime classifier for NSE Indian equities.
Now upgraded to 8-state tracking via RegimeTrackerV2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from engines.regime.regime_tracker import RegimeTrackerV2, RegimeV2Result


# ---------------------------------------------------------------------------
# Backward Compatible Result Container
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """Wrapper to maintain compatibility with legacy RegimeResult."""
    regime:            str
    composite_score:   float
    confidence:        float
    regime_changed:    bool
    score_threshold:   int
    position_multiplier: float
    config:            dict
    signals:           dict[str, float]
    timestamp:         int

    def to_payload(self) -> dict:
        return {
            "regime":            self.regime,
            "composite_score":   round(self.composite_score, 1),
            "confidence":        round(self.confidence, 3),
            "regime_changed":    self.regime_changed,
            "score_threshold":   self.score_threshold,
            "position_multiplier": self.position_multiplier,
            "signals":           self.signals,
            "timestamp":         self.timestamp,
        }


# ---------------------------------------------------------------------------
# Upgraded Regime Engine
# ---------------------------------------------------------------------------

class RegimeEngine:
    """
    Wrapper for RegimeTrackerV2 to maintain existing pipeline interface.
    """

    def __init__(self):
        self._tracker = RegimeTrackerV2()

    def classify(
        self,
        data: dict[str, Any],
        previous_regime: str | None = None,
        nifty_prices: pd.Series | None = None,
    ) -> RegimeResult:
        """
        Classifies market into 1 of 8 states.
        """
        res: RegimeV2Result = self._tracker.classify(
            market_data=data,
            nifty_prices=nifty_prices,
            prev_state=previous_regime
        )

        return RegimeResult(
            regime=res.state,
            composite_score=res.composite_score,
            confidence=res.confidence,
            regime_changed=res.state_changed,
            score_threshold=res.score_threshold,
            position_multiplier=res.position_multiplier,
            config=res.config,
            signals=res.signal_breakdown,
            timestamp=res.timestamp,
        )
