"""
Fractional Kelly position sizing as a post-MVO overlay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class KellyPosition:
    ticker:           str
    mvo_weight:       float      # weight from MVO optimizer
    kelly_weight:     float      # raw Kelly-implied weight (unnormalised)
    adjusted_weight:  float      # final weight after Kelly + normalisation
    win_probability:  float
    win_loss_ratio:   float
    full_kelly:       float      # f* before fraction applied
    fractional_kelly: float      # f* × kelly_fraction
    binding:          bool       # True if Kelly was the binding constraint vs MVO


@dataclass
class KellyResult:
    positions:         list[KellyPosition]
    kelly_fraction:    float
    total_kelly_weight: float    # sum before normalisation (< 1 = conservative regime)
    cash_implied:      float     # 1 - total_kelly_weight (uninvested)
    regime_note:       str

    def weights(self) -> dict[str, float]:
        return {p.ticker: p.adjusted_weight for p in self.positions}

    def summary(self) -> str:
        binding = sum(1 for p in self.positions if p.binding)
        return (
            f"Kelly overlay: {binding}/{len(self.positions)} positions "
            f"constrained by Kelly (not MVO). "
            f"Implied cash {self.cash_implied*100:.1f}%. "
            f"{self.regime_note}"
        )


# ---------------------------------------------------------------------------
# Kelly sizer
# ---------------------------------------------------------------------------

class KellySizer:
    """
    Fractional Kelly position sizing overlay for post-MVO weight adjustment.
    """

    def __init__(
        self,
        kelly_fraction:    float = 0.25,
        max_weight:        float = 0.08,
        min_weight:        float = 0.02,
        default_win_prob:  float = 0.52,
        default_wl_ratio:  float = 1.35,
        score_to_prob:     bool  = True,
    ):
        self.kelly_fraction   = kelly_fraction
        self.max_weight       = max_weight
        self.min_weight       = min_weight
        self.default_win_prob = default_win_prob
        self.default_wl_ratio = default_wl_ratio
        self.score_to_prob    = score_to_prob

    def adjust(
        self,
        weights:        dict[str, float],
        signal_records: list[dict[str, Any]],
        tracker=None,
        model_id:       str | None = None,
        window_days:    int = 60,
    ) -> KellyResult:
        score_map = {r["ticker"]: r.get("total_score", 60) for r in signal_records}
        stats     = self._get_tracker_stats(tracker, model_id, window_days) if tracker else {}

        positions: list[KellyPosition] = []
        raw_kelly_weights: dict[str, float] = {}

        for ticker, mvo_w in weights.items():
            if mvo_w <= 0:
                continue

            win_p, wl_r = self._estimate_edge(ticker, score_map.get(ticker, 60), stats)
            f_star       = self._full_kelly(win_p, wl_r)
            f_frac       = f_star * self.kelly_fraction
            f_capped     = min(max(f_frac, 0.0), self.max_weight)

            binding      = f_capped < mvo_w
            kelly_w      = f_capped if binding else mvo_w

            raw_kelly_weights[ticker] = kelly_w
            positions.append(KellyPosition(
                ticker=ticker, mvo_weight=round(mvo_w, 4),
                kelly_weight=round(kelly_w, 4),
                adjusted_weight=0.0,
                win_probability=round(win_p, 4),
                win_loss_ratio=round(wl_r, 4),
                full_kelly=round(f_star, 4),
                fractional_kelly=round(f_frac, 4),
                binding=binding,
            ))

        for pos in positions:
            if raw_kelly_weights[pos.ticker] < self.min_weight:
                raw_kelly_weights[pos.ticker] = 0.0

        total        = sum(raw_kelly_weights.values())
        cash_implied = max(0.0, 1.0 - total)

        for pos in positions:
            pos.adjusted_weight = round(raw_kelly_weights[pos.ticker], 4)

        if cash_implied > 0.30:
            note = "High implied cash — Kelly sees low edge across most positions."
        elif cash_implied > 0.15:
            note = "Moderate implied cash — Kelly trimming low-conviction positions."
        else:
            note = "Low implied cash — Kelly broadly agrees with MVO sizing."

        return KellyResult(
            positions=positions,
            kelly_fraction=self.kelly_fraction,
            total_kelly_weight=round(total, 4),
            cash_implied=round(cash_implied, 4),
            regime_note=note,
        )

    def _full_kelly(self, p: float, b: float) -> float:
        q = 1.0 - p
        if b <= 0 or p <= 0:
            return 0.0
        f = (p * b - q) / b
        return max(f, 0.0)

    def _estimate_edge(
        self,
        ticker: str,
        score: float,
        stats: dict[str, Any],
    ) -> tuple[float, float]:
        if stats and ticker in stats:
            s = stats[ticker]
            win_p = s.get("win_prob", self.default_win_prob)
            wl_r  = s.get("wl_ratio", self.default_wl_ratio)
            return _clamp(win_p, 0.35, 0.75), _clamp(wl_r, 0.5, 4.0)

        if self.score_to_prob and score is not None:
            win_p = 0.50 + (score - 50) / 40 * 0.12
            win_p = _clamp(win_p, 0.40, 0.65)
            wl_r  = 1.10 + (score - 50) / 40 * 0.60
            wl_r  = _clamp(wl_r, 0.90, 2.0)
            return win_p, wl_r

        return self.default_win_prob, self.default_wl_ratio

    def _get_tracker_stats(
        self,
        tracker,
        model_id: str | None,
        window_days: int,
    ) -> dict[str, dict[str, float]]:
        if tracker is None or model_id is None:
            return {}
        try:
            import sqlite3
            path = tracker.path
            con  = sqlite3.connect(path)
            con.row_factory = sqlite3.Row
            cutoff = _days_ago_str(window_days)
            rows   = con.execute(
                """SELECT ticker,
                          AVG(CASE WHEN hit=1 THEN 1.0 ELSE 0.0 END) AS win_prob,
                          AVG(CASE WHEN actual_return > 0 THEN actual_return ELSE NULL END)
                              / MAX(AVG(CASE WHEN actual_return < 0 THEN ABS(actual_return) ELSE NULL END), 0.001)
                              AS wl_ratio,
                          COUNT(*) AS n
                   FROM predictions
                   WHERE model_id=? AND actual_return IS NOT NULL
                   AND signal_date >= ? AND hit IS NOT NULL
                   GROUP BY ticker HAVING COUNT(*) >= 5""",
                (model_id, cutoff),
            ).fetchall()
            con.close()
            return {
                r["ticker"]: {"win_prob": r["win_prob"] or self.default_win_prob,
                              "wl_ratio": r["wl_ratio"] or self.default_wl_ratio}
                for r in rows
            }
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _days_ago_str(n: int) -> str:
    from datetime import date, timedelta
    return (date.today() - timedelta(days=n)).isoformat()
