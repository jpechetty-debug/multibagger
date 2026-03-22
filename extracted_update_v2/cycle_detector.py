"""
engines/analysis/cycle_detector.py
------------------------------------
NSE sector rotation and business cycle detector.

Identifies which phase of the economic cycle the market is in and
maps it to NSE sector favourability. Used by:
  - engines/multibagger/scorer.py (sector_cycle_stage dimension)
  - engines/regime/regime_tracker.py (sector_strength signal)
  - engines/score_engine/model.py (sector context)

NSE Cycle Model (4 phases)
---------------------------
EARLY     Recovery → capex revival → Financials, Auto, Industrials lead
MID       Expansion → broad participation → IT, Consumer, Capital Goods
LATE      Peak → inflation / commodities → Energy, Metals, Materials
RECESSION Contraction → defensives → FMCG, Healthcare, Pharma, IT (quality)

Detection signals
-----------------
1. Breadth trend       (advancing vs declining stocks, 20-day slope)
2. Sector RS rotation  (cyclicals vs defensives relative strength)
3. Credit conditions   (Gsec yield vs repo rate spread)
4. FII sector flow     (where foreign money is rotating into)
5. Earnings revision   (which sectors have accelerating EPS upgrades)

Confidence scoring
------------------
Each signal votes for a phase with a weight.
Phase with highest weighted vote wins.
Confidence = (winning_score - runner_up) / total_weight.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASES = ("EARLY", "MID", "LATE", "RECESSION")

PHASE_SECTOR_MAP = {
    "EARLY": {
        "leaders":  ["Financial Services", "Automobile and Auto Components",
                     "Capital Goods", "Realty", "Construction"],
        "laggards": ["Fast Moving Consumer Goods", "Healthcare", "Information Technology"],
        "description": "Recovery phase — capex and credit-sensitive sectors lead",
        "position_bias": 1.10,    # slight overweight vs neutral
    },
    "MID": {
        "leaders":  ["Information Technology", "Consumer Durables", "Capital Goods",
                     "Consumer Services", "Telecommunication"],
        "laggards": ["Oil Gas and Consumable Fuels", "Metals and Mining"],
        "description": "Expansion phase — broad participation, quality growth leads",
        "position_bias": 1.0,
    },
    "LATE": {
        "leaders":  ["Oil Gas and Consumable Fuels", "Metals and Mining",
                     "Chemicals", "Power"],
        "laggards": ["Financial Services", "Realty", "Consumer Durables"],
        "description": "Peak phase — commodity and inflation beneficiaries lead",
        "position_bias": 0.85,    # reduce size — cycle is maturing
    },
    "RECESSION": {
        "leaders":  ["Fast Moving Consumer Goods", "Healthcare",
                     "Pharmaceuticals", "Information Technology"],
        "laggards": ["Realty", "Metals and Mining", "Automobile and Auto Components"],
        "description": "Contraction phase — defensives and quality IT lead",
        "position_bias": 0.60,    # significant size reduction
    },
}

# Sector favourability score: +1 = leader, -1 = laggard, 0 = neutral
def sector_score(sector: str, phase: str) -> float:
    pm = PHASE_SECTOR_MAP.get(phase, {})
    if sector in pm.get("leaders", []):
        return 1.0
    if sector in pm.get("laggards", []):
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    phase:           str           # EARLY | MID | LATE | RECESSION
    confidence:      float         # 0.0 – 1.0
    signal_votes:    dict[str, str]  # {signal_name: voted_phase}
    signal_scores:   dict[str, float] # {signal_name: raw score}
    sector_leaders:  list[str]
    sector_laggards: list[str]
    position_bias:   float         # suggested position size multiplier
    description:     str
    timestamp:       int

    def to_dict(self) -> dict:
        return {
            "phase":          self.phase,
            "confidence":     round(self.confidence, 3),
            "signal_votes":   self.signal_votes,
            "sector_leaders": self.sector_leaders,
            "sector_laggards": self.sector_laggards,
            "position_bias":  self.position_bias,
            "description":    self.description,
            "timestamp":      self.timestamp,
        }

    def sector_multiplier(self, sector: str) -> float:
        """
        Return a position size multiplier for a given NSE sector.
        Leaders get 1.2×, laggards get 0.6×, neutral gets 1.0×.
        All scaled by position_bias.
        """
        score = sector_score(sector, self.phase)
        base  = 1.2 if score > 0 else 0.6 if score < 0 else 1.0
        return round(base * self.position_bias, 3)


# ---------------------------------------------------------------------------
# Cycle detector
# ---------------------------------------------------------------------------

class CycleDetector:
    """
    Detects the current NSE market cycle phase from macro and market signals.

    Usage
    -----
        detector = CycleDetector()
        result   = detector.detect(market_data)
        print(result.phase, result.confidence)
        print(result.sector_leaders)       # buy-side sectors
        mult = result.sector_multiplier("Information Technology")
    """

    # Signal weights (sum = 1.0)
    SIGNAL_WEIGHTS = {
        "breadth_trend":    0.25,
        "sector_rotation":  0.30,
        "credit_spread":    0.20,
        "fii_sector_flow":  0.15,
        "earnings_revision": 0.10,
    }

    def detect(self, market_data: dict[str, Any]) -> CycleResult:
        """
        Classify the current cycle phase from market data.

        market_data keys (all optional with sensible defaults):
          breadth_slope_20d     float   slope of advancing/declining ratio
          cyclical_vs_defensive float   RS of cyclicals vs defensives (%)
          gsec_10y              float   10-year Gsec yield
          repo_rate             float   RBI repo rate
          net_fii_20d_cr        float   net FII flows over 20 days (₹ cr)
          fii_cyclical_pct      float   % of FII flow into cyclical sectors
          eps_revision_cyclical float   EPS revision score for cyclicals
          eps_revision_defensive float  EPS revision score for defensives
        """
        votes:  dict[str, str]   = {}
        scores: dict[str, float] = {}

        # ── Signal 1: Breadth trend ────────────────────────────────────────
        breadth = float(market_data.get("breadth_slope_20d", 0.0))
        if breadth > 0.5:
            votes["breadth_trend"] = "EARLY"
            scores["breadth_trend"] = breadth
        elif breadth > 0.0:
            votes["breadth_trend"] = "MID"
            scores["breadth_trend"] = breadth
        elif breadth > -0.5:
            votes["breadth_trend"] = "LATE"
            scores["breadth_trend"] = abs(breadth)
        else:
            votes["breadth_trend"] = "RECESSION"
            scores["breadth_trend"] = abs(breadth)

        # ── Signal 2: Sector rotation ──────────────────────────────────────
        cyc_def = float(market_data.get("cyclical_vs_defensive", 0.0))
        if cyc_def > 5.0:
            votes["sector_rotation"] = "EARLY"
            scores["sector_rotation"] = cyc_def / 10
        elif cyc_def > 0.0:
            votes["sector_rotation"] = "MID"
            scores["sector_rotation"] = cyc_def / 10
        elif cyc_def > -3.0:
            votes["sector_rotation"] = "LATE"
            scores["sector_rotation"] = abs(cyc_def) / 10
        else:
            votes["sector_rotation"] = "RECESSION"
            scores["sector_rotation"] = abs(cyc_def) / 10

        # ── Signal 3: Credit spread ────────────────────────────────────────
        gsec  = float(market_data.get("gsec_10y", 7.0))
        repo  = float(market_data.get("repo_rate", 6.5))
        spread = gsec - repo
        if spread < 0.5:
            votes["credit_spread"] = "EARLY"      # tight spread = easy money
            scores["credit_spread"] = 1.0 - spread * 2
        elif spread < 1.0:
            votes["credit_spread"] = "MID"
            scores["credit_spread"] = 0.5
        elif spread < 1.5:
            votes["credit_spread"] = "LATE"
            scores["credit_spread"] = spread - 0.5
        else:
            votes["credit_spread"] = "RECESSION"  # wide spread = stress
            scores["credit_spread"] = spread - 1.0

        # ── Signal 4: FII sector flow ──────────────────────────────────────
        fii_flow   = float(market_data.get("net_fii_20d_cr", 0.0))
        fii_cyc_pct = float(market_data.get("fii_cyclical_pct", 50.0))
        if fii_flow > 2000 and fii_cyc_pct > 60:
            votes["fii_sector_flow"]  = "EARLY"
            scores["fii_sector_flow"] = fii_cyc_pct / 100
        elif fii_flow > 0 and fii_cyc_pct > 45:
            votes["fii_sector_flow"]  = "MID"
            scores["fii_sector_flow"] = fii_cyc_pct / 100
        elif fii_flow < 0 and fii_cyc_pct < 40:
            votes["fii_sector_flow"]  = "LATE"
            scores["fii_sector_flow"] = (100 - fii_cyc_pct) / 100
        else:
            votes["fii_sector_flow"]  = "RECESSION"
            scores["fii_sector_flow"] = (100 - fii_cyc_pct) / 100

        # ── Signal 5: Earnings revision ────────────────────────────────────
        eps_cyc = float(market_data.get("eps_revision_cyclical", 0.0))
        eps_def = float(market_data.get("eps_revision_defensive", 0.0))
        eps_diff = eps_cyc - eps_def
        if eps_diff > 5.0:
            votes["earnings_revision"]  = "EARLY"
            scores["earnings_revision"] = min(eps_diff / 20, 1.0)
        elif eps_diff > 0.0:
            votes["earnings_revision"]  = "MID"
            scores["earnings_revision"] = eps_diff / 20
        elif eps_diff > -5.0:
            votes["earnings_revision"]  = "LATE"
            scores["earnings_revision"] = abs(eps_diff) / 20
        else:
            votes["earnings_revision"]  = "RECESSION"
            scores["earnings_revision"] = min(abs(eps_diff) / 20, 1.0)

        # ── Weighted vote tally ────────────────────────────────────────────
        phase_weights: dict[str, float] = {p: 0.0 for p in PHASES}
        for sig, voted_phase in votes.items():
            w = self.SIGNAL_WEIGHTS.get(sig, 0.0)
            s = scores.get(sig, 0.5)
            phase_weights[voted_phase] += w * max(s, 0.1)

        total   = sum(phase_weights.values()) or 1.0
        winner  = max(phase_weights, key=phase_weights.get)  # type: ignore[arg-type]
        sorted_phases = sorted(phase_weights.items(), key=lambda x: x[1], reverse=True)
        runner_up_score = sorted_phases[1][1] if len(sorted_phases) > 1 else 0.0
        confidence = (phase_weights[winner] - runner_up_score) / total

        pm = PHASE_SECTOR_MAP[winner]
        return CycleResult(
            phase=winner,
            confidence=round(confidence, 3),
            signal_votes=votes,
            signal_scores={k: round(v, 3) for k, v in scores.items()},
            sector_leaders=pm["leaders"],
            sector_laggards=pm["laggards"],
            position_bias=pm["position_bias"],
            description=pm["description"],
            timestamp=int(time.time()),
        )

    def save_snapshot(self, result: CycleResult) -> None:
        """Persist cycle result to ops database."""
        try:
            from data.db import db
            db.save_market_snapshot("cycle_detector", result.to_dict(), result.timestamp)
        except Exception:
            pass

    def load_latest(self) -> CycleResult | None:
        """Load the most recent cycle snapshot from the database."""
        try:
            from data.db import db
            snap = db.get_latest_market_snapshot("cycle_detector")
            if not snap:
                return None
            p = snap.get("payload", {})
            pm = PHASE_SECTOR_MAP.get(p.get("phase", "MID"), PHASE_SECTOR_MAP["MID"])
            return CycleResult(
                phase=p.get("phase", "MID"),
                confidence=p.get("confidence", 0.5),
                signal_votes=p.get("signal_votes", {}),
                signal_scores={},
                sector_leaders=pm["leaders"],
                sector_laggards=pm["laggards"],
                position_bias=pm["position_bias"],
                description=pm["description"],
                timestamp=p.get("timestamp", int(time.time())),
            )
        except Exception:
            return None
