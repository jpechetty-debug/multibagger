"""
Upgraded 8-state regime classifier with volatility and risk awareness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# State definitions
# ---------------------------------------------------------------------------

STATES = ("BULL", "BEAR", "SIDEWAYS", "LOW_VOL", "HIGH_VOL", "CRISIS", "RISK_ON", "RISK_OFF")

STATE_CONFIG = {
    "BULL":     {"score_threshold": 75, "position_multiplier": 1.00, "stop_multiplier": 1.0,
                 "favour": ["Financials","IT","Auto","Infra"], "avoid": ["Gold","FMCG"]},
    "BEAR":     {"score_threshold": 55, "position_multiplier": 0.25, "stop_multiplier": 0.8,
                 "favour": ["FMCG","Healthcare","Gold"], "avoid": ["Metals","Realty","Auto"]},
    "SIDEWAYS": {"score_threshold": 65, "position_multiplier": 0.55, "stop_multiplier": 1.0,
                 "favour": ["FMCG","Healthcare","IT"], "avoid": ["Metals","Realty"]},
    "LOW_VOL":  {"score_threshold": 68, "position_multiplier": 0.80, "stop_multiplier": 0.8,
                 "favour": ["IT","Financials","Consumer"], "avoid": []},
    "HIGH_VOL": {"score_threshold": 70, "position_multiplier": 0.50, "stop_multiplier": 1.5,
                 "favour": ["Healthcare","FMCG"], "avoid": ["Small Cap","Realty"]},
    "CRISIS":   {"score_threshold": 80, "position_multiplier": 0.10, "stop_multiplier": 0.6,
                 "favour": ["Gold","Cash"], "avoid": ["everything else"]},
    "RISK_ON":  {"score_threshold": 72, "position_multiplier": 0.90, "stop_multiplier": 1.1,
                 "favour": ["Auto","Metals","Infra","PSU"], "avoid": ["FMCG","Pharma"]},
    "RISK_OFF": {"score_threshold": 68, "position_multiplier": 0.60, "stop_multiplier": 0.9,
                 "favour": ["FMCG","Pharma","IT"], "avoid": ["PSU","Metals","Realty"]},
}

# Crisis detection thresholds
CRISIS_VIX_THRESHOLD      = 30.0
CRISIS_BREADTH_THRESHOLD  = 25.0   # % stocks above 200DMA
CRISIS_FII_THRESHOLD      = -3000  # ₹ crore net selling (20-day)

# Volatility regime thresholds
LOW_VOL_ATR_PCT   = 0.008   # < 0.8% daily ATR = low vol
HIGH_VOL_ATR_PCT  = 0.018   # > 1.8% daily ATR = high vol


# ---------------------------------------------------------------------------
# Volatility model
# ---------------------------------------------------------------------------

@dataclass
class VolState:
    realised_vol_10d:  float     # annualised
    realised_vol_30d:  float     # annualised
    vol_of_vol:        float     # std of rolling 10d vol
    atr_pct:           float     # ATR as % of price
    vol_regime:        str       # "LOW" | "NORMAL" | "HIGH" | "EXTREME"
    vol_trend:         str       # "EXPANDING" | "CONTRACTING" | "STABLE"
    regime_stable:     bool      # True if vol regime unchanged for 5+ days


class VolatilityModel:
    """
    Computes current volatility regime from Nifty 50 price series.
    """

    TRADING_DAYS = 252

    def compute(self, prices: pd.Series) -> VolState:
        if len(prices) < 35:
            return VolState(0.15, 0.15, 0.02, 0.01, "NORMAL", "STABLE", True)

        ret = prices.pct_change().dropna()

        # Realised vols
        vol_10 = float(ret.tail(10).std() * np.sqrt(self.TRADING_DAYS))
        vol_30 = float(ret.tail(30).std() * np.sqrt(self.TRADING_DAYS))

        # Vol-of-vol: rolling std of 10-day vol over past 60 days
        if len(ret) >= 60:
            rolling_vol = ret.rolling(10).std() * np.sqrt(self.TRADING_DAYS)
            vov = float(rolling_vol.tail(60).std())
        else:
            vov = 0.02

        # ATR% approximation from daily returns
        atr_pct = float(ret.tail(14).abs().mean())

        # Regime
        if atr_pct < LOW_VOL_ATR_PCT:
            regime = "LOW"
        elif atr_pct > HIGH_VOL_ATR_PCT * 1.5:
            regime = "EXTREME"
        elif atr_pct > HIGH_VOL_ATR_PCT:
            regime = "HIGH"
        else:
            regime = "NORMAL"

        # Vol trend
        if vol_10 > vol_30 * 1.15:
            trend = "EXPANDING"
        elif vol_10 < vol_30 * 0.85:
            trend = "CONTRACTING"
        else:
            trend = "STABLE"

        # Stability: check if past 5 days all in same regime
        if len(ret) >= 15:
            recent_atrs = [float(ret.iloc[i:i+5].abs().mean()) for i in range(-15, -5, 5)]
            recent_regimes = [
                "LOW" if a < LOW_VOL_ATR_PCT else "HIGH" if a > HIGH_VOL_ATR_PCT else "NORMAL"
                for a in recent_atrs
            ]
            stable = len(set(recent_regimes)) == 1
        else:
            stable = True

        return VolState(vol_10, vol_30, vov, atr_pct, regime, trend, stable)


# ---------------------------------------------------------------------------
# 8-state regime tracker
# ---------------------------------------------------------------------------

@dataclass
class RegimeV2Result:
    state:             str
    prev_state:        str | None
    state_changed:     bool
    composite_score:   float        # 0–100 bullishness score
    vol_state:         VolState
    config:            dict
    signal_breakdown:  dict[str, float]
    confidence:        float
    days_in_state:     int
    timestamp:         int

    @property
    def position_multiplier(self) -> float:
        return self.config["position_multiplier"]

    @property
    def score_threshold(self) -> int:
        return self.config["score_threshold"]

    def to_payload(self) -> dict:
        return {
            "state":              self.state,
            "prev_state":         self.prev_state,
            "state_changed":      self.state_changed,
            "composite_score":    round(self.composite_score, 1),
            "confidence":         round(self.confidence, 3),
            "days_in_state":      self.days_in_state,
            "position_multiplier": self.position_multiplier,
            "score_threshold":    self.score_threshold,
            "favour":             self.config["favour"],
            "avoid":              self.config["avoid"],
            "vol_regime":         self.vol_state.vol_regime,
            "vol_trend":          self.vol_state.vol_trend,
            "atr_pct":            round(self.vol_state.atr_pct * 100, 3),
            "signals":            self.signal_breakdown,
            "timestamp":          self.timestamp,
        }


class RegimeTrackerV2:
    """
    8-state regime classifier.
    """

    def __init__(self):
        self._vol_model    = VolatilityModel()
        self._state_history: list[str] = []

    def classify(
        self,
        market_data: dict[str, Any],
        nifty_prices: pd.Series | None = None,
        prev_state: str | None = None,
    ) -> RegimeV2Result:

        # ── Vol state ─────────────────────────────────────────────────────────
        vol = self._vol_model.compute(nifty_prices) if nifty_prices is not None \
              else VolState(0.15, 0.15, 0.02, 0.012, "NORMAL", "STABLE", True)

        # ── Raw signal scores (0–100 bullish) ─────────────────────────────────
        signals = self._score_signals(market_data, vol)
        composite = float(np.average(
            list(signals.values()),
            weights=[0.22, 0.18, 0.16, 0.14, 0.10, 0.08, 0.07, 0.05],
        ))

        # ── State classification ───────────────────────────────────────────────
        state = self._classify(composite, signals, vol, market_data, prev_state)

        self._state_history.append(state)
        if len(self._state_history) > 60:
            self._state_history.pop(0)

        days_in = sum(1 for s in reversed(self._state_history) if s == state)
        changed = prev_state is not None and state != prev_state
        conf    = self._confidence(composite, state)

        return RegimeV2Result(
            state=state, prev_state=prev_state, state_changed=changed,
            composite_score=composite, vol_state=vol,
            config=STATE_CONFIG[state], signal_breakdown=signals,
            confidence=conf, days_in_state=days_in, timestamp=int(time.time()),
        )

    def _score_signals(self, data: dict, vol: VolState) -> dict[str, float]:
        def s(v, lo, hi): return float(max(0, min(100, (v - lo) / (hi - lo) * 100)))

        nifty  = data.get("nifty_close", 23000)
        sma50  = data.get("nifty_sma50",  22000)
        sma200 = data.get("nifty_sma200", 21000)

        return {
            "nifty_trend":       s((nifty - sma200) / sma200 * 100, -15, 15),
            "market_breadth":    s(data.get("pct_above_200dma", 50), 20, 75),
            "vix_signal":        s(data.get("india_vix", 17), 30, 10),
            "fii_flow":          s(data.get("net_fii_20d_cr", 0), -5000, 10000),
            "advance_decline":   s(data.get("ad_ratio_10d_avg", 1.0), 0.4, 2.5),
            "sector_strength":   s(data.get("cyclical_vs_defensive", 0), -8, 8),
            "credit_spread":     s(data.get("gsec_10y", 7.0) - data.get("repo_rate", 6.5), 3.0, 0.5),
            "momentum_breadth":  s(data.get("pct_rsi_above_50", 50), 25, 70),
        }

    def _classify(
        self,
        composite: float,
        signals: dict[str, float],
        vol: VolState,
        data: dict,
        prev: str | None,
    ) -> str:
        # ── Crisis detection (highest priority) ───────────────────────────────
        vix      = data.get("india_vix", 15)
        breadth  = data.get("pct_above_200dma", 50)
        fii_flow = data.get("net_fii_20d_cr", 0)
        if (vix > CRISIS_VIX_THRESHOLD
                and breadth < CRISIS_BREADTH_THRESHOLD
                and fii_flow < CRISIS_FII_THRESHOLD):
            return "CRISIS"

        # ── Volatility regime override ─────────────────────────────────────────
        if vol.vol_regime == "EXTREME":
            return "HIGH_VOL"
        if vol.vol_regime == "HIGH" and composite < 65:
            return "HIGH_VOL"
        if vol.vol_regime == "LOW" and 45 <= composite <= 68:
            return "LOW_VOL"

        # ── Risk appetite (sector rotation signal) ────────────────────────────
        sector_signal = signals.get("sector_strength", 50)
        if composite >= 60 and sector_signal >= 70:
            return "RISK_ON"
        if composite <= 55 and sector_signal <= 35:
            return "RISK_OFF"

        # ── Core 4-state with hysteresis ──────────────────────────────────────
        HYSTERESIS = 5.0
        if composite >= 72:    candidate = "BULL"
        elif composite >= 58:  candidate = "SIDEWAYS"
        elif composite >= 42:  candidate = "SIDEWAYS"
        else:                  candidate = "BEAR"

        if prev and prev in STATES and prev != candidate:
            mid_scores = {"BULL": 80, "SIDEWAYS": 55, "BEAR": 35,
                          "LOW_VOL": 55, "HIGH_VOL": 55, "CRISIS": 20,
                          "RISK_ON": 68, "RISK_OFF": 50}
            dist = abs(composite - (mid_scores.get(prev, 50) + mid_scores.get(candidate, 50)) / 2)
            if dist < HYSTERESIS:
                return prev

        return candidate

    def _confidence(self, score: float, state: str) -> float:
        boundaries = [35, 55, 72]
        min_dist = min(abs(score - b) for b in boundaries)
        return round(min(0.5 + min_dist / 25, 1.0), 3)
