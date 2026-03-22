"""
engines/multibagger/scorer.py
------------------------------
Institutional multibagger scoring engine.
7 dimensions → weighted composite → conviction tier.

Architecture
------------
Each dimension is a standalone scorer returning a 0–100 score
and a breakdown dict for UI transparency. The composite scorer
aggregates with configurable weights and applies a conviction gate
(minimum score per dimension) before awarding MULTIBAGGER status.

Dimension weights (sum to 1.0)
-------------------------------
Quality      0.22   — ROIC trend, EPS stability, FCF, debt trend
Growth       0.20   — revenue CAGR, EPS CAGR, TAM runway
Valuation    0.18   — percentile rank, margin of safety, FCF yield
Ownership    0.15   — promoter pct, pledge, FII delta, insider buying
Momentum     0.10   — price vs 200DMA, relative strength, sector RS
Cycle        0.08   — sector cycle stage, macro tailwind
Risk         0.07   — debt-to-equity trend, earnings variability, governance

Conviction tiers
----------------
ELITE        ≥ 85 composite AND all dimensions ≥ 55
STRONG       ≥ 72 composite AND all dimensions ≥ 40
WATCH        ≥ 58 composite
REJECT       < 58 composite OR any critical gate failed
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS = {
    "quality":   0.22,
    "growth":    0.20,
    "valuation": 0.18,
    "ownership": 0.15,
    "momentum":  0.10,
    "cycle":     0.08,
    "risk":      0.07,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

CONVICTION_TIERS = {
    "ELITE":  {"min_composite": 85, "min_dimension": 55},
    "STRONG": {"min_composite": 72, "min_dimension": 40},
    "WATCH":  {"min_composite": 58, "min_dimension": 0},
    "REJECT": {"min_composite": 0,  "min_dimension": 0},
}

# Critical gates — any gate failure → REJECT regardless of composite score
CRITICAL_GATES = {
    "pledge_pct_max":   15.0,   # promoter pledge > 15% → reject
    "debt_to_equity_max": 3.0,  # D/E > 3 → reject
    "min_roic":         8.0,    # ROIC < 8% → reject (capital destroyers)
    "min_promoter_pct": 25.0,   # promoter holding < 25% → reject
}

# Sector cycle stages — maps sector to current cycle stage
# Update this from macro research, not programmatically
SECTOR_CYCLE = {
    "Financials":    "expansion",
    "IT":            "late_expansion",
    "Healthcare":    "early_expansion",
    "Energy":        "contraction",
    "FMCG":          "recovery",
    "Auto":          "recovery",
    "Metals":        "contraction",
    "Infra":         "early_expansion",
    "Telecom":       "expansion",
    "Chemicals":     "recovery",
    "Defence":       "early_expansion",
    "Realty":        "expansion",
}

CYCLE_SCORE_MAP = {
    "early_expansion": 90,
    "expansion":       75,
    "late_expansion":  50,
    "contraction":     20,
    "recovery":        65,
}


# ---------------------------------------------------------------------------
# Score result container
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    name: str
    score: float             # 0–100
    weight: float
    weighted: float          # score × weight
    breakdown: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)   # warnings or notable signals


@dataclass
class MultibaggerResult:
    ticker: str
    composite: float
    tier: str
    dimensions: dict[str, DimensionScore]
    gates_passed: bool
    gate_failures: list[str]
    narrative: str           # one-sentence human-readable summary

    @property
    def is_multibagger(self) -> bool:
        return self.tier in ("ELITE", "STRONG")

    def to_dict(self) -> dict:
        return {
            "ticker":       self.ticker,
            "composite":    round(self.composite, 1),
            "tier":         self.tier,
            "is_multibagger": self.is_multibagger,
            "gates_passed": self.gates_passed,
            "gate_failures": self.gate_failures,
            "narrative":    self.narrative,
            "dimensions": {
                name: {
                    "score":    round(d.score, 1),
                    "weighted": round(d.weighted, 2),
                    "breakdown": d.breakdown,
                    "flags":    d.flags,
                }
                for name, d in self.dimensions.items()
            },
        }


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------

class QualityScorer:
    """
    ROIC trend, EPS stability, free cash flow, debt trend.
    These are the hardest signals to fake — focus of the engine.
    """
    def score(self, data: dict) -> DimensionScore:
        scores, flags, breakdown = [], [], {}

        # ── ROIC trend (3-year) ──────────────────────────────────────────────
        roic_vals = data.get("roic_history", [])  # list of annual ROIC values, oldest first
        roic_now  = data.get("roic_current")
        if roic_now is not None:
            roic_s = _clamp(_linear_scale(roic_now, lo=0, hi=30), 0, 100)
            scores.append(roic_s * 0.35)
            breakdown["roic_current"] = round(roic_now, 2)
            if roic_now < CRITICAL_GATES["min_roic"]:
                flags.append(f"ROIC {roic_now:.1f}% below minimum threshold")
        if len(roic_vals) >= 3:
            roic_trend = _trend_slope(roic_vals[-4:] if len(roic_vals) >= 4 else roic_vals)
            trend_s = _clamp(50 + roic_trend * 10, 0, 100)
            scores.append(trend_s * 0.20)
            breakdown["roic_trend"] = "improving" if roic_trend > 0.5 else "stable" if roic_trend > -0.5 else "declining"
            if roic_trend < -1.0:
                flags.append("ROIC deteriorating over 3 years")

        # ── EPS growth stability ─────────────────────────────────────────────
        eps_history = data.get("eps_history", [])  # annual EPS, oldest first
        if len(eps_history) >= 4:
            # Penalise volatility — consistent growth beats lumpy growth
            eps_growth_rates = [
                (eps_history[i] - eps_history[i-1]) / abs(eps_history[i-1])
                for i in range(1, len(eps_history))
                if eps_history[i-1] != 0
            ]
            if eps_growth_rates:
                mean_growth = np.mean(eps_growth_rates)
                std_growth  = np.std(eps_growth_rates)
                stability_ratio = mean_growth / (std_growth + 0.01)
                stability_s = _clamp(_linear_scale(stability_ratio, lo=-1, hi=3), 0, 100)
                scores.append(stability_s * 0.25)
                breakdown["eps_mean_growth_pct"] = round(mean_growth * 100, 1)
                breakdown["eps_stability"]       = round(stability_ratio, 2)
                if std_growth > 0.4:
                    flags.append("High EPS variability — lumpy earnings")

        # ── Free cash flow yield and trend ───────────────────────────────────
        fcf_yield   = data.get("fcf_yield")       # FCF / market cap
        fcf_history = data.get("fcf_history", []) # annual FCF ₹ cr, oldest first
        if fcf_yield is not None:
            fcf_s = _clamp(_linear_scale(fcf_yield * 100, lo=-2, hi=8), 0, 100)
            scores.append(fcf_s * 0.12)
            breakdown["fcf_yield_pct"] = round(fcf_yield * 100, 2)
            if fcf_yield < 0:
                flags.append("Negative FCF yield")
        if len(fcf_history) >= 3:
            fcf_positive_yrs = sum(1 for f in fcf_history[-4:] if f > 0)
            fcf_consistency  = fcf_positive_yrs / min(len(fcf_history), 4)
            scores.append(fcf_consistency * 100 * 0.08)
            breakdown["fcf_positive_years_of_4"] = fcf_positive_yrs

        # ── Debt trend ───────────────────────────────────────────────────────
        de_now      = data.get("debt_to_equity")
        de_history  = data.get("de_history", [])
        if de_now is not None:
            # Lower D/E = higher score
            de_s = _clamp(_linear_scale(de_now, lo=3, hi=0, invert=True), 0, 100)
            scores.append(de_s * 0.10)
            breakdown["debt_to_equity"] = round(de_now, 2)
            if de_now > CRITICAL_GATES["debt_to_equity_max"]:
                flags.append(f"D/E {de_now:.1f}x exceeds threshold")
        if len(de_history) >= 3:
            de_trend = _trend_slope(de_history[-3:])
            if de_trend < -0.1:
                flags.append("Debt reduction trend — positive signal")
            breakdown["de_trend"] = "reducing" if de_trend < -0.1 else "stable" if de_trend < 0.1 else "increasing"

        final = sum(scores) if scores else 0.0
        return DimensionScore(
            name="quality", score=final, weight=WEIGHTS["quality"],
            weighted=final * WEIGHTS["quality"], breakdown=breakdown, flags=flags,
        )


class GrowthScorer:
    """Revenue CAGR, EPS CAGR, forward guidance, TAM runway."""

    def score(self, data: dict) -> DimensionScore:
        scores, flags, breakdown = [], [], {}

        # ── Revenue CAGR (3yr) ───────────────────────────────────────────────
        rev_cagr = data.get("revenue_cagr_3y")
        if rev_cagr is not None:
            rev_s = _clamp(_linear_scale(rev_cagr * 100, lo=0, hi=30), 0, 100)
            scores.append(rev_s * 0.30)
            breakdown["revenue_cagr_3y_pct"] = round(rev_cagr * 100, 1)
            if rev_cagr > 0.20:
                flags.append(f"High-growth revenue trajectory {rev_cagr*100:.0f}% CAGR")

        # ── EPS CAGR (3yr) ───────────────────────────────────────────────────
        eps_cagr = data.get("eps_cagr_3y")
        if eps_cagr is not None:
            eps_s = _clamp(_linear_scale(eps_cagr * 100, lo=0, hi=35), 0, 100)
            scores.append(eps_s * 0.30)
            breakdown["eps_cagr_3y_pct"] = round(eps_cagr * 100, 1)

        # ── Forward EPS growth (consensus) ───────────────────────────────────
        fwd_eps_growth = data.get("fwd_eps_growth_pct")
        if fwd_eps_growth is not None:
            fwd_s = _clamp(_linear_scale(fwd_eps_growth, lo=0, hi=30), 0, 100)
            scores.append(fwd_s * 0.20)
            breakdown["fwd_eps_growth_pct"] = round(fwd_eps_growth, 1)

        # ── TAM runway ───────────────────────────────────────────────────────
        tam_runway = data.get("tam_runway_score")  # 0–100 from existing TAM engine
        if tam_runway is not None:
            scores.append(tam_runway * 0.20)
            breakdown["tam_runway"] = round(tam_runway, 0)
            if tam_runway >= 75:
                flags.append("Large addressable market with strong runway")

        final = sum(scores) if scores else 0.0
        return DimensionScore(
            name="growth", score=final, weight=WEIGHTS["growth"],
            weighted=final * WEIGHTS["growth"], breakdown=breakdown, flags=flags,
        )


class ValuationScorer:
    """
    Valuation percentile vs sector peers, margin of safety, FCF yield.
    Avoids the trap of absolute P/E — uses relative positioning instead.
    """

    def score(self, data: dict) -> DimensionScore:
        scores, flags, breakdown = [], [], {}

        # ── Valuation percentile vs sector ───────────────────────────────────
        # 0 = cheapest in sector, 100 = most expensive
        val_pct = data.get("valuation_percentile")
        if val_pct is not None:
            # Invert: cheap = high score
            val_s = _clamp(100 - val_pct, 0, 100)
            scores.append(val_s * 0.40)
            breakdown["valuation_percentile"] = round(val_pct, 1)
            breakdown["vs_sector"] = "cheap" if val_pct < 30 else "fair" if val_pct < 70 else "expensive"
            if val_pct < 20:
                flags.append("Top quartile value vs sector peers")
            if val_pct > 85:
                flags.append("Expensive vs sector — requires exceptional growth")

        # ── DCF margin of safety ─────────────────────────────────────────────
        mos = data.get("margin_of_safety_pct")
        if mos is not None:
            mos_s = _clamp(_linear_scale(mos, lo=-20, hi=40), 0, 100)
            scores.append(mos_s * 0.35)
            breakdown["margin_of_safety_pct"] = round(mos, 1)
            if mos > 25:
                flags.append(f"Strong DCF margin of safety {mos:.0f}%")

        # ── FCF yield ────────────────────────────────────────────────────────
        fcf_yield = data.get("fcf_yield")
        if fcf_yield is not None:
            fcf_s = _clamp(_linear_scale(fcf_yield * 100, lo=0, hi=8), 0, 100)
            scores.append(fcf_s * 0.25)
            breakdown["fcf_yield_pct"] = round(fcf_yield * 100, 2)

        final = sum(scores) if scores else 0.0
        return DimensionScore(
            name="valuation", score=final, weight=WEIGHTS["valuation"],
            weighted=final * WEIGHTS["valuation"], breakdown=breakdown, flags=flags,
        )


class OwnershipScorer:
    """
    Promoter holding, pledge, FII delta, DII delta, insider buying signal.
    Insider buying is the highest-conviction signal in this dimension.
    """

    def score(self, data: dict) -> DimensionScore:
        scores, flags, breakdown = [], [], {}

        # ── Promoter holding ─────────────────────────────────────────────────
        promoter = data.get("promoter_pct")
        if promoter is not None:
            pro_s = _clamp(_linear_scale(promoter, lo=25, hi=75), 0, 100)
            scores.append(pro_s * 0.30)
            breakdown["promoter_pct"] = round(promoter, 1)
            if promoter < CRITICAL_GATES["min_promoter_pct"]:
                flags.append(f"Low promoter holding {promoter:.1f}%")
            if promoter >= 60:
                flags.append("High promoter conviction")

        # ── Pledge ───────────────────────────────────────────────────────────
        pledge = data.get("pledge_pct", 0)
        pledge_s = _clamp(100 - (pledge / CRITICAL_GATES["pledge_pct_max"]) * 100, 0, 100)
        scores.append(pledge_s * 0.20)
        breakdown["pledge_pct"] = round(pledge, 1)
        if pledge > CRITICAL_GATES["pledge_pct_max"]:
            flags.append(f"High pledge {pledge:.1f}% — structural risk")
        elif pledge == 0:
            flags.append("Zero pledge — clean ownership")

        # ── FII delta (quarterly change in FII holding) ───────────────────────
        fii_delta = data.get("fii_delta")
        if fii_delta is not None:
            fii_s = _clamp(50 + fii_delta * 500, 0, 100)  # +1% delta → +50 score
            scores.append(fii_s * 0.20)
            breakdown["fii_delta_pct"] = round(fii_delta * 100, 2)
            if fii_delta > 0.01:
                flags.append("FII accumulation signal")
            elif fii_delta < -0.01:
                flags.append("FII distribution — monitor closely")

        # ── DII delta ────────────────────────────────────────────────────────
        dii_delta = data.get("dii_delta")
        if dii_delta is not None:
            dii_s = _clamp(50 + dii_delta * 400, 0, 100)
            scores.append(dii_s * 0.10)
            breakdown["dii_delta_pct"] = round(dii_delta * 100, 2)

        # ── Insider buying ───────────────────────────────────────────────────
        # High-conviction signal: director/promoter open-market purchases
        insider_buys_90d = data.get("insider_buys_90d", 0)   # count of buy transactions
        insider_qty_pct  = data.get("insider_qty_pct", 0.0)  # shares bought as % of float
        if insider_buys_90d > 0:
            insider_s = _clamp(min(insider_buys_90d * 15, 70) + insider_qty_pct * 3000, 0, 100)
            scores.append(insider_s * 0.20)
            breakdown["insider_buys_90d"]  = insider_buys_90d
            breakdown["insider_qty_pct"]   = round(insider_qty_pct * 100, 3)
            flags.append(f"Insider buying: {insider_buys_90d} transactions in 90 days")
        else:
            scores.append(50 * 0.20)  # neutral if no data

        final = sum(scores) if scores else 0.0
        return DimensionScore(
            name="ownership", score=final, weight=WEIGHTS["ownership"],
            weighted=final * WEIGHTS["ownership"], breakdown=breakdown, flags=flags,
        )


class MomentumScorer:
    """Price vs 200DMA, relative strength vs Nifty, sector RS rank."""

    def score(self, data: dict) -> DimensionScore:
        scores, flags, breakdown = [], [], {}

        # ── Price vs 200 DMA ─────────────────────────────────────────────────
        vs_200dma = data.get("price_vs_200dma_pct")
        if vs_200dma is not None:
            # Prefer stocks above 200DMA but not >30% extended
            if vs_200dma > 30:
                mom_s = _clamp(100 - (vs_200dma - 30) * 2, 40, 100)
                flags.append(f"Extended {vs_200dma:.0f}% above 200DMA")
            elif vs_200dma > 0:
                mom_s = _clamp(60 + vs_200dma, 0, 100)
            else:
                mom_s = _clamp(60 + vs_200dma * 2, 0, 60)  # penalise below 200DMA
            scores.append(mom_s * 0.35)
            breakdown["price_vs_200dma_pct"] = round(vs_200dma, 1)

        # ── Relative strength vs Nifty 50 (3-month) ──────────────────────────
        rs_3m = data.get("relative_strength_3m")
        if rs_3m is not None:
            rs_s = _clamp(50 + rs_3m * 200, 0, 100)
            scores.append(rs_s * 0.35)
            breakdown["relative_strength_3m"] = round(rs_3m, 3)
            if rs_3m > 0.05:
                flags.append("Outperforming Nifty over 3 months")

        # ── Sector RS rank (percentile within sector) ─────────────────────────
        sector_rank_pct = data.get("rank_percentile")
        if sector_rank_pct is not None:
            scores.append(sector_rank_pct * 0.30)
            breakdown["sector_rank_percentile"] = round(sector_rank_pct, 1)
            if sector_rank_pct >= 80:
                flags.append("Top quintile within sector")

        final = sum(scores) if scores else 0.0
        return DimensionScore(
            name="momentum", score=final, weight=WEIGHTS["momentum"],
            weighted=final * WEIGHTS["momentum"], breakdown=breakdown, flags=flags,
        )


class CycleScorer:
    """
    Sector cycle positioning + macro tailwinds.
    Early expansion is the sweet spot for multibaggers.
    """

    def score(self, data: dict) -> DimensionScore:
        scores, flags, breakdown = [], [], {}

        sector = data.get("sector", "")
        cycle_stage = SECTOR_CYCLE.get(sector, "expansion")
        cycle_s = CYCLE_SCORE_MAP.get(cycle_stage, 50)
        scores.append(cycle_s * 0.55)
        breakdown["sector"]      = sector
        breakdown["cycle_stage"] = cycle_stage
        if cycle_stage == "early_expansion":
            flags.append("Sector in early expansion — optimal multibagger entry window")
        elif cycle_stage == "contraction":
            flags.append("Sector in contraction — elevated cycle risk")

        # ── Macro tailwind score ──────────────────────────────────────────────
        # Set from macro research layer (capex cycle, credit growth, etc.)
        macro_tailwind = data.get("macro_tailwind_score", 50)
        scores.append(macro_tailwind * 0.45)
        breakdown["macro_tailwind"] = round(macro_tailwind, 0)
        if macro_tailwind >= 75:
            flags.append("Strong macro tailwind")

        final = sum(scores) if scores else 0.0
        return DimensionScore(
            name="cycle", score=final, weight=WEIGHTS["cycle"],
            weighted=final * WEIGHTS["cycle"], breakdown=breakdown, flags=flags,
        )


class RiskScorer:
    """
    Debt trend, earnings variability, governance proxies.
    This is the ONLY dimension where a high score is bad — invert at composite.
    Score here = riskiness, then invert to reward low-risk names.
    """

    def score(self, data: dict) -> DimensionScore:
        risk_penalties, flags, breakdown = [], [], {}

        # ── Debt trend ───────────────────────────────────────────────────────
        de_history = data.get("de_history", [])
        if len(de_history) >= 3:
            de_trend = _trend_slope(de_history[-3:])
            if de_trend > 0.2:
                risk_penalties.append(30)
                flags.append("Debt rising trend over 3 years")
            elif de_trend < -0.2:
                risk_penalties.append(-10)
                flags.append("Debt reduction — de-leveraging in progress")
            breakdown["de_trend_slope"] = round(de_trend, 3)

        # ── Interest coverage ─────────────────────────────────────────────────
        interest_coverage = data.get("interest_coverage")
        if interest_coverage is not None:
            if interest_coverage < 1.5:
                risk_penalties.append(40)
                flags.append(f"Thin interest coverage {interest_coverage:.1f}x")
            elif interest_coverage < 3.0:
                risk_penalties.append(15)
            breakdown["interest_coverage"] = round(interest_coverage, 1)

        # ── Earnings variability ──────────────────────────────────────────────
        eps_history = data.get("eps_history", [])
        if len(eps_history) >= 4:
            changes = [
                abs((eps_history[i] - eps_history[i-1]) / abs(eps_history[i-1]))
                for i in range(1, len(eps_history))
                if eps_history[i-1] != 0
            ]
            if changes:
                variability = np.std(changes)
                if variability > 0.5:
                    risk_penalties.append(25)
                    flags.append("High earnings variability — unpredictable business")
                breakdown["eps_variability"] = round(variability, 3)

        # ── Governance proxies ────────────────────────────────────────────────
        audit_qualification = data.get("audit_qualified", False)
        related_party_pct   = data.get("related_party_revenue_pct", 0)
        if audit_qualification:
            risk_penalties.append(50)
            flags.append("CRITICAL: Audit qualification on record")
        if related_party_pct > 20:
            risk_penalties.append(20)
            flags.append(f"High related party transactions {related_party_pct:.0f}%")
        breakdown["audit_qualified"]        = audit_qualification
        breakdown["related_party_pct"]      = round(related_party_pct, 1)

        # Aggregate: total penalty → convert to 0–100 risk score → invert
        total_penalty = _clamp(sum(risk_penalties), 0, 100)
        raw_risk      = _clamp(total_penalty, 0, 100)
        final         = _clamp(100 - raw_risk, 0, 100)   # invert: low risk = high score

        return DimensionScore(
            name="risk", score=final, weight=WEIGHTS["risk"],
            weighted=final * WEIGHTS["risk"], breakdown=breakdown, flags=flags,
        )


# ---------------------------------------------------------------------------
# Critical gate checker
# ---------------------------------------------------------------------------

def _check_gates(data: dict) -> tuple[bool, list[str]]:
    failures = []
    pledge    = data.get("pledge_pct", 0)
    de        = data.get("debt_to_equity")
    roic      = data.get("roic_current")
    promoter  = data.get("promoter_pct")
    audit_q   = data.get("audit_qualified", False)

    if pledge > CRITICAL_GATES["pledge_pct_max"]:
        failures.append(f"Pledge {pledge:.1f}% > {CRITICAL_GATES['pledge_pct_max']}%")
    if de is not None and de > CRITICAL_GATES["debt_to_equity_max"]:
        failures.append(f"D/E {de:.1f}x > {CRITICAL_GATES['debt_to_equity_max']}x")
    if roic is not None and roic < CRITICAL_GATES["min_roic"]:
        failures.append(f"ROIC {roic:.1f}% < {CRITICAL_GATES['min_roic']}%")
    if promoter is not None and promoter < CRITICAL_GATES["min_promoter_pct"]:
        failures.append(f"Promoter {promoter:.1f}% < {CRITICAL_GATES['min_promoter_pct']}%")
    if audit_q:
        failures.append("Audit qualification disqualifies MULTIBAGGER status")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Conviction tier assignment
# ---------------------------------------------------------------------------

def _assign_tier(composite: float, dimensions: dict[str, DimensionScore], gates_passed: bool) -> str:
    if not gates_passed:
        return "REJECT"
    min_dim = min(d.score for d in dimensions.values())
    if composite >= 85 and min_dim >= 55:
        return "ELITE"
    if composite >= 72 and min_dim >= 40:
        return "STRONG"
    if composite >= 58:
        return "WATCH"
    return "REJECT"


# ---------------------------------------------------------------------------
# Narrative generator
# ---------------------------------------------------------------------------

def _build_narrative(ticker: str, tier: str, dimensions: dict[str, DimensionScore]) -> str:
    top_dim    = max(dimensions.items(), key=lambda x: x[1].score)
    bottom_dim = min(dimensions.items(), key=lambda x: x[1].score)
    all_flags  = [f for d in dimensions.values() for f in d.flags]
    insider    = any("Insider buying" in f for f in all_flags)
    fii_acc    = any("FII accumulation" in f for f in all_flags)

    strength = f"strongest in {top_dim[0]} ({top_dim[1].score:.0f}/100)"
    weakness = f"weakest in {bottom_dim[0]} ({bottom_dim[1].score:.0f}/100)"
    ownership_note = ""
    if insider:
        ownership_note = " Insider buying in last 90 days adds conviction."
    elif fii_acc:
        ownership_note = " FII accumulation supports institutional interest."

    tier_map = {
        "ELITE":  "Elite multibagger candidate",
        "STRONG": "Strong multibagger candidate",
        "WATCH":  "Watch-list candidate — not yet investable",
        "REJECT": "Does not qualify for multibagger consideration",
    }
    return (
        f"{ticker}: {tier_map.get(tier, tier)}. "
        f"Composite {strength}, {weakness}.{ownership_note}"
    )


# ---------------------------------------------------------------------------
# Main composite scorer
# ---------------------------------------------------------------------------

class MultibaggerScorer:
    """
    Orchestrates all 7 dimension scorers and produces a MultibaggerResult.

    Usage
    -----
        scorer = MultibaggerScorer()
        result = scorer.score("HDFCBANK", data_dict)
        print(result.tier, result.composite)
        print(result.to_dict())
    """

    def __init__(self):
        self._scorers = {
            "quality":   QualityScorer(),
            "growth":    GrowthScorer(),
            "valuation": ValuationScorer(),
            "ownership": OwnershipScorer(),
            "momentum":  MomentumScorer(),
            "cycle":     CycleScorer(),
            "risk":      RiskScorer(),
        }

    def score(self, ticker: str, data: dict) -> MultibaggerResult:
        gates_passed, gate_failures = _check_gates(data)

        dimensions: dict[str, DimensionScore] = {}
        for name, scorer in self._scorers.items():
            try:
                dimensions[name] = scorer.score(data)
            except Exception as exc:
                dimensions[name] = DimensionScore(
                    name=name, score=0.0,
                    weight=WEIGHTS[name], weighted=0.0,
                    flags=[f"Scorer error: {exc}"],
                )

        composite = sum(d.weighted for d in dimensions.values())
        composite = _clamp(composite, 0, 100)

        if not gates_passed:
            tier = "REJECT"
        else:
            tier = _assign_tier(composite, dimensions, gates_passed)

        narrative = _build_narrative(ticker, tier, dimensions)

        return MultibaggerResult(
            ticker=ticker,
            composite=composite,
            tier=tier,
            dimensions=dimensions,
            gates_passed=gates_passed,
            gate_failures=gate_failures,
            narrative=narrative,
        )

    def score_batch(self, records: list[dict]) -> list[MultibaggerResult]:
        results = []
        for rec in records:
            ticker = rec.get("ticker", "UNKNOWN")
            try:
                results.append(self.score(ticker, rec))
            except Exception as exc:
                results.append(MultibaggerResult(
                    ticker=ticker, composite=0.0, tier="REJECT",
                    dimensions={}, gates_passed=False,
                    gate_failures=[f"Scoring error: {exc}"],
                    narrative=f"{ticker}: Scoring failed — {exc}",
                ))
        results.sort(key=lambda r: r.composite, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def _linear_scale(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """Scale value linearly to 0–100 between lo and hi."""
    if hi == lo:
        return 50.0
    scaled = (value - lo) / (hi - lo) * 100.0
    return (100.0 - scaled) if invert else scaled


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _trend_slope(values: list[float]) -> float:
    """
    Returns the linear regression slope of a short time series.
    Positive = improving trend, negative = deteriorating.
    """
    if len(values) < 2:
        return 0.0
    n  = len(values)
    xs = np.arange(n, dtype=float)
    ys = np.array(values, dtype=float)
    xs -= xs.mean()
    ys -= ys.mean()
    denom = (xs ** 2).sum()
    return float((xs * ys).sum() / denom) if denom > 0 else 0.0
