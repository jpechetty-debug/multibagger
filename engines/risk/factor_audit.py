"""
Factor exposure audit for quant portfolios.
Compares portfolio factor loadings (Value, Quality, Momentum, Size)
against the Nifty 500 benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FactorAuditReport:
    portfolio_name:  str
    audit_date:      str
    factor_loadings: dict[str, float]  # factor -> z-score/loading
    benchmark_diff:  dict[str, float]  # portfolio loading - benchmark loading
    active_share:    float             # 0 to 1
    concentration:   float             # Herfindahl-Hirschman Index (HHI)
    top_exposures:   list[dict]        # top 5 names by weight
    warnings:        list[str]
    timestamp:       int

    def to_dict(self) -> dict:
        return {
            "portfolio":      self.portfolio_name,
            "date":           self.audit_date,
            "active_share":   round(self.active_share, 4),
            "hhi":            round(self.concentration, 4),
            "loadings":       {k: round(v, 3) for k, v in self.factor_loadings.items()},
            "active_loadings": {k: round(v, 3) for k, v in self.benchmark_diff.items()},
            "warnings":       self.warnings,
            "top_names":      self.top_exposures,
        }


class FactorAudit:
    """
    Analyzes portfolio factor risk and benchmark deviation.
    """

    def __init__(
        self,
        benchmark_name: str = "Nifty 500",
        risk_threshold: float = 2.0,  # warn if factor > 2.0 z-score
    ):
        self.benchmark_name = benchmark_name
        self.risk_threshold = risk_threshold

    def run(
        self,
        weights: dict[str, float],
        factor_scores: pd.DataFrame,  # columns: ticker, value, quality, momentum, size (z-scores)
        benchmark_weights: dict[str, float] | None = None,
    ) -> FactorAuditReport:
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers])
        
        # Ensure factor scores exist for tickers
        df = factor_scores[factor_scores["ticker"].isin(tickers)].set_index("ticker")
        df = df.reindex(tickers).fillna(0)

        # ── Compute portfolio factor loadings (weighted averages) ─────────────
        loadings: dict[str, float] = {}
        for factor in ["value", "quality", "momentum", "size"]:
            if factor in df.columns:
                loadings[factor] = float(np.sum(df[factor] * w))

        # ── Benchmark comparison ──────────────────────────────────────────────
        benchmark_diff = {}
        if benchmark_weights:
            b_df = factor_scores[factor_scores["ticker"].isin(benchmark_weights)].set_index("ticker")
            b_tickers = list(benchmark_weights.keys())
            bw = np.array([benchmark_weights[t] for t in b_tickers])
            b_df = b_df.reindex(b_tickers).fillna(0)
            
            for factor in loadings:
                b_loading = float(np.sum(b_df[factor] * bw))
                benchmark_diff[factor] = loadings[factor] - b_loading
        else:
            # If no benchmark provided, assume zero-neutral benchmark (standardised)
            benchmark_diff = {k: v for k, v in loadings.items()}

        # ── Active Share ──────────────────────────────────────────────────────
        all_tickers = set(weights.keys()) | set((benchmark_weights or {}).keys())
        active_share = 0.5 * sum(
            abs(weights.get(t, 0) - (benchmark_weights or {}).get(t, 0))
            for t in all_tickers
        )

        # ── Concentration (HHI) ───────────────────────────────────────────────
        hhi = float(np.sum(w**2))

        # ── Top Exposures ─────────────────────────────────────────────────────
        top_5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        top_names = [{"ticker": t, "weight": round(wt, 4)} for t, wt in top_5]

        # ── Warnings ──────────────────────────────────────────────────────────
        warnings = []
        if active_share < 0.5 and benchmark_weights:
            warnings.append("Low active share — portfolio mimics benchmark")
        if hhi > 0.15:
            warnings.append("High concentration (HHI > 0.15) — specific risk elevated")
        for factor, loading in loadings.items():
            if abs(loading) > self.risk_threshold:
                side = "overweight" if loading > 0 else "underweight"
                warnings.append(f"Excessive {factor} {side} ({loading:.2f} z-score)")

        return FactorAuditReport(
            portfolio_name="Main Quant",
            audit_date=time.strftime("%Y-%m-%d"),
            factor_loadings=loadings,
            benchmark_diff=benchmark_diff,
            active_share=active_share,
            concentration=hhi,
            top_exposures=top_names,
            warnings=warnings,
            timestamp=int(time.time()),
        )
