"""Conviction scorer and tranche planner for multibagger candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.multibagger.early_signal_detector import EarlySignalDetector
from engines.multibagger.quality_filter import QualityFilter
from engines.multibagger.tam_scorer import TAMScorer
from models.schemas import (
    FundamentalData,
    MultibaggerCandidate,
    QualityResult,
    EarlySignalResult,
    TAMResult,
)


@dataclass
class MultibaggerDependencies:
    """Dependency bundle for the conviction scorer."""

    fetcher: DataFetcher
    quality_filter: QualityFilter
    early_signal_detector: EarlySignalDetector
    tam_scorer: TAMScorer


from engines.multibagger.scorer import MultibaggerScorer, MultibaggerResult

class ConvictionScorer:
    """
    Institutional conviction scorer for multibagger candidates.
    Uses the 7-dimensional MultibaggerScorer for deep analysis and critical gates.
    """

    def __init__(self, dependencies: MultibaggerDependencies | None = None) -> None:
        if dependencies is None:
            fetcher = DataFetcher()
            dependencies = MultibaggerDependencies(
                fetcher=fetcher,
                quality_filter=QualityFilter(),
                early_signal_detector=EarlySignalDetector(),
                tam_scorer=TAMScorer(),
            )
        self.deps = dependencies
        self.master_scorer = MultibaggerScorer()

    def score_ticker(self, ticker: str, data: FundamentalData | None = None) -> MultibaggerCandidate:
        """Evaluate an end-to-end multibagger conviction score for *ticker*."""

        normalized = ticker.strip().upper()
        now_ts = int(time.time())

        if data is None:
            data = self.deps.fetcher.fetch(normalized)

        # 1. Run legacy component filters for backward compatibility and specialized signals
        quality_res = self.deps.quality_filter.filter(normalized, data)
        early_res   = self.deps.early_signal_detector.detect(normalized, data)
        tam_res     = self.deps.tam_scorer.score(normalized, data)

        # 2. Bridge to the new Master Scorer
        # We build a comprehensive dict combining raw data + pre-computed component scores
        score_input = self._build_score_input(data, quality_res, early_res, tam_res)
        master_res = self.master_scorer.score(normalized, score_input)

        return self._map_to_candidate(master_res, data, now_ts)

    def _build_score_input(
        self, 
        data: FundamentalData, 
        quality: QualityResult, 
        early: EarlySignalResult, 
        tam: TAMResult
    ) -> dict:
        """Map Pydantic schemas and component results to the master scorer's input dict."""
        
        # Base fundamentals
        out = {
            "ticker": data.ticker,
            "sector": data.sector or "Unknown",
            "price": data.price,
            "promoter_pct": data.promoter_pct,
            "pledge_pct": data.pledge_pct or 0.0,
            "debt_to_equity": data.debt_equity,
            "fii_delta": data.fii_delta,
            "dii_delta": data.dii_delta,
        }

        # Quality & ROIC (Pull from source_metadata if available, or use current)
        # In a real impl, we'd fetch history from DB. Here we use what we have.
        out["roic_current"] = data.roe_ttm  # ROE as proxy if ROIC missing
        out["roic_history"] = data.source_metadata.get("roic_history", [data.roe_5y or 0.0] * 4)
        out["eps_history"]  = data.source_metadata.get("eps_history", [0.0] * 4)
        out["de_history"]   = data.source_metadata.get("de_history", [data.debt_equity or 0.0] * 3)

        # Growth
        out["revenue_cagr_3y"] = data.sales_growth_5y / 100.0 if data.sales_growth_5y else 0.0
        out["eps_cagr_3y"]     = data.eps_growth_ttm / 100.0 if data.eps_growth_ttm else 0.0
        out["tam_runway_score"] = tam.tam_runway_score

        # Valuation
        out["valuation_percentile"] = data.source_metadata.get("valuation_percentile", 50.0)
        out["margin_of_safety_pct"] = data.source_metadata.get("margin_of_safety_pct", 0.0)
        out["fcf_yield"]            = data.cfo_to_pat * 0.1 if data.cfo_to_pat else 0.0 # simple proxy

        # Ownership & Insider signals from EarlySignalDetector
        out["insider_buys_90d"] = 1 if early.promoter_buying else 0
        out["insider_qty_pct"]  = 0.001 if early.promoter_buying else 0.0

        # Momentum snapshot
        # (Usually populated by a separate momentum_engine.py / pipeline.py)
        out["price_vs_200dma_pct"]  = data.source_metadata.get("price_vs_200dma_pct", 5.0)
        out["relative_strength_3m"] = data.source_metadata.get("relative_strength_3m", 0.0)
        out["rank_percentile"]      = data.source_metadata.get("rank_percentile", 50.0)

        # Risk
        out["audit_qualified"] = data.source_metadata.get("audit_qualified", False)
        out["related_party_revenue_pct"] = data.source_metadata.get("related_party_revenue_pct", 0.0)
        
        return out

    def _map_to_candidate(self, res: MultibaggerResult, data: FundamentalData, now_ts: int) -> MultibaggerCandidate:
        """Convert a MultibaggerResult to the standard MultibaggerCandidate schema."""
        
        # Reasoning combines programmatic signals and gate failures
        reasoning = [res.narrative]
        if not res.gates_passed:
            reasoning.extend([f"GATE FAILED: {f}" for f in res.gate_failures])
        
        # Detailed flags from dimensions
        for dim in res.dimensions.values():
            if dim.flags:
                reasoning.extend(dim.flags[:2])

        # Tranche plan logic (identical to previous version)
        tranche_plan = []
        if res.is_multibagger and data.price:
            for i in range(config.MB_TRANCHE_COUNT):
                discount = 1.0 - (i * 0.03)
                entry_price = round(data.price * discount, 2)
                tranche_plan.append({
                    "tranche": i + 1,
                    "entry_price": entry_price,
                    "risk_pct": config.MB_TRANCHE_RISK_PCT,
                })

        candidate = MultibaggerCandidate(
            ticker=res.ticker,
            quality_score=res.dimensions["quality"].score,
            early_signal_score=res.dimensions["growth"].score, # mapped to growth for now
            tam_score=res.dimensions["cycle"].score,          # mapped to cycle
            conviction_score=round(res.composite, 2),
            tranche_plan=tranche_plan,
            action=res.tier if res.tier in ("BUY", "WATCH", "REJECT") else "REJECT", # Adjust for naming
            reasoning=reasoning,
            generated_at=now_ts,
        )
        
        # Specific override for Streamlit readability if it's a "STRONG/ELITE" tier
        if res.tier in ("ELITE", "STRONG"):
            candidate.action = "BUY"
        elif res.tier == "WATCH":
            candidate.action = "WATCH"
        else:
            candidate.action = "REJECT"

        db.log_engine_event("INFO", "engines.multibagger.conviction_scorer", "institutional multibagger scored", res.to_dict())
        return candidate


if __name__ == "__main__":
    print(ConvictionScorer().score_ticker("RELIANCE").model_dump())
