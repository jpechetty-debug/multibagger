"""
engines/multibagger — public API

    from engines.multibagger import MultibaggerScorer, MultibaggerResult
    scorer = MultibaggerScorer()
    result = scorer.score("HDFCBANK", data_dict)
    result.tier          # "ELITE" | "STRONG" | "WATCH" | "REJECT"
    result.composite     # 0–100
    result.is_multibagger  # True if ELITE or STRONG
    result.to_dict()     # full breakdown for DB storage / UI
"""

from .scorer import (
    MultibaggerScorer,
    MultibaggerResult,
    DimensionScore,
    QualityScorer,
    GrowthScorer,
    ValuationScorer,
    OwnershipScorer,
    MomentumScorer,
    CycleScorer,
    RiskScorer,
    WEIGHTS,
    SECTOR_CYCLE,
    CONVICTION_TIERS,
    CRITICAL_GATES,
)

__all__ = [
    "MultibaggerScorer",
    "MultibaggerResult",
    "DimensionScore",
    "QualityScorer",
    "GrowthScorer",
    "ValuationScorer",
    "OwnershipScorer",
    "MomentumScorer",
    "CycleScorer",
    "RiskScorer",
    "WEIGHTS",
    "SECTOR_CYCLE",
    "CONVICTION_TIERS",
    "CRITICAL_GATES",
]
