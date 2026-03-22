"""Factor weights for each market regime."""

from __future__ import annotations

import config
from models.schemas import MarketRegime


FACTOR_DEFINITIONS = {
    "fundamentals": "ROE, EPS growth, cash conversion, and Piotroski quality",
    "earnings_revision": "Beat streak and estimate revisions",
    "momentum": "Relative strength, DMA posture, and volume confirmation",
    "valuation": "Proxy valuation derived from PEG, PE, and growth quality",
    "ownership": "Promoter quality, pledge risk, and institutional flows",
    "sector_strength": "Sector-relative rank and DuPont quality",
    "risk": "Volatility, beta, and drawdown resilience",
}


def get_weights(regime: MarketRegime) -> dict[str, float]:
    """Return the configured factor weights for a regime."""

    return dict(config.REGIME_FACTOR_WEIGHTS[regime.value])


if __name__ == "__main__":
    print(get_weights(MarketRegime.BULL))
