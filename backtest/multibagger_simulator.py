"""Multibagger return simulator: 3-year projected return simulation."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.multibagger.conviction_scorer import ConvictionScorer
from models.schemas import MultibaggerCandidate


@dataclass
class SimulationOutcome:
    """Result for a single candidate over the simulation period."""

    ticker: str
    conviction_score: float
    initial_price: float
    projected_3y_price: float
    projected_return_pct: float
    cagr_pct: float
    tranche_avg_cost: float
    total_invested: float
    projected_value: float


class MultibaggerSimulator:
    """Simulate 3-year returns for multibagger candidates.

    Uses the conviction score and fundamental growth rates to project
    price appreciation.  This is a rough estim ation model, not a
    precise forecast.

    Growth projection:
    - Base growth = sales growth CAGR × PEG-adjusted multiplier
    - Conviction premium = conviction score / 100 × 15% bonus
    - Risk discount = debt/equity ratio × 5% penalty
    """

    YEARS = 3

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        self.fetcher = fetcher or DataFetcher()
        self.scorer = ConvictionScorer()

    def simulate(
        self,
        tickers: list[str],
        years: int = 3,
    ) -> list[SimulationOutcome]:
        """Run the multibagger simulation for the given tickers.

        Parameters
        ----------
        tickers:
            Candidate ticker symbols.
        years:
            Projection horizon in years.
        """

        outcomes: list[SimulationOutcome] = []

        for ticker in tickers:
            try:
                data = self.fetcher.fetch(ticker)
                if data.price is None or data.price <= 0:
                    continue

                candidate = self.scorer.score_ticker(ticker, data)
                if candidate.action == "REJECT":
                    continue

                outcome = self._project_returns(ticker, candidate, data, years)
                if outcome is not None:
                    outcomes.append(outcome)
            except Exception as exc:
                db.log_engine_event("WARN", "backtest.multibagger_simulator", "simulation failed", {"ticker": ticker, "error": str(exc)})

        # Sort by projected return, descending
        outcomes.sort(key=lambda o: o.projected_return_pct, reverse=True)

        db.log_engine_event("INFO", "backtest.multibagger_simulator", "simulation completed", {
            "candidates": len(tickers), "projected": len(outcomes),
            "top": [{"ticker": o.ticker, "return": f"{o.projected_return_pct:.1f}%"} for o in outcomes[:5]],
        })
        return outcomes

    def _project_returns(
        self,
        ticker: str,
        candidate: MultibaggerCandidate,
        data,
        years: int,
    ) -> SimulationOutcome | None:
        """Project returns for a single candidate."""

        price = data.price
        sales_growth = data.sales_growth_5y or 0.10
        conviction = candidate.conviction_score
        debt_eq = data.debt_equity or 0.0

        # --- Growth projection ---
        base_growth = sales_growth * 1.2  # slight premium for quality
        conviction_premium = (conviction / 100.0) * 0.15
        risk_discount = min(0.10, debt_eq * 0.05)
        annual_growth = base_growth + conviction_premium - risk_discount
        annual_growth = max(0.05, min(0.50, annual_growth))  # clamp 5-50%

        projected_price = price * ((1 + annual_growth) ** years)
        projected_return_pct = ((projected_price / price) - 1) * 100
        cagr_pct = (((projected_price / price) ** (1 / years)) - 1) * 100

        # --- Tranche average cost ---
        tranche_count = config.MB_TRANCHE_COUNT
        tranche_entries = [price * (1 - 0.03 * i) for i in range(tranche_count)]
        tranche_avg_cost = sum(tranche_entries) / len(tranche_entries)
        invested_per_tranche = config.DEFAULT_PORTFOLIO_CAPITAL * config.MB_TRANCHE_RISK_PCT
        total_invested = invested_per_tranche * tranche_count
        shares = sum(invested_per_tranche / entry for entry in tranche_entries)
        projected_value = shares * projected_price

        return SimulationOutcome(
            ticker=ticker.strip().upper(),
            conviction_score=conviction,
            initial_price=round(price, 2),
            projected_3y_price=round(projected_price, 2),
            projected_return_pct=round(projected_return_pct, 2),
            cagr_pct=round(cagr_pct, 2),
            tranche_avg_cost=round(tranche_avg_cost, 2),
            total_invested=round(total_invested, 2),
            projected_value=round(projected_value, 2),
        )


if __name__ == "__main__":
    outcomes = MultibaggerSimulator().simulate(["RELIANCE", "TCS", "INFY"])
    for o in outcomes:
        print(f"{o.ticker}: {o.projected_return_pct:.1f}% over 3Y (CAGR {o.cagr_pct:.1f}%)")
