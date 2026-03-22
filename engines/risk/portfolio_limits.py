"""Portfolio limit checks."""

from __future__ import annotations

from collections import defaultdict

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.risk.correlation import CorrelationAnalyzer
from models.schemas import FundamentalData, PortfolioLimitResult, PortfolioPosition


class PortfolioLimitsChecker:
    """Checks stock, sector, liquidity, and correlation constraints."""

    def __init__(self, fetcher: DataFetcher | None = None, correlation_analyzer: CorrelationAnalyzer | None = None) -> None:
        """Initialize the checker."""

        self.fetcher = fetcher or DataFetcher()
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()

    def check(
        self,
        candidate_ticker: str,
        candidate_value: float,
        positions: list[PortfolioPosition] | None = None,
        data: FundamentalData | None = None,
        portfolio_total_value: float | None = None,
    ) -> PortfolioLimitResult:
        """Return whether a candidate trade passes portfolio constraints."""

        positions = positions if positions is not None else db.list_portfolio_positions()
        normalized_ticker = candidate_ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        current_total = sum(position.market_value for position in positions)
        projected_total = portfolio_total_value if portfolio_total_value is not None else current_total + candidate_value
        projected_stock_weight = candidate_value / projected_total if projected_total else 0.0
        stock_weight_ok = projected_stock_weight <= config.MAX_SINGLE_STOCK_WEIGHT

        sector_exposure = defaultdict(float)
        for position in positions:
            sector_exposure[position.sector or "Unknown"] += position.market_value
        sector_exposure[data.sector or "Unknown"] += candidate_value
        projected_sector_weight = sector_exposure[data.sector or "Unknown"] / projected_total if projected_total else 0.0
        sector_weight_ok = projected_sector_weight <= config.MAX_SECTOR_WEIGHT

        liquidity_value = (data.avg_volume or 0.0) * (data.price or 0.0)
        liquidity_ok = liquidity_value >= config.MIN_LIQUIDITY_VALUE

        correlation_ok = True
        violations: list[str] = []
        correlation_tickers = [position.ticker for position in positions] + [normalized_ticker]
        if len(correlation_tickers) > 1:
            try:
                correlation_result = self.correlation_analyzer.correlation_matrix(correlation_tickers)
            except ValueError:
                correlation_result = None
                db.log_engine_event(
                    "WARN",
                    "engines.risk.portfolio_limits",
                    "correlation check skipped due to missing history",
                    {"ticker": normalized_ticker, "correlation_tickers": correlation_tickers},
                )
            if correlation_result is not None:
                correlation_ok = normalized_ticker in correlation_result.allowed_tickers
                if not correlation_ok:
                    violations.append("CORRELATION_LIMIT")

        if not stock_weight_ok:
            violations.append("MAX_SINGLE_STOCK_WEIGHT")
        if not sector_weight_ok:
            violations.append("MAX_SECTOR_WEIGHT")
        if not liquidity_ok:
            violations.append("MIN_LIQUIDITY_VALUE")

        result = PortfolioLimitResult(
            passed=stock_weight_ok and sector_weight_ok and liquidity_ok and correlation_ok,
            stock_weight_ok=stock_weight_ok,
            sector_weight_ok=sector_weight_ok,
            liquidity_ok=liquidity_ok,
            correlation_ok=correlation_ok,
            violations=violations,
        )
        db.log_engine_event("INFO", "engines.risk.portfolio_limits", "portfolio limits checked", {"ticker": normalized_ticker, "violations": violations})
        return result


if __name__ == "__main__":
    print(PortfolioLimitsChecker().check("RELIANCE", candidate_value=100000.0).model_dump())
