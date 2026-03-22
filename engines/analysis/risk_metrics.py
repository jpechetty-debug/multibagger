"""Risk metrics analysis engine."""

from __future__ import annotations

from data.db import db
from data.fetcher import DataFetcher
from engines.analysis._common import (
    annualized_volatility,
    beta,
    compute_max_drawdown,
    compute_returns,
    current_timestamp,
    load_benchmark_history,
    load_price_history,
    log_event,
    normalize_score,
)
from models.schemas import FundamentalData, RiskMetricsAnalysis


class RiskMetricsAnalyzer:
    """Computes volatility, beta, and drawdown metrics."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the analyzer."""

        self.fetcher = fetcher or DataFetcher()

    def analyze(self, ticker: str, data: FundamentalData | None = None) -> RiskMetricsAnalysis:
        """Return 6M risk metrics for a ticker."""

        normalized_ticker = ticker.strip().upper()
        _ = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        history = load_price_history(normalized_ticker).tail(126)
        benchmark = load_benchmark_history().tail(126)
        returns = compute_returns(history)
        benchmark_returns = compute_returns(benchmark)
        volatility_6m = annualized_volatility(returns)
        beta_vs_nifty = beta(returns, benchmark_returns)
        max_drawdown_6m = compute_max_drawdown(history)
        volatility_score = normalize_score(volatility_6m, 0.15, 0.60, inverse=True)
        beta_score = normalize_score(abs(beta_vs_nifty - 1.0) if beta_vs_nifty is not None else 0.0, 0.0, 1.0, inverse=True)
        drawdown_score = normalize_score(abs(max_drawdown_6m) if max_drawdown_6m is not None else 0.10, 0.05, 0.50, inverse=True)
        score = volatility_score * 0.4 + beta_score * 0.2 + drawdown_score * 0.4

        result = RiskMetricsAnalysis(
            ticker=normalized_ticker,
            volatility_6m=volatility_6m,
            beta_vs_nifty=beta_vs_nifty,
            max_drawdown_6m=max_drawdown_6m,
            score=score,
            as_of=current_timestamp(),
        )
        log_event("engines.analysis.risk_metrics", "risk metrics analysis complete", result.model_dump())
        return result


if __name__ == "__main__":
    print(RiskMetricsAnalyzer().analyze("RELIANCE").model_dump())
