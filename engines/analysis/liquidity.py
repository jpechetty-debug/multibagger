"""Liquidity analysis engine."""

from __future__ import annotations

from data.db import db
from data.fetcher import DataFetcher
from engines.analysis._common import current_timestamp, load_price_history, log_event, normalize_score
from models.schemas import FundamentalData, LiquidityAnalysis


class LiquidityAnalyzer:
    """Computes trading liquidity metrics."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the analyzer."""

        self.fetcher = fetcher or DataFetcher()

    def analyze(self, ticker: str, data: FundamentalData | None = None) -> LiquidityAnalysis:
        """Return liquidity metrics for a ticker."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        history = load_price_history(normalized_ticker)
        if history.empty:
            raise ValueError(f"Insufficient history for liquidity analysis: {normalized_ticker}")

        avg_daily_volume_20d = float(history["Volume"].tail(20).mean()) if "Volume" in history else data.avg_volume
        latest_close = float(history["Close"].dropna().iloc[-1])
        turnover_value_20d = avg_daily_volume_20d * latest_close if avg_daily_volume_20d is not None else None
        delivery_pct = None
        liquidity_ok = bool(
            (avg_daily_volume_20d or 0.0) >= 10_000.0
            and (turnover_value_20d or 0.0) >= 10_000_000.0
        )
        volume_score = normalize_score(avg_daily_volume_20d, 10_000.0, 5_000_000.0)
        turnover_score = normalize_score(turnover_value_20d, 10_000_000.0, 5_000_000_000.0)
        score = volume_score * 0.5 + turnover_score * 0.5

        result = LiquidityAnalysis(
            ticker=normalized_ticker,
            avg_daily_volume_20d=avg_daily_volume_20d,
            turnover_value_20d=turnover_value_20d,
            delivery_pct=delivery_pct,
            liquidity_ok=liquidity_ok,
            score=score,
            as_of=current_timestamp(),
        )
        log_event("engines.analysis.liquidity", "liquidity analysis complete", result.model_dump())
        return result


if __name__ == "__main__":
    print(LiquidityAnalyzer().analyze("RELIANCE").model_dump())
