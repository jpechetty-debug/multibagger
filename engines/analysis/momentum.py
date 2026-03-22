"""Momentum analysis engine."""

from __future__ import annotations

import pandas as pd

from data.db import db
from data.fetcher import DataFetcher
from engines.analysis._common import current_timestamp, load_benchmark_history, load_price_history, log_event, normalize_score
from models.schemas import FundamentalData, MomentumAnalysis


class MomentumAnalyzer:
    """Computes relative strength, moving-average position, and volume acceleration."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the analyzer."""

        self.fetcher = fetcher or DataFetcher()

    def analyze(self, ticker: str, data: FundamentalData | None = None) -> MomentumAnalysis:
        """Return momentum metrics for a ticker."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        history = load_price_history(normalized_ticker)
        benchmark = load_benchmark_history()
        if history.empty or benchmark.empty:
            raise ValueError(f"Insufficient history for momentum analysis: {normalized_ticker}")

        close = history["Close"].dropna()
        volume = history["Volume"].dropna() if "Volume" in history else pd.Series(dtype=float)
        benchmark_close = benchmark["Close"].dropna()

        lookback = min(63, len(close) - 1, len(benchmark_close) - 1)
        if lookback <= 0:
            raise ValueError(f"Insufficient lookback window for momentum analysis: {normalized_ticker}")

        price_return_3m = float(close.iloc[-1] / close.iloc[-(lookback + 1)] - 1)
        benchmark_return_3m = float(benchmark_close.iloc[-1] / benchmark_close.iloc[-(lookback + 1)] - 1)
        relative_strength_3m = price_return_3m - benchmark_return_3m
        dma_50 = close.rolling(window=50).mean().iloc[-1]
        price_vs_50dma_pct = float(close.iloc[-1] / dma_50 - 1) if dma_50 else 0.0
        short_volume = float(volume.tail(10).mean()) if len(volume) >= 10 else float(volume.mean()) if not volume.empty else 0.0
        long_volume = float(volume.tail(50).mean()) if len(volume) >= 50 else float(volume.mean()) if not volume.empty else 1.0
        volume_acceleration = short_volume / long_volume if long_volume else 1.0
        above_50dma = bool(close.iloc[-1] > dma_50) if dma_50 else False

        rs_score = normalize_score(relative_strength_3m, -0.25, 0.25)
        dma_score = normalize_score(price_vs_50dma_pct, -0.20, 0.20)
        volume_score = normalize_score(volume_acceleration, 0.50, 2.00)
        score = rs_score * 0.5 + dma_score * 0.3 + volume_score * 0.2

        result = MomentumAnalysis(
            ticker=normalized_ticker,
            price_return_3m=price_return_3m,
            benchmark_return_3m=benchmark_return_3m,
            relative_strength_3m=relative_strength_3m,
            price_vs_50dma_pct=price_vs_50dma_pct,
            volume_acceleration=volume_acceleration,
            above_50dma=above_50dma,
            score=score,
            as_of=current_timestamp(),
        )
        log_event("engines.analysis.momentum", "momentum analysis complete", result.model_dump())
        return result


if __name__ == "__main__":
    print(MomentumAnalyzer().analyze("RELIANCE").model_dump())
