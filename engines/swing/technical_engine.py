"""Technical indicator engine for swing trades: RSI, MACD, Bollinger, EMA, ATR."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config
from data.db import db
from models.schemas import TechnicalAnalysis


class TechnicalEngine:
    """Computes RSI, MACD, Bollinger Bands, EMA, and ATR from price history."""

    def analyze(self, ticker: str, price_df: pd.DataFrame) -> TechnicalAnalysis:
        """Return a ``TechnicalAnalysis`` snapshot for *ticker*.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        price_df:
            DataFrame with at least ``Close``, ``High``, ``Low`` columns and a
            DatetimeIndex.  Must contain enough rows for the longest look-back
            window (MACD slow period + signal).
        """

        close = price_df["Close"].astype(float)
        high = price_df["High"].astype(float)
        low = price_df["Low"].astype(float)

        rsi = self._rsi(close, config.SWING_RSI_PERIOD)
        macd, macd_signal, macd_hist = self._macd(
            close, config.SWING_MACD_FAST, config.SWING_MACD_SLOW, config.SWING_MACD_SIGNAL
        )
        bb_upper, bb_middle, bb_lower = self._bollinger(close, config.SWING_BB_PERIOD, config.SWING_BB_STD)
        ema_short = self._ema(close, config.SWING_EMA_SHORT)
        ema_long = self._ema(close, config.SWING_EMA_LONG)
        atr = self._atr(high, low, close, period=14)

        trend_bullish = bool(
            ema_short is not None
            and ema_long is not None
            and ema_short > ema_long
            and macd_hist is not None
            and macd_hist > 0
        )

        result = TechnicalAnalysis(
            ticker=ticker.strip().upper(),
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            ema_short=ema_short,
            ema_long=ema_long,
            atr=atr,
            trend_bullish=trend_bullish,
            as_of=int(time.time()),
        )
        db.log_engine_event("INFO", "engines.swing.technical_engine", "technical analysis computed", result.model_dump())
        return result

    # ------------------------------------------------------------------
    # Indicator implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float | None:
        """Compute the Relative Strength Index."""

        if len(close) < period + 1:
            return None
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else None

    @staticmethod
    def _macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute MACD line, signal line, and histogram."""

        if len(close) < slow + signal:
            return None, None, None
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1]),
        )

    @staticmethod
    def _bollinger(
        close: pd.Series,
        period: int = 20,
        num_std: int = 2,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute Bollinger Bands (upper, middle, lower)."""

        if len(close) < period:
            return None, None, None
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return (
            float(upper.iloc[-1]),
            float(middle.iloc[-1]),
            float(lower.iloc[-1]),
        )

    @staticmethod
    def _ema(close: pd.Series, span: int) -> float | None:
        """Compute an exponential moving average."""

        if len(close) < span:
            return None
        return float(close.ewm(span=span, adjust=False).mean().iloc[-1])

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float | None:
        """Compute the Average True Range."""

        if len(close) < period + 1:
            return None
        tr = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])


if __name__ == "__main__":
    print("TechnicalEngine ready")
