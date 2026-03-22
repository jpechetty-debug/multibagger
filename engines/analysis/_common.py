"""Shared helpers for analysis modules."""

from __future__ import annotations

from typing import Any
import time

import numpy as np
import pandas as pd
import yfinance as yf

import config
from data.cache import cache_manager
from data.db import db
from ticker_list import to_yfinance


def log_event(component: str, message: str, context: dict[str, Any] | None = None) -> None:
    """Write an engine log event."""

    db.log_engine_event("INFO", component, message, context or {})


def normalize_score(value: float | None, low: float, high: float, inverse: bool = False) -> float:
    """Map a raw value into a clipped 0-100 score."""

    if value is None:
        return 50.0
    if high == low:
        return 50.0
    clipped = max(low, min(high, float(value)))
    ratio = (clipped - low) / (high - low)
    if inverse:
        ratio = 1.0 - ratio
    return max(0.0, min(100.0, ratio * 100.0))


def _history_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize price history into cache-friendly records."""

    frame = _dedupe_history_columns(frame)
    serialized = frame.reset_index()
    serialized["Date"] = serialized["Date"].astype(str)
    return serialized.to_dict(orient="records")


def _history_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Load price history records from cache."""

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index("Date")
    return _dedupe_history_columns(frame)


def _dedupe_history_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate history columns while preserving the first occurrence."""

    if frame.empty or not frame.columns.duplicated().any():
        return frame
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def load_price_history(ticker: str, period: str | None = None, interval: str | None = None) -> pd.DataFrame:
    """Load cached or live price history for a ticker."""

    period = period or config.PRICE_HISTORY_PERIOD
    interval = interval or config.PRICE_HISTORY_INTERVAL
    source = f"history_{period}"
    cached = cache_manager.get(ticker, source)
    if cached and "records" in cached:
        return _history_from_records(cached["records"])

    history = yf.download(
        to_yfinance(ticker),
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    history = _dedupe_history_columns(history).dropna(how="all")
    cache_manager.set(ticker, source, {"records": _history_to_records(history)}, ttl=config.CACHE_TTL_HISTORY)
    return history


def load_benchmark_history(period: str | None = None, interval: str | None = None) -> pd.DataFrame:
    """Load cached or live benchmark history for Nifty."""

    period = period or config.PRICE_HISTORY_PERIOD
    interval = interval or config.PRICE_HISTORY_INTERVAL
    source = f"benchmark_history_{period}"
    cached = cache_manager.get(config.NIFTY_YFINANCE_SYMBOL, source)
    if cached and "records" in cached:
        return _history_from_records(cached["records"])

    history = yf.download(
        config.NIFTY_YFINANCE_SYMBOL,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    history = _dedupe_history_columns(history).dropna(how="all")
    cache_manager.set(
        config.NIFTY_YFINANCE_SYMBOL,
        source,
        {"records": _history_to_records(history)},
        ttl=config.CACHE_TTL_HISTORY,
    )
    return history


def load_financial_statements(ticker: str) -> dict[str, pd.DataFrame]:
    """Load annual financial statements from cache or yfinance."""

    source = "financials_annual"
    cached = cache_manager.get(ticker, source)
    if cached and {"financials", "balance_sheet", "cashflow"} <= set(cached):
        result: dict[str, pd.DataFrame] = {}
        for key in ("financials", "balance_sheet", "cashflow"):
            frame = pd.DataFrame.from_dict(cached[key], orient="index")
            frame.columns = pd.to_datetime(frame.columns)
            result[key] = frame
        return result

    yf_ticker = yf.Ticker(to_yfinance(ticker))
    result = {
        "financials": yf_ticker.financials if isinstance(yf_ticker.financials, pd.DataFrame) else pd.DataFrame(),
        "balance_sheet": yf_ticker.balance_sheet if isinstance(yf_ticker.balance_sheet, pd.DataFrame) else pd.DataFrame(),
        "cashflow": yf_ticker.cashflow if isinstance(yf_ticker.cashflow, pd.DataFrame) else pd.DataFrame(),
    }
    serialized = {
        name: {
            str(index): {str(column): value for column, value in row.items()}
            for index, row in frame.to_dict(orient="index").items()
        }
        for name, frame in result.items()
    }
    cache_manager.set(ticker, source, serialized, ttl=config.CACHE_TTL_FINANCIALS)
    return result


def load_earnings_data(ticker: str) -> dict[str, Any]:
    """Load earnings-related data from yfinance with cache."""

    source = "earnings_meta"
    cached = cache_manager.get(ticker, source)
    if cached:
        result: dict[str, Any] = {}
        for key, value in cached.items():
            if isinstance(value, list):
                frame = pd.DataFrame(value)
                if "index" in frame.columns:
                    if key == "earnings_dates":
                        frame["index"] = pd.to_datetime(frame["index"], errors="coerce", utc=True)
                    frame = frame.set_index("index")
                result[key] = frame
            else:
                result[key] = value
        return result

    yf_ticker = yf.Ticker(to_yfinance(ticker))
    earnings_dates = yf_ticker.get_earnings_dates(limit=8)
    eps_trend = yf_ticker.eps_trend
    eps_revisions = yf_ticker.eps_revisions
    payload = {
        "earnings_dates": earnings_dates.reset_index().astype({"Earnings Date": str}).rename(columns={"Earnings Date": "index"}).to_dict(orient="records")
        if isinstance(earnings_dates, pd.DataFrame)
        else [],
        "eps_trend": eps_trend.reset_index().astype({"period": str}).rename(columns={"period": "index"}).to_dict(orient="records")
        if isinstance(eps_trend, pd.DataFrame)
        else [],
        "eps_revisions": eps_revisions.reset_index().astype({"period": str}).rename(columns={"period": "index"}).to_dict(orient="records")
        if isinstance(eps_revisions, pd.DataFrame)
        else [],
    }
    cache_manager.set(ticker, source, payload, ttl=config.CACHE_TTL_HISTORY)
    return {
        "earnings_dates": earnings_dates if isinstance(earnings_dates, pd.DataFrame) else pd.DataFrame(),
        "eps_trend": eps_trend if isinstance(eps_trend, pd.DataFrame) else pd.DataFrame(),
        "eps_revisions": eps_revisions if isinstance(eps_revisions, pd.DataFrame) else pd.DataFrame(),
    }


def latest_close(history: pd.DataFrame) -> float | None:
    """Return the latest close price from a history frame."""

    if history.empty or "Close" not in history:
        return None
    value = history["Close"].dropna()
    return float(value.iloc[-1]) if not value.empty else None


def compute_returns(history: pd.DataFrame) -> pd.Series:
    """Compute daily returns from a history frame."""

    if history.empty or "Close" not in history:
        return pd.Series(dtype=float)
    return history["Close"].pct_change().dropna()


def compute_max_drawdown(history: pd.DataFrame) -> float | None:
    """Compute the maximum drawdown for a history frame."""

    if history.empty or "Close" not in history:
        return None
    close = history["Close"].dropna()
    if close.empty:
        return None
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    return float(drawdown.min())


def percentile_rank(values: list[float], current: float) -> float:
    """Return the percentile rank of a value within a peer set."""

    if not values:
        return 1.0
    less_or_equal = sum(value <= current for value in values)
    return less_or_equal / len(values)


def annualized_volatility(returns: pd.Series) -> float | None:
    """Return annualized volatility from daily returns."""

    if returns.empty:
        return None
    return float(returns.std(ddof=0) * np.sqrt(252))


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """Return beta of a series against a benchmark."""

    if returns.empty or benchmark_returns.empty:
        return None
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty or aligned.iloc[:, 1].var(ddof=0) == 0:
        return None
    covariance = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=0)[0, 1]
    variance = aligned.iloc[:, 1].var(ddof=0)
    return float(covariance / variance)


def current_timestamp() -> int:
    """Return the current Unix timestamp."""

    return int(time.time())
