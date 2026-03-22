"""Phase 6 integration tests for the production pipeline path."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

import config
from data.cache import cache_manager
from data.db import db
from data.fetcher import DataFetcher, DataQualitySkip
from engines.pipeline import PipelineOrchestrator
from models.schemas import SourceSnapshot, SourceSnapshotBundle


TEST_TICKERS = ["P6BUY", "P6WATCH", "P6WEAK", "P6REJECT", "P6BUY2"]


def _cleanup_tickers(tickers: list[str]) -> None:
    """Remove integration-test rows for the provided tickers."""

    with db.connection("stocks") as conn:
        for ticker in tickers:
            conn.execute("DELETE FROM manual_overrides WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM fundamentals WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM scores WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM score_history WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM valuations WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM signals WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM analysis_snapshots WHERE ticker = ?", (ticker,))
        conn.execute("DELETE FROM market_snapshots")
    with db.connection("ops") as conn:
        for ticker in tickers:
            conn.execute("DELETE FROM audit_findings WHERE run_id IN (SELECT run_id FROM audit_runs WHERE ticker = ?)", (ticker,))
            conn.execute("DELETE FROM audit_runs WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM audit_actions WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM audit_log WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM logs WHERE context_json LIKE ?", (f"%{ticker}%",))
        conn.execute("DELETE FROM run_history WHERE command_name = 'pipeline.run'")
    for ticker in tickers:
        cache_manager.invalidate(ticker)
    cache_manager.invalidate("INDIAVIX")
    cache_manager.invalidate(config.NIFTY_YFINANCE_SYMBOL)


def _symbol_name(symbol: str) -> str:
    """Return the base ticker from a yfinance symbol."""

    return symbol.replace(".NS", "")


def _price_profile(symbol: str) -> tuple[float, float]:
    """Return start/end prices for mocked history."""

    ticker = _symbol_name(symbol)
    profiles = {
        "P6BUY": (80.0, 100.0),
        "P6BUY2": (70.0, 95.0),
        "P6WATCH": (110.0, 112.0),
        "P6WEAK": (120.0, 115.0),
        "P6REJECT": (150.0, 90.0),
        "^NSEI": (100.0, 108.0),
    }
    return profiles.get(ticker, (100.0, 105.0))


def _info_for_symbol(symbol: str) -> dict[str, object]:
    """Return a mocked yfinance info payload."""

    ticker = _symbol_name(symbol)
    profiles = {
        "P6BUY": {"price": 100.0, "market_cap": 5_000_000_000.0, "avg_volume": 200_000.0, "roe": 0.24, "sales_growth": 0.18, "eps_growth": 0.22, "debt_equity": 0.30, "pe": 12.0, "promoter": 0.62},
        "P6BUY2": {"price": 95.0, "market_cap": 4_500_000_000.0, "avg_volume": 180_000.0, "roe": 0.23, "sales_growth": 0.16, "eps_growth": 0.20, "debt_equity": 0.35, "pe": 11.0, "promoter": 0.60},
        "P6WATCH": {"price": 112.0, "market_cap": 4_800_000_000.0, "avg_volume": 160_000.0, "roe": 0.18, "sales_growth": 0.10, "eps_growth": 0.08, "debt_equity": 0.50, "pe": 18.0, "promoter": 0.55},
        "P6WEAK": {"price": 115.0, "market_cap": 3_200_000_000.0, "avg_volume": 120_000.0, "roe": 0.10, "sales_growth": 0.04, "eps_growth": 0.01, "debt_equity": 0.90, "pe": 22.0, "promoter": 0.40},
        "P6REJECT": {"price": 90.0, "market_cap": 2_000_000_000.0, "avg_volume": 90_000.0, "roe": 0.02, "sales_growth": -0.05, "eps_growth": -0.08, "debt_equity": 3.20, "pe": 35.0, "promoter": 0.18},
    }
    profile = profiles[ticker]
    return {
        "shortName": ticker,
        "sector": "Financial Services",
        "currentPrice": profile["price"],
        "regularMarketPrice": profile["price"],
        "marketCap": profile["market_cap"],
        "averageVolume": profile["avg_volume"],
        "earningsGrowth": profile["eps_growth"],
        "revenueGrowth": profile["sales_growth"],
        "trailingPE": profile["pe"],
        "heldPercentInsiders": profile["promoter"],
        "returnOnEquity": profile["roe"],
        "debtToEquity": profile["debt_equity"] * 100,
    }


def _financials_for_symbol(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return mocked annual statements for a ticker."""

    ticker = _symbol_name(symbol)
    base = {
        "P6BUY": (1_000_000.0, 1_300_000.0, 10_000_000.0, 13_000_000.0, 1_350_000.0),
        "P6BUY2": (900_000.0, 1_150_000.0, 9_000_000.0, 11_500_000.0, 1_200_000.0),
        "P6WATCH": (850_000.0, 940_000.0, 8_500_000.0, 9_300_000.0, 890_000.0),
        "P6WEAK": (700_000.0, 710_000.0, 7_000_000.0, 7_100_000.0, 500_000.0),
        "P6REJECT": (600_000.0, 550_000.0, 6_500_000.0, 6_000_000.0, 180_000.0),
    }[ticker]
    prev_income, latest_income, prev_revenue, latest_revenue, latest_cash = base
    columns = [pd.Timestamp("2024-03-31"), pd.Timestamp("2025-03-31")]
    financials = pd.DataFrame(
        {
            columns[0]: {"Net Income": prev_income, "Total Revenue": prev_revenue, "Gross Profit": prev_revenue * 0.40, "Operating Income": prev_revenue * 0.18},
            columns[1]: {"Net Income": latest_income, "Total Revenue": latest_revenue, "Gross Profit": latest_revenue * 0.42, "Operating Income": latest_revenue * 0.20},
        }
    )
    balance_sheet = pd.DataFrame(
        {
            columns[0]: {"Total Assets": prev_revenue * 1.8, "Common Stock Equity": prev_revenue * 0.9, "Total Debt": prev_revenue * 0.20, "Current Assets": prev_revenue * 0.45, "Current Liabilities": prev_revenue * 0.20, "Ordinary Shares Number": 100_000.0},
            columns[1]: {"Total Assets": latest_revenue * 1.8, "Common Stock Equity": latest_revenue * 0.95, "Total Debt": latest_revenue * (0.18 if ticker != "P6REJECT" else 0.60), "Current Assets": latest_revenue * 0.48, "Current Liabilities": latest_revenue * 0.18, "Ordinary Shares Number": 100_000.0},
        }
    )
    cashflow = pd.DataFrame(
        {
            columns[0]: {"Operating Cash Flow": prev_income * 1.05, "Free Cash Flow": prev_income * 0.70},
            columns[1]: {"Operating Cash Flow": latest_cash, "Free Cash Flow": latest_cash * 0.75},
        }
    )
    return financials, balance_sheet, cashflow


class _FakeTicker:
    """Mocked yfinance.Ticker object."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        financials, balance_sheet, cashflow = _financials_for_symbol(symbol)
        self._financials = financials
        self._balance_sheet = balance_sheet
        self._cashflow = cashflow

    @property
    def info(self) -> dict[str, object]:
        return _info_for_symbol(self.symbol)

    @property
    def financials(self) -> pd.DataFrame:
        return self._financials.copy()

    @property
    def balance_sheet(self) -> pd.DataFrame:
        return self._balance_sheet.copy()

    @property
    def cashflow(self) -> pd.DataFrame:
        return self._cashflow.copy()

    def get_earnings_dates(self, limit: int = 8) -> pd.DataFrame:
        index = pd.to_datetime(["2025-02-01", "2024-11-01", "2024-08-01", "2024-05-01"], utc=True)
        ticker = _symbol_name(self.symbol)
        surprise = 6.0 if ticker.startswith("P6BUY") else 2.0 if ticker == "P6WATCH" else -4.0 if ticker == "P6REJECT" else 0.5
        return pd.DataFrame(
            {
                "EPS Estimate": [10.0, 9.8, 9.5, 9.1],
                "Reported EPS": [10.5 + surprise / 20.0, 10.1, 9.6, 9.0],
                "Surprise(%)": [surprise, 3.0, 1.0, -1.0],
            },
            index=pd.Index(index, name="Earnings Date"),
        )

    @property
    def eps_trend(self) -> pd.DataFrame:
        ticker = _symbol_name(self.symbol)
        current = 12.0 if ticker.startswith("P6BUY") else 10.5 if ticker == "P6WATCH" else 8.0 if ticker == "P6REJECT" else 9.0
        prior = 10.0 if ticker != "P6REJECT" else 9.5
        return pd.DataFrame({"current": [current], "90daysAgo": [prior]}, index=pd.Index(["0q"], name="period"))

    @property
    def eps_revisions(self) -> pd.DataFrame:
        ticker = _symbol_name(self.symbol)
        up = 4 if ticker.startswith("P6BUY") else 2 if ticker == "P6WATCH" else 0
        down = 3 if ticker == "P6REJECT" else 1
        return pd.DataFrame({"upLast30days": [up], "downLast30days": [down]}, index=pd.Index(["0q"], name="period"))


def _fake_download(symbol: str, *args, **kwargs) -> pd.DataFrame:
    """Return deterministic price history for a ticker or benchmark."""

    start_price, end_price = _price_profile(symbol)
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    close = pd.Series(np.linspace(start_price, end_price, len(dates)), index=dates)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": 100_000,
        },
        index=dates,
    ).rename_axis("Date")


def _fake_quote_equity(ticker: str) -> dict[str, object]:
    """Return deterministic NSE payloads for the fetcher."""

    info = _info_for_symbol(ticker)
    return {
        "priceInfo": {"lastPrice": info["currentPrice"]},
        "securityInfo": {"issuedSize": 50_000_000},
        "metadata": {"lastUpdateTime": "2026-03-14T09:15:00+05:30", "pdSymbolPe": info["trailingPE"]},
        "industryInfo": {"sector": info["sector"]},
        "info": {"companyName": ticker},
    }


def _fake_advances_declines(mode: str = "pandas") -> pd.DataFrame:
    """Return deterministic breadth data."""

    return pd.DataFrame({"pChange": [1.2, 0.8, 0.5, -0.2, 0.1, -0.4]})


def test_fetcher_raises_data_quality_skip_for_empty_bundle(monkeypatch) -> None:
    """Fetcher should reject records whose ingestion quality falls below the configured minimum."""

    monkeypatch.setattr(DataFetcher, "fetch_source_snapshots", lambda self, ticker, refresh=False: SourceSnapshotBundle(ticker=ticker, snapshots=[]))
    with pytest.raises(DataQualitySkip):
        DataFetcher().fetch("P6LOWQ", refresh=True)


def test_pipeline_integration_with_mocked_yfinance(monkeypatch) -> None:
    """Pipeline should run end-to-end on mocked provider data and return well-formed results."""

    _cleanup_tickers(TEST_TICKERS)
    monkeypatch.setattr(config, "ALPHA_VANTAGE_API_KEY", None)
    monkeypatch.setattr(config, "TELEGRAM_BOT_TOKEN", None)
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", None)
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)
    monkeypatch.setattr(yf, "download", _fake_download)
    monkeypatch.setattr("data.fetcher.quote_equity", _fake_quote_equity)
    monkeypatch.setattr("data.fetcher.indiavix", lambda: 14.0)
    monkeypatch.setattr("engines.score_engine.regime.indiavix", lambda: 14.0)
    monkeypatch.setattr("engines.score_engine.regime.nse_get_advances_declines", _fake_advances_declines)
    monkeypatch.setattr(
        DataFetcher,
        "_fetch_bse_ownership",
        lambda self, ticker: SourceSnapshot(source=config.SOURCE_NAME_BSE, ticker=ticker, fetched_at=int(time.time()), fields={}),
    )

    result = asyncio.run(PipelineOrchestrator().run(TEST_TICKERS, triggered_by="phase6_integration"))
    assert result.tickers_requested == 5
    assert result.error_count == 0
    assert len(result.results) == 5
    assert sum(result.action_counts.values()) == 5
    assert db.get_latest_market_snapshot("regime") is not None
    assert db.list_run_history(limit=1)[0]["command_name"] == "pipeline.run"

    allowed_actions = {"BUY", "WATCH", "WEAK", "REJECT", "SKIP"}
    for row in result.results:
        assert row.ticker in TEST_TICKERS
        assert row.action in allowed_actions
        assert row.generated_at > 0
        if row.action != "SKIP":
            assert db.get_latest_signal(row.ticker) is not None
