"""Fundamental quality analysis engine."""

from __future__ import annotations

from typing import Any
import pandas as pd

from data.db import db
from data.fetcher import DataFetcher
from engines.analysis._common import current_timestamp, load_financial_statements, log_event, normalize_score
from models.schemas import FundamentalAnalysis, FundamentalData


def _row(frame: pd.DataFrame, names: list[str]) -> pd.Series | None:
    """Return the first matching statement row."""

    for name in names:
        if name in frame.index:
            series = frame.loc[name]
            if isinstance(series, pd.Series):
                return series.dropna()
    return None


def _latest_two(series: pd.Series | None) -> tuple[float | None, float | None]:
    """Return the latest two values from a statement row."""

    if series is None or series.empty:
        return None, None
    ordered = series.sort_index(ascending=False)
    latest = float(ordered.iloc[0]) if len(ordered) >= 1 else None
    previous = float(ordered.iloc[1]) if len(ordered) >= 2 else None
    return latest, previous


class FundamentalAnalyzer:
    """Computes quality metrics and Piotroski F-score inputs."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the analyzer."""

        self.fetcher = fetcher or DataFetcher()

    def analyze(self, ticker: str, data: FundamentalData | None = None) -> FundamentalAnalysis:
        """Return the fundamental quality profile for a ticker."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        statements = load_financial_statements(normalized_ticker)
        financials = statements["financials"]
        balance_sheet = statements["balance_sheet"]
        cashflow = statements["cashflow"]

        checks = self._piotroski_checks(financials, balance_sheet, cashflow)
        piotroski_score = int(sum(checks.values())) if checks else data.piotroski_score

        roe_score = normalize_score(data.roe_5y, -0.10, 0.30)
        eps_score = normalize_score(data.eps_growth_ttm, -0.20, 0.40)
        cfo_score = normalize_score(data.cfo_to_pat, 0.20, 2.00)
        piotroski_component = normalize_score(piotroski_score, 0.0, 9.0)
        score = roe_score * 0.30 + eps_score * 0.20 + cfo_score * 0.20 + piotroski_component * 0.30

        result = FundamentalAnalysis(
            ticker=normalized_ticker,
            roe_5y=data.roe_5y,
            eps_growth_ttm=data.eps_growth_ttm,
            cfo_to_pat=data.cfo_to_pat,
            piotroski_score=piotroski_score,
            piotroski_checks=checks,
            score=score,
            as_of=current_timestamp(),
        )
        log_event("engines.analysis.fundamentals", "fundamental analysis complete", result.model_dump())
        return result

    def _piotroski_checks(
        self,
        financials: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> dict[str, bool]:
        """Compute Piotroski's nine binary checks from statements."""

        assets = _row(balance_sheet, ["Total Assets"])
        net_income = _row(financials, ["Net Income", "Net Income From Continuing Operation Net Minority Interest"])
        cash_ops = _row(cashflow, ["Operating Cash Flow"])
        long_debt = _row(balance_sheet, ["Long Term Debt", "Total Debt"])
        current_assets = _row(balance_sheet, ["Current Assets"])
        current_liabilities = _row(balance_sheet, ["Current Liabilities"])
        gross_profit = _row(financials, ["Gross Profit"])
        revenue = _row(financials, ["Total Revenue"])
        shares = _row(balance_sheet, ["Ordinary Shares Number", "Share Issued"])

        latest_assets, previous_assets = _latest_two(assets)
        latest_income, previous_income = _latest_two(net_income)
        latest_cash, _ = _latest_two(cash_ops)
        latest_debt, previous_debt = _latest_two(long_debt)
        latest_current_assets, previous_current_assets = _latest_two(current_assets)
        latest_current_liabilities, previous_current_liabilities = _latest_two(current_liabilities)
        latest_gross_profit, previous_gross_profit = _latest_two(gross_profit)
        latest_revenue, previous_revenue = _latest_two(revenue)
        latest_shares, previous_shares = _latest_two(shares)

        if latest_assets in (None, 0) or latest_income is None or latest_cash is None:
            return {}

        current_roa = latest_income / latest_assets
        previous_roa = previous_income / previous_assets if previous_income is not None and previous_assets not in (None, 0) else None
        leverage_now = latest_debt / latest_assets if latest_debt is not None and latest_assets else None
        leverage_prev = previous_debt / previous_assets if previous_debt is not None and previous_assets not in (None, 0) else None
        current_ratio_now = (
            latest_current_assets / latest_current_liabilities
            if latest_current_assets is not None and latest_current_liabilities not in (None, 0)
            else None
        )
        current_ratio_prev = (
            previous_current_assets / previous_current_liabilities
            if previous_current_assets is not None and previous_current_liabilities not in (None, 0)
            else None
        )
        gross_margin_now = latest_gross_profit / latest_revenue if latest_gross_profit is not None and latest_revenue not in (None, 0) else None
        gross_margin_prev = (
            previous_gross_profit / previous_revenue
            if previous_gross_profit is not None and previous_revenue not in (None, 0)
            else None
        )
        asset_turnover_now = latest_revenue / latest_assets if latest_revenue is not None and latest_assets not in (None, 0) else None
        asset_turnover_prev = (
            previous_revenue / previous_assets
            if previous_revenue is not None and previous_assets not in (None, 0)
            else None
        )

        return {
            "positive_roa": bool(latest_income > 0),
            "positive_operating_cash_flow": bool(latest_cash > 0),
            "improving_roa": bool(previous_roa is not None and current_roa > previous_roa),
            "cash_flow_exceeds_income": bool(latest_cash > latest_income),
            "lower_leverage": bool(leverage_now is not None and leverage_prev is not None and leverage_now < leverage_prev),
            "improving_liquidity": bool(current_ratio_now is not None and current_ratio_prev is not None and current_ratio_now > current_ratio_prev),
            "no_equity_dilution": bool(latest_shares is not None and previous_shares is not None and latest_shares <= previous_shares),
            "improving_gross_margin": bool(gross_margin_now is not None and gross_margin_prev is not None and gross_margin_now > gross_margin_prev),
            "improving_asset_turnover": bool(asset_turnover_now is not None and asset_turnover_prev is not None and asset_turnover_now > asset_turnover_prev),
        }


if __name__ == "__main__":
    print(FundamentalAnalyzer().analyze("RELIANCE").model_dump())
