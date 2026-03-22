"""Sector ranking analysis engine."""

from __future__ import annotations

import pandas as pd

from data.db import db
from data.fetcher import DataFetcher
from engines.analysis._common import current_timestamp, load_financial_statements, log_event, percentile_rank
from models.schemas import FundamentalData, SectorRankAnalysis


def _statement_value(frame: pd.DataFrame, names: list[str]) -> float | None:
    """Return the latest numeric value for the first matching row."""

    for name in names:
        if name in frame.index:
            series = frame.loc[name]
            if isinstance(series, pd.Series):
                series = series.dropna().sort_index(ascending=False)
                if not series.empty:
                    return float(series.iloc[0])
    return None


class SectorRankAnalyzer:
    """Ranks a ticker against stored peers in the same sector."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the analyzer."""

        self.fetcher = fetcher or DataFetcher()

    def analyze(self, ticker: str, data: FundamentalData | None = None) -> SectorRankAnalysis:
        """Return peer-relative ranking and DuPont metrics."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        peers = [record for record in db.list_fundamentals(effective=True) if record.sector == data.sector]
        if not peers:
            peers = [data]

        peer_scores = []
        for peer in peers:
            growth = peer.sales_growth_5y or 0.0
            profitability = peer.roe_ttm or 0.0
            quality = peer.cfo_to_pat or 0.0
            leverage_penalty = peer.debt_equity or 0.0
            composite = profitability * 0.45 + growth * 0.25 + quality * 0.20 - leverage_penalty * 0.10
            peer_scores.append((peer.ticker, composite))
        sorted_scores = sorted(peer_scores, key=lambda item: item[1], reverse=True)
        sector_rank = next((index for index, (peer_ticker, _) in enumerate(sorted_scores, start=1) if peer_ticker == normalized_ticker), 1)
        rank_percentile = percentile_rank([score for _, score in sorted_scores], next(score for peer_ticker, score in sorted_scores if peer_ticker == normalized_ticker))

        statements = load_financial_statements(normalized_ticker)
        financials = statements["financials"]
        balance_sheet = statements["balance_sheet"]
        revenue = _statement_value(financials, ["Total Revenue"])
        gross_profit = _statement_value(financials, ["Gross Profit"])
        operating_income = _statement_value(financials, ["Operating Income", "Operating Revenue"])
        net_income = _statement_value(financials, ["Net Income", "Net Income From Continuing Operation Net Minority Interest"])
        assets = _statement_value(balance_sheet, ["Total Assets"])
        equity = _statement_value(balance_sheet, ["Common Stock Equity"])

        dupont = {
            "net_margin": (net_income / revenue) if net_income is not None and revenue not in (None, 0) else None,
            "asset_turnover": (revenue / assets) if revenue is not None and assets not in (None, 0) else None,
            "equity_multiplier": (assets / equity) if assets is not None and equity not in (None, 0) else None,
        }
        dupont["roe"] = (
            dupont["net_margin"] * dupont["asset_turnover"] * dupont["equity_multiplier"]
            if all(value is not None for value in dupont.values())
            else None
        )
        common_size = {
            "gross_margin": (gross_profit / revenue) if gross_profit is not None and revenue not in (None, 0) else None,
            "operating_margin": (operating_income / revenue) if operating_income is not None and revenue not in (None, 0) else None,
            "net_margin": (net_income / revenue) if net_income is not None and revenue not in (None, 0) else None,
        }
        score = max(0.0, min(100.0, (1.0 - ((sector_rank - 1) / max(1, len(sorted_scores) - 1))) * 100.0 if len(sorted_scores) > 1 else 100.0))

        result = SectorRankAnalysis(
            ticker=normalized_ticker,
            sector=data.sector,
            peer_count=len(sorted_scores),
            sector_rank=sector_rank,
            rank_percentile=rank_percentile,
            top_3=sector_rank <= 3,
            dupont=dupont,
            common_size=common_size,
            score=score,
            as_of=current_timestamp(),
        )
        log_event("engines.analysis.sector_rank", "sector rank analysis complete", result.model_dump())
        return result


if __name__ == "__main__":
    print(SectorRankAnalyzer().analyze("RELIANCE").model_dump())
