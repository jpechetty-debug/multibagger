"""Ownership analysis engine."""

from __future__ import annotations

from data.db import db
from data.fetcher import DataFetcher
from engines.analysis._common import current_timestamp, log_event, normalize_score
from models.schemas import FundamentalData, OwnershipAnalysis


class OwnershipAnalyzer:
    """Evaluates promoter quality, pledge risk, and institutional flows."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the analyzer."""

        self.fetcher = fetcher or DataFetcher()

    def analyze(self, ticker: str, data: FundamentalData | None = None) -> OwnershipAnalysis:
        """Return ownership quality metrics for a ticker."""

        normalized_ticker = ticker.strip().upper()
        data = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        promoter_score = normalize_score(data.promoter_pct, 20.0, 75.0)
        pledge_score = normalize_score(data.pledge_pct if data.pledge_pct is not None else 5.0, 0.0, 30.0, inverse=True)
        fii_score = normalize_score(data.fii_delta if data.fii_delta is not None else 0.0, -10.0, 10.0)
        dii_score = normalize_score(data.dii_delta if data.dii_delta is not None else 0.0, -10.0, 10.0)
        ownership_clean = bool(
            (data.promoter_pct or 0.0) >= 50.0
            and (data.pledge_pct or 0.0) < 5.0
            and (data.fii_delta is None or data.fii_delta >= 0.0)
            and (data.dii_delta is None or data.dii_delta >= 0.0)
        )
        score = promoter_score * 0.4 + pledge_score * 0.3 + fii_score * 0.15 + dii_score * 0.15

        result = OwnershipAnalysis(
            ticker=normalized_ticker,
            promoter_pct=data.promoter_pct,
            pledge_pct=data.pledge_pct,
            fii_delta=data.fii_delta,
            dii_delta=data.dii_delta,
            ownership_clean=ownership_clean,
            score=score,
            as_of=current_timestamp(),
        )
        log_event("engines.analysis.ownership", "ownership analysis complete", result.model_dump())
        return result


if __name__ == "__main__":
    print(OwnershipAnalyzer().analyze("RELIANCE").model_dump())
