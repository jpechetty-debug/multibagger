"""Earnings revision analysis engine."""

from __future__ import annotations

import pandas as pd

from data.db import db
from data.fetcher import DataFetcher
from engines.analysis._common import current_timestamp, load_earnings_data, log_event, normalize_score
from models.schemas import EarningsRevisionAnalysis, FundamentalData


class EarningsRevisionAnalyzer:
    """Computes earnings beat streaks and estimate revision trends."""

    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        """Initialize the analyzer."""

        self.fetcher = fetcher or DataFetcher()

    def analyze(self, ticker: str, data: FundamentalData | None = None) -> EarningsRevisionAnalysis:
        """Return earnings revision metrics for a ticker."""

        normalized_ticker = ticker.strip().upper()
        _ = data or db.get_fundamental(normalized_ticker, effective=True) or self.fetcher.fetch(normalized_ticker)
        earnings_data = load_earnings_data(normalized_ticker)
        earnings_dates = earnings_data["earnings_dates"]
        eps_trend = earnings_data["eps_trend"]
        eps_revisions = earnings_data["eps_revisions"]

        beat_streak = self._beat_streak(earnings_dates)
        estimate_trend_pct = self._estimate_trend(eps_trend)
        surprise_mean = self._surprise_mean(earnings_dates)
        revision_signal = self._revision_signal(estimate_trend_pct, eps_revisions)

        beat_score = normalize_score(float(beat_streak), 0.0, 4.0)
        trend_score = normalize_score(estimate_trend_pct, -0.20, 0.20)
        surprise_score = normalize_score(surprise_mean, -20.0, 20.0)
        score = beat_score * 0.4 + trend_score * 0.35 + surprise_score * 0.25

        result = EarningsRevisionAnalysis(
            ticker=normalized_ticker,
            beat_streak=beat_streak,
            revision_signal=revision_signal,
            estimate_trend_pct=estimate_trend_pct,
            surprise_mean=surprise_mean,
            score=score,
            as_of=current_timestamp(),
        )
        log_event("engines.analysis.earnings_revision", "earnings revision analysis complete", result.model_dump())
        return result

    def _beat_streak(self, earnings_dates: pd.DataFrame) -> int:
        """Return consecutive positive earnings surprises."""

        if earnings_dates.empty or {"EPS Estimate", "Reported EPS"} - set(earnings_dates.columns):
            return 0
        frame = earnings_dates.dropna(subset=["EPS Estimate", "Reported EPS"]).sort_index(ascending=False)
        streak = 0
        for _, row in frame.iterrows():
            if float(row["Reported EPS"]) > float(row["EPS Estimate"]):
                streak += 1
            else:
                break
        return streak

    def _estimate_trend(self, eps_trend: pd.DataFrame) -> float | None:
        """Return current quarter estimate trend versus 90 days ago."""

        if eps_trend.empty or "0q" not in eps_trend.index:
            return None
        row = eps_trend.loc["0q"]
        current = row.get("current")
        prior = row.get("90daysAgo")
        if current in (None, 0) or prior in (None, 0):
            return None
        return float(current / prior - 1)

    def _surprise_mean(self, earnings_dates: pd.DataFrame) -> float | None:
        """Return mean surprise percentage over recent quarters."""

        if earnings_dates.empty or "Surprise(%)" not in earnings_dates.columns:
            return None
        values = earnings_dates["Surprise(%)"].dropna().head(4)
        return float(values.mean()) if not values.empty else None

    def _revision_signal(self, estimate_trend_pct: float | None, eps_revisions: pd.DataFrame) -> str:
        """Return upgrade, downgrade, or neutral revision flag."""

        if eps_revisions.empty or "0q" not in eps_revisions.index:
            if estimate_trend_pct is None:
                return "NEUTRAL"
            return "UPGRADE" if estimate_trend_pct > 0 else "DOWNGRADE" if estimate_trend_pct < 0 else "NEUTRAL"
        row = eps_revisions.loc["0q"]
        up = float(row.get("upLast30days", 0))
        down = float(row.get("downLast30days", 0))
        if estimate_trend_pct is not None and estimate_trend_pct > 0 and up >= down:
            return "UPGRADE"
        if estimate_trend_pct is not None and estimate_trend_pct < 0 and down > up:
            return "DOWNGRADE"
        return "NEUTRAL"


if __name__ == "__main__":
    print(EarningsRevisionAnalyzer().analyze("RELIANCE").model_dump())
