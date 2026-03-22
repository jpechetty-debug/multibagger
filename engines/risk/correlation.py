"""Correlation analysis for candidate portfolios."""

from __future__ import annotations

import pandas as pd

import config
from data.db import db
from engines.analysis._common import compute_returns, current_timestamp, load_price_history
from models.schemas import CorrelationFilterResult


class CorrelationAnalyzer:
    """Computes correlations and filters highly correlated candidates."""

    def correlation_matrix(self, tickers: list[str]) -> CorrelationFilterResult:
        """Return the 6M correlation matrix for the given tickers."""

        normalized = [ticker.strip().upper() for ticker in tickers]
        frames = {}
        for ticker in normalized:
            history = load_price_history(ticker).tail(config.RISK_LOOKBACK_DAYS)
            returns = compute_returns(history)
            if not returns.empty:
                frames[ticker] = returns
        if not frames:
            raise ValueError("No valid return history available for correlation analysis")

        returns_frame = pd.DataFrame(frames).dropna(how="all")
        corr = returns_frame.corr().fillna(0.0)
        rejected_pairs: list[tuple[str, str, float]] = []
        allowed: list[str] = []
        for ticker in corr.columns:
            can_add = True
            for existing in allowed:
                correlation = float(corr.loc[ticker, existing])
                if abs(correlation) > config.CORRELATION_LIMIT:
                    rejected_pairs.append((existing, ticker, correlation))
                    can_add = False
                    break
            if can_add:
                allowed.append(ticker)
        result = CorrelationFilterResult(
            tickers=list(corr.columns),
            correlation_matrix={column: {idx: float(value) for idx, value in corr[column].items()} for column in corr.columns},
            rejected_pairs=rejected_pairs,
            allowed_tickers=allowed,
            as_of=current_timestamp(),
        )
        db.log_engine_event("INFO", "engines.risk.correlation", "correlation analysis complete", {"tickers": normalized, "allowed": allowed, "rejected_pairs": rejected_pairs})
        return result


if __name__ == "__main__":
    print(CorrelationAnalyzer().correlation_matrix(["RELIANCE", "TCS", "HDFCBANK"]).model_dump())
