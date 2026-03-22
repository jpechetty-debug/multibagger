"""Feature engineering for ML training and prediction."""

from __future__ import annotations

import numpy as np

from models.schemas import FundamentalData, ScoreResult


class MLFeatureEngineer:
    """Builds ML feature arrays from fundamentals and score outputs."""

    def build(self, data: FundamentalData, score: ScoreResult) -> tuple[np.ndarray, list[str]]:
        """Return a NumPy feature vector and aligned feature names."""

        names = [
            "total_score",
            "weighted_score",
            "meta_model_score",
            "roe_5y",
            "roe_ttm",
            "sales_growth_5y",
            "eps_growth_ttm",
            "cfo_to_pat",
            "debt_equity",
            "peg_ratio",
            "pe_ratio",
            "piotroski_score",
            "promoter_pct",
            "pledge_pct",
            "fii_delta",
            "dii_delta",
        ]
        values = np.asarray(
            [
                score.total_score,
                score.weighted_score,
                score.meta_model_score,
                data.roe_5y or 0.0,
                data.roe_ttm or 0.0,
                data.sales_growth_5y or 0.0,
                data.eps_growth_ttm or 0.0,
                data.cfo_to_pat or 0.0,
                data.debt_equity or 0.0,
                data.peg_ratio or 0.0,
                data.pe_ratio or 0.0,
                float(data.piotroski_score or 0.0),
                data.promoter_pct or 0.0,
                data.pledge_pct or 0.0,
                data.fii_delta or 0.0,
                data.dii_delta or 0.0,
            ],
            dtype=float,
        )
        return values, names


if __name__ == "__main__":
    print("MLFeatureEngineer import ok")
