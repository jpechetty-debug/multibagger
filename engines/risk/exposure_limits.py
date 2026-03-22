"""Per-strategy exposure limits enforcer."""

from __future__ import annotations

import config
from models.schemas import ExposureLimitResult, StrategyTag


# Maximum portfolio allocation per strategy
STRATEGY_CAPS = {
    StrategyTag.SWING: config.SWING_MAX_PORTFOLIO_PCT,
    StrategyTag.POSITIONAL: config.POSITIONAL_MAX_PORTFOLIO_PCT,
    StrategyTag.MULTIBAGGER: config.MULTIBAGGER_MAX_PORTFOLIO_PCT,
}


class ExposureLimitsChecker:
    """Ensure that a proposed trade does not exceed the strategy's portfolio cap."""

    def check(
        self,
        strategy_tag: StrategyTag,
        current_strategy_value: float,
        proposed_value: float,
        total_portfolio_value: float,
    ) -> ExposureLimitResult:
        """Validate that the proposed trade stays within the strategy's exposure cap.

        Parameters
        ----------
        current_strategy_value:
            Total market value currently held under this strategy.
        proposed_value:
            Market value of the new position being added.
        total_portfolio_value:
            Total portfolio value (cash + equity across all strategies).
        """

        max_pct = STRATEGY_CAPS.get(strategy_tag, 0.30)

        if total_portfolio_value <= 0:
            return ExposureLimitResult(
                strategy_tag=strategy_tag,
                current_exposure_pct=0.0,
                proposed_exposure_pct=0.0,
                max_allowed_pct=max_pct,
                within_limit=False,
                headroom_pct=0.0,
            )

        current_pct = current_strategy_value / total_portfolio_value
        proposed_total = current_strategy_value + proposed_value
        proposed_pct = proposed_total / total_portfolio_value
        within_limit = proposed_pct <= max_pct
        headroom_pct = max(0.0, max_pct - proposed_pct)

        return ExposureLimitResult(
            strategy_tag=strategy_tag,
            current_exposure_pct=round(current_pct, 4),
            proposed_exposure_pct=round(proposed_pct, 4),
            max_allowed_pct=max_pct,
            within_limit=within_limit,
            headroom_pct=round(headroom_pct, 4),
        )


if __name__ == "__main__":
    result = ExposureLimitsChecker().check(
        StrategyTag.SWING,
        current_strategy_value=250_000.0,
        proposed_value=50_000.0,
        total_portfolio_value=1_000_000.0,
    )
    print(result.model_dump())
