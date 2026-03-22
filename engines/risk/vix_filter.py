"""India VIX portfolio filter with per-strategy routing."""

from __future__ import annotations

from nsepython import indiavix

import config
from data.db import db
from models.schemas import StrategyTag, VixFilterResult, VixState


class VixFilter:
    """Maps India VIX levels into exposure states, with optional per-strategy routing."""

    def evaluate(self) -> VixFilterResult:
        """Return current VIX exposure guidance (default positional thresholds)."""

        try:
            vix_value = float(indiavix())
        except (TypeError, ValueError):
            vix_value = None
        if vix_value is None:
            state = VixState.HALF
            multiplier = 0.5
            reason = "India VIX unavailable; using defensive half exposure"
        elif vix_value >= config.VIX_HALT_THRESHOLD:
            state = VixState.HALT
            multiplier = 0.0
            reason = "India VIX above halt threshold"
        elif vix_value >= config.VIX_HALF_THRESHOLD:
            state = VixState.HALF
            multiplier = 0.5
            reason = "India VIX above caution threshold"
        else:
            state = VixState.NORMAL
            multiplier = 1.0
            reason = "India VIX within normal range"
        result = VixFilterResult(vix_value=vix_value, state=state, position_multiplier=multiplier, reason=reason, as_of=int(__import__("time").time()))
        db.log_engine_event("INFO", "engines.risk.vix_filter", "vix filter evaluated", result.model_dump())
        return result

    def evaluate_for_strategy(self, strategy_tag: StrategyTag) -> VixFilterResult:
        """Return VIX exposure guidance tuned to a specific strategy.

        Thresholds:
        - **Swing**: HALT at VIX >= 25, HALF at VIX >= 20
        - **Positional**: HALT at VIX >= 35, HALF at VIX >= 25 (default)
        - **Multibagger**: Never halts (always NORMAL)
        """

        if strategy_tag is StrategyTag.MULTIBAGGER:
            base = self.evaluate()
            return VixFilterResult(
                vix_value=base.vix_value,
                state=VixState.NORMAL,
                position_multiplier=1.0,
                reason="Multibagger strategy ignores VIX",
                as_of=base.as_of,
            )

        if strategy_tag is StrategyTag.SWING:
            try:
                vix_value = float(indiavix())
            except (TypeError, ValueError):
                vix_value = None
            if vix_value is None:
                state = VixState.HALT
                multiplier = 0.0
                reason = "India VIX unavailable; swing halted defensively"
            elif vix_value >= config.SWING_VIX_HALT:
                state = VixState.HALT
                multiplier = 0.0
                reason = f"India VIX {vix_value:.1f} >= swing halt {config.SWING_VIX_HALT}"
            elif vix_value >= 20.0:
                state = VixState.HALF
                multiplier = 0.5
                reason = f"India VIX {vix_value:.1f} elevated for swing"
            else:
                state = VixState.NORMAL
                multiplier = 1.0
                reason = "India VIX within swing-safe range"
            result = VixFilterResult(vix_value=vix_value, state=state, position_multiplier=multiplier, reason=reason, as_of=int(__import__("time").time()))
            db.log_engine_event("INFO", "engines.risk.vix_filter", "vix filter evaluated (swing)", result.model_dump())
            return result

        # Default: positional thresholds
        return self.evaluate()


if __name__ == "__main__":
    print(VixFilter().evaluate().model_dump())
    print(VixFilter().evaluate_for_strategy(StrategyTag.SWING).model_dump())
