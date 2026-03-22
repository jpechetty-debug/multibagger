"""ATR-based stop-loss and target calculator for swing trades."""

from __future__ import annotations

import config
from models.schemas import StopTarget


class StopTargetEngine:
    """Compute stop-loss and target prices from ATR."""

    def compute(
        self,
        entry_price: float,
        atr: float,
        stop_mult: float = config.SWING_ATR_STOP_MULT,
        reward_risk: float = config.SWING_REWARD_RISK,
    ) -> StopTarget:
        """Return a ``StopTarget`` given entry price and ATR.

        Parameters
        ----------
        entry_price:
            The price at which the swing trade is entered.
        atr:
            Average True Range for the instrument.
        stop_mult:
            Multiplier applied to ATR for the stop distance (default 1.5).
        reward_risk:
            Minimum reward-to-risk ratio (default 2.0 → 1:2 R/R).
        """

        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if atr <= 0:
            raise ValueError("atr must be positive")

        stop_distance = atr * stop_mult
        stop_loss = entry_price - stop_distance
        target_distance = stop_distance * reward_risk
        target_price = entry_price + target_distance

        return StopTarget(
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            target_price=round(target_price, 2),
            atr=atr,
            reward_risk=reward_risk,
        )


if __name__ == "__main__":
    result = StopTargetEngine().compute(entry_price=100.0, atr=3.5)
    print(result.model_dump())
