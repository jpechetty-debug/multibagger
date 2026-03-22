"""Risk-based position sizing."""

from __future__ import annotations

import config
from data.db import db
from models.schemas import PositionSizingResult, VixState


class PositionSizer:
    """Computes Kelly-aware position sizing."""

    def size_position(
        self,
        ticker: str,
        entry_price: float,
        stop_loss_price: float,
        capital: float = config.DEFAULT_PORTFOLIO_CAPITAL,
        confidence_score: float = 50.0,
        vix_state: VixState = VixState.NORMAL,
        reward_to_risk: float = config.DEFAULT_REWARD_TO_RISK,
    ) -> PositionSizingResult:
        """Return position size based on risk budget and Kelly fraction."""

        if stop_loss_price >= entry_price:
            raise ValueError("stop_loss_price must be below entry_price for long positions")
        loss_per_share = entry_price - stop_loss_price
        implied_win_rate = max(0.05, min(0.95, confidence_score / 100.0))
        kelly_fraction = max(0.0, min(config.MAX_KELLY_FRACTION, implied_win_rate - (1 - implied_win_rate) / reward_to_risk))
        conviction = confidence_score >= 85.0
        risk_fraction = config.CONVICTION_RISK_FRACTION if conviction else config.NORMAL_RISK_FRACTION
        if vix_state is VixState.HALF:
            risk_fraction = min(risk_fraction, config.HIGH_VIX_RISK_FRACTION)
        elif vix_state is VixState.HALT:
            risk_fraction = 0.0
        target_risk_value = capital * risk_fraction
        target_position_value = min(capital * kelly_fraction, capital * config.MAX_SINGLE_STOCK_WEIGHT)
        quantity = int(min(target_risk_value / loss_per_share if loss_per_share else 0, target_position_value / entry_price if entry_price else 0))
        result = PositionSizingResult(
            ticker=ticker.strip().upper(),
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            capital=capital,
            risk_fraction=risk_fraction,
            kelly_fraction=kelly_fraction,
            target_position_value=max(0.0, quantity * entry_price),
            quantity=max(0, quantity),
            conviction=conviction,
            vix_state=vix_state,
            as_of=int(__import__("time").time()),
        )
        db.log_engine_event("INFO", "engines.risk.position_sizing", "position sizing computed", result.model_dump())
        return result


if __name__ == "__main__":
    print(PositionSizer().size_position("RELIANCE", entry_price=100.0, stop_loss_price=92.0, confidence_score=82.0).model_dump())
