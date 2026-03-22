"""Strategy-aware position sizing for swing, positional, and multibagger trades."""

from __future__ import annotations

import time

import config
from data.db import db
from models.schemas import StrategyTag, StrategySizingResult, VixState


class StrategySizer:
    """Compute position size based on the trading strategy.

    Risk fractions per strategy:
    - **Swing**: 0.5–1% of capital per trade
    - **Positional**: 1–2% of capital per trade
    - **Multibagger**: 0.5% per tranche × 3 tranches
    """

    RISK_FRACTIONS = {
        StrategyTag.SWING: (0.005, 0.01),       # 0.5% – 1%
        StrategyTag.POSITIONAL: (0.01, 0.02),    # 1% – 2%
        StrategyTag.MULTIBAGGER: (0.005, 0.005), # 0.5% per tranche (fixed)
    }

    def size(
        self,
        ticker: str,
        strategy_tag: StrategyTag,
        entry_price: float,
        stop_loss: float,
        capital: float = config.DEFAULT_PORTFOLIO_CAPITAL,
        confidence: float = 50.0,
        vix_state: VixState = VixState.NORMAL,
    ) -> StrategySizingResult:
        """Return strategy-specific position sizing.

        Parameters
        ----------
        confidence:
            0–100 confidence score.  High confidence uses the upper risk
            fraction for the strategy; low confidence uses the lower.
        """

        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if stop_loss >= entry_price:
            raise ValueError("stop_loss must be below entry_price for long positions")

        min_risk, max_risk = self.RISK_FRACTIONS.get(strategy_tag, (0.01, 0.01))

        # Scale risk fraction by confidence (0→min, 100→max)
        confidence_pct = max(0.0, min(1.0, confidence / 100.0))
        risk_pct = min_risk + (max_risk - min_risk) * confidence_pct

        # VIX adjustment
        if vix_state is VixState.HALF:
            risk_pct *= 0.5
        elif vix_state is VixState.HALT:
            risk_pct = 0.0

        # Swing-specific VIX halt at lower threshold
        if strategy_tag is StrategyTag.SWING and vix_state is VixState.HALF:
            risk_pct = 0.0  # Swing fully halts at HALF (VIX ≥ 25)

        # Multibagger never halts on VIX
        if strategy_tag is StrategyTag.MULTIBAGGER:
            risk_pct = max(min_risk, risk_pct)  # always at least minimum

        risk_value = capital * risk_pct
        loss_per_share = entry_price - stop_loss
        quantity = int(risk_value / loss_per_share) if loss_per_share > 0 else 0
        position_value = quantity * entry_price

        result = StrategySizingResult(
            ticker=ticker.strip().upper(),
            strategy_tag=strategy_tag,
            risk_pct=risk_pct,
            position_value=position_value,
            quantity=max(0, quantity),
            entry_price=entry_price,
            stop_loss=stop_loss,
            as_of=int(time.time()),
        )
        db.log_engine_event("INFO", "engines.risk.strategy_sizing", "strategy sizing computed", result.model_dump())
        return result


if __name__ == "__main__":
    result = StrategySizer().size("RELIANCE", StrategyTag.SWING, 100.0, 95.0, confidence=80.0)
    print(result.model_dump())
