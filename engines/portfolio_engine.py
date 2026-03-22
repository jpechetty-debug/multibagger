"""Portfolio management engine."""

from __future__ import annotations

from collections import defaultdict

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.risk.portfolio_limits import PortfolioLimitsChecker
from engines.risk.position_sizing import PositionSizer
from engines.risk.vix_filter import VixFilter
from models.schemas import PortfolioPosition, PortfolioSnapshot, SignalResult


class PortfolioEngine:
    """Manages open positions, cash, and rebalancing."""

    def __init__(
        self,
        fetcher: DataFetcher | None = None,
        limits_checker: PortfolioLimitsChecker | None = None,
        position_sizer: PositionSizer | None = None,
        vix_filter: VixFilter | None = None,
    ) -> None:
        """Initialize the portfolio engine."""

        self.fetcher = fetcher or DataFetcher()
        self.limits_checker = limits_checker or PortfolioLimitsChecker(self.fetcher)
        self.position_sizer = position_sizer or PositionSizer()
        self.vix_filter = vix_filter or VixFilter()

    def get_cash(self) -> float:
        """Return available portfolio cash from transaction history."""

        with db.connection("stocks") as conn:
            row = conn.execute("SELECT COALESCE(SUM(cash_delta), 0) AS cash_delta FROM portfolio_transactions").fetchone()
        return config.DEFAULT_PORTFOLIO_CAPITAL + float(row["cash_delta"] if row is not None else 0.0)

    def snapshot(self) -> PortfolioSnapshot:
        """Return the current portfolio snapshot."""

        positions = []
        for position in db.list_portfolio_positions():
            latest = self.fetcher.fetch(position.ticker)
            updated_position = PortfolioPosition(
                ticker=position.ticker,
                sector=latest.sector or position.sector,
                quantity=position.quantity,
                avg_cost=position.avg_cost,
                last_price=latest.price or position.last_price,
                market_value=position.quantity * (latest.price or position.last_price),
                stop_loss=position.stop_loss,
                conviction=position.conviction,
                opened_at=position.opened_at,
                updated_at=int(__import__("time").time()),
            )
            db.upsert_portfolio_position(updated_position)
            positions.append(updated_position)

        equity_value = sum(position.market_value for position in positions)
        cash = self.get_cash()
        sector_exposure = defaultdict(float)
        total = cash + equity_value
        for position in positions:
            sector_exposure[position.sector or "Unknown"] += position.market_value / total if total else 0.0
        return PortfolioSnapshot(
            cash=cash,
            equity_value=equity_value,
            total_value=cash + equity_value,
            positions=positions,
            sector_exposure=dict(sector_exposure),
            as_of=int(__import__("time").time()),
        )

    def add_position(self, signal: SignalResult, stop_loss: float | None = None) -> PortfolioPosition:
        """Open or increase a position from a signal."""

        if signal.action not in {"BUY", "WATCH"}:
            raise ValueError("Only BUY or WATCH signals can be added to the portfolio")
        data = self.fetcher.fetch(signal.ticker)
        if data.price is None:
            raise ValueError(f"No price available for {signal.ticker}")
        snapshot = self.snapshot()
        vix_result = self.vix_filter.evaluate()
        stop_loss = stop_loss or data.price * (1.0 - config.DEFAULT_STOP_LOSS_PCT)
        sizing = self.position_sizer.size_position(
            ticker=signal.ticker,
            entry_price=data.price,
            stop_loss_price=stop_loss,
            capital=snapshot.total_value,
            confidence_score=signal.confidence_score,
            vix_state=vix_result.state,
        )
        if sizing.quantity <= 0:
            raise ValueError("Position size resolved to zero shares")
        candidate_value = sizing.quantity * data.price
        if candidate_value > snapshot.cash:
            raise ValueError(f"Insufficient cash for {signal.ticker}: required {candidate_value:.2f}, available {snapshot.cash:.2f}")
        limits = self.limits_checker.check(
            signal.ticker,
            candidate_value,
            positions=snapshot.positions,
            data=data,
            portfolio_total_value=snapshot.total_value,
        )
        if not limits.passed:
            raise ValueError(f"Portfolio limits violated: {', '.join(limits.violations)}")

        existing = next((position for position in snapshot.positions if position.ticker == signal.ticker), None)
        now_ts = int(__import__("time").time())
        if existing:
            new_quantity = existing.quantity + sizing.quantity
            avg_cost = ((existing.avg_cost * existing.quantity) + candidate_value) / new_quantity
            position = PortfolioPosition(
                ticker=signal.ticker,
                sector=data.sector,
                quantity=new_quantity,
                avg_cost=avg_cost,
                last_price=data.price,
                market_value=new_quantity * data.price,
                stop_loss=min(existing.stop_loss, stop_loss),
                conviction=existing.conviction or sizing.conviction,
                opened_at=existing.opened_at,
                updated_at=now_ts,
            )
        else:
            position = PortfolioPosition(
                ticker=signal.ticker,
                sector=data.sector,
                quantity=sizing.quantity,
                avg_cost=data.price,
                last_price=data.price,
                market_value=sizing.quantity * data.price,
                stop_loss=stop_loss,
                conviction=sizing.conviction,
                opened_at=now_ts,
                updated_at=now_ts,
            )

        db.upsert_portfolio_position(position)
        db.log_portfolio_transaction(signal.ticker, "BUY", sizing.quantity, data.price, -candidate_value, {"signal_action": signal.action})
        db.log_engine_event("INFO", "engines.portfolio_engine", "position added", position.model_dump())
        return position

    def remove_position(self, ticker: str, quantity: int | None = None) -> PortfolioPosition:
        """Reduce or close an existing position."""

        normalized_ticker = ticker.strip().upper()
        snapshot = self.snapshot()
        existing = next((position for position in snapshot.positions if position.ticker == normalized_ticker), None)
        if existing is None:
            raise ValueError(f"No open position found for {normalized_ticker}")
        quantity = quantity or existing.quantity
        if quantity <= 0 or quantity > existing.quantity:
            raise ValueError("Invalid quantity to remove")
        latest = self.fetcher.fetch(normalized_ticker)
        sale_value = quantity * (latest.price or existing.last_price)
        if quantity == existing.quantity:
            db.delete_portfolio_position(normalized_ticker)
            result = existing
        else:
            remaining_quantity = existing.quantity - quantity
            result = PortfolioPosition(
                ticker=existing.ticker,
                sector=existing.sector,
                quantity=remaining_quantity,
                avg_cost=existing.avg_cost,
                last_price=latest.price or existing.last_price,
                market_value=remaining_quantity * (latest.price or existing.last_price),
                stop_loss=existing.stop_loss,
                conviction=existing.conviction,
                opened_at=existing.opened_at,
                updated_at=int(__import__("time").time()),
            )
            db.upsert_portfolio_position(result)
        db.log_portfolio_transaction(normalized_ticker, "SELL", quantity, latest.price or existing.last_price, sale_value, {})
        db.log_engine_event("INFO", "engines.portfolio_engine", "position removed", {"ticker": normalized_ticker, "quantity": quantity})
        return result

    def rebalance(self, target_weights: dict[str, float]) -> PortfolioSnapshot:
        """Rebalance existing positions toward target weights."""

        snapshot = self.snapshot()
        total_value = snapshot.total_value
        for position in snapshot.positions:
            target_weight = target_weights.get(position.ticker, 0.0)
            target_value = total_value * target_weight
            drift = position.market_value - target_value
            if abs(drift) / total_value <= config.DEFAULT_REBALANCE_TOLERANCE:
                continue
            if drift > 0:
                quantity_to_sell = int(drift / position.last_price)
                if quantity_to_sell > 0:
                    self.remove_position(position.ticker, quantity=min(quantity_to_sell, position.quantity))
        return self.snapshot()


if __name__ == "__main__":
    print(PortfolioEngine().snapshot().model_dump())
