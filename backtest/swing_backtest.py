"""Swing trade backtester: 6-month historical win rate and R/R analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

import config
from data.db import db
from engines.swing.technical_engine import TechnicalEngine
from engines.swing.breakout_scanner import BreakoutScanner
from engines.swing.stop_target import StopTargetEngine
from models.schemas import BacktestResult


@dataclass
class SwingTrade:
    """Record of a single simulated swing trade."""

    ticker: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target: float
    exit_date: str | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    outcome: str = "OPEN"  # WIN / LOSS / TIMEOUT


class SwingBacktester:
    """Simulate swing trades over historical data and compute performance metrics.

    Walk-forward methodology:
    1. For each day, compute technicals and breakout signals.
    2. On ENTRY conditions, open a simulated position.
    3. Exit when stop-loss / target is hit, or after 5 trading days (timeout).
    """

    MAX_HOLD_DAYS = 5

    def __init__(self) -> None:
        self.technical_engine = TechnicalEngine()
        self.breakout_scanner = BreakoutScanner()
        self.stop_target_engine = StopTargetEngine()

    def run(
        self,
        tickers: list[str],
        lookback_months: int = 6,
    ) -> BacktestResult:
        """Run a swing backtest across multiple tickers.

        Parameters
        ----------
        tickers:
            Stock ticker symbols to backtest.
        lookback_months:
            How many months of history to use.
        """

        all_trades: list[SwingTrade] = []
        period = f"{lookback_months}mo"

        for ticker in tickers:
            try:
                df = yf.download(f"{ticker}.NS", period=period, interval="1d", progress=False)
                if df is None or df.empty or len(df) < 50:
                    continue
                trades = self._simulate_ticker(ticker, df)
                all_trades.extend(trades)
            except Exception as exc:
                db.log_engine_event("WARN", "backtest.swing_backtest", "backtest ticker failed", {"ticker": ticker, "error": str(exc)})

        # --- Compute metrics ---
        if not all_trades:
            return BacktestResult(
                start="", end="", win_rate=0.0, total_pnl=0.0,
                best_ticker=None, worst_ticker=None,
            )

        closed_trades = [t for t in all_trades if t.outcome != "OPEN"]
        wins = sum(1 for t in closed_trades if t.outcome == "WIN")
        win_rate = wins / len(closed_trades) * 100 if closed_trades else 0.0
        total_pnl = sum(t.pnl for t in closed_trades)

        pnl_by_ticker: dict[str, float] = {}
        for t in closed_trades:
            pnl_by_ticker[t.ticker] = pnl_by_ticker.get(t.ticker, 0.0) + t.pnl
        best_ticker = max(pnl_by_ticker, key=lambda k: pnl_by_ticker[k]) if pnl_by_ticker else None
        worst_ticker = min(pnl_by_ticker, key=lambda k: pnl_by_ticker[k]) if pnl_by_ticker else None

        trade_outcomes = [
            {"ticker": t.ticker, "entry": t.entry_date, "exit": t.exit_date or "", "pnl": t.pnl, "outcome": t.outcome}
            for t in closed_trades[-50:]  # last 50 trades
        ]

        result = BacktestResult(
            start=str(closed_trades[0].entry_date) if closed_trades else "",
            end=str(closed_trades[-1].exit_date or closed_trades[-1].entry_date) if closed_trades else "",
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            best_ticker=best_ticker,
            worst_ticker=worst_ticker,
            trade_outcomes=trade_outcomes,
        )
        db.log_engine_event("INFO", "backtest.swing_backtest", "backtest completed", {
            "tickers": len(tickers), "trades": len(closed_trades), "win_rate": win_rate
        })
        return result

    def _simulate_ticker(self, ticker: str, df: pd.DataFrame) -> list[SwingTrade]:
        """Walk through daily bars and simulate swing entries/exits."""

        trades: list[SwingTrade] = []
        in_trade = False
        current_trade: SwingTrade | None = None
        bars_held = 0

        for i in range(50, len(df)):
            window = df.iloc[:i + 1]
            close = float(df["Close"].iloc[i])
            date_str = str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i])

            if in_trade and current_trade is not None:
                bars_held += 1
                # Check stop loss
                if close <= current_trade.stop_loss:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = close
                    current_trade.pnl = close - current_trade.entry_price
                    current_trade.outcome = "LOSS"
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                # Check target
                elif close >= current_trade.target:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = close
                    current_trade.pnl = close - current_trade.entry_price
                    current_trade.outcome = "WIN"
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
                # Timeout
                elif bars_held >= self.MAX_HOLD_DAYS:
                    current_trade.exit_date = date_str
                    current_trade.exit_price = close
                    current_trade.pnl = close - current_trade.entry_price
                    current_trade.outcome = "WIN" if current_trade.pnl > 0 else "LOSS"
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None
            elif not in_trade:
                # Evaluate entry signals
                try:
                    technicals = self.technical_engine.analyze(ticker, window)
                    breakout = self.breakout_scanner.scan(ticker, window)

                    entry_signals = 0
                    if technicals.rsi is not None and technicals.rsi <= config.SWING_RSI_OVERSOLD:
                        entry_signals += 1
                    if technicals.trend_bullish:
                        entry_signals += 1
                    if breakout.volume_surge:
                        entry_signals += 1
                    if breakout.consolidation_breakout:
                        entry_signals += 1

                    if entry_signals >= 3 and technicals.atr and technicals.atr > 0:
                        st = self.stop_target_engine.compute(close, technicals.atr)
                        current_trade = SwingTrade(
                            ticker=ticker,
                            entry_date=date_str,
                            entry_price=close,
                            stop_loss=st.stop_loss,
                            target=st.target_price,
                        )
                        in_trade = True
                        bars_held = 0
                except Exception:
                    continue

        return trades


if __name__ == "__main__":
    result = SwingBacktester().run(["RELIANCE", "TCS", "INFY"], lookback_months=6)
    print(result.model_dump())
