"""Simple portfolio backtest simulator."""

from __future__ import annotations

from statistics import mean

import pandas as pd
import yfinance as yf

from models.schemas import BacktestResult, SignalResult
from ticker_list import to_yfinance


class PortfolioSimulator:
    """Runs a simple equal-weight backtest from signals."""

    def run(self, signals: list[SignalResult], start: str, end: str) -> BacktestResult:
        """Backtest BUY and WATCH signals over a date range."""

        trade_signals = [signal for signal in signals if signal.action in {"BUY", "WATCH"}]
        if not trade_signals:
            return BacktestResult(start=start, end=end, win_rate=0.0, total_pnl=0.0, best_ticker=None, worst_ticker=None, equity_curve=[], trade_outcomes=[])

        price_map: dict[str, pd.Series] = {}
        trade_outcomes: list[dict[str, float | str]] = []
        for signal in trade_signals:
            history = yf.download(to_yfinance(signal.ticker), start=start, end=end, progress=False, auto_adjust=True, threads=False)
            if history.empty:
                continue
            if isinstance(history.columns, pd.MultiIndex):
                history.columns = history.columns.get_level_values(0)
            close = history["Close"].dropna()
            if close.empty:
                continue
            price_map[signal.ticker] = close
            pnl = float(close.iloc[-1] / close.iloc[0] - 1)
            trade_outcomes.append({"ticker": signal.ticker, "pnl": pnl, "action": signal.action})

        if not price_map:
            return BacktestResult(start=start, end=end, win_rate=0.0, total_pnl=0.0, best_ticker=None, worst_ticker=None, equity_curve=[], trade_outcomes=[])

        frame = pd.concat({ticker: series for ticker, series in price_map.items()}, axis=1).dropna(how="all").ffill().dropna()
        normalized = frame / frame.iloc[0]
        portfolio_curve = normalized.mean(axis=1)
        equity_curve = [{"date": str(index.date()), "equity": float(value)} for index, value in portfolio_curve.items()]
        pnl_values = [float(outcome["pnl"]) for outcome in trade_outcomes]
        best = max(trade_outcomes, key=lambda outcome: float(outcome["pnl"]))
        worst = min(trade_outcomes, key=lambda outcome: float(outcome["pnl"]))
        return BacktestResult(
            start=start,
            end=end,
            win_rate=sum(value > 0 for value in pnl_values) / len(pnl_values) if pnl_values else 0.0,
            total_pnl=float(portfolio_curve.iloc[-1] - 1.0),
            best_ticker=str(best["ticker"]),
            worst_ticker=str(worst["ticker"]),
            equity_curve=equity_curve,
            trade_outcomes=trade_outcomes,
        )


if __name__ == "__main__":
    sample = [SignalResult(ticker="RELIANCE", action="BUY", confidence_score=85.0, reason_code="sample", generated_at=0)]
    print(PortfolioSimulator().run(sample, "2025-01-01", "2025-03-01").model_dump())
