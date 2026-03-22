"""
Realistic event-driven backtest with full transaction cost modelling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    brokerage_pct:    float = 0.0003     # 0.03% each side (discount broker)
    stt_pct:          float = 0.001      # 0.1% on sell (NSE equity delivery)
    exchange_pct:     float = 0.00003    # NSE turnover charges
    gst_on_brokerage: float = 0.18       # GST on brokerage
    slippage_adv_pct: float = 0.10       # participate at max 10% of ADV
    impact_coefficient: float = 0.15     # market impact = 0.15 × sqrt(participation)
    min_slippage_pct: float = 0.001      # minimum 0.1% slippage

    def total_cost(
        self,
        value: float,
        side: str,           # "BUY" or "SELL"
        adv_cr: float = 50,  # stock's average daily value ₹ crore
    ) -> tuple[float, dict]:
        """Returns (total_cost_rupees, breakdown_dict)."""
        brok  = value * self.brokerage_pct
        gst   = brok * self.gst_on_brokerage
        exch  = value * self.exchange_pct
        stt   = value * self.stt_pct if side == "SELL" else 0.0

        # Market impact — increases with participation rate
        participation = (value / 1e7) / (adv_cr + 1e-9)  # crore units
        impact_pct = max(
            self.impact_coefficient * math.sqrt(participation),
            self.min_slippage_pct,
        )
        slippage = value * impact_pct

        total = brok + gst + exch + stt + slippage
        return total, {
            "brokerage": round(brok, 2), "gst": round(gst, 2),
            "exchange":  round(exch, 2), "stt": round(stt, 2),
            "slippage":  round(slippage, 2), "total": round(total, 2),
        }


# ---------------------------------------------------------------------------
# Trade and position containers
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    ticker:       str
    entry_date:   str
    exit_date:    str | None
    entry_price:  float
    exit_price:   float | None
    shares:       int
    side:         str          # "LONG"
    entry_cost:   float
    exit_cost:    float
    pnl:          float        # net of all costs
    pnl_pct:      float
    holding_days: int
    regime_at_entry: str
    signal_score: float
    exit_reason:  str          # "TARGET" | "STOP" | "SIGNAL_EXIT" | "REBALANCE"


@dataclass
class BacktestResult:
    equity_curve:      pd.Series
    trade_log:         list[Trade]
    daily_returns:     pd.Series
    cost_summary:      dict[str, float]
    performance:       dict[str, float]
    regime_breakdown:  dict[str, dict]   # per-regime performance

    def summary(self) -> str:
        p = self.performance
        return (
            f"CAGR {p.get('cagr_pct',0):.1f}% | "
            f"Sharpe {p.get('sharpe',0):.2f} | "
            f"Calmar {p.get('calmar',0):.2f} | "
            f"Max DD {p.get('max_drawdown_pct',0):.1f}% | "
            f"Win rate {p.get('win_rate_pct',0):.1f}% | "
            f"Trades {p.get('total_trades',0)} | "
            f"Total costs ₹{self.cost_summary.get('total',0):,.0f}"
        )


# ---------------------------------------------------------------------------
# Main backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Daily bar event-driven backtest.
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        capital:          float = 1_000_000,
        signal_delay:     int   = 1,
        max_position_pct: float = 0.08,
        stop_loss_pct:    float = 0.07,
        take_profit_pct:  float = 0.15,
        max_positions:    int   = 15,
        cost_model:       CostModel | None = None,
    ):
        self.capital          = capital
        self.signal_delay     = signal_delay
        self.max_position_pct = max_position_pct
        self.stop_loss_pct    = stop_loss_pct
        self.take_profit_pct  = take_profit_pct
        self.max_positions    = max_positions
        self.costs            = cost_model or CostModel()

    def run(
        self,
        signals_df:  pd.DataFrame,   # columns: date, ticker, score, action
        prices_df:   pd.DataFrame,   # MultiIndex (date, ticker) with close, volume_cr
        regime_df:   pd.DataFrame | None = None,  # columns: date, state, multiplier
    ) -> BacktestResult:
        dates = sorted(signals_df["date"].unique())
        cash  = self.capital
        positions: dict[str, dict] = {}    # ticker → position info
        equity_curve:  list[tuple[str, float]] = []
        trade_log:     list[Trade] = []
        pending:       list[dict] = []     # delayed signals
        cost_totals    = {"brokerage": 0.0, "stt": 0.0, "slippage": 0.0, "total": 0.0}

        regime_map: dict[str, tuple[str, float]] = {}
        if regime_df is not None:
            for _, r in regime_df.iterrows():
                regime_map[str(r["date"])[:10]] = (r["state"], float(r.get("multiplier", 1.0)))

        for date in dates:
            date_str = str(date)[:10]
            regime, multiplier = regime_map.get(date_str, ("QUALITY", 0.80))

            # ── Price lookup for this date ────────────────────────────────────
            def get_price(ticker: str) -> float | None:
                try:
                    if isinstance(prices_df.index, pd.MultiIndex):
                        return float(prices_df.loc[(date, ticker), "close"])
                    elif date in prices_df.index and ticker in prices_df.columns:
                        return float(prices_df.loc[date, ticker])
                except (KeyError, TypeError):
                    return None

            def get_adv(ticker: str) -> float:
                try:
                    if isinstance(prices_df.index, pd.MultiIndex):
                        return float(prices_df.loc[(date, ticker), "volume_cr"])
                except (KeyError, TypeError):
                    pass
                return 50.0  # default ₹50cr ADV

            # ── Evaluate pending delayed signals ──────────────────────────────
            ready   = [p for p in pending if p["execute_date"] <= date_str]
            pending = [p for p in pending if p["execute_date"] >  date_str]

            for sig in ready:
                ticker = sig["ticker"]
                if ticker in positions or len(positions) >= self.max_positions:
                    continue
                price = get_price(ticker)
                if price is None:
                    continue

                size_pct = self.max_position_pct * multiplier
                alloc    = min(cash * size_pct, cash * 0.95)
                shares   = max(1, int(alloc / price))
                value    = shares * price
                cost, breakdown = self.costs.total_cost(value, "BUY", get_adv(ticker))

                if value + cost > cash:
                    continue

                cash -= (value + cost)
                positions[ticker] = {
                    "shares":    shares, "entry_price": price,
                    "entry_date": date_str, "entry_cost": cost,
                    "stop":      price * (1 - self.stop_loss_pct),
                    "target":    price * (1 + self.take_profit_pct),
                    "regime":    regime, "score": sig.get("score", 0),
                }
                for k in ("brokerage", "stt", "slippage", "total"):
                    cost_totals[k] += breakdown.get(k, 0)

            # ── Check exits for open positions ────────────────────────────────
            to_close: list[tuple[str, str]] = []
            for ticker, pos in positions.items():
                price = get_price(ticker)
                if price is None:
                    continue
                reason = None
                if price <= pos["stop"]:
                    reason = "STOP"
                elif price >= pos["target"]:
                    reason = "TARGET"

                # Check if signal reversed
                today_sigs = signals_df[
                    (signals_df["date"] == date) & (signals_df["ticker"] == ticker)
                ]
                if not today_sigs.empty:
                    act = today_sigs.iloc[0].get("action", "BUY")
                    if act in ("WEAK", "REJECT"):
                        reason = "SIGNAL_EXIT"

                if reason:
                    to_close.append((ticker, reason))

            for ticker, reason in to_close:
                pos   = positions.pop(ticker)
                price = get_price(ticker) or pos["entry_price"]
                value = pos["shares"] * price
                cost, breakdown = self.costs.total_cost(value, "SELL", get_adv(ticker))
                cash += (value - cost)
                for k in ("brokerage", "stt", "slippage", "total"):
                    cost_totals[k] += breakdown.get(k, 0)

                gross_pnl = (price - pos["entry_price"]) * pos["shares"]
                net_pnl   = gross_pnl - pos["entry_cost"] - cost
                holding   = len([d for d in dates if pos["entry_date"] <= str(d)[:10] <= date_str])

                trade_log.append(Trade(
                    ticker=ticker, entry_date=pos["entry_date"], exit_date=date_str,
                    entry_price=pos["entry_price"], exit_price=price,
                    shares=pos["shares"], side="LONG",
                    entry_cost=pos["entry_cost"], exit_cost=cost,
                    pnl=net_pnl, pnl_pct=net_pnl / (pos["entry_price"] * pos["shares"]),
                    holding_days=holding, regime_at_entry=pos["regime"],
                    signal_score=pos["score"], exit_reason=reason,
                ))

            # ── Queue new signals with delay ──────────────────────────────────
            new_sigs = signals_df[
                (signals_df["date"] == date) & (signals_df["action"] == "BUY")
            ]
            for _, sig in new_sigs.iterrows():
                ticker = sig["ticker"]
                if ticker not in positions:
                    exec_idx = min(dates.index(date) + self.signal_delay, len(dates) - 1)
                    pending.append({
                        "ticker": ticker, "score": sig.get("score", 0),
                        "execute_date": str(dates[exec_idx])[:10],
                    })

            # ── Mark-to-market portfolio value ────────────────────────────────
            port_value = cash
            for ticker, pos in positions.items():
                p = get_price(ticker)
                if p:
                    port_value += pos["shares"] * p
            equity_curve.append((date_str, port_value))

        # ── Performance metrics ───────────────────────────────────────────────
        eq = pd.Series(
            dict(equity_curve), dtype=float
        ).sort_index()
        ret   = eq.pct_change().dropna()
        years = len(ret) / self.TRADING_DAYS

        cagr     = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 0.01)) - 1 if len(eq) > 1 else 0
        sharpe   = (ret.mean() / ret.std() * math.sqrt(self.TRADING_DAYS)) if ret.std() > 0 else 0
        max_dd   = float((eq / eq.cummax() - 1).min())
        calmar   = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0
        wins     = [t for t in trade_log if t.pnl > 0]
        win_rate = len(wins) / len(trade_log) if trade_log else 0

        # Per-regime breakdown
        regime_breakdown: dict[str, dict] = {}
        for trade in trade_log:
            r = trade.regime_at_entry
            if r not in regime_breakdown:
                regime_breakdown[r] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
            regime_breakdown[r]["trades"] += 1
            regime_breakdown[r]["total_pnl"] += trade.pnl
            if trade.pnl > 0:
                regime_breakdown[r]["wins"] += 1
        for r, d in regime_breakdown.items():
            d["win_rate_pct"] = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0

        perf = {
            "cagr_pct":         round(cagr * 100, 2),
            "sharpe":           round(sharpe, 3),
            "calmar":           round(calmar, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct":     round(win_rate * 100, 1),
            "total_trades":     len(trade_log),
            "avg_holding_days": round(
                sum(t.holding_days for t in trade_log) / max(len(trade_log), 1), 1
            ),
            "total_return_pct": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2) if len(eq) > 1 else 0,
        }

        return BacktestResult(
            equity_curve=eq, trade_log=trade_log,
            daily_returns=ret, cost_summary={k: round(v, 2) for k, v in cost_totals.items()},
            performance=perf, regime_breakdown=regime_breakdown,
        )
