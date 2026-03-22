"""
engines/backtest/backtest_engine.py
-------------------------------------
Realistic event-driven backtest — upgraded with:
  • Circuit filter (NSE upper/lower circuit price capping)
  • Partial fill model (fills scale with ADV participation rate)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from engines.backtest.circuit_filter import CircuitFilter


@dataclass
class CostModel:
    brokerage_pct:      float = 0.0003
    stt_pct:            float = 0.001
    exchange_pct:       float = 0.00003
    gst_on_brokerage:   float = 0.18
    impact_coefficient: float = 0.15
    min_slippage_pct:   float = 0.001

    def total_cost(self, value: float, side: str, adv_cr: float = 50.0) -> tuple[float, dict]:
        brok  = value * self.brokerage_pct
        gst   = brok  * self.gst_on_brokerage
        exch  = value * self.exchange_pct
        stt   = value * self.stt_pct if side == "SELL" else 0.0
        part  = (value / 1e7) / (adv_cr + 1e-9)
        slip  = value * max(self.impact_coefficient * math.sqrt(part), self.min_slippage_pct)
        total = brok + gst + exch + stt + slip
        return total, {"brokerage": round(brok,2), "gst": round(gst,2),
                       "exchange": round(exch,2), "stt": round(stt,2),
                       "slippage": round(slip,2), "total": round(total,2)}


class PartialFillModel:
    """Scales shares filled to ADV participation limit."""
    def __init__(self, max_adv_participation: float = 0.10):
        self.max_adv_participation = max_adv_participation

    def apply(self, requested: int, price: float, adv_cr: float) -> int:
        if requested <= 0:
            return 0
        max_fill = self.max_adv_participation * adv_cr * 1e7
        frac = min(1.0, max_fill / max(requested * price, 1.0))
        return max(1, int(requested * frac))


@dataclass
class Trade:
    ticker:           str
    entry_date:       str
    exit_date:        str | None
    entry_price:      float
    exit_price:       float | None
    shares_requested: int
    shares_filled:    int
    fill_pct:         float
    side:             str
    entry_cost:       float
    exit_cost:        float
    pnl:              float
    pnl_pct:          float
    holding_days:     int
    regime_at_entry:  str
    signal_score:     float
    exit_reason:      str
    circuit_adjusted: bool


@dataclass
class BacktestResult:
    equity_curve:       pd.Series
    trade_log:          list[Trade]
    daily_returns:      pd.Series
    cost_summary:       dict[str, float]
    performance:        dict[str, float]
    regime_breakdown:   dict[str, dict]
    partial_fill_stats: dict[str, float]
    circuit_events:     int

    def summary(self) -> str:
        p = self.performance
        pf = self.partial_fill_stats
        return (
            f"CAGR {p.get('cagr_pct',0):.1f}% | Sharpe {p.get('sharpe',0):.2f} | "
            f"Calmar {p.get('calmar',0):.2f} | Max DD {p.get('max_drawdown_pct',0):.1f}% | "
            f"Win {p.get('win_rate_pct',0):.1f}% | Trades {p.get('total_trades',0)} | "
            f"Avg fill {pf.get('avg_fill_pct',100):.0f}% | Circuit {self.circuit_events}"
        )


class BacktestEngine:
    TRADING_DAYS = 252

    def __init__(
        self,
        capital:               float = 1_000_000,
        signal_delay:          int   = 1,
        max_position_pct:      float = 0.08,
        stop_loss_pct:         float = 0.07,
        take_profit_pct:       float = 0.15,
        max_positions:         int   = 15,
        cost_model:            CostModel | None = None,
        enable_circuit_filter: bool  = True,
        enable_partial_fills:  bool  = True,
        max_adv_participation: float = 0.10,
    ):
        self.capital          = capital
        self.signal_delay     = signal_delay
        self.max_position_pct = max_position_pct
        self.stop_loss_pct    = stop_loss_pct
        self.take_profit_pct  = take_profit_pct
        self.max_positions    = max_positions
        self.costs            = cost_model or CostModel()
        self.circuit          = CircuitFilter() if enable_circuit_filter else None
        self.partial_fill     = PartialFillModel(max_adv_participation) if enable_partial_fills else None

    def run(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame,
            regime_df: pd.DataFrame | None = None) -> BacktestResult:

        dates = sorted(signals_df["date"].unique())
        cash  = self.capital
        positions:    dict[str, dict] = {}
        equity_curve: list[tuple[str, float]] = []
        trade_log:    list[Trade] = []
        pending:      list[dict] = []
        cost_totals   = {"brokerage": 0.0, "stt": 0.0, "slippage": 0.0, "total": 0.0}
        circuit_events = 0
        prev_close_map: dict[str, float] = {}

        regime_map: dict[str, tuple[str, float]] = {}
        if regime_df is not None:
            for _, r in regime_df.iterrows():
                regime_map[str(r["date"])[:10]] = (r["state"], float(r.get("multiplier", 1.0)))

        for date in dates:
            date_str = str(date)[:10]
            regime, multiplier = regime_map.get(date_str, ("QUALITY", 0.80))

            def get_price(t: str) -> float | None:
                try:
                    if isinstance(prices_df.index, pd.MultiIndex):
                        return float(prices_df.loc[(date, t), "close"])
                    elif date in prices_df.index and t in prices_df.columns:
                        return float(prices_df.loc[date, t])
                except (KeyError, TypeError):
                    return None

            def get_adv(t: str) -> float:
                try:
                    if isinstance(prices_df.index, pd.MultiIndex):
                        return float(prices_df.loc[(date, t), "volume_cr"])
                except (KeyError, TypeError):
                    pass
                return 50.0

            def get_open(t: str) -> float | None:
                try:
                    if isinstance(prices_df.index, pd.MultiIndex):
                        return float(prices_df.loc[(date, t), "open"])
                except (KeyError, TypeError):
                    pass
                return get_price(t)

            def circ_adjust(t: str, intended: float, side: str) -> tuple[float, bool]:
                nonlocal circuit_events
                if self.circuit is None:
                    return intended, False
                open_px  = get_open(t) or intended
                prev_cls = prev_close_map.get(t, open_px)
                fill_px, avail = self.circuit.apply_to_fill(t, date_str, intended, open_px, prev_cls, side)
                if not avail:
                    return 0.0, True
                if abs(fill_px - intended) > 0.001:
                    circuit_events += 1
                    return fill_px, True
                return intended, False

            # Execute pending entries
            ready   = [p for p in pending if p["execute_date"] <= date_str]
            pending = [p for p in pending if p["execute_date"] >  date_str]

            for sig in ready:
                t = sig["ticker"]
                if t in positions or len(positions) >= self.max_positions:
                    continue
                price = get_price(t)
                if price is None:
                    continue
                fill_px, cadj = circ_adjust(t, price, "BUY")
                if fill_px == 0.0:
                    continue
                adv    = get_adv(t)
                alloc  = min(cash * self.max_position_pct * multiplier, cash * 0.95)
                req    = max(1, int(alloc / fill_px))
                actual = self.partial_fill.apply(req, fill_px, adv) if self.partial_fill else req
                value  = actual * fill_px
                cost, bd = self.costs.total_cost(value, "BUY", adv)
                if value + cost > cash:
                    continue
                cash -= (value + cost)
                positions[t] = {
                    "shares": actual, "shares_requested": req,
                    "entry_price": fill_px, "entry_date": date_str,
                    "entry_cost": cost,
                    "stop":   fill_px * (1 - self.stop_loss_pct),
                    "target": fill_px * (1 + self.take_profit_pct),
                    "regime": regime, "score": sig.get("score", 0), "cadj": cadj,
                }
                for k in ("brokerage", "stt", "slippage", "total"):
                    cost_totals[k] += bd.get(k, 0)

            # Check exits
            to_close: list[tuple[str, str]] = []
            for t, pos in positions.items():
                price = get_price(t)
                if price is None:
                    continue
                reason = None
                if price <= pos["stop"]:     reason = "STOP"
                elif price >= pos["target"]: reason = "TARGET"
                sigs = signals_df[(signals_df["date"] == date) & (signals_df["ticker"] == t)]
                if not sigs.empty and sigs.iloc[0].get("action") in ("WEAK", "REJECT"):
                    reason = "SIGNAL_EXIT"
                if reason:
                    to_close.append((t, reason))

            for t, reason in to_close:
                pos   = positions.pop(t)
                price = get_price(t) or pos["entry_price"]
                adv   = get_adv(t)
                fill_px, cadj = circ_adjust(t, price, "SELL")
                if fill_px == 0.0:
                    positions[t] = pos  # lower circuit — hold, retry tomorrow
                    continue
                value = pos["shares"] * fill_px
                cost, bd = self.costs.total_cost(value, "SELL", adv)
                cash += (value - cost)
                for k in ("brokerage", "stt", "slippage", "total"):
                    cost_totals[k] += bd.get(k, 0)
                gross    = (fill_px - pos["entry_price"]) * pos["shares"]
                net_pnl  = gross - pos["entry_cost"] - cost
                holding  = len([d for d in dates if pos["entry_date"] <= str(d)[:10] <= date_str])
                fill_pct = pos["shares"] / max(pos["shares_requested"], 1)
                trade_log.append(Trade(
                    ticker=t, entry_date=pos["entry_date"], exit_date=date_str,
                    entry_price=pos["entry_price"], exit_price=fill_px,
                    shares_requested=pos["shares_requested"], shares_filled=pos["shares"],
                    fill_pct=round(fill_pct, 3), side="LONG",
                    entry_cost=pos["entry_cost"], exit_cost=cost,
                    pnl=net_pnl, pnl_pct=net_pnl / (pos["entry_price"] * pos["shares"]),
                    holding_days=holding, regime_at_entry=pos["regime"],
                    signal_score=pos["score"], exit_reason=reason,
                    circuit_adjusted=cadj or pos.get("cadj", False),
                ))

            # Queue new signals
            new_sigs = signals_df[(signals_df["date"] == date) & (signals_df["action"] == "BUY")]
            for _, sig in new_sigs.iterrows():
                t = sig["ticker"]
                if t not in positions:
                    ei = min(dates.index(date) + self.signal_delay, len(dates) - 1)
                    pending.append({"ticker": t, "score": sig.get("score", 0),
                                    "execute_date": str(dates[ei])[:10]})

            for t in list(positions) + [s["ticker"] for s in pending]:
                p = get_price(t)
                if p:
                    prev_close_map[t] = p

            pv = cash + sum(pos["shares"] * (get_price(t) or pos["entry_price"])
                            for t, pos in positions.items())
            equity_curve.append((date_str, pv))

        # Performance
        eq    = pd.Series(dict(equity_curve), dtype=float).sort_index()
        ret   = eq.pct_change().dropna()
        years = len(ret) / self.TRADING_DAYS
        cagr  = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 0.01)) - 1 if len(eq) > 1 else 0
        sharpe = ret.mean() / ret.std() * math.sqrt(self.TRADING_DAYS) if ret.std() > 0 else 0
        max_dd = float((eq / eq.cummax() - 1).min())
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0
        wins   = [t for t in trade_log if t.pnl > 0]
        wr     = len(wins) / len(trade_log) if trade_log else 0

        rbd: dict[str, dict] = {}
        for tr in trade_log:
            r = tr.regime_at_entry
            if r not in rbd:
                rbd[r] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
            rbd[r]["trades"] += 1
            rbd[r]["total_pnl"] += tr.pnl
            if tr.pnl > 0:
                rbd[r]["wins"] += 1
        for r, d in rbd.items():
            d["win_rate_pct"] = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0

        fp = [t.fill_pct for t in trade_log]
        pf_stats = {
            "avg_fill_pct":       round(float(np.mean(fp)) * 100, 1) if fp else 100.0,
            "partial_fill_count": sum(1 for f in fp if f < 0.99),
            "full_fill_count":    sum(1 for f in fp if f >= 0.99),
        }
        perf = {
            "cagr_pct":         round(cagr * 100, 2),
            "sharpe":           round(sharpe, 3),
            "calmar":           round(calmar, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "win_rate_pct":     round(wr * 100, 1),
            "total_trades":     len(trade_log),
            "avg_holding_days": round(sum(t.holding_days for t in trade_log) / max(len(trade_log), 1), 1),
            "total_return_pct": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 2) if len(eq) > 1 else 0,
        }
        return BacktestResult(
            equity_curve=eq, trade_log=trade_log, daily_returns=ret,
            cost_summary={k: round(v, 2) for k, v in cost_totals.items()},
            performance=perf, regime_breakdown=rbd,
            partial_fill_stats=pf_stats, circuit_events=circuit_events,
        )
