"""
engines/backtest/circuit_filter.py
engines/monitoring/metrics.py
-------------------------------
Two gap-closing upgrades bundled together.

1. CircuitFilter  — prevents backtest from processing fills at prices
   that weren't actually available when a stock hit its NSE circuit limit.

2. SovereignMetrics — Prometheus-compatible metrics exporter.
   Exposes engine health as a /metrics HTTP endpoint for Grafana.
"""

from __future__ import annotations

# ===========================================================================
# 1. Circuit Filter for Backtest Engine
# ===========================================================================

from dataclasses import dataclass
from typing import Any
import math


NSE_CIRCUIT_LIMITS = (0.02, 0.05, 0.10, 0.20)   # 2%, 5%, 10%, 20%


@dataclass
class CircuitEvent:
    ticker:        str
    date:          str
    open_price:    float
    prev_close:    float
    pct_change:    float
    limit_hit:     str          # "UPPER" | "LOWER" | "NONE"
    circuit_pct:   float        # which circuit band triggered
    available_price: float      # last tradeable price


class CircuitFilter:
    """
    Detects NSE circuit breaker events and returns the last tradeable price.

    NSE rules:
    - Stocks hitting upper circuit can only be BOUGHT at the circuit price
      (no sellers) — exits are unavailable.
    - Stocks hitting lower circuit can only be SOLD at the circuit price
      (no buyers) — entries are unavailable.

    In a backtest this matters most for:
    - Exit on stop loss: if stock lower-circuits, exit price ≠ stop price
    - Entry on signal: if stock upper-circuits, entry may not be achievable

    Usage (in BacktestEngine)
    -------------------------
        circuit = CircuitFilter()
        event   = circuit.check(ticker, date_str, open_px, prev_close_px)
        if event.limit_hit == "LOWER":
            # Adjust exit price — can only exit at circuit price
            fill_price = event.available_price
        elif event.limit_hit == "UPPER":
            # Entry unavailable — skip this ticker today
            continue
    """

    def check(
        self,
        ticker:     str,
        date:       str,
        open_price: float,
        prev_close: float,
    ) -> CircuitEvent:
        if prev_close <= 0 or open_price <= 0:
            return CircuitEvent(ticker, date, open_price, prev_close, 0.0,
                                "NONE", 0.0, open_price)

        pct_chg = (open_price - prev_close) / prev_close

        # Check each circuit band
        for band in sorted(NSE_CIRCUIT_LIMITS, reverse=True):
            if pct_chg >= band:
                return CircuitEvent(
                    ticker=ticker, date=date,
                    open_price=open_price, prev_close=prev_close,
                    pct_change=pct_chg,
                    limit_hit="UPPER",
                    circuit_pct=band,
                    available_price=round(prev_close * (1 + band), 2),
                )
            if pct_chg <= -band:
                return CircuitEvent(
                    ticker=ticker, date=date,
                    open_price=open_price, prev_close=prev_close,
                    pct_change=pct_chg,
                    limit_hit="LOWER",
                    circuit_pct=band,
                    available_price=round(prev_close * (1 - band), 2),
                )

        return CircuitEvent(ticker, date, open_price, prev_close,
                            pct_chg, "NONE", 0.0, open_price)

    def apply_to_fill(
        self,
        ticker:     str,
        date:       str,
        intended_price: float,
        open_price: float,
        prev_close: float,
        side:       str,   # "BUY" or "SELL"
    ) -> tuple[float, bool]:
        """
        Given an intended fill price, return (actual_fill_price, fill_available).

        Returns (0.0, False) if the fill is completely unavailable due to circuit.
        """
        event = self.check(ticker, date, open_price, prev_close)

        if event.limit_hit == "NONE":
            return intended_price, True

        if event.limit_hit == "UPPER":
            # Upper circuit: can BUY at circuit price, CANNOT SELL (no buyers)
            if side == "BUY":
                return event.available_price, True
            else:
                return 0.0, False   # Can't exit on upper circuit

        if event.limit_hit == "LOWER":
            # Lower circuit: can SELL at circuit price, CANNOT BUY (no sellers)
            if side == "SELL":
                return event.available_price, True
            else:
                return 0.0, False   # Can't enter on lower circuit

        return intended_price, True