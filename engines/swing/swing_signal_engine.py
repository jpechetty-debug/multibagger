"""Swing signal engine: combines technical indicators and breakout patterns to emit ENTRY/EXIT/HOLD."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

import config
from data.db import db
from data.fetcher import DataFetcher
from engines.risk.vix_filter import VixFilter
from engines.swing.breakout_scanner import BreakoutScanner
from engines.swing.stop_target import StopTargetEngine
from engines.swing.technical_engine import TechnicalEngine
from models.schemas import SwingAction, SwingSignal, StrategyTag, VixState


@dataclass
class SwingDependencies:
    """Dependency bundle for the swing signal engine."""

    fetcher: DataFetcher
    technical_engine: TechnicalEngine
    breakout_scanner: BreakoutScanner
    stop_target_engine: StopTargetEngine
    vix_filter: VixFilter


class SwingSignalEngine:
    """Evaluates swing trade signals (ENTRY / EXIT / HOLD).

    Parameters
    ----------
    dependencies:
        Optional pre-built dependency bundle.  If *None*, defaults are
        constructed automatically.
    """

    def __init__(self, dependencies: SwingDependencies | None = None) -> None:
        if dependencies is None:
            fetcher = DataFetcher()
            dependencies = SwingDependencies(
                fetcher=fetcher,
                technical_engine=TechnicalEngine(),
                breakout_scanner=BreakoutScanner(),
                stop_target_engine=StopTargetEngine(),
                vix_filter=VixFilter(),
            )
        self.deps = dependencies

    def evaluate(self, ticker: str, price_df: pd.DataFrame | None = None) -> SwingSignal:
        """Evaluate a swing trade signal for *ticker*.

        Parameters
        ----------
        ticker:
            Stock ticker.
        price_df:
            Optional pre-fetched price history DataFrame.
        """

        normalized = ticker.strip().upper()
        now_ts = int(time.time())

        # ---- VIX gate (swing uses stricter 25 threshold) ----
        vix_result = self.deps.vix_filter.evaluate()
        if vix_result.vix_value is not None and vix_result.vix_value >= config.SWING_VIX_HALT:
            return SwingSignal(
                ticker=normalized,
                action=SwingAction.HOLD,
                reason=f"VIX {vix_result.vix_value:.1f} >= swing halt {config.SWING_VIX_HALT}",
                generated_at=now_ts,
            )

        # ---- Fetch price history if not provided ----
        if price_df is None:
            import yfinance as yf

            price_df = yf.download(
                f"{normalized}.NS",
                period=config.PRICE_HISTORY_PERIOD,
                interval=config.PRICE_HISTORY_INTERVAL,
                progress=False,
            )
        if price_df is None or price_df.empty or len(price_df) < 30:
            return SwingSignal(
                ticker=normalized,
                action=SwingAction.HOLD,
                reason="Insufficient price history",
                generated_at=now_ts,
            )

        # ---- Technical analysis ----
        technicals = self.deps.technical_engine.analyze(normalized, price_df)
        breakout = self.deps.breakout_scanner.scan(normalized, price_df)

        # ---- Decision logic ----
        entry_conditions: list[str] = []
        exit_conditions: list[str] = []

        # RSI oversold + turning up → entry
        if technicals.rsi is not None and technicals.rsi <= config.SWING_RSI_OVERSOLD:
            entry_conditions.append(f"RSI oversold ({technicals.rsi:.1f})")

        # RSI overbought → exit
        if technicals.rsi is not None and technicals.rsi >= config.SWING_RSI_OVERBOUGHT:
            exit_conditions.append(f"RSI overbought ({technicals.rsi:.1f})")

        # MACD bullish crossover (histogram just turned positive)
        if technicals.macd_histogram is not None and technicals.macd_histogram > 0 and technicals.trend_bullish:
            entry_conditions.append("MACD bullish crossover")

        # MACD bearish crossover
        if technicals.macd_histogram is not None and technicals.macd_histogram < 0:
            exit_conditions.append("MACD bearish")

        # Price near or below lower Bollinger → entry
        if technicals.bb_lower is not None:
            latest_price = float(price_df["Close"].iloc[-1])
            if latest_price <= technicals.bb_lower * 1.01:
                entry_conditions.append("Price at lower Bollinger")

        # Breakout signals
        if breakout.volume_surge:
            entry_conditions.append("Volume surge")
        if breakout.near_52w_high:
            entry_conditions.append("Near 52W high")
        if breakout.consolidation_breakout:
            entry_conditions.append("Consolidation breakout")

        # EMA trend
        if technicals.trend_bullish:
            entry_conditions.append("EMA trend bullish")

        # ---- Determine action ----
        entry_score = len(entry_conditions)
        exit_score = len(exit_conditions)
        latest_price = float(price_df["Close"].iloc[-1])

        if exit_score >= 2 or (exit_score >= 1 and entry_score == 0):
            action = SwingAction.EXIT
            reason = "EXIT: " + "; ".join(exit_conditions)
            confidence = min(100.0, exit_score * 25.0)
        elif entry_score >= 3:
            action = SwingAction.ENTRY
            reason = "ENTRY: " + "; ".join(entry_conditions)
            confidence = min(100.0, entry_score * 15.0 + breakout.breakout_score * 0.3)
        else:
            action = SwingAction.HOLD
            reason = f"HOLD: {entry_score} entry / {exit_score} exit signals"
            confidence = 0.0

        # ---- Compute stop/target for ENTRY signals ----
        stop_loss: float | None = None
        target_price: float | None = None
        if action is SwingAction.ENTRY and technicals.atr and technicals.atr > 0:
            st = self.deps.stop_target_engine.compute(latest_price, technicals.atr)
            stop_loss = st.stop_loss
            target_price = st.target_price

        result = SwingSignal(
            ticker=normalized,
            action=action,
            entry_price=latest_price if action is SwingAction.ENTRY else None,
            stop_loss=stop_loss,
            target_price=target_price,
            atr=technicals.atr,
            rsi=technicals.rsi,
            macd_signal_value=technicals.macd_signal,
            breakout_score=breakout.breakout_score,
            confidence=confidence,
            reason=reason,
            generated_at=now_ts,
        )
        db.log_engine_event("INFO", "engines.swing.swing_signal_engine", "swing signal evaluated", result.model_dump())
        return result


if __name__ == "__main__":
    print(SwingSignalEngine().evaluate("RELIANCE").model_dump())
