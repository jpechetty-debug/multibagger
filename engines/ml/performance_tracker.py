"""
Out-of-sample vs live performance comparison.
Closes the ML feedback loop by tracking whether backtest predictions
hold up in live trading — and triggering retraining when they degrade.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REGISTRY_PATH = Path("runtime/model_registry.db")
TRADING_DAYS  = 252


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@contextmanager
def _conn(path: Path = REGISTRY_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _init(path: Path = REGISTRY_PATH) -> None:
    with _conn(path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id         TEXT    NOT NULL,
            ticker           TEXT    NOT NULL,
            signal_date      TEXT    NOT NULL,
            predicted_score  REAL    NOT NULL,
            predicted_return REAL,
            outcome_date     TEXT,
            actual_return    REAL,
            hit              INTEGER,
            created_at       INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_pred_ticker_date
            ON predictions (ticker, signal_date);
        CREATE INDEX IF NOT EXISTS idx_pred_model
            ON predictions (model_id);
        CREATE INDEX IF NOT EXISTS idx_pred_outcome
            ON predictions (outcome_date);

        CREATE TABLE IF NOT EXISTS performance_snapshots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT    NOT NULL,
            model_id      TEXT    NOT NULL,
            window_days   INTEGER NOT NULL,
            live_sharpe   REAL,
            oos_sharpe    REAL,
            sharpe_decay  REAL,
            live_hit_rate REAL,
            oos_hit_rate  REAL,
            return_corr   REAL,
            sample_size   INTEGER,
            decay_alert   INTEGER DEFAULT 0,
            created_at    INTEGER NOT NULL
        );
        """)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class DecayReport:
    def __init__(self, data: dict):
        self._d = data

    @property
    def sharpe_decay(self) -> float:
        return self._d.get("sharpe_decay", 0.0)

    @property
    def live_sharpe(self) -> float:
        return self._d.get("live_sharpe", 0.0)

    @property
    def oos_sharpe(self) -> float:
        return self._d.get("oos_sharpe", 0.0)

    @property
    def live_hit_rate(self) -> float:
        return self._d.get("live_hit_rate", 0.0)

    @property
    def return_corr(self) -> float:
        return self._d.get("return_corr", 0.0)

    @property
    def sample_size(self) -> int:
        return self._d.get("sample_size", 0)

    @property
    def decay_alert(self) -> bool:
        return bool(self._d.get("decay_alert", False))

    @property
    def action(self) -> str:
        decay = abs(self.sharpe_decay)
        if decay > 0.40 or self.return_corr < 0.30:
            return "RETRAIN_NOW"
        if decay > 0.20:
            return "MONITOR_CLOSELY"
        return "HEALTHY"

    @property
    def summary(self) -> str:
        return (
            f"Live Sharpe {self.live_sharpe:.3f} vs OOS {self.oos_sharpe:.3f} "
            f"(decay {self.sharpe_decay*100:+.1f}%) | "
            f"Hit rate {self.live_hit_rate*100:.1f}% | "
            f"Return corr {self.return_corr:.3f} | "
            f"n={self.sample_size} | Action: {self.action}"
        )

    def to_dict(self) -> dict:
        return {**self._d, "action": self.action, "summary": self.summary}


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """
    Records signal predictions and their eventual outcomes,
    then compares live performance against OOS backtest benchmarks.
    """

    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        _init(path)

    def record_prediction(
        self,
        model_id:         str,
        ticker:           str,
        signal_date:      str,
        predicted_score:  float,
        predicted_return: float | None = None,
        holding_days:     int = 20,
    ) -> int:
        outcome_date = _add_trading_days(signal_date, holding_days)
        with _conn(self.path) as con:
            cur = con.execute(
                """INSERT INTO predictions
                   (model_id, ticker, signal_date, predicted_score,
                    predicted_return, outcome_date, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (model_id, ticker, signal_date, predicted_score,
                 predicted_return, outcome_date, int(time.time())),
            )
        return cur.lastrowid

    def record_batch_predictions(self, records: list[dict[str, Any]]) -> int:
        rows = []
        for r in records:
            hd = r.get("holding_days", 20)
            rows.append((
                r["model_id"], r["ticker"], r["signal_date"],
                r["predicted_score"], r.get("predicted_return"),
                _add_trading_days(r["signal_date"], hd),
                int(time.time()),
            ))
        with _conn(self.path) as con:
            con.executemany(
                """INSERT INTO predictions
                   (model_id, ticker, signal_date, predicted_score,
                    predicted_return, outcome_date, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def record_outcome(
        self,
        ticker:        str,
        signal_date:   str,
        actual_return: float,
        model_id:      str | None = None,
    ) -> int:
        hit = None
        with _conn(self.path) as con:
            rows = con.execute(
                """SELECT id, predicted_return FROM predictions
                   WHERE ticker=? AND signal_date=?
                   AND actual_return IS NULL
                   AND (? IS NULL OR model_id=?)""",
                (ticker, signal_date, model_id, model_id),
            ).fetchall()

            updated = 0
            for row in rows:
                pred_ret = row["predicted_return"]
                if pred_ret is not None:
                    hit = int(
                        (pred_ret > 0 and actual_return > 0) or
                        (pred_ret < 0 and actual_return < 0)
                    )
                con.execute(
                    """UPDATE predictions
                       SET actual_return=?, hit=? WHERE id=?""",
                    (actual_return, hit, row["id"]),
                )
                updated += 1
        return updated

    def backfill_outcomes(
        self,
        prices_df: pd.DataFrame,   # date × ticker close prices
        model_id: str | None = None,
    ) -> int:
        with _conn(self.path) as con:
            pending = con.execute(
                """SELECT id, ticker, signal_date, outcome_date
                   FROM predictions
                   WHERE actual_return IS NULL
                   AND outcome_date <= date('now')
                   AND (? IS NULL OR model_id=?)""",
                (model_id, model_id),
            ).fetchall()

        updated = 0
        for row in pending:
            ticker      = row["ticker"]
            sig_date    = row["signal_date"]
            out_date    = row["outcome_date"]
            pred_id     = row["id"]

            if ticker not in prices_df.columns:
                continue
            try:
                entry_px = _nearest_price(prices_df, ticker, sig_date)
                exit_px  = _nearest_price(prices_df, ticker, out_date)
                if entry_px and exit_px and entry_px > 0:
                    actual_ret = (exit_px - entry_px) / entry_px
                    with _conn(self.path) as con:
                        row_sel = con.execute(
                            "SELECT predicted_return FROM predictions WHERE id=?",
                            (pred_id,),
                        ).fetchone()
                        pred_ret = row_sel["predicted_return"] if row_sel else None
                        hit = None
                        if pred_ret is not None:
                            hit = int(
                                (pred_ret > 0 and actual_ret > 0) or
                                (pred_ret < 0 and actual_ret < 0)
                            )
                        con.execute(
                            "UPDATE predictions SET actual_return=?, hit=? WHERE id=?",
                            (actual_ret, hit, pred_id),
                        )
                    updated += 1
            except Exception:
                continue

        return updated

    def compare(
        self,
        model_id:      str,
        oos_sharpe:    float,
        oos_hit_rate:  float,
        window_days:   int = 60,
    ) -> DecayReport:
        cutoff = _days_ago_str(window_days)

        with _conn(self.path) as con:
            rows = con.execute(
                """SELECT predicted_return, actual_return, hit
                   FROM predictions
                   WHERE model_id=?
                   AND actual_return IS NOT NULL
                   AND signal_date >= ?
                   ORDER BY signal_date""",
                (model_id, cutoff),
            ).fetchall()

        if len(rows) < 10:
            return DecayReport({
                "model_id": model_id, "window_days": window_days,
                "sample_size": len(rows), "oos_sharpe": oos_sharpe,
                "oos_hit_rate": oos_hit_rate,
                "live_sharpe": 0.0, "sharpe_decay": 0.0,
                "live_hit_rate": 0.0, "return_corr": 0.0,
                "decay_alert": False,
                "note": f"Only {len(rows)} outcomes — need ≥10 for reliable comparison",
            })

        actual_rets  = np.array([r["actual_return"] for r in rows])
        pred_rets    = np.array([r["predicted_return"] or 0 for r in rows])
        hits         = [r["hit"] for r in rows if r["hit"] is not None]

        live_mean    = actual_rets.mean()
        live_std     = actual_rets.std()
        live_sharpe  = (live_mean / live_std * np.sqrt(TRADING_DAYS)) if live_std > 1e-9 else 0.0

        sharpe_decay = ((oos_sharpe - live_sharpe) / oos_sharpe) if oos_sharpe > 1e-9 else 0.0
        live_hit_rate = float(np.mean(hits)) if hits else 0.0

        valid = [(p, a) for p, a in zip(pred_rets, actual_rets) if p != 0]
        if len(valid) >= 5:
            pv, av    = zip(*valid)
            return_corr = float(np.corrcoef(pv, av)[0, 1])
        else:
            return_corr = 0.0

        decay_alert = (
            abs(sharpe_decay) > 0.20
            or (oos_hit_rate - live_hit_rate) > 0.10
            or return_corr < 0.30
        )

        snapshot = {
            "snapshot_date": _today_str(),
            "model_id":      model_id,
            "window_days":   window_days,
            "live_sharpe":   round(live_sharpe, 4),
            "oos_sharpe":    round(oos_sharpe, 4),
            "sharpe_decay":  round(sharpe_decay, 4),
            "live_hit_rate": round(live_hit_rate, 4),
            "oos_hit_rate":  round(oos_hit_rate, 4),
            "return_corr":   round(return_corr, 4),
            "sample_size":   len(rows),
            "decay_alert":   int(decay_alert),
            "created_at":    int(time.time()),
        }

        with _conn(self.path) as con:
            con.execute(
                """INSERT INTO performance_snapshots
                   (snapshot_date, model_id, window_days, live_sharpe, oos_sharpe,
                    sharpe_decay, live_hit_rate, oos_hit_rate, return_corr,
                    sample_size, decay_alert, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                tuple(snapshot.values()),
            )

        return DecayReport(snapshot)

    def decay_history(self, model_id: str, limit: int = 30) -> pd.DataFrame:
        with _conn(self.path) as con:
            rows = con.execute(
                """SELECT snapshot_date, live_sharpe, oos_sharpe,
                          sharpe_decay, live_hit_rate, return_corr,
                          sample_size, decay_alert
                   FROM performance_snapshots
                   WHERE model_id=?
                   ORDER BY snapshot_date DESC LIMIT ?""",
                (model_id, limit),
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def pending_outcomes(self, as_of_date: str | None = None) -> list[dict]:
        cutoff = as_of_date or _today_str()
        with _conn(self.path) as con:
            rows = con.execute(
                """SELECT ticker, signal_date, outcome_date, model_id
                   FROM predictions
                   WHERE actual_return IS NULL AND outcome_date <= ?
                   ORDER BY outcome_date""",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    from datetime import date
    return date.today().isoformat()


def _days_ago_str(n: int) -> str:
    from datetime import date, timedelta
    return (date.today() - timedelta(days=n)).isoformat()


def _add_trading_days(date_str: str, n: int) -> str:
    from datetime import date, timedelta
    dt = date.fromisoformat(date_str)
    added = 0
    while added < n:
        dt += timedelta(days=1)
        if dt.weekday() < 5:   # Mon–Fri
            added += 1
    return dt.isoformat()


def _nearest_price(df: pd.DataFrame, ticker: str, date_str: str) -> float | None:
    try:
        idx = df.index[df.index >= date_str]
        if idx.empty:
            return None
        return float(df.loc[idx[0], ticker])
    except (KeyError, IndexError, TypeError):
        return None
