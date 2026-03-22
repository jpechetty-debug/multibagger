"""
Institutional model registry with experiment tracking, rollback,
and regression detection. Every trained model is versioned, scored,
and compared against its predecessor before promotion.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path("runtime/model_registry.db")
MODEL_DIR     = Path("runtime/models")


# ---------------------------------------------------------------------------
# DB bootstrap
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
        CREATE TABLE IF NOT EXISTS models (
            model_id         TEXT PRIMARY KEY,
            dataset_hash     TEXT NOT NULL,
            algorithm        TEXT NOT NULL,
            params           TEXT NOT NULL,
            auc              REAL,
            brier            REAL,
            regime           TEXT,
            features_version TEXT NOT NULL,
            artifact_path    TEXT,
            promoted         INTEGER DEFAULT 0,
            created_at       INTEGER NOT NULL,
            notes            TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_models_algo    ON models(algorithm);
        CREATE INDEX IF NOT EXISTS idx_models_regime  ON models(regime);
        CREATE INDEX IF NOT EXISTS idx_models_promoted ON models(promoted);

        CREATE TABLE IF NOT EXISTS promotions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id     TEXT NOT NULL,
            promoted_at  INTEGER NOT NULL,
            promoted_by  TEXT DEFAULT 'system',
            reason       TEXT,
            prev_auc     REAL,
            new_auc      REAL
        );

        CREATE TABLE IF NOT EXISTS regression_alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            challenger_id TEXT NOT NULL,
            champion_id   TEXT NOT NULL,
            challenger_auc REAL,
            champion_auc   REAL,
            delta          REAL,
            flagged_at     INTEGER NOT NULL,
            resolved       INTEGER DEFAULT 0
        );
        """)


# ---------------------------------------------------------------------------
# Core registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Versioned model store. Each training run produces a record here.
    Only one model per (algorithm, regime) pair is promoted at a time.
    """

    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _init(path)

    # ── Write ──────────────────────────────────────────────────────────────

    def save_model(
        self,
        model: Any,
        dataset_hash: str,
        algorithm: str,
        params: dict,
        auc: float,
        brier: float,
        features_version: str,
        regime: str | None = None,
        notes: str | None = None,
    ) -> str:
        model_id      = str(uuid.uuid4())
        artifact_name = f"{algorithm}_{model_id[:8]}_{int(time.time())}.pkl"
        artifact_path = MODEL_DIR / artifact_name

        with open(artifact_path, "wb") as f:
            pickle.dump({"model": model, "model_id": model_id, "algorithm": algorithm,
                         "features_version": features_version}, f)

        with _conn(self.path) as con:
            con.execute(
                """INSERT INTO models
                   (model_id, dataset_hash, algorithm, params, auc, brier,
                    regime, features_version, artifact_path, created_at, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (model_id, dataset_hash, algorithm, json.dumps(params),
                 auc, brier, regime, features_version,
                 str(artifact_path), int(time.time()), notes),
            )
        return model_id

    def promote_model(
        self,
        model_id: str,
        promoted_by: str = "system",
        reason: str | None = None,
    ) -> None:
        """Promote a model. Demotes all other models for same algorithm+regime."""
        with _conn(self.path) as con:
            row = con.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
            if not row:
                raise ValueError(f"model_id {model_id} not found in registry")

            # Get current champion AUC for promotion log
            champion = con.execute(
                "SELECT auc FROM models WHERE algorithm=? AND regime IS ? AND promoted=1",
                (row["algorithm"], row["regime"]),
            ).fetchone()
            prev_auc = champion["auc"] if champion else None

            # Demote all current promoted models for this algorithm+regime
            con.execute(
                "UPDATE models SET promoted=0 WHERE algorithm=? AND regime IS ?",
                (row["algorithm"], row["regime"]),
            )
            # Promote this one
            con.execute("UPDATE models SET promoted=1 WHERE model_id=?", (model_id,))
            # Log promotion
            con.execute(
                """INSERT INTO promotions (model_id, promoted_at, promoted_by, reason, prev_auc, new_auc)
                   VALUES (?,?,?,?,?,?)""",
                (model_id, int(time.time()), promoted_by, reason, prev_auc, row["auc"]),
            )

    def rollback(self, algorithm: str, regime: str | None = None) -> str | None:
        """
        Demote the current promoted model and promote the previous one.
        Returns the rolled-back model_id, or None if nothing to roll back to.
        """
        with _conn(self.path) as con:
            # Find last two promotions for this algorithm+regime
            history = con.execute(
                """SELECT model_id FROM promotions p
                   JOIN models m USING (model_id)
                   WHERE m.algorithm=? AND m.regime IS ?
                   ORDER BY p.promoted_at DESC LIMIT 2""",
                (algorithm, regime),
            ).fetchall()

            if len(history) < 2:
                return None

            rollback_id = history[1]["model_id"]
            # Demote current
            con.execute(
                "UPDATE models SET promoted=0 WHERE algorithm=? AND regime IS ? AND promoted=1",
                (algorithm, regime),
            )
            # Restore previous
            con.execute("UPDATE models SET promoted=1 WHERE model_id=?", (rollback_id,))
            con.execute(
                """INSERT INTO promotions (model_id, promoted_at, promoted_by, reason)
                   VALUES (?,?,?,?)""",
                (rollback_id, int(time.time()), "rollback", f"Rolled back from {history[0]['model_id'][:8]}"),
            )
        return rollback_id

    # ── Read ───────────────────────────────────────────────────────────────

    def load_best_model(
        self,
        algorithm: str,
        regime: str | None = None,
    ) -> tuple[Any, dict] | tuple[None, None]:
        """
        Load the currently promoted model for (algorithm, regime).
        Returns (model_object, metadata_dict) or (None, None).
        """
        with _conn(self.path) as con:
            row = con.execute(
                "SELECT * FROM models WHERE algorithm=? AND regime IS ? AND promoted=1",
                (algorithm, regime),
            ).fetchone()
            if not row:
                # Fallback: best AUC even if not promoted
                row = con.execute(
                    "SELECT * FROM models WHERE algorithm=? ORDER BY auc DESC LIMIT 1",
                    (algorithm,),
                ).fetchone()
            if not row:
                return None, None

        path = Path(row["artifact_path"])
        if not path.exists():
            return None, dict(row)

        with open(path, "rb") as f:
            artifact = pickle.load(f)

        return artifact["model"], dict(row)

    def compare_models(
        self,
        algorithm: str | None = None,
        regime: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Rank all models by AUC. Use to decide which challenger to promote.
        """
        with _conn(self.path) as con:
            clauses = []
            params: list[Any] = []
            if algorithm:
                clauses.append("algorithm=?")
                params.append(algorithm)
            if regime is not None:
                clauses.append("regime IS ?")
                params.append(regime)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = con.execute(
                f"""SELECT model_id, algorithm, regime, auc, brier,
                           features_version, promoted, created_at, notes
                    FROM models {where}
                    ORDER BY auc DESC LIMIT ?""",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def regression_check(
        self,
        challenger_id: str,
        threshold: float = 0.03,
    ) -> dict:
        """
        Compare challenger to current promoted model.
        Flags a regression if challenger AUC is more than `threshold` below champion.
        Returns dict with 'regression': bool and details.
        """
        with _conn(self.path) as con:
            challenger = con.execute(
                "SELECT * FROM models WHERE model_id=?", (challenger_id,)
            ).fetchone()
            if not challenger:
                return {"regression": False, "reason": "Challenger not found"}

            champion = con.execute(
                """SELECT * FROM models
                   WHERE algorithm=? AND regime IS ? AND promoted=1
                   AND model_id != ?""",
                (challenger["algorithm"], challenger["regime"], challenger_id),
            ).fetchone()

            if not champion:
                return {"regression": False, "reason": "No champion to compare against"}

            delta = challenger["auc"] - champion["auc"]
            is_regression = delta < -threshold

            if is_regression:
                con.execute(
                    """INSERT INTO regression_alerts
                       (challenger_id, champion_id, challenger_auc, champion_auc, delta, flagged_at)
                       VALUES (?,?,?,?,?,?)""",
                    (challenger_id, champion["model_id"],
                     challenger["auc"], champion["auc"],
                     delta, int(time.time())),
                )

        return {
            "regression":      is_regression,
            "challenger_auc":  round(challenger["auc"], 4),
            "champion_auc":    round(champion["auc"], 4),
            "delta":           round(delta, 4),
            "threshold":       threshold,
            "recommendation":  "Do NOT promote — regression detected" if is_regression
                               else "Safe to promote",
        }

    def audit_trail(self, limit: int = 30) -> list[dict]:
        with _conn(self.path) as con:
            rows = con.execute(
                """SELECT p.promoted_at, p.promoted_by, p.reason,
                          p.prev_auc, p.new_auc, m.model_id, m.algorithm,
                          m.regime, m.auc, m.features_version
                   FROM promotions p JOIN models m USING(model_id)
                   ORDER BY p.promoted_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
