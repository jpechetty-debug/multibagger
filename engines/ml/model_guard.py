"""
engines/ml/model_guard.py
--------------------------
Automatic model disable with rule-based fallback.

When ModelDecayTracker.is_decayed() fires, the ML score is unreliable.
This guard intercepts ScoreEngine.score_ticker() calls and:
  1. Checks decay state from the active ModelDecayTracker
  2. If decayed → bypasses the XGBoost meta-model, uses
     pure rule-based weights from config.BASE_FACTOR_WEIGHTS
  3. Logs a DB event and Telegram alert on first transition
  4. Clears the fallback automatically once a fresh model is promoted

Usage — in engines/score_engine/model.py score_ticker():

    from engines.ml.model_guard import ModelGuard
    guard = ModelGuard()          # singleton, shares state across calls

    # At the top of score_ticker():
    if guard.is_fallback_active():
        return self._rule_based_score(ticker, data, weights)

    # After normal scoring:
    guard.record_outcome(predicted_score, actual_return)
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class ModelGuard:
    """
    Thread-safe model decay guard with rule-based fallback.

    Singleton pattern — all ScoreEngine instances share one guard so that
    decay detected during one ticker scan is immediately visible to all
    subsequent calls in the same process.
    """

    _instance:  "ModelGuard | None" = None
    _lock:      Lock = Lock()

    # ── Singleton ──────────────────────────────────────────────────────────

    def __new__(cls) -> "ModelGuard":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
        return cls._instance

    def _init(self) -> None:
        self._fallback_active:  bool  = False
        self._fallback_since:   float = 0.0
        self._fallback_reason:  str   = ""
        self._outcomes:         list[tuple[float, float]] = []  # (predicted, actual)
        self._baseline_auc:     float | None = None
        self._current_model_id: str | None = None
        self._alert_sent:       bool  = False

    # ── Public API ─────────────────────────────────────────────────────────

    def is_fallback_active(self) -> bool:
        """True if the ML model is disabled and rule-based scoring is in use."""
        return self._fallback_active

    def record_outcome(self, predicted_score: float, actual_forward_return: float) -> None:
        """
        Call this after each live signal to feed the decay detector.
        predicted_score: 0-100 from ScoreEngine
        actual_forward_return: realised N-day return (fraction)
        """
        self._outcomes.append((predicted_score, actual_forward_return))
        # Keep only last 100 outcomes (rolling window)
        if len(self._outcomes) > 100:
            self._outcomes = self._outcomes[-100:]
        self._check_decay()

    def set_baseline(self, model_id: str, baseline_auc: float) -> None:
        """
        Call after promoting a new model to reset the guard.
        Clears fallback if a fresh model has been promoted.
        """
        if model_id != self._current_model_id:
            self._current_model_id = model_id
            self._baseline_auc     = baseline_auc
            self._outcomes         = []
            self._alert_sent       = False
            if self._fallback_active:
                logger.info(
                    "ModelGuard: new model %s promoted — clearing fallback mode", model_id[:8]
                )
                self._fallback_active = False
                self._fallback_reason = ""
                self._log_event("INFO", f"Fallback cleared — new model {model_id[:8]} promoted")

    def force_fallback(self, reason: str = "manual") -> None:
        """Manually activate fallback (e.g. from CLI or health-check)."""
        self._activate_fallback(reason)

    def clear_fallback(self) -> None:
        """Manually clear fallback (e.g. after emergency retrain)."""
        self._fallback_active = False
        self._fallback_reason = ""
        self._alert_sent      = False
        self._log_event("INFO", "Fallback manually cleared")

    def status(self) -> dict[str, Any]:
        return {
            "fallback_active":  self._fallback_active,
            "fallback_since":   self._fallback_since if self._fallback_active else None,
            "fallback_reason":  self._fallback_reason,
            "outcome_samples":  len(self._outcomes),
            "baseline_auc":     self._baseline_auc,
            "current_model_id": self._current_model_id,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _check_decay(self) -> None:
        """Check current outcome window for decay and activate fallback if needed."""
        if len(self._outcomes) < 20:
            return   # need minimum sample for reliable detection

        try:
            from sklearn.metrics import roc_auc_score
            preds   = [p for p, _ in self._outcomes]
            actuals = [1 if r > 0 else 0 for _, r in self._outcomes]

            if len(set(actuals)) < 2:
                return   # need both classes for AUC

            live_auc = float(roc_auc_score(actuals, preds))

            # Correlation check: if predicted score has near-zero correlation
            # with actual returns, the model has decoupled from reality
            import numpy as np
            pred_arr = np.array(preds)
            ret_arr  = np.array([r for _, r in self._outcomes])
            if pred_arr.std() > 0 and ret_arr.std() > 0:
                corr = float(np.corrcoef(pred_arr, ret_arr)[0, 1])
            else:
                corr = 0.0

            # Decay triggers:
            # 1. AUC drops > 5% below baseline
            # 2. Return correlation < 0.20 with 30+ samples
            auc_decayed  = (self._baseline_auc is not None and
                            self._baseline_auc - live_auc > 0.05)
            corr_decayed = (len(self._outcomes) >= 30 and corr < 0.20)

            if auc_decayed or corr_decayed:
                reason = (
                    f"AUC decay {self._baseline_auc:.3f}→{live_auc:.3f}"
                    if auc_decayed else
                    f"Return correlation {corr:.3f} < 0.20 threshold"
                )
                self._activate_fallback(reason)

        except Exception as exc:
            logger.debug("ModelGuard decay check error: %s", exc)

    def _activate_fallback(self, reason: str) -> None:
        if self._fallback_active:
            return   # already in fallback, don't spam logs
        self._fallback_active = True
        self._fallback_since  = time.time()
        self._fallback_reason = reason
        logger.warning("ModelGuard: FALLBACK ACTIVATED — %s", reason)
        self._log_event("WARN", f"ML fallback activated: {reason}")
        if not self._alert_sent:
            self._send_alert(reason)
            self._alert_sent = True

    def _log_event(self, level: str, message: str) -> None:
        try:
            from data.db import db
            db.log_engine_event(level, "engines.ml.model_guard", message,
                                {"model_id": self._current_model_id, "samples": len(self._outcomes)})
        except Exception:
            pass

    def _send_alert(self, reason: str) -> None:
        try:
            from engines.alert_engine import AlertEngine
            AlertEngine().send_custom_alert(
                "⚠️ ML Fallback Active",
                f"Model guard activated rule-based fallback.\nReason: {reason}\n"
                f"Samples: {len(self._outcomes)}"
            )
        except Exception:
            pass
