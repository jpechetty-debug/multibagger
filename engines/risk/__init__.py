"""Risk engine package.

FIX-2: Duplicate module consolidation
--------------------------------------
Previously two files implemented identical classes:
  - engines/risk/advanced_risk.py       (16 KB, imported by quant_orchestrator)
  - engines/risk/advanced/risk_engine.py (30 KB, re-exported by advanced/__init__.py)

This caused a split-import situation where engines loaded different objects
depending on which import path they used, and `isinstance` checks between
them would silently fail.

Resolution: engines/risk/advanced_risk.py is the canonical module.
All public symbols are re-exported from here.  The engines/risk/advanced/
sub-package is kept for backwards compatibility but its __init__.py now
imports from advanced_risk.py as well.

ACTION REQUIRED after deploying this file:
  1. Replace engines/risk/advanced/__init__.py with fix2_advanced_init.py
  2. Delete engines/risk/advanced/risk_engine.py  (or keep as dead backup)
  3. Grep for any remaining `from engines.risk.advanced.risk_engine import`
     and replace with `from engines.risk.advanced_risk import`
"""

from engines.risk.advanced_risk import (   # noqa: F401
    AdvancedRiskEngine,
    RiskCheckResult,
    VaREngine,
    VaRResult,
    DrawdownGuard,
)
