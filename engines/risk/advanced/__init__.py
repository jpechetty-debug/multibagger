"""engines/risk/advanced — backwards-compatibility shim.

FIX-2: This sub-package previously defined its own copy of VaREngine,
DrawdownGuard, etc. in risk_engine.py (30 KB).  The canonical copy is now
engines/risk/advanced_risk.py.

All imports are redirected there.  Any code that previously used:
    from engines.risk.advanced import AdvancedRiskEngine
    from engines.risk.advanced.risk_engine import VaREngine
will continue to work without modification.
"""

from engines.risk.advanced_risk import (   # noqa: F401
    AdvancedRiskEngine,
    RiskCheckResult,
    VaREngine,
    VaRResult,
    DrawdownGuard,
)

__all__ = [
    "AdvancedRiskEngine",
    "RiskCheckResult",
    "VaREngine",
    "VaRResult",
    "DrawdownGuard",
]
