"""cloture: Sub-microsecond anytime-valid consensus stopping, correlated-panel
auditing, sequential agent racing, and active prediction-powered inference.

Uses PEP 562 lazy attribute loading so `import cloture` executes in <10 ms
with zero external dependencies.
"""
from __future__ import annotations

__version__ = "0.1.0"

_NATIVE_EXPORTS = {
    "AbortCascade",
    "ActivePPI",
    "AgentRace",
    "AscdStopper",
    "BettingCS",
    "CiteStopper",
    "CsResult",
    "EmpiricalBernsteinCS",
    "EvidenceSketch",
    "ModeDecision",
    "OnlineConformal",
    "OnlinePanelTracker",
    "PairedEProcess",
    "PairedResult",
    "PanelReport",
    "PopDecision",
    "PopulationCert",
    "PpiResult",
    "SafeFuse",
    "audit_panel",
    "e_bh_discoveries",
}

_ASYNC_EXPORTS = {
    "AsyncConsensusResult",
    "consensus",
    "consensus_dist",
    "extract_logprob_distribution",
    "race_agents",
}

__all__ = sorted(_NATIVE_EXPORTS | _ASYNC_EXPORTS | {"__version__"})


def __getattr__(name: str):
    if name in _NATIVE_EXPORTS:
        from . import _native
        return getattr(_native, name)
    if name in _ASYNC_EXPORTS:
        from . import async_tools
        return getattr(async_tools, name)
    raise AttributeError(f"module 'cloture' has no attribute {name!r}")
