"""Early stabilization and late admission remain distinct."""
from __future__ import annotations

from .config import RuntimeConfig
from .stance import currently_valid
from .types import Tier, Trace

# Version 2: late (systems) admission requires early stabilization from every
# tier. Cards certified under an older or unknown criterion are re-checked.
VERTICAL_GATE_VERSION = 2


def early_allowed(trace: Trace, config: RuntimeConfig) -> bool:
    return (config.mechanisms.vertical_gates and currently_valid(trace)
            and (trace.recurrence >= config.min_recurrence
                 or trace.replay_count >= config.min_recurrence))


def late_allowed(trace: Trace, config: RuntimeConfig) -> bool:
    """Systems consolidation follows early stabilization; it never replaces it.

    The two gates are sequential stages of one vertical path. A trace may
    lose expression (cold) after it stabilized, but a trace that never met
    the early criterion cannot enter a card from any tier.
    """
    return (early_allowed(trace, config) and consolidatable_tier(trace, config)
            and trace.replay_score >= config.min_replay_score)


def consolidatable_tier(trace: Trace, config: RuntimeConfig) -> bool:
    """Expression and storage are different axes.

    A warm trace is expressed. A cold trace that lost expression only to the
    reader budget (homeostatic demotion) is a silent engram: still stored,
    still reachable through the encoding index, and still eligible for
    offline systems consolidation once replay has stabilized it. A silent
    trace is not read, so it is reactivated offline through its encoding
    episode when a bound member is replayed (runtime._replay). An explicit
    soft forget suppresses expression *and* consolidation, so its content is
    never re-expressed through a card. Hot traces have not passed early
    stabilization and never qualify.
    """
    if trace.suppressed:
        return False
    if trace.tier == Tier.WARM:
        return True
    return config.mechanisms.silent_consolidation and trace.tier == Tier.COLD
