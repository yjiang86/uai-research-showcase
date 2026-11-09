"""Explicit candidate generation, gating, ranking and bounded reactivation."""
from __future__ import annotations

from .config import RuntimeConfig
from .graph import TraceGraph
from .signals import replay_score
from .stance import currently_valid
from .types import Tier, Trace


def rank_candidates(traces: dict[str, Trace], graph: TraceGraph, operation: int,
                    config: RuntimeConfig, feedback: dict[str, float]) -> list[tuple[float, str]]:
    scored = []
    for key, trace in traces.items():
        if not currently_valid(trace) or trace.merged_into is not None:
            continue
        # An explicit forget withdraws a trace from rehearsal. Only a new
        # encounter (encode) may lift suppression; replay alone must never
        # promote, stabilize or coactivate suppressed content.
        if trace.suppressed:
            continue
        if trace.tier == Tier.COLD and trace.last_access is None:
            continue
        if trace.last_replay == operation:
            continue
        conflicts = len(graph.neighbors(key, "conflicts_with"))
        score = replay_score(trace, operation, conflicts, config, feedback.get(key))
        scored.append((score, key))
    # Reactivation coverage first prevents already stable traces from taking
    # every replay slot through their own stability feedback. Derived score
    # orders traces at the same reactivation count; no new tuning constant.
    return sorted(scored, key=lambda x: (traces[x[1]].replay_count, -x[0], x[1]))
