"""One exposed mechanism threshold per layer; resource bounds are separate."""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Mechanisms:
    replay: bool = True
    vertical_gates: bool = True
    consolidation: bool = True
    reconsolidation: bool = True
    soft_forgetting: bool = True
    competition: bool = True
    context_integration: bool = True
    history_cueing: bool = True
    usage_priority: bool = True
    conflict_detection: bool = True
    novelty_priority: bool = True
    repetition_adaptation: bool = True
    context_prediction: bool = True
    pattern_completion: bool = True
    episodic_binding: bool = True
    concept_binding: bool = True
    silent_consolidation: bool = True
    specific_reinstatement: bool = True
    episode_time: bool = True
    temporal_order: bool = True
    discourse_context: bool = True
    directed_forgetting: bool = True
    temporal_window: bool = True
    event_segmentation: bool = True
    withdrawal_support: bool = True

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in vars(self).values()):
            raise ValueError("MECHANISM_SWITCH_NOT_BOOLEAN")


@dataclass(frozen=True)
class RuntimeConfig:
    write_support_threshold: float = 0.0
    replay_threshold: float = 0.15
    min_recurrence: int = 2
    min_replay_score: float = 0.41
    cluster_size: int = 3
    inhibition_beta: float = 0.15
    history_cue_overlap: float = 0.05
    # Suppression raises the retrieval threshold of an explicitly suppressed
    # trace: only a cue about that trace itself (sharing at least a third of
    # the combined terms) recovers it; a neighbouring cue does not.
    suppression_cue_overlap: float = 0.33
    reader_words: int = 2048
    reader_bytes: int = 16384
    reader_records: int = 128
    replay_limit: int = 12
    cluster_limit: int = 12
    graph_neighbors: int = 12
    candidate_limit: int = 128
    recency_scale: int = 64
    # Slow consolidation separates episodic encoding from semantic card
    # formation.  The legacy default remains immediate so existing callers
    # retain their deterministic behavior; production successor runners opt
    # into the delayed path explicitly.
    slow_consolidation: bool = False
    consolidation_quiet_epochs: int = 1
    mechanisms: Mechanisms = field(default_factory=Mechanisms)

    def __post_init__(self) -> None:
        for name in ("write_support_threshold", "replay_threshold", "min_replay_score",
                     "inhibition_beta", "history_cue_overlap", "suppression_cue_overlap"):
            x = getattr(self, name)
            if type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(f"CONFIG_THRESHOLD_INVALID:{name}")
        for name in ("min_recurrence", "cluster_size", "reader_words", "reader_bytes",
                     "reader_records", "replay_limit", "cluster_limit", "graph_neighbors",
                     "candidate_limit", "recency_scale", "consolidation_quiet_epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"CONFIG_BOUND_INVALID:{name}")
        if type(self.slow_consolidation) is not bool:
            raise ValueError("CONFIG_SWITCH_NOT_BOOLEAN:slow_consolidation")
        if self.cluster_size < 2 or self.cluster_size > self.cluster_limit:
            raise ValueError("CONFIG_CLUSTER_BOUNDS_INVALID")


def legacy_config(value: dict) -> dict:
    """Upgrade a runtime snapshot's config dict without changing what it recalls.

    Configs persisted before explicit suppression had its own retrieval
    threshold recovered suppressed traces at the history cue threshold; a
    restored runtime snapshot keeps that threshold. Application journals are
    bound to their state format and are not migrated (R5-06, Claude r5 F5).
    """
    conf = dict(value)
    if "suppression_cue_overlap" not in conf and "history_cue_overlap" in conf:
        conf["suppression_cue_overlap"] = conf["history_cue_overlap"]
    # Snapshots persisted before event segmentation cued whole traces only.
    mechanisms = conf.get("mechanisms")
    if isinstance(mechanisms, dict) and "event_segmentation" not in mechanisms:
        conf["mechanisms"] = mechanisms = {**mechanisms, "event_segmentation": False}
    # Snapshots persisted before withdrawal support evaluation accepted any
    # source-present withdrawal quote.
    if isinstance(mechanisms, dict) and "withdrawal_support" not in mechanisms:
        conf["mechanisms"] = {**mechanisms, "withdrawal_support": False}
    return conf
