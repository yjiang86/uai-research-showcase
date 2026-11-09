"""Deterministic counters/events; no clocks or mutable external sinks."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ``capacity_rejections`` is retained only for v1 receipt compatibility. It is
# not a v2 evidence field; capacity homeostasis is measured from accepted
# writes after demotion and separate stored/expression counts.
COUNTERS = ("writes_proposed", "writes_accepted", "suppressed_by_boundary",
            "replay_candidates", "replay_triggered", "promotions",
            "promotions.hot_to_warm", "promotions.cold_to_warm",
            "promotions.warm_to_consolidated", "promotions.cold_to_consolidated",
            "vertical_consolidation_allowed", "vertical_consolidation_denied",
            "consolidations.requested", "consolidations.accepted", "consolidations.rejected",
            "reconsolidations", "demotions", "conflicts_detected", "forget.soft", "forget.hard",
            "accesses", "prediction_observations", "semantic_schema_observations",
            "dependencies_pruned", "capacity_rejections")


@dataclass
class Telemetry:
    counts: Counter[str] = field(default_factory=lambda: Counter(dict.fromkeys(COUNTERS, 0)))
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, count: int = 1) -> None:
        self.counts[name] += count

    def record(self, operation: int, action: str, **fields: Any) -> None:
        self.events.append({"operation": operation, "action": action, **fields})

    def summary(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))
