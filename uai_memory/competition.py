"""Deterministic coupled competition over a typed directed graph.

The global operator matches the public SYNAPSE Eq. 3 implementation shape:
every node is inhibited by the same globally highest-potential nodes.  The
typed operator changes only the competitor set. Optional mass matching keeps
the typed penalty total equal to the global operator evaluated on the same
current typed potential state at each step. It does not copy the realized
penalty total from the separately evolving global trajectory. This within-state
counterfactual control prevents a typed result from being explained solely by
weaker inhibition on that state.

This module intentionally contains no embedding, LLM, benchmark, or storage
dependency.  A later adapter must bind immutable graph and candidate artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from math import comb
from typing import Iterable, Mapping, Sequence


class CompetitionInvariantError(ValueError):
    """Raised when an input would make the competition result ambiguous."""


class CompetitionMode(str, Enum):
    NONE = "none"
    GLOBAL = "global"
    TYPED = "typed"


@dataclass(frozen=True, order=True)
class GraphNode:
    node_id: str
    competition_class: str
    protected: bool = False

    def __post_init__(self) -> None:
        if not self.node_id:
            raise CompetitionInvariantError("node_id must be non-empty")
        if not self.competition_class:
            raise CompetitionInvariantError("competition_class must be non-empty")


@dataclass(frozen=True, order=True)
class DirectedEdge:
    source: str
    target: str
    weight: float

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise CompetitionInvariantError("edge endpoints must be non-empty")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise CompetitionInvariantError("edge weight must be finite and non-negative")


@dataclass(frozen=True)
class GraphSpec:
    nodes: tuple[GraphNode, ...]
    edges: tuple[DirectedEdge, ...] = ()

    def __post_init__(self) -> None:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise CompetitionInvariantError("node ids must be unique")
        if tuple(node_ids) != tuple(sorted(node_ids)):
            raise CompetitionInvariantError("nodes must use canonical node-id order")
        known = set(node_ids)
        seen_edges: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise CompetitionInvariantError("edge endpoint is not a graph node")
            key = (edge.source, edge.target)
            if key in seen_edges:
                raise CompetitionInvariantError("directed edges must be unique")
            seen_edges.add(key)
        edge_keys = tuple((edge.source, edge.target) for edge in self.edges)
        if edge_keys != tuple(sorted(edge_keys)):
            raise CompetitionInvariantError("edges must use canonical endpoint order")


@dataclass(frozen=True)
class CompetitionConfig:
    spreading_factor: float = 0.8
    node_decay_delta: float = 0.5
    inhibition_beta: float = 0.15
    inhibition_m: int = 7
    sigmoid_gamma: float = 5.0
    sigmoid_theta: float = 0.5
    steps: int = 3
    mass_match_to_global: bool = False

    def __post_init__(self) -> None:
        for name in (
            "spreading_factor",
            "node_decay_delta",
            "inhibition_beta",
            "sigmoid_gamma",
            "sigmoid_theta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise CompetitionInvariantError(f"{name} must be finite")
        if not 0.0 <= self.node_decay_delta <= 1.0:
            raise CompetitionInvariantError("node_decay_delta must be in [0, 1]")
        if self.spreading_factor < 0.0 or self.inhibition_beta < 0.0:
            raise CompetitionInvariantError("spreading and inhibition must be non-negative")
        if self.sigmoid_gamma <= 0.0:
            raise CompetitionInvariantError("sigmoid_gamma must be positive")
        if self.inhibition_m < 0 or self.steps <= 0:
            raise CompetitionInvariantError("inhibition_m and steps are invalid")


@dataclass(frozen=True)
class NodeSuppressionTrace:
    node_id: str
    competition_class: str
    protected: bool
    potential: float
    raw_penalty: float
    scaled_penalty: float
    inhibited_potential: float
    competitors: tuple[str, ...]


@dataclass(frozen=True)
class StepTrace:
    step: int
    mass_match_scale: float
    global_raw_penalty_total: float
    applied_raw_penalty_total: float
    nodes: tuple[NodeSuppressionTrace, ...]


@dataclass(frozen=True)
class DynamicsResult:
    mode: CompetitionMode
    final_activation: tuple[tuple[str, float], ...]
    max_inhibited_potential: tuple[tuple[str, float], ...]
    trace: tuple[StepTrace, ...]

    def activation_map(self) -> dict[str, float]:
        return dict(self.final_activation)

    def potential_map(self) -> dict[str, float]:
        return dict(self.max_inhibited_potential)


@dataclass(frozen=True)
class PairedBinarySummary:
    pair_count: int
    both_zero: int
    both_one: int
    first_only: int
    second_only: int
    discordant_count: int
    exact_two_sided_p: Fraction

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "both_zero": self.both_zero,
            "both_one": self.both_one,
            "first_only": self.first_only,
            "second_only": self.second_only,
            "discordant_count": self.discordant_count,
            "exact_two_sided_p": {
                "numerator": self.exact_two_sided_p.numerator,
                "denominator": self.exact_two_sided_p.denominator,
            },
        }


def _validated_initial(
    graph: GraphSpec, initial_activation: Mapping[str, float]
) -> dict[str, float]:
    known = {node.node_id for node in graph.nodes}
    unknown = sorted(set(initial_activation) - known)
    if unknown:
        raise CompetitionInvariantError(f"unknown initial-activation nodes: {unknown}")
    result: dict[str, float] = {}
    for node in graph.nodes:
        value = float(initial_activation.get(node.node_id, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise CompetitionInvariantError(
                "initial activation must be finite and non-negative"
            )
        result[node.node_id] = value
    return result


def _propagate(
    graph: GraphSpec, activation: Mapping[str, float], config: CompetitionConfig
) -> dict[str, float]:
    fan = {node.node_id: 0 for node in graph.nodes}
    for edge in graph.edges:
        fan[edge.source] += 1
    potential = {
        node.node_id: (1.0 - config.node_decay_delta) * activation[node.node_id]
        for node in graph.nodes
    }
    for edge in graph.edges:
        divisor = max(1, fan[edge.source])
        potential[edge.target] += (
            config.spreading_factor
            * float(edge.weight)
            * activation[edge.source]
            / divisor
        )
    return potential


def _top_competitors(
    graph: GraphSpec,
    potential: Mapping[str, float],
    mode: CompetitionMode,
    node: GraphNode,
    m: int,
) -> tuple[str, ...]:
    if mode is CompetitionMode.NONE or m == 0:
        return ()
    if mode is CompetitionMode.GLOBAL:
        eligible = graph.nodes
    elif mode is CompetitionMode.TYPED:
        eligible = tuple(
            candidate
            for candidate in graph.nodes
            if candidate.competition_class == node.competition_class
        )
    else:  # pragma: no cover - Enum exhaustiveness guard
        raise CompetitionInvariantError(f"unsupported competition mode: {mode}")
    ordered = sorted(eligible, key=lambda item: (-potential[item.node_id], item.node_id))
    return tuple(item.node_id for item in ordered[: min(m, len(ordered))])


def _raw_penalties(
    graph: GraphSpec,
    potential: Mapping[str, float],
    mode: CompetitionMode,
    m: int,
) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    penalties: dict[str, float] = {}
    competitors: dict[str, tuple[str, ...]] = {}
    for node in graph.nodes:
        pool = _top_competitors(graph, potential, mode, node, m)
        competitors[node.node_id] = pool
        if node.protected:
            penalties[node.node_id] = 0.0
            continue
        penalties[node.node_id] = sum(
            max(0.0, potential[winner] - potential[node.node_id])
            for winner in pool
        )
    return penalties, competitors


def _sigmoid(value: float, gamma: float, theta: float) -> float:
    z = -gamma * (value - theta)
    z = min(60.0, max(-60.0, z))
    return 1.0 / (1.0 + math.exp(z))


def run_dynamics(
    graph: GraphSpec,
    initial_activation: Mapping[str, float],
    *,
    mode: CompetitionMode | str,
    config: CompetitionConfig = CompetitionConfig(),
) -> DynamicsResult:
    """Run propagation -> inhibition -> sigmoid with deterministic tie order."""

    try:
        normalized_mode = CompetitionMode(mode)
    except ValueError as exc:
        raise CompetitionInvariantError(f"unknown competition mode: {mode}") from exc
    if config.mass_match_to_global and normalized_mode is not CompetitionMode.TYPED:
        raise CompetitionInvariantError(
            "mass_match_to_global is defined only for typed competition"
        )

    activation = _validated_initial(graph, initial_activation)
    max_potential = {node.node_id: 0.0 for node in graph.nodes}
    traces: list[StepTrace] = []

    for step in range(config.steps):
        potential = _propagate(graph, activation, config)
        global_penalty, _ = _raw_penalties(
            graph, potential, CompetitionMode.GLOBAL, config.inhibition_m
        )
        penalty, competitors = _raw_penalties(
            graph, potential, normalized_mode, config.inhibition_m
        )
        global_total = math.fsum(global_penalty.values())
        applied_total = math.fsum(penalty.values())
        scale = 1.0
        if config.mass_match_to_global:
            if applied_total == 0.0 and global_total > 0.0:
                raise CompetitionInvariantError(
                    "typed penalties are zero and cannot be mass-matched"
                )
            scale = 0.0 if global_total == 0.0 else global_total / applied_total

        next_activation: dict[str, float] = {}
        node_traces: list[NodeSuppressionTrace] = []
        for node in graph.nodes:
            raw = penalty[node.node_id]
            scaled = raw * scale
            inhibited = max(
                0.0, potential[node.node_id] - config.inhibition_beta * scaled
            )
            max_potential[node.node_id] = max(max_potential[node.node_id], inhibited)
            next_activation[node.node_id] = _sigmoid(
                inhibited, config.sigmoid_gamma, config.sigmoid_theta
            )
            node_traces.append(
                NodeSuppressionTrace(
                    node_id=node.node_id,
                    competition_class=node.competition_class,
                    protected=node.protected,
                    potential=potential[node.node_id],
                    raw_penalty=raw,
                    scaled_penalty=scaled,
                    inhibited_potential=inhibited,
                    competitors=competitors[node.node_id],
                )
            )
        traces.append(
            StepTrace(
                step=step,
                mass_match_scale=scale,
                global_raw_penalty_total=global_total,
                applied_raw_penalty_total=math.fsum(
                    item.scaled_penalty for item in node_traces
                ),
                nodes=tuple(node_traces),
            )
        )
        activation = next_activation

    return DynamicsResult(
        mode=normalized_mode,
        final_activation=tuple((node.node_id, activation[node.node_id]) for node in graph.nodes),
        max_inhibited_potential=tuple(
            (node.node_id, max_potential[node.node_id]) for node in graph.nodes
        ),
        trace=tuple(traces),
    )


def global_inhibition_preserves_order(
    potential: Mapping[str, float], inhibited_potential: Mapping[str, float]
) -> bool:
    """Check the weak-order preservation implied by one global competitor set."""

    ids = sorted(potential)
    if set(ids) != set(inhibited_potential):
        raise CompetitionInvariantError("potential maps must have identical keys")
    return all(
        potential[left] < potential[right]
        or inhibited_potential[left] >= inhibited_potential[right] - 1e-15
        for left in ids
        for right in ids
    )


def select_top_k(scores: Mapping[str, float], k: int) -> tuple[str, ...]:
    if k < 0:
        raise CompetitionInvariantError("k must be non-negative")
    for value in scores.values():
        if not math.isfinite(float(value)):
            raise CompetitionInvariantError("candidate scores must be finite")
    return tuple(
        node_id
        for node_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    )


def conditional_tirf(
    gold_ids: Iterable[str],
    pre_inhibition_ids: Iterable[str],
    post_inhibition_ids: Iterable[str],
) -> bool | None:
    """Return TIRF on the shared eligible set, or None if the condition fails."""

    gold = set(gold_ids)
    if not gold:
        raise CompetitionInvariantError("gold_ids must be non-empty")
    pre = set(pre_inhibition_ids)
    if not gold.intersection(pre):
        return None
    return not bool(gold.intersection(post_inhibition_ids))


def paired_binary_summary(
    first: Sequence[bool], second: Sequence[bool]
) -> PairedBinarySummary:
    """Exact paired-binary summary with sign test conditioned on nonzero pairs."""

    if len(first) != len(second):
        raise CompetitionInvariantError("paired vectors must have equal length")
    both_zero = both_one = first_only = second_only = 0
    for left, right in zip(first, second):
        if left and right:
            both_one += 1
        elif left:
            first_only += 1
        elif right:
            second_only += 1
        else:
            both_zero += 1
    discordant = first_only + second_only
    if discordant == 0:
        p_value = Fraction(1, 1)
    else:
        tail = min(first_only, second_only)
        numerator = 2 * sum(comb(discordant, k) for k in range(tail + 1))
        denominator = 2**discordant
        p_value = min(Fraction(1, 1), Fraction(numerator, denominator))
    return PairedBinarySummary(
        pair_count=len(first),
        both_zero=both_zero,
        both_one=both_one,
        first_only=first_only,
        second_only=second_only,
        discordant_count=discordant,
        exact_two_sided_p=p_value,
    )
