"""Sparse typed edges with deterministic lexical neighbor bounds."""
from __future__ import annotations

from collections import defaultdict
from .signals import overlap, terms
from .types import Trace


class TraceGraph:
    def __init__(self) -> None:
        self.edges: dict[tuple[str, str, str], float] = {}
        self.index: dict[str, set[str]] = defaultdict(set)
        self.tokens: dict[str, frozenset[str]] = {}
        self.pools: dict[str, set[str]] = defaultdict(set)
        self._out: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._in: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._pool: dict[str, str] = {}

    def link(self, kind: str, left: str, right: str, weight: float = 1.0) -> None:
        if left != right:
            self.edges[kind, left, right] = weight
            self._out[kind, left].add(right)
            self._in[kind, right].add(left)

    def unlink(self, kind: str, left: str, right: str) -> None:
        self.edges.pop((kind, left, right), None)
        self._out[kind, left].discard(right)
        self._in[kind, right].discard(left)

    def set_edges(self, rows: list[list]) -> None:
        self.edges.clear()
        self._out.clear()
        self._in.clear()
        seen = set()
        for kind, a, b, weight in rows:
            if (kind not in {"depends_on", "merged_from", "term_overlap", "co_activation", "conflicts_with"}
                    or a == b or (kind, a, b) in seen):
                raise ValueError("GRAPH_EDGE_SCHEMA_INVALID")
            seen.add((kind, a, b))
            self.link(kind, a, b, weight)

    def neighbors(self, key: str, kind: str, *, reverse: bool = False) -> list[str]:
        return sorted((self._in if reverse else self._out).get((kind, key), ()))

    def update(self, trace: Trace, traces: dict[str, Trace], limit: int) -> None:
        key = trace.key
        old_tokens = self.tokens.get(key, frozenset())
        for token in old_tokens:
            self.index[token].discard(key)
        if key in self._pool:
            self.pools[self._pool[key]].discard(key)
        self.pools[trace.pool].add(key)
        self._pool[key] = trace.pool
        self.tokens[key] = terms(trace.current.text)
        peers = set().union(*(self.index[t] for t in self.tokens[key])) if self.tokens[key] else set()
        for token in self.tokens[key]:
            self.index[token].add(key)
        for kind in ("term_overlap", "conflicts_with"):
            for peer in self.neighbors(key, kind):
                self.unlink(kind, key, peer)
            for peer in self.neighbors(key, kind, reverse=True):
                self.unlink(kind, peer, key)
        for kind in ("depends_on", "merged_from"):
            for peer in self.neighbors(key, kind):
                self.unlink(kind, key, peer)
        scored = sorted(((overlap(trace.current.text, traces[p].current.text), p)
                         for p in peers if p != key), key=lambda x: (-x[0], x[1]))
        for score, peer in scored[:limit]:
            if score:
                self.link("term_overlap", key, peer, score)
                self.link("term_overlap", peer, key, score)
        for dep in trace.depends_on:
            self.link("depends_on", key, dep)
        for ref in trace.merged_from:
            self.link("merged_from", key, ref.key)

    def coactivate(self, keys: list[str], *, cohorts: list[list[str]] | None = None,
                   episodes: list[list[str]] | None = None,
                   hubs: list[tuple[str, list[str]]] | None = None) -> None:
        # Preserve the global sparse path and also connect the actual contextual
        # cohorts. A single path can alternate between unrelated pools/tasks,
        # leaving no same-context neighbors even though all were reactivated.
        # An episode is reactivated as a bound whole: its members join the
        # joint event even when only one of them was selected.
        selected = set(keys)
        chains = [sorted(selected)]
        chains.extend(sorted(selected.intersection(cohort)) for cohort in (cohorts or []))
        chains.extend(sorted(set(episode)) for episode in (episodes or []))
        pairs = {pair for chain in chains for pair in zip(chain, chain[1:])}
        # A concept hub joins the reactivated attribute to each sibling.
        pairs.update(tuple(sorted((center, member))) for center, members in (hubs or [])
                     for member in members if member != center)
        # One joint event counts once even if the same edge belongs to both
        # paths. Disjoint cohorts add at most another N-1 edges, not a clique.
        for left, right in sorted(pairs):
            value = self.edges.get(("co_activation", left, right), 0) + 1
            self.link("co_activation", left, right, value)
            self.link("co_activation", right, left, value)
