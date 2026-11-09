"""Lossless factored cards, cluster-driven requests and version-bound coverage."""
from __future__ import annotations

import os
from difflib import SequenceMatcher
from functools import lru_cache
from .admission import KEY
from .config import RuntimeConfig
from .graph import TraceGraph
from .stance import currently_valid
from .tiers import VERTICAL_GATE_VERSION, late_allowed
from .types import ConsolidationProposal, ConsolidationRequest, SemanticFrame, SourceRef, Trace


def semantic_frame(claims: tuple[str, ...]) -> SemanticFrame | None:
    """Learn a shared schema from ordered evidence, rather than only a prefix.

    This is state-level, lossless schema induction. It does not certify free
    paraphrase, infer a new fact, or train provider weights.
    """
    # A pure function of the claim texts returning an immutable frame; repeated
    # assimilation scans re-test the same card/candidate texts every step.
    return _semantic_frame(tuple(claims))


@lru_cache(maxsize=65536)
def _semantic_frame(claims: tuple[str, ...]) -> SemanticFrame | None:
    sequences = [claim.split() for claim in claims]
    if not sequences:
        return None
    common = list(sequences[0])
    for sequence in sequences[1:]:
        matcher = SequenceMatcher(a=common, b=sequence, autojunk=False)
        common = [token for block in matcher.get_matching_blocks()
                  for token in common[block.a:block.a + block.size]]
    if not common:
        return None
    positions = []
    for sequence in sequences:
        selected, start = [], 0
        for token in common:
            offset = sequence.index(token, start)
            selected.append(offset)
            start = offset + 1
        positions.append(selected)
    boundaries = [0] + [i for i in range(1, len(common))
                       if any(p[i] != p[i - 1] + 1 for p in positions)] + [len(common)]
    segments = tuple(" ".join(common[a:b]) for a, b in zip(boundaries, boundaries[1:]))
    rows = []
    for sequence, indices in zip(sequences, positions):
        row, start = [], 0
        for a, b in zip(boundaries, boundaries[1:]):
            row.append(" ".join(sequence[start:indices[a]]))
            start = indices[b - 1] + 1
        row.append(" ".join(sequence[start:]))
        rows.append(tuple(row))
    return SemanticFrame(segments, tuple(rows))


def frame_claims(frame: SemanticFrame) -> tuple[str, ...]:
    if (not isinstance(frame, SemanticFrame) or not frame.segments
            or any(type(s) is not str or not s for s in frame.segments)
            or any(len(row) != len(frame.segments) + 1 or any(type(s) is not str for s in row)
                   for row in frame.rows)):
        raise ValueError("SEMANTIC_FRAME_INVALID")
    return tuple(" ".join(part for i, segment in enumerate(frame.segments)
                           for part in (row[i], segment)) + (" " + row[-1] if row[-1] else "")
                 for row in frame.rows)


def frame_text(frame: SemanticFrame) -> str:
    frame_claims(frame)
    varying = [i for i in range(len(frame.segments) + 1) if any(row[i] for row in frame.rows)]
    parts = []
    for i in range(len(frame.segments) + 1):
        if i in varying:
            parts.append("{" + str(varying.index(i) + 1) + "}")
        if i < len(frame.segments):
            parts.append(frame.segments[i])
    cases = tuple(dict.fromkeys(tuple(row[i] for i in varying) for row in frame.rows))
    # JSON preserves boundaries of strings that contain commas or separators.
    import json
    rendered = ";".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in cases)
    return " ".join(parts) + (" :: " + rendered if varying else "")


def factored_text(claims: tuple[str, ...]) -> str:
    """Factor common textual schema without inventing an inferred proposition.

    A shared prefix is explicitly a prefix, not an assertion of a new fact.
    The coverage vector is retained and reconstitutes each source verbatim.
    """
    unique = tuple(dict.fromkeys(claims))
    if not unique:
        return ""
    prefix = os.path.commonprefix(unique)
    prefix = prefix[:prefix.rfind(" ") + 1] if " " in prefix else ""
    normal = " ; ".join(unique)
    compressed = prefix + "{" + " ; ".join(t[len(prefix):] for t in unique) + "}"
    candidates = [normal]
    if prefix:
        candidates.append(compressed)
    frame = semantic_frame(unique)
    if frame is not None:
        candidates.append(frame_text(frame))
    return min(candidates, key=lambda text: (len(text.encode("utf-8")), text))


@lru_cache(maxsize=65536)
def compressible(claims: tuple[str, ...]) -> bool:
    """Whether these claims share structure that a lossless card can factor.

    Consolidation abstracts regularities. Facts that were reactivated
    together but share no schema stay linked by their co-activation edges;
    proposing them as a card can only fail the compression gate, and every
    superset of such a group fails as well.
    """
    return (len(factored_text(claims).encode("utf-8"))
            < sum(len(claim.encode("utf-8")) for claim in claims))


def assimilating_card(sources, traces: dict[str, Trace]) -> Trace | None:
    """Return the live frame card whose complete coverage these sources extend."""
    owners = {traces[ref.key].merged_into for ref in sources
              if ref.key in traces and traces[ref.key].merged_into is not None}
    if len(owners) != 1:
        return None
    card = traces.get(next(iter(owners)))
    if (card is None or card.kind != "semantic_card" or card.frame is None
            or not currently_valid(card) or card.merged_into is not None or card.suppressed
            or not set(card.merged_from) < set(sources)):
        return None
    return card


def find_assimilation(traces: dict[str, Trace], graph: TraceGraph, config: RuntimeConfig,
                      incompressible: set[tuple[SourceRef, ...]] | None = None
                      ) -> ConsolidationRequest | None:
    """Schema-consistent reconsolidation of an existing frame card.

    A learned frame is an existing cortical structure: a late-admitted
    episode that instantiates the same relational segments extends that card
    (the card becomes labile and is rewritten as its next version) instead of
    waiting for a fresh cluster of new instances or forming a duplicate card.
    Membership is exact frame structure, never lexical similarity.
    """
    if not (config.mechanisms.consolidation and config.mechanisms.reconsolidation):
        return None
    cards = sorted(key for key, trace in traces.items()
                   if trace.kind == "semantic_card" and trace.frame is not None
                   and trace.merged_into is None and currently_valid(trace)
                   and not trace.suppressed)
    for card_key in cards:
        card = traces[card_key]
        if len(card.merged_from) >= config.cluster_limit:
            continue
        if any(ref.key not in traces or traces[ref.key].merged_into != card_key
               or traces[ref.key].revision != ref.revision for ref in card.merged_from):
            continue
        # A card certified under an older or unknown criterion is extended
        # only when every member passes the current one (R4-01); otherwise
        # the request would be refused and block fresh cluster formation.
        if (card.gate_version < VERTICAL_GATE_VERSION
                and not all(late_allowed(traces[ref.key], config) for ref in card.merged_from)):
            continue
        claims = tuple(traces[ref.key].current.text for ref in card.merged_from)
        members = {ref.key for ref in card.merged_from}
        context = (card.current.scope, card.current.task)
        for key in sorted(graph.pools.get(card.pool, ())):
            trace = traces.get(key)
            if (trace is None or key in members or trace.kind != "episodic"
                    or trace.merged_into is not None or not late_allowed(trace, config)
                    or graph.neighbors(key, "conflicts_with")
                    or (trace.current.scope, trace.current.task) != context):
                continue
            frame = semantic_frame(claims + (trace.current.text,))
            if frame is None or frame.segments != card.frame.segments:
                continue
            refs = card.merged_from + (SourceRef(key, trace.revision),)
            # The incompressible memo is keyed by member identity, not by the
            # card's insertion order, so an extension the gate already refused
            # is never proposed again.
            if incompressible and canonical_refs(refs) in incompressible:
                continue
            # Assimilation obeys the same precondition as a fresh cluster: an
            # extension whose factored text no longer saves bytes is not an
            # abstraction, so it is never requested (and never blocks fresh
            # cluster formation behind a request the gate must reject).
            if not compressible(claims + (trace.current.text,)):
                continue
            return ConsolidationRequest(refs, claims + (trace.current.text,), card.about)
    return None


def canonical_refs(refs) -> tuple[SourceRef, ...]:
    return tuple(sorted(refs, key=lambda ref: ref.key))


def find_cluster(traces: dict[str, Trace], graph: TraceGraph,
                 config: RuntimeConfig,
                 incompressible: set[tuple[SourceRef, ...]] | None = None) -> ConsolidationRequest | None:
    if not config.mechanisms.consolidation:
        return None
    assimilation = find_assimilation(traces, graph, config, incompressible)
    if assimilation is not None:
        return assimilation
    best = None
    for pool in sorted(graph.pools):
        members = sorted(k for k in graph.pools[pool] if k in traces
                         and late_allowed(traces[k], config)
                         and traces[k].kind == "episodic" and traces[k].merged_into is None
                         and not graph.neighbors(k, "conflicts_with"))
        # Each member must be linked to a reactivated cohort, independently
        # of reader capacity. Recurrence alone never bypasses late replay.
        for scope, task in sorted({(traces[k].current.scope, traces[k].current.task)
                                   for k in members}):
            scoped = {k for k in members if (traces[k].current.scope, traces[k].current.task)
                      == (scope, task)}
            active = {k for k in scoped if set(graph.neighbors(k, "co_activation")) & scoped}
            for anchor in sorted(active):
                peers = (set(graph.neighbors(anchor, "co_activation"))
                         | set(graph.neighbors(anchor, "term_overlap"))) & active - {anchor}
                ordered = sorted(peers, key=lambda k: (
                    -graph.edges.get(("term_overlap", anchor, k), 0), k))
                cohort = [anchor] + ordered[:config.cluster_limit - 1]
                for size in range(len(cohort), config.cluster_size - 1, -1):
                    keys = sorted(cohort[:size])
                    # Prefer shared structure over arbitrary sorted pool heads.
                    shared = sum(graph.edges.get(("term_overlap", anchor, k), 0)
                                 for k in cohort[1:size]) / (size - 1)
                    rank = (-shared, -size, tuple(keys))
                    if best is not None and rank >= best[0]:
                        continue
                    refs = tuple(SourceRef(k, traces[k].revision) for k in keys)
                    if incompressible and refs in incompressible:
                        continue
                    texts = tuple(traces[k].current.text for k in keys)
                    # Select the largest sub-cohort with shared structure
                    # rather than requesting a group the gate must reject.
                    if not compressible(texts):
                        continue
                    request = ConsolidationRequest(refs, texts, (pool,))
                    best = (rank, request)
    return best[1] if best else None


def proposal_for(request: ConsolidationRequest, key: str) -> ConsolidationProposal:
    return ConsolidationProposal(key, request.sources, request.texts,
                                 factored_text(request.texts), request.about, semantic_frame(request.texts))


def validate_proposal(proposal: ConsolidationProposal, traces: dict[str, Trace],
                      graph: TraceGraph, config: RuntimeConfig,
                      revising: Trace | None = None) -> str | None:
    if revising is None:
        if not KEY.fullmatch(proposal.key) or proposal.key in traces:
            return "CARD_KEY_INVALID_OR_EXISTS"
    elif proposal.key != revising.key:
        return "CARD_KEY_INVALID_OR_EXISTS"
    # Members already certified by the card being reconsolidated passed the
    # late gate when that card formed; only the new member must pass it now.
    # Membership is proof of certification only under the current criterion:
    # a card certified under an older or unknown one re-checks every member.
    certified_members = (set(revising.merged_from) if revising is not None
                         and revising.gate_version >= VERTICAL_GATE_VERSION else set())
    if len(proposal.sources) < config.cluster_size or len(proposal.sources) > config.cluster_limit:
        return "CARD_CLUSTER_SIZE_INVALID"
    if len({r.key for r in proposal.sources}) != len(proposal.sources):
        return "CARD_SOURCE_DUPLICATE"
    if len(proposal.sources) != len(proposal.claims):
        return "CARD_COVERAGE_INCOMPLETE"
    scopes = {(traces[r.key].current.scope, traces[r.key].current.task)
              for r in proposal.sources if r.key in traces}
    if len(scopes) > 1:
        return "CARD_CONTEXT_MIXED"
    for ref, claim in zip(proposal.sources, proposal.claims):
        trace = traces.get(ref.key)
        if trace is None or trace.revision != ref.revision:
            return "CARD_SOURCE_REVISION_MISMATCH"
        # A source revision supports exactly one live abstraction: a new card
        # may only take uncarried members, and a revision only its own.
        if trace.merged_into is not None and (revising is None
                                              or trace.merged_into != revising.key):
            return "CARD_SOURCE_ALREADY_CARRIED"
        if not late_allowed(trace, config) and ref not in certified_members:
            return "CARD_LATE_ADMISSION_DENIED"
        if graph.neighbors(ref.key, "conflicts_with"):
            return "CARD_CONFLICT_UNRESOLVED"
        if claim != trace.current.text:
            return "CARD_CLAIM_NOT_LOSSLESS"
    certified = {factored_text(proposal.claims)}
    if proposal.frame is not None:
        try:
            restored = tuple(" ".join(s.split()) for s in frame_claims(proposal.frame))
            expected = tuple(" ".join(s.split()) for s in proposal.claims)
            if restored != expected:
                return "CARD_FRAME_COVERAGE_MISMATCH"
            certified.add(frame_text(proposal.frame))
        except (ValueError, TypeError):
            return "CARD_FRAME_INVALID"
    if proposal.text not in certified:
        return "CARD_TEXT_NOT_CERTIFIED"
    if len(proposal.text.encode("utf-8")) >= sum(len(c.encode("utf-8")) for c in proposal.claims):
        return "CARD_NO_COMPRESSION"
    return None
