"""Temporal eligibility, typed competition and indivisible dependency closure."""
from __future__ import annotations

from .competition import CompetitionConfig, DirectedEdge, GraphNode, GraphSpec, run_dynamics
from .config import RuntimeConfig
from .forgetting import expressed
from .graph import TraceGraph
from .language import cue_identity_match
from .signals import cue_match, priority, retrieval_gain
from .stance import (currently_valid, eligible_stances, in_window, present_day, render_history,
                     selected_stance, window_start)
from .types import Evidence, Query, Stance, Tier, Trace


def window_since(traces: dict[str, Trace], query: Query,
                 config: RuntimeConfig) -> tuple[str, str] | None:
    """Inclusive (first day, present) of a relative window, or None.

    The window is a retrieval context: it gates which experienced episodes
    may be expressed, never what is true now. A memory with no dated
    experience has no present, so a relative window cannot be placed. An
    explicit as-of day moves the present back to that day.
    """
    if query.within is None or not config.mechanisms.temporal_window:
        return None
    present = query.at_when or present_day(traces.values())
    if present is None:
        return None
    return window_start(query.within, present), present


def windowed(trace: Trace, stance: Stance, traces: dict[str, Trace],
             since: tuple[str, str] | None) -> bool:
    """Whether an expressed stance belongs to the window's episodes.

    A card is a later rewrite of earlier episodes, so its time is that of its
    sources. A gist that also covers episodes outside the window is not a
    representation of the window: the window cue reinstates the specific
    in-window sources instead, as a specific cue does for one member.
    """
    if since is None:
        return True
    if trace.kind == "semantic_card" and trace.merged_from:
        return all(ref.key in traces and in_window(traces[ref.key].current, since)
                   for ref in trace.merged_from)
    return in_window(stance, since)


def represented(trace: Trace, traces: dict[str, Trace]) -> bool:
    """Whether a merged source is expressed through its live card.

    Consolidation rewrites sources into a card; it does not withdraw them.
    Such a source stays as reachable as the card that carries it, so a cue
    to the source is a cue to the card and can reinstate the source itself.
    """
    card = traces.get(trace.merged_into) if trace.merged_into else None
    return (card is not None and card.kind == "semantic_card" and expressed(card)
            and not card.stale and currently_valid(trace) and not trace.suppressed
            and any(ref.key == trace.key and ref.revision == trace.revision
                    for ref in card.merged_from))


def reader_history_allowed(trace: Trace, traces: dict[str, Trace]) -> bool:
    """Keep derived audit history out of reader output after hard forgetting.

    A semantic card retains its own history for audit and recovery, but that
    history is a derived representation of its source refs.  Once any source
    was explicitly hard-forgotten, exposing the card's old text would bypass
    the hard-forget boundary.  Other stale cards remain historical evidence;
    this gate is deliberately limited to the irreversible hard-forget case.
    """
    if trace.kind != "semantic_card":
        return True
    return not any(ref.key not in traces or traces[ref.key].hard_forgotten
                   for ref in trace.merged_from)


def content_time(evidence: Evidence, traces: dict[str, Trace]) -> int:
    """When the expressed content was experienced, as an encoding operation.

    A live card is a later rewrite of earlier episodes; its content belongs
    to the time of its newest source, not to the moment it was consolidated.
    """
    trace = traces[evidence.key]
    if trace.kind == "semantic_card" and evidence.current and trace.merged_from:
        sources = [traces[ref.key].current.operation for ref in trace.merged_from if ref.key in traces]
        if sources:
            return max(sources)
    return evidence.operation


def chronological(items: list[Evidence], traces: dict[str, Trace]) -> list[Evidence]:
    """Express an already selected evidence set in experienced order.

    Selection, budget and competition are finished; only the serial order of
    the same lines changes. Statements about one thing written under
    different keys are not merged or withdrawn by inferred identity. The
    reader receives them in the order they happened, so the newest stance is
    last and its reasoning, not lexical overlap, decides what holds now.
    """
    return sorted(items, key=lambda e: (content_time(e, traces), e.key))


def exclusive_claim_label(key: str, stance: Stance, traces: dict[str, Trace],
                          graph: TraceGraph, ordered: bool) -> str:
    """Warn that a single-valued slot has competing claims, and state their order.

    Competing claims are never withdrawn here: a later claim may be a change
    of state ("moved to the garden") or a contradiction, and only reasoning
    over the kept history can tell which.  What the core does know is the
    order in which the claims were experienced, and the chronological view
    already lists them oldest first.  Calling time-ordered claims
    "unresolved" would withhold that order from the reader, so the label
    states the order instead.  It never calls a claim the latest: slots are
    derived only from explicit metadata or a few single-valued predicates, so
    a later statement about the same slot ("journeyed to the bedroom") may
    exist without being detected as a competitor.  Claims experienced in the
    same encoding operation have no order and stay unresolved, as do all
    claims when temporal ordering is disabled.
    """
    if not ordered:
        return "UNRESOLVED EXCLUSIVE-SLOT CLAIM: "
    rivals = [traces[other].current for other in graph.neighbors(key, "conflicts_with")
              if other in traces]
    if not rivals or any(rival.operation == stance.operation for rival in rivals):
        return "UNRESOLVED EXCLUSIVE-SLOT CLAIM: "
    later = sum(rival.operation > stance.operation for rival in rivals)
    earlier = len(rivals) - later
    return (f"COMPETING EXCLUSIVE-SLOT CLAIM (follows {earlier}, followed by {later} "
            f"competing claim(s); lines are in experienced order): ")


def evidence_set(traces: dict[str, Trace], graph: TraceGraph, query: Query,
                 operation: int, config: RuntimeConfig,
                 stats: dict[str, int] | None = None) -> list[Evidence]:
    candidates = []
    # Lexical candidate admission and evidence eligibility serve different
    # purposes. An explicit dependency is itself a cue to its source; the
    # source need not share a query token or occupy the expression window.
    # Context, temporal validity and indivisible packet bounds still apply.
    eligible_by_key = {}
    cue_by_key = {}
    via_card = set()
    indegree = {k: len(graph.neighbors(k, "depends_on", reverse=True)) for k in traces}
    historical = query.view == "history" or query.at_operation is not None or query.at_when is not None
    since = window_since(traces, query, config)
    outside = 0
    split_cards = {key for key, trace in traces.items()
                   if since is not None and trace.kind == "semantic_card" and trace.merged_from
                   and not windowed(trace, trace.current, traces, since)}
    for key, trace in traces.items():
        if trace.hard_forgotten:
            continue
        if not reader_history_allowed(trace, traces):
            continue
        stance = selected_stance(trace, query)
        if stance is None:
            continue
        # Only a silent trace depends on its cue alone to be reached; an
        # expressed trace is already in the window, and segmenting its cue
        # would only let long traces crowd out the rest of the window.
        segmented = config.mechanisms.event_segmentation and not expressed(trace)
        cued = cue_match(query.text, stance.text, segmented) if query.text else 0.0
        if historical:
            history = [s for s in eligible_stances(trace, query)
                       if trace.kind == "semantic_card" or in_window(s, since)]
            if not history or not windowed(trace, stance, traces, since):
                outside += 1
                continue
            cued = (max((cue_match(query.text, s.text, segmented) for s in history), default=0.0)
                    if query.text else 0.0)
            if not config.mechanisms.history_cueing or (query.text and cued < config.history_cue_overlap):
                continue
            text = "\n".join(render_history(s) for s in history)
            current = False
        else:
            if stance.action != "assert" or trace.stale:
                continue
            if not windowed(trace, stance, traces, since):
                outside += 1
                continue
            text, current = stance.text, True
        # A complete direct cue is stronger than lexical similarity for a
        # suppressed trace.  Jaccard alone treats same-subject/same-predicate
        # neighbours (``Alex likes cocoa``) as a cue for ``Alex likes tea``;
        # the identity gate keeps history, current and writer retrieval on the
        # same exact-claim boundary while leaving ordinary fuzzy retrieval
        # unchanged for unparsed prose.  Apply this after the historical branch
        # has recomputed its maximum cue over the selected history.
        if query.text and trace.suppressed:
            identity_match = cue_identity_match(query.text, stance.text)
            if identity_match is False:
                cued = 0.0
            elif identity_match is True:
                cued = 1.0
        score = (retrieval_gain(priority(trace, operation, indegree[key], config))
                 * (cued if query.text else 1.0))
        ambiguous = bool(graph.neighbors(key, "conflicts_with"))
        if ambiguous and current:
            text = exclusive_claim_label(key, stance, traces, graph,
                                         config.mechanisms.temporal_order) + text
        evidence = Evidence(key, text, trace.tier.value, current, score, ambiguous,
                            stance.operation, stance.when)
        eligible_by_key[key] = evidence
        cue_by_key[key] = cued
        if historical and trace.suppressed and (not query.text
                                                or cued < config.suppression_cue_overlap):
            # Suppression governs expression in every view (R5-04): a broad
            # or uncued history request does not bring back what the user
            # asked to stop mentioning. The stored history is untouched and
            # a cue about the trace itself still recovers it.
            continue
        if not historical and not expressed(trace):
            if query.text and cued > 0 and represented(trace, traces):
                # When the window splits the gist, the cued in-window episode
                # is expressed itself rather than through the card.
                if trace.merged_into not in split_cards:
                    via_card.add(key)
            elif not query.text or cued < config.history_cue_overlap:
                continue
            elif trace.suppressed and cued < config.suppression_cue_overlap:
                # Explicit suppression is not budget silence: a neighbouring
                # cue does not bring it back, a cue about the trace does. It
                # stays eligible as support of a claim that is expressed.
                continue
        if query.text and score <= 0:
            continue
        candidates.append(evidence)
    recruited: dict[str, set[str]] = {}
    if config.mechanisms.pattern_completion and query.text and not historical:
        # Pattern completion: a partial cue reactivates what was learned
        # together with the cued traces. The recruited trace inherits a
        # discounted cue through one learned co-activation link; lexical
        # similarity alone is never an association. Weak (cold, merged or
        # soft-forgotten) traces still need their own direct cue, and the
        # context, time and stance gates above have already applied.
        spreading = CompetitionConfig().spreading_factor
        cued_keys = {e.key for e in candidates if e.current and cue_by_key[e.key] > 0}
        for key in sorted(eligible_by_key):
            evidence = eligible_by_key[key]
            trace = traces[key]
            if (key in cued_keys or cue_by_key[key] > 0 or not evidence.current
                    or not expressed(trace)):
                continue
            peers = {peer for peer in graph.neighbors(key, "co_activation") if peer in cued_keys}
            inherited = max((cue_by_key[peer] * w / (1 + w) for peer in peers
                             for w in (graph.edges[("co_activation", key, peer)],)), default=0.0)
            if inherited <= 0:
                continue
            score = (retrieval_gain(priority(trace, operation, indegree[key], config))
                     * spreading * inherited)
            candidates.append(Evidence(key, evidence.text, evidence.tier, True, score,
                                       evidence.ambiguous, evidence.operation, evidence.when))
            recruited[key] = peers
    candidates.sort(key=lambda e: (-e.score, e.key))
    # A live lossless card is the current representation of its sources, not
    # an additional copy of that knowledge. Source cues still find the card;
    # original histories remain independent and are never routed this way.
    representatives = {}
    for key, evidence in eligible_by_key.items():
        trace = traces[key]
        card = traces.get(trace.merged_into)
        if (evidence.current and card is not None and card.key in eligible_by_key
                and eligible_by_key[card.key].current and card.kind == "semantic_card"
                and any(ref.key == key and ref.revision == trace.revision for ref in card.merged_from)):
            representatives[key] = card.key
    # A source admitted only through its card is never expressed in the
    # card's place when the card itself is not an eligible current line.
    candidates = [e for e in candidates if e.key not in via_card or e.key in representatives]
    direct = {e.key: e for e in candidates}
    preferred, fallbacks, cue_of = {}, {}, {}
    for evidence in candidates:
        key = representatives.get(evidence.key, evidence.key)
        chosen = eligible_by_key[key]
        score = max(chosen.score, evidence.score)
        cue_of[key] = max(cue_of.get(key, 0.0), cue_by_key[evidence.key])
        if key != evidence.key:
            fallbacks.setdefault(key, []).append(evidence.key)
        if key not in preferred or score > preferred[key].score:
            preferred[key] = Evidence(key, chosen.text, chosen.tier, chosen.current, score,
                                      chosen.ambiguous, chosen.operation, chosen.when)
    candidates = sorted(preferred.values(), key=lambda e: (-e.score, e.key))
    # This intervention disables associative support cueing, not the
    # requirement to carry a claim's complete eligible evidence. Directly
    # cued support remains available even below the ranking limit.
    support_by_key = (eligible_by_key if config.mechanisms.context_integration
                      else {**direct, **{e.key: e for e in candidates}})
    candidates = candidates[:config.candidate_limit]
    if config.mechanisms.competition and candidates:
        # The cue completes the pattern it matches best: that trace drives
        # the lateral inhibition and is not inhibited by traces the cue
        # matches less (R5-02). Accessibility and learned links only decide
        # among the rest, so neither a replayed cohort nor its spreading
        # activation can silence the most specific match.
        best = min(candidates, key=lambda e: (-cue_of.get(e.key, 0.0), -e.score, e.key))
        anchor = best.key if query.text and cue_of.get(best.key, 0.0) > 0 else None
        nodes = tuple(GraphNode(e.key, ("history:" if not e.current else
                                      "semantic:" if traces[e.key].kind == "semantic_card" else
                                      "episodic:") + traces[e.key].pool, e.key == anchor)
                      for e in sorted(candidates, key=lambda e: e.key))
        # Lexical similarity is what makes traces compete; it is not an
        # association, so only learned links (dependency, co-activation)
        # spread. Inflow is divisively normalized so a replay hub cannot
        # outrun a cue, and inhibition scales the cue-driven score rather
        # than replacing relevance with network state.
        ids = {e.key for e in candidates}
        edges = {}
        for (kind, a, b), w in graph.edges.items():
            if a in ids and b in ids and kind in ("depends_on", "co_activation"):
                edges[a, b] = max(edges.get((a, b), 0), w / (1 + w) if kind == "co_activation" else w)
        indegree = {}
        for _, target in edges:
            indegree[target] = indegree.get(target, 0) + 1
        spec = GraphSpec(nodes, tuple(DirectedEdge(a, b, w / indegree[b]) for (a, b), w in sorted(edges.items())))
        top = max(e.score for e in candidates) or 1.0
        dynamics = run_dynamics(spec, {e.key: e.score / top for e in candidates}, mode="typed",
                                config=CompetitionConfig(inhibition_beta=config.inhibition_beta))
        first = {n.node_id: n for n in dynamics.trace[0].nodes}
        scores = {e.key: e.score * (first[e.key].inhibited_potential / first[e.key].potential
                                    if first[e.key].potential > 0 else 0.0) for e in candidates}
        candidates.sort(key=lambda e: (-scores[e.key], -e.score, e.key))
        # A trace whose activity the competition drives to zero has lost:
        # it is not expressed on its own, whatever room the window has left.
        # It stays reachable as a dependency of a claim that won.
        candidates = [Evidence(e.key, e.text, e.tier, e.current, scores[e.key], e.ambiguous,
                               e.operation, e.when) for e in candidates if scores[e.key] > 0]
    # Dependencies that rank below the limit are still eligible for closure.
    by_key = {**support_by_key, **{e.key: e for e in candidates}}
    selected = []
    used = set()
    words = size = records = 0
    for evidence in candidates:
        # Completion is relative to what the cue actually retrieved: an
        # associate is never expressed in place of a cue that did not fit.
        if evidence.key in recruited and not recruited[evidence.key] & used:
            continue
        # Prefer the compressed representation. If its complete proof cannot
        # fit, try the already cued original sources with the same proof gate.
        roots, modes = (evidence.key, *fallbacks.get(evidence.key, ())), (True, False)
        cued = fallbacks.get(evidence.key, ())
        if (config.mechanisms.specific_reinstatement and cued
                and traces[evidence.key].kind == "semantic_card"
                and all(sum(cost(by_key[k].text) for k in cued) < cost(evidence.text)
                        for cost in (lambda t: len(t.split()), lambda t: len(t.encode("utf-8"))))):
            # Specific reinstatement: a cue that reaches only part of a
            # schema reinstates those episodes, the cheapest complete
            # expression of what was cued (cheaper in both reader budgets).
            # The gist card is expressed when the cue spans enough of it that
            # the card costs no more.
            roots, modes = (*cued, evidence.key), (False, True)
        for root in roots:
            for use_cards in modes:
                group, visiting, built = [], set(), set()

                def enter(key: str) -> bool | tuple[str, list[str]]:
                    """True when KEY is already expressed, False when it
                    cannot be, else KEY as visited with its dependencies."""
                    if key in used or key in built:
                        return True
                    represented = representatives.get(key) if use_cards else None
                    if represented in by_key:
                        key = represented
                    if key in used or key in built:
                        return True
                    if key in visiting or key not in by_key:
                        return False
                    trace = traces[key]
                    internal = ({ref.key for ref in trace.merged_from}
                                if by_key[key].current and trace.kind == "semantic_card" else set())
                    # A previous original fallback already occupies this
                    # representation. Do not append its card as another vote
                    # or duplicate text; cued originals remain available.
                    if internal & (used | built):
                        return False
                    visiting.add(key)
                    dependencies: set[str] = set()
                    if by_key[key].current:
                        dependencies = set(trace.depends_on)
                        if trace.kind == "semantic_card":
                            dependencies.update(dep for ref in trace.merged_from
                                                for dep in traces[ref.key].depends_on)
                            dependencies.difference_update(internal)
                    return key, sorted(dependencies)

                def closure(root_key: str) -> bool:
                    """Every dependency of ROOT_KEY, then the key itself, joins
                    GROUP in dependency order.  The walk keeps its own stack,
                    so a long depends_on chain is read without nesting a call
                    per link (r80 Pro F1)."""
                    entered = enter(root_key)
                    if entered is True or entered is False:
                        return entered
                    stack = [(entered[0], iter(entered[1]))]
                    while stack:
                        key, pending = stack[-1]
                        for dep in pending:
                            entered = enter(dep)
                            if entered is False:
                                return False
                            if entered is not True:
                                stack.append((entered[0], iter(entered[1])))
                                break
                        else:
                            stack.pop()
                            group.append(by_key[key])
                            visiting.discard(key)
                            built.add(key)
                    return True

                if not closure(root):
                    continue
                unique = [e for e in group if e.key not in used]
                lines = [f"[{e.key}] {e.text}" for e in unique]
                extra_words = sum(len(line.split()) for line in lines)
                extra_records = sum(sum(bool(part.strip()) for part in line.splitlines()) for line in lines)
                # Exact newlines, including those joining the existing packet.
                extra_bytes = sum(len(line.encode("utf-8")) for line in lines) + max(0, len(lines) - 1)
                if selected and lines:
                    extra_bytes += 1
                if (records + extra_records > config.reader_records
                        or words + extra_words > config.reader_words or size + extra_bytes > config.reader_bytes):
                    continue
                selected.extend(unique)
                used.update(e.key for e in unique)
                for item in unique:
                    if item.current and traces[item.key].kind == "semantic_card":
                        used.update(ref.key for ref in traces[item.key].merged_from)
                words += extra_words
                size += extra_bytes
                records += extra_records
                break
            if evidence.key in used:
                break
    if stats is not None:
        stats["recruited"] = len(recruited)
        stats["recruited_selected"] = sum(1 for e in selected if e.key in recruited)
        if since is not None:
            stats["window_applied"] = 1
            stats["window_outside"] = outside
    return selected
