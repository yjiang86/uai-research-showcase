"""Deterministic state-level memory runtime; all inference stays in adapters."""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from .admission import KEY, normalize, prepare
from .config import Mechanisms, RuntimeConfig, legacy_config
from .conflict import incompatible, revises
from .consolidation import (assimilating_card, canonical_refs, factored_text, find_cluster,
                            frame_claims, frame_text, proposal_for, validate_proposal)
from .forgetting import fits, usage, victims
from .graph import TraceGraph
from .integration import chronological, evidence_set
from .language import interpret_query, natural_slot, revision_references
from .prediction import ContextPredictor, event_for, schema_identity
from .replay import rank_candidates
from .signals import novelty, overlap, replay_score, terms
from .stance import currently_valid, valid_day
from .telemetry import Telemetry
from .tiers import VERTICAL_GATE_VERSION, consolidatable_tier, early_allowed
from .types import (BatchResult, ConsolidationProposal, ConsolidationRequest, Evidence,
                    MemoryInvariantError, Query, SemanticFrame, SourceRef, Stance, Tier, Trace)

SCHEMA = "uai-core-v2.runtime.v1"
_REVISION_MAX_TARGETS = 4


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class UAIMemoryRuntimeV2:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.operation = 0
        self._traces: dict[str, Trace] = {}
        self.graph = TraceGraph()
        self.telemetry = Telemetry()
        self.metadata_keys: set[str] = set()
        self.source_hashes: dict[str, str] = {}
        self._requests: dict[str, BatchResult] = {}
        self.predictions: dict[str, dict[str, Any]] = {}
        self.feedback: dict[str, float] = {}
        self.pending_consolidation: ConsolidationRequest | None = None
        # A pending request is born at the current maintenance epoch.  In the
        # delayed path it cannot be consumed until an independent maintenance
        # transition advances that epoch.  The fields are persisted so close /
        # restore cannot collapse the two time scales.
        self.maintenance_epoch = 0
        self.pending_consolidation_epoch: int | None = None
        self._requested_clusters: set[str] = set()
        self._incompressible_clusters: set[tuple[SourceRef, ...]] = set()
        self.context_prediction = ContextPredictor()

    def trace(self, key: str) -> Trace | None:
        return copy.deepcopy(self._traces.get(key))

    def traces(self) -> dict[str, Trace]:
        return copy.deepcopy(self._traces)

    def traces_view(self) -> Mapping[str, Trace]:
        """Read-only view for pure validators such as admission ``prepare``.

        ``traces()`` returns a detached copy; its cost grows with the store
        and is paid on every write. Callers of this view must not mutate it.
        """
        return MappingProxyType(self._traces)

    def _promote(self, trace: Trace) -> None:
        # Promotion is a consequence of stabilization, never a forget decision:
        # an explicitly suppressed trace stays suppressed until a new
        # independent encounter or a changed assertion lifts it.
        if trace.suppressed:
            return
        # A merged source is represented by its card; it stays a cold member
        # until the card is retired or reconsolidated (F12).
        if trace.merged_into is not None:
            return
        if trace.tier in (Tier.HOT, Tier.COLD) and early_allowed(trace, self.config):
            previous_tier = trace.tier.value
            trace.tier = Tier.WARM
            self.telemetry.add("promotions")
            self.telemetry.add(f"promotions.{previous_tier}_to_warm")
            self.telemetry.record(self.operation, "promotion", key=trace.key,
                                  from_tier=previous_tier, to_tier=trace.tier.value)

    def _count_consolidated_sources(self, refs, previous) -> None:
        """Count each source that newly enters a card by its real tier.

        "promotions" counts trace tier transitions, one per trace, so it
        equals the sum of its "promotions.*" sub-counters (R4 F8).
        """
        for ref in refs:
            tier = previous.get(ref.key)
            if tier is None:
                continue
            self.telemetry.add("promotions")
            self.telemetry.add(f"promotions.{tier}_to_consolidated")

    def _entry_stability(self, refs) -> dict[str, dict[str, int | str]]:
        """Raw stabilization evidence of each source as it enters a card.

        Recorded so an external audit can re-evaluate the early criterion
        from numbers rather than trusting the gate that applied it.
        """
        return {ref.key: {"tier": self._traces[ref.key].tier.value,
                          "recurrence": self._traces[ref.key].recurrence,
                          "replay_count": self._traces[ref.key].replay_count}
                for ref in refs}

    def _graph_update(self, trace: Trace) -> None:
        self.graph.update(trace, self._traces, self.config.graph_neighbors)
        if self.config.mechanisms.conflict_detection and trace.exclusive and currently_valid(trace):
            for other in self._traces.values():
                if other.key != trace.key and incompatible(trace, other):
                    already_linked = (("conflicts_with", trace.key, other.key) in self.graph.edges
                                      and ("conflicts_with", other.key, trace.key) in self.graph.edges)
                    self.graph.link("conflicts_with", trace.key, other.key)
                    self.graph.link("conflicts_with", other.key, trace.key)
                    if not already_linked:
                        self.telemetry.add("conflicts_detected")
                    # A card was certified only for conflict-free sources. Once
                    # a represented claim is contested, the gist would present
                    # one side unqualified; the card is retired so each source
                    # is expressed with its own unresolved-conflict warning.
                    self._retire_carrier(trace, "conflict")
                    self._retire_carrier(other, "conflict")

    def _refresh_conflicts(self) -> None:
        """Remove conflict edges whose claims are no longer jointly valid.

        Dependency invalidation can stale a child without changing its own
        text, so ``TraceGraph.update`` is not reached for that child.  Conflict
        edges are eligibility edges, not audit history: once either endpoint is
        stale, withdrawn, or hard-forgotten, the edge must stop affecting
        reader ambiguity and consolidation.  Rebuild the small conflict
        relation from the current valid traces after propagation.
        """
        for kind, left, right in list(self.graph.edges):
            if kind != "conflicts_with":
                continue
            a, b = self._traces.get(left), self._traces.get(right)
            if a is None or b is None or not incompatible(a, b):
                self.graph.unlink(kind, left, right)
        if not self.config.mechanisms.conflict_detection:
            return
        valid = [trace for trace in self._traces.values()
                 if trace.exclusive and currently_valid(trace)]
        for index, left in enumerate(valid):
            for right in valid[index + 1:]:
                if incompatible(left, right):
                    self.graph.link("conflicts_with", left.key, right.key)
                    self.graph.link("conflicts_with", right.key, left.key)

    def _retire_carrier(self, trace: Trace, reason: str) -> None:
        """Retire the live card that currently represents this trace, if any."""
        card = self._traces.get(trace.merged_into) if trace.merged_into else None
        if card is not None and card.kind == "semantic_card" and not card.stale:
            self._retire_card(card, reason)

    def _retire_card(self, card: Trace, reason: str) -> None:
        """Retire a derived card: its sources become independent traces again.

        The card's history is kept; its schema support is withdrawn and its
        identity is never resurrected. Unsuppressed sources become expressible
        on their own; suppressed sources stay silent.
        """
        self._drop_requested_cluster(card)
        card.stale = True
        card.tier = Tier.COLD
        self._release_card_support(card)
        self.telemetry.add("cards_retired." + reason)
        self.telemetry.record(self.operation, "card_retired", key=card.key, reason=reason)
        self._invalidate_dependents(card.key)

    def _drop_requested_cluster(self, card: Trace) -> None:
        """Allow a retired card's evidence cluster to be born again.

        ``_requested_clusters`` holds pending requests only; a consumed request
        leaves it (_consume_pending_request).  A retired, invalidated or
        hard-forgotten card is a new lifecycle boundary: its sources are
        released and a later replay may legitimately form the same structural
        cluster again.  Dropping the card's cluster identity here also clears
        one kept by a snapshot written before consumption removed it, so the
        reborn request gets its own birth receipt (delayed-path history
        invariant).
        """
        if card.kind != "semantic_card" or not card.merged_from:
            return
        request = ConsolidationRequest(
            tuple(card.merged_from),
            tuple(self._traces[ref.key].current.text
                  for ref in card.merged_from if ref.key in self._traces),
            tuple(card.about),
        )
        if len(request.sources) == len(request.texts):
            self._requested_clusters.discard(digest(asdict(request)))

    def _release_card_support(self, card: Trace) -> None:
        """Retire a derived expectation without erasing independent evidence."""
        if card.kind != 'semantic_card':
            return
        prototypes = {key: self._traces[key].current.text
                      for row in self.context_prediction.schemas.values()
                      for key in row['card_ids'] if key != card.key}
        self.context_prediction.forget_schema_card(card.key, prototypes=prototypes)
        for ref in card.merged_from:
            source = self._traces[ref.key]
            if source.merged_into == card.key:
                source.merged_into = None
                if currently_valid(source) and not source.suppressed:
                    source.tier = Tier.WARM

    def _invalidate_dependents(self, key: str) -> None:
        frontier, seen = [key], set()
        while frontier:
            parent = frontier.pop()
            for child in self.graph.neighbors(parent, "depends_on", reverse=True):
                if child not in seen:
                    seen.add(child)
                    self._traces[child].stale = True
                    self._drop_requested_cluster(self._traces[child])
                    self._release_card_support(self._traces[child])
                    frontier.append(child)
            if not self.config.mechanisms.reconsolidation:
                continue
            for card_key in self.graph.neighbors(parent, "merged_from", reverse=True):
                card = self._traces[card_key]
                # An already retired card no longer represents its sources.
                if card_key in seen or card.stale:
                    continue
                seen.add(card_key)
                card.stale = True
                card.tier = Tier.COLD if card.suppressed else Tier.WARM
                # A stale card no longer represents its sources, so its
                # cluster may be born again with its own receipt (r54 F3).
                self._drop_requested_cluster(card)
                self._release_card_support(card)
                self.telemetry.add("reconsolidations")
                frontier.append(card_key)
        self._refresh_conflicts()

    def _bind_revisions(self, plan: Any, applied: list[str], source_id: str,
                        observed: str | None, observations: list) -> list[str]:
        """Withdraw earlier traces that a new statement explicitly revises.

        "User used to like X", "User no longer enjoys X", "以前喜欢X" and
        "Budget: B (changed from A)" are themselves evidence that the earlier
        plain proposition no longer holds, even when the writer stored them
        under a new key.  The earlier trace keeps its whole history and gains
        a negate stance quoted from the revising statement.  A new positive
        value alone ("likes Y") never withdraws anything.  A reference that
        matches more than ``_REVISION_MAX_TARGETS`` traces identifies no
        single proposition and is left unbound.
        """
        if not self.config.mechanisms.conflict_detection:
            return []
        excluded = set(applied) | {row["key"] for row in plan.negates}
        bound: list[str] = []
        for key in applied:
            reviser = self._traces[key]
            if not currently_valid(reviser):
                continue
            refs = revision_references(reviser.current.text)
            if not refs:
                continue
            context = (reviser.current.scope, reviser.current.task)
            for ref in refs:
                matches = [trace for trace in self._traces.values()
                           if trace.key not in excluded and trace.kind != "semantic_card"
                           and currently_valid(trace)
                           and (trace.current.scope, trace.current.task) == context
                           and revises(ref, reviser.current.text, trace.current.text)]
                if not matches:
                    continue
                matches = [trace for trace in matches if trace.pool == reviser.pool] or matches
                if len(matches) > _REVISION_MAX_TARGETS:
                    self.telemetry.add("revisions_ambiguous")
                    continue
                for trace in sorted(matches, key=lambda item: item.key):
                    event = event_for({"entity": trace.entity, "attribute": trace.attribute},
                                      "negate", trace.current.text, trace.current.scope,
                                      trace.pool, task=trace.current.task)
                    signal = self.context_prediction.preview(event, source_id, self.config.mechanisms)
                    observations.append((trace.key, event, signal))
                    self._invalidate_dependents(trace.key)
                    trace.history.append(Stance(self.operation, "negate", trace.current.text,
                                                reviser.current.quote, source_id,
                                                reviser.current.when, trace.current.scope,
                                                trace.current.task, observed))
                    trace.tier = Tier.COLD
                    trace.support_ids = set()
                    self._graph_update(trace)
                    excluded.add(trace.key)
                    bound.append(trace.key)
                    self.telemetry.add("revisions_bound")
                    self.telemetry.record(self.operation, "revision_bound", key=trace.key,
                                          revised_by=key, kind=ref.kind, target=ref.target)
        return bound

    def request_identity(self, source: str, raw: str, *, source_id: str | None = None,
                         observed: str | None = None) -> str:
        """Return the idempotency identity used by :meth:`encode`.

        Application surfaces use this to distinguish a cached replay of an
        identical saved writer response from a new source event.  Keeping the
        calculation in the core prevents a wrapper from manufacturing a
        prediction observation for a request the core will return from its
        cache.
        """
        if type(source) is not str or type(raw) is not str:
            raise TypeError("ENCODE_REQUIRES_TEXT")
        resolved = source_id if source_id is not None else digest(source)
        if type(resolved) is not str or not resolved:
            raise MemoryInvariantError("SOURCE_ID_INVALID")
        source_hash = digest(source)
        if observed is not None and not self.config.mechanisms.episode_time:
            observed = None
        return digest((resolved, source_hash, raw) if observed is None
                      else (resolved, source_hash, raw, observed))

    def has_request(self, source: str, raw: str, *, source_id: str | None = None,
                    observed: str | None = None) -> bool:
        return self.request_identity(source, raw, source_id=source_id,
                                     observed=observed) in self._requests

    def encode(self, source: str, raw: str, *, source_id: str | None = None,
               observed: str | None = None) -> BatchResult:
        """Apply a saved/externally supplied writer proposal, with no model call.

        ``observed`` is the episode date from the application source envelope.
        It timestamps every stance written from this source.
        """
        if type(source) is not str or type(raw) is not str:
            raise TypeError("ENCODE_REQUIRES_TEXT")
        if observed is not None and not valid_day(observed):
            raise MemoryInvariantError("OBSERVED_DATE_INVALID")
        if not self.config.mechanisms.episode_time:
            observed = None
        source_id = source_id if source_id is not None else digest(source)
        if type(source_id) is not str or not source_id:
            raise MemoryInvariantError("SOURCE_ID_INVALID")
        source_hash = digest(source)
        if source_id in self.source_hashes and self.source_hashes[source_id] != source_hash:
            raise MemoryInvariantError("SOURCE_ID_CONTENT_COLLISION")
        request = self.request_identity(source, raw, source_id=source_id, observed=observed)
        if request in self._requests:
            return self._requests[request]
        self.operation += 1
        self.source_hashes[source_id] = source_hash
        plan = prepare(source, raw, self._traces, self.metadata_keys,
                       withdrawal_support=self.config.mechanisms.withdrawal_support)
        if plan.whole_error:
            result = BatchResult("MODEL_WRITE_REJECTED", self.operation, reason=plan.whole_error)
            self.telemetry.add("writes_rejected." + plan.whole_error)
            self.telemetry.record(self.operation, "encode", status=result.status, reason=result.reason,
                                  source_id=source_id, request=request)
            self._requests[request] = result
            return result
        self.metadata_keys = set(plan.metadata_keys)
        applied, negated, drops = [], [], list(plan.dropped)
        observations = []
        failed_updates: set[str] = set()
        for row in plan.upserts:
            key, text = row["key"], row["text"].strip()
            trace = self._traces.get(key)
            scope = row.get("scope", "default")
            task = row.get("task", "default")
            if trace is not None and trace.kind == "semantic_card":
                drops.append({"action": "upsert", "key": key, "reason": "ITEM_DERIVED_CARD_READ_ONLY"})
                failed_updates.add(key)
                continue
            if trace is not None and trace.hard_forgotten:
                drops.append({"action": "upsert", "key": key, "reason": "ITEM_HARD_FORGOTTEN"})
                failed_updates.add(key)
                continue
            if overlap(text, row["quote"]) < self.config.write_support_threshold:
                drops.append({"action": "upsert", "key": key, "reason": "ITEM_WRITE_SUPPORT_GATE"})
                self.telemetry.add("suppressed_by_boundary")
                failed_updates.add(key)
                continue
            if trace is not None and trace.current.scope != scope:
                drops.append({"action": "upsert", "key": key, "reason": "ITEM_SCOPE_KEY_COLLISION"})
                failed_updates.add(key)
                continue
            if trace is not None and trace.current.task != task:
                drops.append({"action": "upsert", "key": key, "reason": "ITEM_TASK_KEY_COLLISION"})
                failed_updates.add(key)
                continue
            if any(d in failed_updates or d not in self._traces or not currently_valid(self._traces[d])
                   for d in row.get("depends_on", [])):
                drops.append({"action": "upsert", "key": key, "reason": "ITEM_DEPENDENCY_UNRESOLVED"})
                failed_updates.add(key)
                continue
            # A different string is a new version, even at high lexical overlap.
            # Similarity is never sufficient to replace a conflicting value.
            stance = Stance(self.operation, "assert", text, row["quote"], source_id,
                            row.get("when"), scope, task, observed)
            about = tuple(a.strip().casefold() for a in row.get("about", []))
            deps = tuple(row.get("depends_on", []))
            # Structured slot metadata is authoritative.  When a writer omits
            # it, derive only the tiny, explicitly exclusive natural-language
            # relation set; likes/preferences and generic lexical neighbours
            # remain non-exclusive and cannot become conflicts by similarity.
            derived_slot = (natural_slot(text)
                            if row.get("entity") is None and row.get("attribute") is None
                            and "exclusive" not in row else None)
            entity = row.get("entity")
            attribute = row.get("attribute")
            exclusive = row.get("exclusive", False)
            if derived_slot is not None:
                entity, attribute, exclusive = derived_slot
            peers = [self._traces[k] for k in sorted(self.graph.pools.get(
                     about[0] if about else re.split(r"[_.:/-]", key)[0].casefold(), ()))
                     if k != key and currently_valid(self._traces[k])]
            novel = novelty(text, peers)
            event = event_for(row, "assert", text, scope,
                             about[0] if about else re.split(r"[_.:/-]", key)[0].casefold(),
                             task=task)
            signal = self.context_prediction.preview(event, source_id, self.config.mechanisms)
            if trace is None:
                trace = Trace(key, [stance], about, deps, support_ids={source_id}, novelty=novel,
                              entity=entity, attribute=attribute, exclusive=exclusive)
                self._traces[key] = trace
            else:
                changed = trace.current.action != "assert" or trace.current.text != text
                if changed:
                    self._invalidate_dependents(key)
                    trace.support_ids = {source_id}
                    trace.replay_count, trace.replay_score, trace.stability = 0, 0.0, 0.0
                    trace.last_replay = None
                    trace.tier = Tier.HOT
                    trace.suppressed = False
                    trace.novelty = novel
                    trace.merged_into = None
                else:
                    # Only a new independent source is a re-encounter. Resubmitting
                    # the same source (for example with reformatted JSON) adds no
                    # support and cannot undo an explicit forget.
                    reencounter = source_id not in trace.support_ids
                    if reencounter:
                        # A genuine re-encounter is allowed to restore this
                        # member, but it must not silently resurrect a
                        # suppressed semantic carrier that still contains
                        # forgotten siblings. Retire that representation so
                        # the active source is expressed independently while
                        # the other members retain their own suppression.
                        carrier = self._traces.get(trace.merged_into) if trace.suppressed else None
                        if (carrier is not None and carrier.kind == "semantic_card"
                                and not carrier.stale):
                            self._retire_card(carrier, "reencounter")
                        trace.support_ids.add(source_id)
                        if trace.suppressed:
                            trace.suppressed = False
                            trace.tier = Tier.HOT
                            self.telemetry.add("suppression_lifted.reencounter")
                # The same source restating the same assertion is one stance;
                # a duplicate would inflate history and recency (F9).
                if changed or reencounter:
                    trace.history.append(stance)
                trace.about = about or trace.about
                trace.depends_on = deps
                trace.entity = entity if entity is not None else trace.entity
                trace.attribute = attribute if attribute is not None else trace.attribute
                trace.exclusive = exclusive if "exclusive" in row or derived_slot is not None else trace.exclusive
            trace.stale = any(not currently_valid(self._traces[d]) for d in trace.depends_on)
            if signal["independent"]:
                trace.processing_gain, trace.context_error = signal["processing_gain"], signal["context_error"]
                observations.append((key, event, signal))
            self._graph_update(trace)
            self._promote(trace)
            applied.append(key)
            self.telemetry.add("writes_accepted")
            self.telemetry.record(self.operation, "signal", key=key, novelty=trace.novelty,
                                  support=trace.recurrence, prediction_error=self.feedback.get(key))
        revised = self._bind_revisions(plan, applied, source_id, observed, observations)
        negated.extend(revised)
        for row in plan.negates:
            trace = self._traces[row["key"]]
            event = event_for({"entity": trace.entity, "attribute": trace.attribute}, "negate",
                              trace.current.text, trace.current.scope, trace.pool,
                              task=trace.current.task)
            signal = self.context_prediction.preview(event, source_id, self.config.mechanisms)
            observations.append((trace.key, event, signal))
            self._invalidate_dependents(trace.key)
            trace.history.append(Stance(self.operation, "negate", trace.current.text, row["quote"],
                                        source_id, row.get("when"), trace.current.scope,
                                        trace.current.task, observed))
            trace.tier = Tier.COLD
            trace.support_ids = set()
            self._graph_update(trace)
            negated.append(trace.key)
            self.telemetry.add("writes_accepted")
        # Every prediction in this source batch is frozen before *any* of its
        # events are learned. Key sorting must not fabricate temporal exposure.
        for key, event, signal in observations:
            if self.context_prediction.learn(event, source_id):
                self.telemetry.add("context_observations")
                # ``event`` and ``signal`` both expose the selected schema.
                # Merge explicitly so telemetry remains a single JSON object
                # instead of raising on duplicate keyword expansion.  The
                # event schema is the admitted input; the signal copy is
                # retained under a distinct name for auditability.
                fields = {"key": key, "source_id": source_id, **event}
                signal_fields = dict(signal)
                if "schema" in signal_fields:
                    signal_fields["predicted_schema"] = signal_fields.pop("schema")
                fields.update(signal_fields)
                self.telemetry.record(self.operation, "context_observed", **fields)
        self.telemetry.add("writes_proposed", len(plan.upserts) + len(plan.negates) + len(plan.dropped))
        for item in drops:
            self.telemetry.add("writes_dropped." + item["reason"])
        if plan.ignored_evictions:
            self.telemetry.add("writer_evictions_ignored", plan.ignored_evictions)
        self.telemetry.add("dependencies_pruned", sum(len(p["dependencies"]) for p in plan.pruned))
        self._replay()
        self.pending_consolidation = self._request_consolidation()
        demoted = self._regulate()
        result = BatchResult("APPLIED", self.operation, tuple(applied), tuple(negated),
                             tuple(drops), plan.pruned)
        self._requests[request] = result
        self.telemetry.record(self.operation, "encode", status=result.status, source_id=source_id,
                              request=request, applied=applied, negated=negated, dropped=drops,
                              pruned=list(plan.pruned), demoted=demoted, window=self.window_usage(),
                              ignored_evictions=plan.ignored_evictions)
        return result

    def _coactivate(self, keys: list[str], episodes: list[list[str]] | None = None,
                    hubs: list[tuple[str, list[str]]] | None = None) -> None:
        cohorts = {}
        for key in keys:
            trace = self._traces[key]
            context = (trace.pool, trace.current.scope, trace.current.task)
            cohorts.setdefault(context, []).append(key)
        self.graph.coactivate(keys, cohorts=list(cohorts.values()), episodes=episodes, hubs=hubs)

    def _episodes(self, keys: list[str]) -> tuple[list[list[str]], list[tuple[str, list[str]]]]:
        """Episodes bound at encoding that contain a replayed trace.

        Items admitted from one source event share its support id: that is
        the encoding-time binding (the hippocampal event index). Replaying
        one member reactivates the bound episode in the same scope and task,
        so learned associations follow what was experienced together rather
        than which traces happened to share a replay batch. Episode siblings
        receive reactivation credit (see _replay) but never recurrence or
        support; merged, stale, withdrawn, soft-forgotten and hard-forgotten
        traces are not reactivated.

        Concept binding is the semantic hub of the same index: attributes
        the writer declared to describe one entity (``Trace.concept``) are
        reactivated with a replayed attribute even when they were encoded in
        different sessions, under the same exclusions and context gate. The
        hub links the replayed attribute to each sibling (one link per
        sibling), so a later cue on any attribute completes the others. Hub
        members are linked only and receive no reactivation credit: they were
        not experienced in the replayed event.

        The index also reaches silent engrams: siblings that lost expression
        only to the reader budget stay linkable (see consolidatable_tier), so
        offline consolidation can still compress them. Explicitly
        soft-forgotten siblings are not reactivated, and reading never
        recruits a cold trace through these links.
        """
        episodic = self.config.mechanisms.episodic_binding
        conceptual = self.config.mechanisms.concept_binding
        if not (episodic or conceptual) or not keys:
            return [], []
        wanted, concepts = set(), {}
        for key in keys:
            trace = self._traces[key]
            context = (trace.current.scope, trace.current.task)
            if episodic:
                wanted.update((source, *context) for source in trace.support_ids)
            if conceptual and trace.kind == "episodic" and trace.concept is not None:
                concepts.setdefault((trace.concept, *context), []).append(key)
        bound: dict[tuple[str, str, str], set[str]] = {}
        hubs: dict[tuple[str, str, str], set[str]] = {}
        for key, trace in self._traces.items():
            if (trace.kind != "episodic" or trace.merged_into is not None
                    or (trace.tier == Tier.COLD and not consolidatable_tier(trace, self.config))
                    or not currently_valid(trace)):
                continue
            for source in trace.support_ids:
                identity = (source, trace.current.scope, trace.current.task)
                if identity in wanted:
                    bound.setdefault(identity, set()).add(key)
            if trace.concept is not None:
                identity = (trace.concept, trace.current.scope, trace.current.task)
                if identity in concepts:
                    hubs.setdefault(identity, set()).add(key)
        episodes = [sorted(members) for _, members in sorted(bound.items()) if len(members) > 1]
        if episodes:
            self.telemetry.add("episodic_binding.episodes", len(episodes))
            self.telemetry.add("episodic_binding.reactivated",
                               len(set().union(*map(set, episodes)) - set(keys)))
        stars = [(center, sorted(members - {center}))
                 for identity, members in sorted(hubs.items())
                 for center in sorted(concepts[identity]) if members - {center}]
        if stars:
            self.telemetry.add("concept_binding.concepts",
                               len({i for i, m in hubs.items() if len(m) > 1}))
            self.telemetry.add("concept_binding.reactivated",
                               len(set().union(*(set(m) for _, m in stars)) - set(keys)))
        return episodes, stars

    def _replay(self) -> tuple[str, ...]:
        if not self.config.mechanisms.replay:
            return ()
        ranked = rank_candidates(self._traces, self.graph, self.operation, self.config, self.feedback)
        self.telemetry.add("replay_candidates", len(ranked))
        selected = [(s, k) for s, k in ranked if s >= self.config.replay_threshold][:self.config.replay_limit]
        for score, key in selected:
            trace = self._traces[key]
            trace.replay_count += 1
            trace.last_replay = self.operation
            trace.replay_score = score
            trace.stability = 1 - 1 / (1 + trace.replay_count)
            self._promote(trace)
        keys = [key for _, key in selected]
        episodes, hubs = self._episodes(keys)
        # Replay of one member reactivates its encoding-time episode as a
        # whole (behavioural tagging): bound members are reactivated too and
        # receive the same reactivation credit, so an item that lost
        # expression to the reader budget can still stabilize offline and be
        # compressed. Credit is reactivation only: recurrence and source
        # support are unchanged, concept hubs across sessions get no credit,
        # and a silent member stays silent (offline reactivation is not
        # expression); an expressed hot member is promoted like any replayed
        # trace because promotion between expressed tiers changes no output.
        ensemble = sorted({k for episode in episodes for k in episode} - set(keys))
        for key in ensemble:
            trace = self._traces[key]
            trace.replay_count += 1
            trace.last_replay = self.operation
            trace.replay_score = replay_score(
                trace, self.operation, len(self.graph.neighbors(key, "conflicts_with")),
                self.config, self.feedback.get(key))
            trace.stability = 1 - 1 / (1 + trace.replay_count)
            if trace.tier == Tier.HOT:
                self._promote(trace)
        if ensemble:
            self.telemetry.add("episodic_binding.credited", len(ensemble))
        self._coactivate(keys, episodes, hubs)
        self.telemetry.add("replay_triggered", len(selected))
        if selected:
            self.telemetry.record(self.operation, "replay", ranked=selected)
        return tuple(keys)

    def replay(self) -> tuple[str, ...]:
        self.operation += 1
        selected = self._replay()
        self.pending_consolidation = self._request_consolidation()
        self._regulate()
        return selected

    def _request_consolidation(self) -> ConsolidationRequest | None:
        request = find_cluster(self._traces, self.graph, self.config, self._incompressible_clusters)
        identity = None if request is None else digest(asdict(request))
        previous = self.pending_consolidation
        if previous is not None and digest(asdict(previous)) != identity:
            # A pending request that is cancelled or replaced ends its
            # lifecycle.  Its identity leaves the idempotency cache so a later
            # reappearance is a new birth with its own receipt and quiet
            # window, matching the delayed-path history invariant.
            self._requested_clusters.discard(digest(asdict(previous)))
        if request is None:
            self.pending_consolidation = None
            self.pending_consolidation_epoch = None
            return None
        if identity not in self._requested_clusters:
            self._requested_clusters.add(identity)
            self.pending_consolidation_epoch = self.maintenance_epoch
            self.telemetry.add("consolidations.requested")
            self.telemetry.record(self.operation, "consolidation_request",
                                  request=asdict(request),
                                  pending_epoch=self.pending_consolidation_epoch)
        elif self.pending_consolidation_epoch is None:
            # Legacy snapshots may contain a request but no epoch.  Treat the
            # restored state as born now; it still needs a fresh maintenance
            # transition before a delayed consumer can use it.
            self.pending_consolidation_epoch = self.maintenance_epoch
        self.pending_consolidation = request
        return request

    def consolidate(self, proposal: ConsolidationProposal | None = None) -> str | None:
        if not self.config.mechanisms.consolidation:
            return None
        self.operation += 1
        request = self.pending_consolidation
        if proposal is None:
            # Revalidate the pending request against the live graph.  A
            # restored request may outlive a replay/edge change; consuming it
            # without this refresh would bypass the late admission gate.
            # The request identity is retained by _request_consolidation, so
            # an unchanged request keeps its birth epoch.
            request = self._request_consolidation()
        elif proposal is not None and request is None and self.config.slow_consolidation:
            # Slow mode never accepts a caller-created proposal without a
            # live request and its independently born maintenance epoch.
            request = self._request_consolidation()
        if proposal is None and request is None:
            return None
        if self.config.slow_consolidation:
            born = self.pending_consolidation_epoch
            if (born is None
                    or self.maintenance_epoch - born < self.config.consolidation_quiet_epochs):
                self.telemetry.add("consolidations.deferred")
                self.telemetry.record(
                    self.operation, "consolidation_deferred",
                    maintenance_epoch=self.maintenance_epoch,
                    pending_epoch=born,
                    required_epochs=self.config.consolidation_quiet_epochs,
                )
                return None
        revising = None
        if proposal is None:
            revising = assimilating_card(request.sources, self._traces)
        else:
            if (request is not None
                    and tuple(ref.key for ref in proposal.sources) != tuple(ref.key for ref in request.sources)):
                self.telemetry.add("consolidations.rejected")
                self.telemetry.record(self.operation, "consolidation_rejected",
                                      key=proposal.key, reason="CONSOLIDATION_REQUEST_MISMATCH")
                return None
            candidate = assimilating_card(proposal.sources, self._traces)
            revising = candidate if candidate is not None and candidate.key == proposal.key else None
        if proposal is None and revising is not None:
            proposal = proposal_for(request, revising.key)
        elif proposal is None:
            card_key = "semantic_" + digest(asdict(request))[:20]
            previous = self._traces.get(card_key)
            if (previous is not None and previous.kind == 'semantic_card'
                    and (previous.stale or previous.hard_forgotten)):
                # Independent source traces can support a fresh abstraction;
                # a retired card identity itself must never be resurrected.
                card_key += '_' + str(self.operation)
            proposal = proposal_for(request, card_key)
        reason = validate_proposal(proposal, self._traces, self.graph, self.config, revising)
        if reason:
            if reason == "CARD_NO_COMPRESSION":
                self._incompressible_clusters.add(canonical_refs(proposal.sources))
                self._consume_pending_request()
            self.telemetry.add("vertical_consolidation_denied")
            self.telemetry.add("consolidations.rejected")
            self.telemetry.record(self.operation, "consolidation_rejected", key=proposal.key, reason=reason)
            return None
        if revising is not None:
            return self._reconsolidate_card(revising, proposal)
        evidence_id = "derived:" + digest(asdict(proposal))
        source_trace = self._traces[proposal.sources[0].key]
        card = Trace(proposal.key, [Stance(self.operation, "assert", proposal.text,
                     proposal.text, evidence_id, scope=source_trace.current.scope,
                     task=source_trace.current.task)],
                     proposal.about, tier=Tier.CONSOLIDATED, kind="semantic_card",
                     merged_from=proposal.sources, support_ids={evidence_id}, frame=proposal.frame,
                     gate_version=VERTICAL_GATE_VERSION)
        self._traces[card.key] = card
        entering = {ref.key: self._traces[ref.key].tier.value for ref in proposal.sources}
        stability = self._entry_stability(proposal.sources)
        for ref in proposal.sources:
            trace = self._traces[ref.key]
            trace.tier, trace.merged_into = Tier.COLD, card.key
        self._graph_update(card)
        if proposal.frame is not None:
            schema = schema_identity(proposal.frame.segments)
            if self.context_prediction.learn_schema(
                    schema, scope=card.current.scope, channel=card.pool,
                    card_id=card.key, prototype=proposal.text,
                    task=card.current.task):
                self.telemetry.add("semantic_schema_observations")
                self.telemetry.record(self.operation, "semantic_schema_observed",
                                      schema=schema, card_key=card.key,
                                      source_keys=[ref.key for ref in proposal.sources])
        self.telemetry.add("vertical_consolidation_allowed")
        self.telemetry.add("consolidations.accepted")
        self._count_consolidated_sources(proposal.sources, entering)
        self.telemetry.record(self.operation, "consolidation_accepted", key=card.key,
                              sources=[asdict(ref) for ref in proposal.sources],
                              entering=stability, gate_version=VERTICAL_GATE_VERSION)
        self._consume_pending_request()
        self._regulate()
        return card.key

    def _consume_pending_request(self) -> None:
        """End the pending request's lifecycle when a card consumes it.

        ``_requested_clusters`` holds only requests that are still waiting, so
        a consumed identity leaves it.  A later reappearance of the same
        cluster (its card went stale or was retired) is then a new birth with
        its own receipt and quiet window (r55 F2).
        """
        if self.pending_consolidation is not None:
            self._requested_clusters.discard(digest(asdict(self.pending_consolidation)))
        self.pending_consolidation = None
        self.pending_consolidation_epoch = None

    def _learn_card_schema(self, card: Trace) -> None:
        if card.frame is None:
            return
        schema = schema_identity(card.frame.segments)
        if self.context_prediction.learn_schema(
                schema, scope=card.current.scope, channel=card.pool,
                card_id=card.key, prototype=card.current.text, task=card.current.task):
            self.telemetry.add("semantic_schema_observations")
            self.telemetry.record(self.operation, "semantic_schema_observed",
                                  schema=schema, card_key=card.key,
                                  source_keys=[ref.key for ref in card.merged_from])

    def _reconsolidate_card(self, card: Trace, proposal: ConsolidationProposal) -> str:
        """Rewrite a reactivated frame card as its next version, keeping history."""
        previous = card.merged_from
        entering = {ref.key: self._traces[ref.key].tier.value for ref in proposal.sources
                    if ref not in previous}
        # The audit record covers exactly the members validate_proposal had to
        # gate: new members, and every member of a card certified under an
        # older or unknown criterion (R4-01).
        certified = set(previous) if card.gate_version >= VERTICAL_GATE_VERSION else set()
        stability = self._entry_stability(ref for ref in proposal.sources if ref not in certified)
        certified_before = card.gate_version
        self._release_card_support(card)
        self._invalidate_dependents(card.key)
        evidence_id = "derived:" + digest(asdict(proposal))
        card.history.append(Stance(self.operation, "assert", proposal.text, proposal.text,
                                   evidence_id, scope=card.current.scope, task=card.current.task))
        card.merged_from, card.frame, card.about = proposal.sources, proposal.frame, proposal.about
        card.support_ids, card.tier = {evidence_id}, Tier.CONSOLIDATED
        # validate_proposal checked every member not certified under the
        # current criterion, so the revised card is certified under it.
        card.gate_version = VERTICAL_GATE_VERSION
        for ref in proposal.sources:
            trace = self._traces[ref.key]
            trace.tier, trace.merged_into = Tier.COLD, card.key
        self._graph_update(card)
        self._learn_card_schema(card)
        self.telemetry.add("consolidations.assimilated")
        self.telemetry.add("vertical_consolidation_allowed")
        self.telemetry.add("consolidations.accepted")
        self._count_consolidated_sources(proposal.sources, entering)
        self.telemetry.record(self.operation, "consolidation_assimilated", key=card.key,
                              added=[asdict(ref) for ref in proposal.sources if ref not in previous],
                              sources=[asdict(ref) for ref in proposal.sources],
                              entering=stability, gate_version=VERTICAL_GATE_VERSION,
                              certified_before=certified_before)
        self._consume_pending_request()
        self._regulate()
        return card.key

    def _regulate(self) -> list[str]:
        demoted = []
        if self.config.mechanisms.soft_forgetting and not fits(self.window_usage()):
            for key in victims(self._traces, self.operation, self.config):
                self._traces[key].tier = Tier.COLD
                # Losing expression to the budget is not a forget decision:
                # victims are expressed, hence never suppressed, and remain so.
                demoted.append(key)
                # A homeostatic demotion is the runtime's automatic soft
                # forgetting event. Keep it distinct from the explicit
                # ``forget(key, mode="soft")`` API, while exposing both under
                # the published forget.soft telemetry family.
                self.telemetry.add("forget.soft")
                self.telemetry.add("forget.soft.homeostatic")
                if fits(self.window_usage()):
                    break
            self.telemetry.add("demotions", len(demoted))
        return demoted

    def window_usage(self) -> dict[str, int]:
        return usage(self._traces, self.config)

    def peek(self, query: Query | str | None = None, *,
             stats: dict[str, int] | None = None) -> list[Evidence]:
        if query is None:
            query = Query()
        elif isinstance(query, str):
            scopes, tasks = self.known_contexts()
            query = interpret_query(query, known_scopes=scopes, known_tasks=tasks).query
        return evidence_set(self._traces, self.graph, query, self.operation, self.config, stats)

    def known_contexts(self) -> tuple[frozenset[str], frozenset[str]]:
        """Scopes and tasks this memory has learned from its own stances."""
        scopes, tasks = set(), set()
        for trace in self._traces.values():
            for stance in trace.history:
                scopes.add(stance.scope)
                tasks.add(stance.task)
        return frozenset(scopes - {"default"}), frozenset(tasks - {"default"})

    def retrieve(self, query: Query | str | None = None) -> list[Evidence]:
        self.operation += 1
        stats: dict[str, int] = {}
        result = self.peek(query, stats=stats)
        if stats.get("recruited"):
            self.telemetry.add("pattern_completion.recruited", stats["recruited"])
            self.telemetry.add("pattern_completion.selected", stats["recruited_selected"])
        if stats.get("window_applied"):
            self.telemetry.add("temporal_window.applied")
            self.telemetry.add("temporal_window.outside", stats["window_outside"])
        if self.config.mechanisms.usage_priority:
            for item in result:
                trace = self._traces[item.key]
                trace.access_count += 1
                trace.last_access = self.operation
            self.telemetry.add("accesses", len(result))
        self._coactivate([item.key for item in result if item.current])
        self.telemetry.add("evidence_set_size", len(result))
        self.telemetry.record(self.operation, "retrieve", keys=[item.key for item in result],
                              words=sum(len(item.text.split()) for item in result))
        self._regulate()
        return result

    def maintenance(self) -> dict[str, Any]:
        """Advance the independent zero-call maintenance/replay epoch.

        The transition only replays and schedules work.  Application facades
        may call :meth:`consolidate` after this step; in the slow path that is
        the first point at which the pending request is eligible.
        """
        self.operation += 1
        self.maintenance_epoch += 1
        replayed = self._replay()
        self.pending_consolidation = self._request_consolidation()
        demoted = self._regulate()
        self.telemetry.record(self.operation, "maintenance", replayed=list(replayed),
                              demoted=demoted, maintenance_epoch=self.maintenance_epoch,
                              pending_epoch=self.pending_consolidation_epoch)
        return {"replayed": replayed, "demoted": tuple(demoted),
                "consolidation": self.pending_consolidation}

    def read(self, query: Query | str | None = None, *, touch: bool = True) -> str:
        items = self.retrieve(query) if touch else self.peek(query)
        if self.config.mechanisms.temporal_order:
            items = chronological(items, self._traces)
        return "\n".join(f"[{item.key}] {item.text}" for item in items)

    def forget(self, key: str, *, mode: str = "soft") -> None:
        if key not in self._traces or mode not in ("soft", "hard"):
            raise MemoryInvariantError("FORGET_ARGUMENT_INVALID")
        if mode == "soft" and not self.config.mechanisms.soft_forgetting:
            return
        self.operation += 1
        trace = self._traces[key]
        trace.tier = Tier.COLD
        trace.suppressed = True
        if mode == "hard":
            trace.hard_forgotten = True
            # Forgetting a derived card is a semantic lifecycle boundary, not
            # merely a visibility change. Permit a fresh request birth from
            # the now-independent source traces.
            if trace.kind == "semantic_card":
                self._drop_requested_cluster(trace)
            self._release_card_support(trace)
            self._invalidate_dependents(key)
            self._graph_update(trace)
        elif trace.kind == "semantic_card" and not trace.stale:
            # Forgetting a gist suppresses the knowledge it carries, not the
            # learned schema: the card stays live but silent, and each
            # represented source is suppressed with it, so a later release of
            # the card (a sibling changes) cannot re-express them. Only a new
            # encounter lifts a source; a strong direct cue can still recover
            # it in a read (cue recovery).
            for ref in trace.merged_from:
                source = self._traces[ref.key]
                if source.merged_into == trace.key:
                    source.suppressed = True
                    source.tier = Tier.COLD
        else:
            # A suppressed source must not stay expressed through its card.
            self._retire_carrier(trace, "soft_forget")
        self._refresh_conflicts()
        self.pending_consolidation = self._request_consolidation()
        self._regulate()
        self.telemetry.add("forget." + mode)
        if mode == "soft":
            self.telemetry.add("forget.soft.explicit")
        self.telemetry.record(self.operation, "forget", key=key, mode=mode)

    def predict(self, prediction_id: str, key: str, expected: Any) -> None:
        if not prediction_id or prediction_id in self.predictions or key not in self._traces:
            raise MemoryInvariantError("PREDICTION_ID_OR_KEY_INVALID")
        # JSON validation/copy prevents caller mutation and NaN pseudo-feedback.
        expected_copy = json.loads(json.dumps(expected, allow_nan=False))
        self.operation += 1
        self.predictions[prediction_id] = {"key": key, "expected": expected_copy,
                                           "operation": self.operation, "observed": False}

    def observe(self, prediction_id: str, observed: Any) -> float:
        prediction = self.predictions.get(prediction_id)
        if prediction is None or prediction["observed"]:
            raise MemoryInvariantError("PREDICTION_UNKNOWN_OR_ALREADY_OBSERVED")
        value = json.loads(json.dumps(observed, allow_nan=False))
        self.operation += 1
        residual = float(digest(value) != digest(prediction["expected"]))
        prediction.update(observed=True, value=value, prediction_error=residual)
        self.feedback[prediction["key"]] = residual
        self.telemetry.add("prediction_observations")
        self.telemetry.record(self.operation, "prediction_observed", prediction_id=prediction_id,
                              prediction_error=residual)
        return residual

    def snapshot(self) -> dict[str, Any]:
        rows = {}
        for key, trace in sorted(self._traces.items()):
            row = asdict(trace)
            row["tier"] = trace.tier.value
            row["support_ids"] = sorted(trace.support_ids)
            rows[key] = row
        value = {"schema": SCHEMA, "config": asdict(self.config), "operation": self.operation,
                 "traces": rows, "metadata_keys": sorted(self.metadata_keys),
                 "source_hashes": self.source_hashes, "requests": {k: asdict(v) for k, v in self._requests.items()},
                 "predictions": self.predictions, "feedback": self.feedback,
                 "edges": [[*e, w] for e, w in sorted(self.graph.edges.items())],
                 "telemetry": self.telemetry.summary(), "events": self.telemetry.events,
                 # Only a waiting request belongs in the idempotency cache
                 # (_consume_pending_request); restore applies the same rule,
                 # so a hook that cleared only the request object still
                 # round-trips byte-equal.
                 "requested_clusters": sorted(
                     self._requested_clusters
                     & ({digest(asdict(self.pending_consolidation))}
                        if self.pending_consolidation is not None else set())),
                 "maintenance_epoch": self.maintenance_epoch,
                 # A caller that has consumed/cleared a pending request must
                 # not leave a detached birth epoch in the durable snapshot.
                 # The live slow path clears both fields together; normalizing
                 # here also keeps snapshots faithful for controlled test or
                 # adapter hooks that clear only the request object.
                 "pending_consolidation_epoch": (
                     self.pending_consolidation_epoch
                     if self.pending_consolidation is not None else None),
                  "maintenance_history": [
                      event.get("maintenance_epoch") for event in self.telemetry.events
                      if event.get("action") == "maintenance"],
                  "pending_birth_history": [
                      {"request_digest": digest(event.get("request")),
                       "pending_epoch": event.get("pending_epoch")}
                      for event in self.telemetry.events
                      if event.get("action") == "consolidation_request"],
                 "context_prediction": self.context_prediction.snapshot(),
                 "pending_consolidation": (None if self.pending_consolidation is None
                                           else asdict(self.pending_consolidation)),
                 "incompressible_clusters": [[asdict(ref) for ref in refs] for refs in sorted(
                     self._incompressible_clusters, key=lambda refs: tuple((r.key, r.revision) for r in refs))]}
        # Force a detached JSON-domain value, including tuple/list normalization.
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

    @classmethod
    def restore(cls, value: dict[str, Any]) -> UAIMemoryRuntimeV2:
        try:
            if value["schema"] != SCHEMA:
                raise ValueError("schema")
            for name, kind in (("traces", dict), ("requests", dict), ("events", list),
                               ("edges", list), ("feedback", dict), ("predictions", dict)):
                if not isinstance(value[name], kind):
                    raise TypeError(name)
            conf = legacy_config(value["config"])
            conf["mechanisms"] = Mechanisms(**conf["mechanisms"])
            runtime = cls(RuntimeConfig(**conf))
            runtime.operation = value["operation"]
            runtime.maintenance_epoch = value.get("maintenance_epoch", 0)
            runtime.pending_consolidation_epoch = value.get("pending_consolidation_epoch")
            if (type(runtime.maintenance_epoch) is not int or runtime.maintenance_epoch < 0
                    or (runtime.pending_consolidation_epoch is not None
                        and (type(runtime.pending_consolidation_epoch) is not int
                             or runtime.pending_consolidation_epoch < 0
                             or runtime.pending_consolidation_epoch > runtime.maintenance_epoch))):
                raise ValueError("maintenance_epoch")
            # Snapshots written before suppression was persisted carry no
            # "suppressed" field. Their explicit forgets are still recorded
            # as events. Suppression is re-derived with the live rule: the
            # latest forget of the trace itself, or of a card that carried
            # its current revision, applies until a stance after it would
            # have lifted it live (_lifted_after). An upgrade therefore cannot
            # make soft-forgotten content expressible or consolidatable again.
            legacy_forgets: dict[str, int] = {}
            soft_card_forgets: dict[str, int] = {}
            retired_cards: dict[str, int] = {}
            for event in value["events"]:
                if event.get("action") == "card_retired":
                    retired_cards[event["key"]] = min(event["operation"],
                                                      retired_cards.get(event["key"], event["operation"]))
                if event.get("action") == "forget":
                    legacy_forgets[event["key"]] = max(event["operation"],
                                                       legacy_forgets.get(event["key"], 0))
                    # Only a soft forget of a card suppresses its members; a
                    # hard forget releases them (live forget()).
                    if event.get("mode") == "soft":
                        soft_card_forgets[event["key"]] = max(
                            event["operation"], soft_card_forgets.get(event["key"], 0))
            inferred: set[str] = set()
            for key, raw in value["traces"].items():
                row = dict(raw)
                row["history"] = [Stance(**s) for s in row["history"]]
                row["merged_from"] = tuple(SourceRef(**ref) for ref in row["merged_from"])
                if row.get("frame") is not None:
                    row["frame"] = SemanticFrame(tuple(row["frame"]["segments"]),
                                                  tuple(tuple(r) for r in row["frame"]["rows"]))
                row["tier"] = Tier(row["tier"])
                if "suppressed" not in row:
                    inferred.add(key)
                    row["suppressed"] = False
                row["support_ids"] = set(row["support_ids"])
                row["about"], row["depends_on"] = tuple(row["about"]), tuple(row["depends_on"])
                if key != row["key"]:
                    raise ValueError("key")
                runtime._traces[key] = Trace(**row)
            # A soft forget of a live card suppressed every member the card
            # carried at that operation (F11), including members it released
            # later (R4 F5); a forget of an already retired card touched none
            # (live forget(), Claude r5 F1). Carriage is judged at the forget:
            # the card must have been created and not yet retired, and the
            # member must then have been at the revision the card carried. A
            # retirement is read from its event or, for reconsolidation, which
            # records none, from the persisted stale flag together with a
            # member that had already left the carried revision at the
            # forget: that change retired the card. A later revision of the
            # member is judged afterwards by the live lifting rule, so a
            # negation keeps the suppression (R5-05). A member now carried by
            # another live card cannot be suppressed (SUPPRESSED_SOURCE_
            # CARRIED), so it was released live. The card's own merged_from is
            # the only membership record a legacy snapshot has; any remaining
            # ambiguity errs toward silence, never toward re-expression.
            reconsolidates = runtime.config.mechanisms.reconsolidation
            carried_forgets: dict[str, int] = {}
            for key, operation in soft_card_forgets.items():
                card = runtime._traces.get(key)
                if (card is None or card.kind != "semantic_card"
                        or operation < card.history[0].operation
                        or retired_cards.get(key, operation + 1) <= operation):
                    continue
                members = [(ref, runtime._traces.get(ref.key)) for ref in card.merged_from]
                if card.stale and reconsolidates and any(
                        member is not None and _revision_at(member.history, operation) != ref.revision
                        for ref, member in members):
                    continue
                for ref, member in members:
                    if member is None or _revision_at(member.history, operation) != ref.revision:
                        continue
                    carrier = runtime._traces.get(member.merged_into) if member.merged_into else None
                    if (carrier is not None and carrier.key != key and carrier.kind == "semantic_card"
                            and not carrier.stale):
                        continue
                    carried_forgets[ref.key] = max(operation, carried_forgets.get(ref.key, 0))
            for key in inferred:
                trace = runtime._traces[key]
                operation = max(legacy_forgets.get(key, 0), carried_forgets.get(key, 0))
                if operation and not _lifted_after(trace.history, operation):
                    trace.suppressed, trace.tier = True, Tier.COLD
            for trace in runtime._traces.values():
                runtime.graph.update(trace, runtime._traces, runtime.config.graph_neighbors)
            runtime.graph.set_edges(value["edges"])
            runtime.metadata_keys = set(value["metadata_keys"])
            runtime.source_hashes = dict(value["source_hashes"])
            if "context_prediction" in value:
                runtime.context_prediction = ContextPredictor.restore(value["context_prediction"], runtime.source_hashes)
            for key, raw in value["requests"].items():
                row = dict(raw)
                for name in ("applied", "negated", "dropped", "pruned"):
                    row[name] = tuple(row[name])
                runtime._requests[key] = BatchResult(**row)
            runtime.predictions = copy.deepcopy(value["predictions"])
            runtime.feedback = dict(value["feedback"])
            runtime.telemetry.counts.update(value["telemetry"])
            runtime.telemetry.events = copy.deepcopy(value["events"])
            maintenance_history = [event.get("maintenance_epoch")
                                  for event in runtime.telemetry.events
                                  if event.get("action") == "maintenance"]
            request_history = [
                {"request_digest": digest(event.get("request")),
                 "pending_epoch": event.get("pending_epoch")}
                for event in runtime.telemetry.events
                if event.get("action") == "consolidation_request"]
            if "maintenance_history" in value:
                if value["maintenance_history"] != maintenance_history:
                    raise ValueError("maintenance_history")
            elif runtime.maintenance_epoch != 0:
                raise ValueError("maintenance_history")
            if "pending_birth_history" in value:
                if value["pending_birth_history"] != request_history:
                    raise ValueError("pending_birth_history")
            elif request_history:
                raise ValueError("pending_birth_history")
            # Only a pending request is waiting (_consume_pending_request), so
            # the restored cache holds at most the restored request's
            # identity.  A snapshot written before consumption left the cache
            # (r55) may still list consumed identities; keeping them would
            # suppress the receipt of a later rebirth of the same cluster.
            persisted_requests = set(value["requested_clusters"])
            runtime._incompressible_clusters = {
                tuple(SourceRef(**ref) for ref in refs) for refs in value.get("incompressible_clusters", [])}
            if "pending_consolidation" in value:
                pending = value["pending_consolidation"]
                runtime.pending_consolidation = None if pending is None else ConsolidationRequest(
                    tuple(SourceRef(**ref) for ref in pending["sources"]),
                    tuple(pending["texts"]), tuple(pending["about"]))
            else:  # legacy snapshot: the request is a pure function of the restored state
                runtime.pending_consolidation = find_cluster(runtime._traces, runtime.graph,
                                                             runtime.config,
                                                             runtime._incompressible_clusters)
            # Preserve the persisted birth epoch even when a caller has
            # cleared the in-memory request after consumption.  The epoch is
            # part of the durable temporal state; erasing it on restore makes
            # snapshot round-trips observably non-identical and can collapse
            # the delayed eligibility boundary.  Only legacy snapshots that
            # lack the field need a derived value for a live request.
            if (runtime.pending_consolidation is not None
                    and runtime.pending_consolidation_epoch is None):
                runtime.pending_consolidation_epoch = runtime.maintenance_epoch
            pending = runtime.pending_consolidation
            pending_identity = None if pending is None else digest(asdict(pending))
            runtime._requested_clusters = (
                {pending_identity} if pending_identity in persisted_requests else set())
            if pending is not None:
                runtime._validate_pending_consolidation()
            # Validate the final pending request after it has been decoded;
            # checking only before parsing allowed malformed delayed work to
            # survive a snapshot round-trip.
            runtime.check_invariants()
            return runtime
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
            raise MemoryInvariantError("SNAPSHOT_INVALID") from error

    def _validate_pending_consolidation(self) -> None:
        """Validate the durable request that bridges encoding and maintenance.

        A request is derived from live traces, but it is persisted so the
        delayed path can survive close/restore.  Validate every persisted
        field against the referenced revision before allowing maintenance to
        consume it; source revisions alone do not protect altered text or
        cluster shape.
        """
        pending = self.pending_consolidation
        if pending is None:
            return
        sources = pending.sources
        if (not isinstance(sources, tuple)
                or not self.config.mechanisms.consolidation
                or not sources or len(sources) > self.config.cluster_limit
                or len(sources) != len(pending.texts)
                or len({(ref.key, ref.revision) for ref in sources}) != len(sources)):
            raise MemoryInvariantError("PENDING_CONSOLIDATION_INVALID")
        if (not isinstance(pending.about, tuple)
                or any(type(value) is not str or not value for value in pending.about)):
            raise MemoryInvariantError("PENDING_CONSOLIDATION_ABOUT_INVALID")
        for ref, text in zip(sources, pending.texts):
            trace = self._traces.get(ref.key)
            if (trace is None or type(ref.revision) is not int or ref.revision < 1
                    or trace.revision != ref.revision
                    or type(text) is not str or not text
                    or trace.current.text != text):
                raise MemoryInvariantError("PENDING_CONSOLIDATION_SOURCE_INVALID")
            if trace.kind != "episodic":
                raise MemoryInvariantError("PENDING_CONSOLIDATION_KIND_INVALID")
        contexts = {(self._traces[ref.key].current.scope,
                     self._traces[ref.key].current.task) for ref in sources}
        if len(contexts) != 1:
            raise MemoryInvariantError("PENDING_CONSOLIDATION_CONTEXT_INVALID")

    def _validate_maintenance_history(self) -> None:
        """Bind delayed-consolidation epochs to their telemetry history.

        The epoch is a temporal gate, not an ordinary counter.  A snapshot that
        increases it without the corresponding independent maintenance event
        could consume a newly born request immediately after restore.  Every
        current-format maintenance/request event therefore carries the epoch
        that created it, and the live state must agree with that history.
        """
        maintenance_events = [event for event in self.telemetry.events
                               if event.get("action") == "maintenance"]
        request_events = [event for event in self.telemetry.events
                          if event.get("action") == "consolidation_request"]
        # A maintenance receipt is part of the runtime operation timeline.
        # Without this bound, a caller could append a syntactically valid
        # maintenance row to a snapshot and raise ``maintenance_epoch``
        # without having advanced the runtime at all.  Application journals
        # provide the stronger external chain; the low-level snapshot still
        # must reject an internally impossible operation number.
        for event in self.telemetry.events:
            if not isinstance(event, dict):
                raise MemoryInvariantError("TELEMETRY_EVENT_INVALID")
            operation = event.get("operation")
            if (type(operation) is not int or operation < 0
                    or operation > self.operation):
                raise MemoryInvariantError("TELEMETRY_OPERATION_INVALID")
        if maintenance_events and any("maintenance_epoch" not in event
                                      for event in maintenance_events):
            raise MemoryInvariantError("MAINTENANCE_HISTORY_MISSING")
        epochs = [event.get("maintenance_epoch") for event in maintenance_events]
        if epochs and epochs != list(range(1, len(epochs) + 1)):
            raise MemoryInvariantError("MAINTENANCE_HISTORY_NOT_SEQUENTIAL")
        if self.maintenance_epoch != len(maintenance_events):
            # A legacy epoch of zero with no maintenance events is harmless;
            # every nonzero epoch must have an auditable transition.
            raise MemoryInvariantError("MAINTENANCE_EPOCH_HISTORY_MISMATCH")
        request_births = {}
        for event in request_events:
            if "request" not in event or type(event.get("pending_epoch")) is not int:
                raise MemoryInvariantError("PENDING_BIRTH_HISTORY_MISSING")
            request_births[digest(event["request"])] = event["pending_epoch"]
        if self.pending_consolidation is None:
            if self.pending_consolidation_epoch is not None:
                raise MemoryInvariantError("PENDING_BIRTH_WITHOUT_REQUEST")
            return
        identity = digest(asdict(self.pending_consolidation))
        if (self.pending_consolidation_epoch is None
                or request_births.get(identity) != self.pending_consolidation_epoch):
            # Non-slow adapters may supply a deterministic request hook for a
            # serialization/consumer check without emitting the internal
            # request telemetry.  There is no delayed eligibility boundary in
            # that mode, so absence of a birth receipt is harmless; slow mode
            # remains fail-closed and requires the journaled epoch.
            if (not self.config.slow_consolidation
                    and self.pending_consolidation_epoch is None
                    and not request_births):
                return
            raise MemoryInvariantError("PENDING_BIRTH_HISTORY_MISMATCH")

    def check_invariants(self) -> None:
        if type(self.operation) is not int or self.operation < 0:
            raise MemoryInvariantError("OPERATION_INVALID")
        if (type(self.maintenance_epoch) is not int or self.maintenance_epoch < 0
                or (self.pending_consolidation_epoch is not None
                    and (type(self.pending_consolidation_epoch) is not int
                         or self.pending_consolidation_epoch < 0
                         or self.pending_consolidation_epoch > self.maintenance_epoch))):
            raise MemoryInvariantError("MAINTENANCE_EPOCH_INVALID")
        self._validate_pending_consolidation()
        self._validate_maintenance_history()
        for key, trace in self._traces.items():
            if not KEY.fullmatch(key) or not trace.history or trace.kind not in ("episodic", "semantic_card"):
                raise MemoryInvariantError("TRACE_INVALID")
            if (any(type(n) is not int or n < 0 for n in (trace.access_count, trace.replay_count))
                    or any(type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1
                           for n in (trace.replay_score, trace.stability, trace.novelty,
                                     trace.processing_gain, trace.context_error))
                    or any(type(n) is not bool for n in (trace.stale, trace.hard_forgotten, trace.exclusive,
                                                         trace.suppressed))
                    or type(trace.gate_version) is not int
                    or not 0 <= trace.gate_version <= VERTICAL_GATE_VERSION
                    or (trace.kind != "semantic_card" and trace.gate_version != 0)):
                raise MemoryInvariantError("TRACE_COUNTER_OR_SIGNAL_INVALID")
            if len({(s.scope, s.task) for s in trace.history}) != 1:
                raise MemoryInvariantError("TRACE_CONTEXT_MIXED")
            last = 0
            for stance in trace.history:
                if (type(stance.operation) is not int or not max(1, last) <= stance.operation <= self.operation
                        or stance.action not in ("assert", "negate") or type(stance.text) is not str
                        or not stance.text or type(stance.scope) is not str or not stance.scope
                        or type(stance.task) is not str or not stance.task
                        or not stance.source_id
                        or (stance.observed is not None and not valid_day(stance.observed))):
                    raise MemoryInvariantError("STANCE_HISTORY_INVALID")
                last = stance.operation
            for ref in trace.merged_from:
                if (ref.key not in self._traces or type(ref.revision) is not int
                        or not 1 <= ref.revision <= self._traces[ref.key].revision):
                    raise MemoryInvariantError("CARD_PROVENANCE_INVALID")
            if any(dep not in self._traces for dep in trace.depends_on):
                raise MemoryInvariantError("DEPENDENCY_UNKNOWN")
            if trace.merged_into is not None and trace.merged_into not in self._traces:
                raise MemoryInvariantError("MERGED_INTO_UNKNOWN")
            if trace.kind == "semantic_card" and not trace.stale and self.config.mechanisms.reconsolidation:
                for ref in trace.merged_from:
                    source = self._traces[ref.key]
                    if ref.revision != source.revision or not currently_valid(source):
                        raise MemoryInvariantError("LIVE_CARD_SOURCE_INVALID")
            if any(type(source_id) is not str or not source_id for source_id in trace.support_ids):
                raise MemoryInvariantError("SUPPORT_ID_INVALID")
            if trace.kind == "episodic" and any(sid not in self.source_hashes for sid in trace.support_ids):
                raise MemoryInvariantError("SUPPORT_SOURCE_UNKNOWN")
            if trace.kind == "episodic":
                # Support is provenance of the current assertion: exactly the
                # sources whose stances restated it since its last change. A
                # known but unrelated source cannot inflate recurrence.
                expected = set()
                if trace.current.action == "assert":
                    identity = (trace.current.action, trace.current.text,
                                trace.current.scope, trace.current.task)
                    for stance in reversed(trace.history):
                        if (stance.action, stance.text, stance.scope, stance.task) != identity:
                            break
                        expected.add(stance.source_id)
                if trace.support_ids != expected:
                    raise MemoryInvariantError("SUPPORT_PROVENANCE_INVALID")
            if trace.suppressed and trace.tier != Tier.COLD:
                raise MemoryInvariantError("SUPPRESSED_TRACE_EXPRESSIBLE")
            if trace.suppressed and trace.merged_into is not None:
                carrier = self._traces.get(trace.merged_into)
                if carrier is not None and not carrier.stale and not carrier.suppressed:
                    raise MemoryInvariantError("SUPPRESSED_SOURCE_CARRIED")
            if (trace.kind == "semantic_card" and not trace.stale and currently_valid(trace)
                    and any(self.graph.neighbors(ref.key, "conflicts_with")
                            for ref in trace.merged_from
                            if self._traces[ref.key].merged_into == key)):
                raise MemoryInvariantError("CARD_CONFLICT_UNRESOLVED")
            if trace.frame is not None:
                if trace.kind != "semantic_card":
                    raise MemoryInvariantError("FRAME_NOT_SEMANTIC")
                covered = []
                for ref in trace.merged_from:
                    revision, identity, statement = 0, None, None
                    for stance in self._traces[ref.key].history:
                        current = (stance.action, stance.text, stance.scope, stance.task)
                        if current != identity:
                            revision += 1
                            identity = current
                        if revision == ref.revision:
                            statement = stance.text
                    covered.append(" ".join(statement.split()) if statement is not None else None)
                if list(map(lambda s: " ".join(s.split()), frame_claims(trace.frame))) != covered:
                    raise MemoryInvariantError("FRAME_SOURCE_COVERAGE_INVALID")
                if trace.current.text not in (factored_text(tuple(covered)), frame_text(trace.frame)):
                    raise MemoryInvariantError("FRAME_CARD_TEXT_INVALID")
            stack, seen = list(trace.depends_on), set()
            while stack:
                other = stack.pop()
                if other == key:
                    raise MemoryInvariantError("DEPENDENCY_CYCLE")
                if other not in seen:
                    seen.add(other)
                    stack.extend(self._traces[other].depends_on)
        for schema, row in self.context_prediction.schemas.items():
            # A cache is not an independent knowledge source. Bind every
            # supported prototype to a live semantic card in this runtime.
            expected = ContextPredictor()
            if not row['card_ids']:
                raise MemoryInvariantError('SCHEMA_SUPPORT_EMPTY')
            for card_key in row['card_ids']:
                card = self._traces.get(card_key)
                if (card is None or card.kind != 'semantic_card' or card.frame is None
                        or not currently_valid(card)):
                    raise MemoryInvariantError('SCHEMA_CARD_SUPPORT_INVALID')
                expected.learn_schema(schema_identity(card.frame.segments),
                                      scope=card.current.scope, task=card.current.task,
                                      channel=card.pool, card_id=card.key,
                                      prototype=card.current.text)
            if (set(expected.schemas) != {schema}
                    or any(row[field] != expected.schemas[schema][field]
                           for field in ('scope', 'task', 'channel', 'card_ids', 'prototypes'))
                    or row['prototype'] not in row['prototypes']):
                raise MemoryInvariantError('SCHEMA_CARD_BINDING_INVALID')
        for (_, left, right), weight in self.graph.edges.items():
            if (left not in self._traces or right not in self._traces
                    or type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0):
                raise MemoryInvariantError("GRAPH_EDGE_INVALID")
        structural = {(kind, left, right) for kind, left, right in self.graph.edges
                      if kind in ("depends_on", "merged_from")}
        required = {("depends_on", t.key, d) for t in self._traces.values() for d in t.depends_on}
        required |= {("merged_from", t.key, ref.key) for t in self._traces.values() for ref in t.merged_from}
        if structural != required:
            raise MemoryInvariantError("GRAPH_STRUCTURE_MISMATCH")
        if any(type(s) is not str or not s or type(h) is not str or len(h) != 64
               or any(c not in "0123456789abcdef" for c in h) for s, h in self.source_hashes.items()):
            raise MemoryInvariantError("SOURCE_HASH_INVALID")
        if any(type(n) is not int or n < 0 for n in self.telemetry.counts.values()):
            raise MemoryInvariantError("TELEMETRY_COUNTER_INVALID")
        if any(type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1
               for n in self.feedback.values()):
            raise MemoryInvariantError("FEEDBACK_INVALID")
        for refs in self._incompressible_clusters:
            if (not self.config.cluster_size <= len(refs) <= self.config.cluster_limit
                    or len({ref.key for ref in refs}) != len(refs)
                    or any(ref.key not in self._traces or type(ref.revision) is not int
                           or not 1 <= ref.revision <= self._traces[ref.key].revision for ref in refs)):
                raise MemoryInvariantError("INCOMPRESSIBLE_CLUSTER_INVALID")
        if self.config.mechanisms.soft_forgetting and not fits(self.window_usage()):
            raise MemoryInvariantError("EXPRESSION_WINDOW_OVERRUN")


def _revision_at(history: list[Stance], operation: int) -> int:
    """Content revision (Trace.revision) of a history as of *operation*."""
    last, count = None, 0
    for stance in history:
        if stance.operation > operation:
            break
        identity = (stance.action, stance.text, stance.scope, stance.task)
        if identity != last:
            count += 1
            last = identity
    return count


def _lifted_after(history: list[Stance], operation: int) -> bool:
    """Whether a stance after a forget at *operation* lifted it live.

    The live rule (encode): a changed assertion, or the same assertion from a
    source outside the support set of the current assertion, lifts the
    suppression. A negation does not lift it, and a same-source restatement
    is no new encounter.
    """
    before = [stance for stance in history if stance.operation <= operation]
    current = before[-1] if before else None
    support: set[str] = set()
    if current is not None and current.action == "assert":
        for stance in reversed(before):
            if stance.action != "assert" or stance.text != current.text:
                break
            support.add(stance.source_id)
    for stance in history:
        if stance.operation <= operation:
            continue
        if stance.action == "negate":
            current, support = stance, set()
            continue
        if (current is None or current.action != "assert" or stance.text != current.text
                or stance.source_id not in support):
            return True
    return False
