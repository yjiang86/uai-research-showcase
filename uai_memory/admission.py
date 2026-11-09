"""Item-by-item source-bound admission, independent of available capacity."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any
from .conflict import (capitalised_words, named_mentions, other_referent, shares_frame,
                       statement_names, subject_frame)
from .stance import currently_valid
from .types import Trace


KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,95}\Z")
FIELDS = {"key", "text", "quote", "about", "when", "depends_on", "scope", "task",
          "entity", "attribute", "exclusive"}
META_HEAD = re.compile(r"^\W*(session|sessions|date|dated|day|time|timestamp|conversation|"
                       r"conversations|dialog|dialogue|chat|meeting|talk)\b", re.I)
META_FILLER = re.compile(
    r"\b(session|sessions|date|dated|day|days|time|timestamp|conversation|conversations|dialog|"
    r"dialogue|chat|meeting|talk|on|at|of|in|the|a|an|is|was|were|held|took|take|place|occurred|"
    r"happened|number|no|id|user|assistant|with|and|this|that|it|for|from|to|by|our|their|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|"
    r"may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|"
    r"sep|sept|oct|nov|dec|am|pm|utc|today|yesterday|tomorrow|week|month|year|morning|"
    r"afternoon|evening|night)\b", re.I)
PUNCTUATION = dict(zip("‘’‚‛′“”„‟″‐‑‒–—―−\u00a0\u2007\u202f…",
                       ["'"] * 5 + ['"'] * 5 + ["-"] * 7 + [" "] * 3 + ["..."]))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join("".join(PUNCTUATION.get(ch, ch) for ch in text).replace("|", " ").split())


def supported(quote: Any, source: str) -> bool:
    return type(quote) is str and bool(normalize(quote)) and normalize(quote) in normalize(source)


def metadata_only(text: str) -> bool:
    if META_HEAD.match(text) is None:
        return False
    rest = re.sub(r"[\d:/.,#()-]+", " ", META_FILLER.sub(" ", text))
    return len(re.findall(r"[^\W\d_]{2,}", rest)) < 2


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


@dataclass(frozen=True)
class Admission:
    upserts: tuple[dict[str, Any], ...] = ()
    negates: tuple[dict[str, Any], ...] = ()
    dropped: tuple[dict[str, Any], ...] = ()
    pruned: tuple[dict[str, Any], ...] = ()
    metadata_keys: frozenset[str] = frozenset()
    whole_error: str | None = None
    # Writer-proposed capacity evictions are never executed: capacity belongs
    # to the core's usage/replay-driven regulation. They are counted, not lost.
    ignored_evictions: int = 0


def parse_writer_payload(raw: str) -> Any:
    """Decode a writer response using the admission parser's exact semantics.

    Writer responses may be plain JSON or a single fenced JSON block.  The
    fence is transport syntax and must be removed before decoding; duplicate
    object keys remain rejected in both forms.  Callers that need the full
    admission decision should continue to use :func:`prepare`.

    A payload nested deeper than the decoder can follow is invalid JSON for
    the writer contract: it is reported as INVALID_WRITER_JSON, so a caller
    rejects it on its ordinary path (r80 Pro F1).
    """
    clean = raw.strip()
    if clean.startswith("```") and clean.endswith("```"):
        try:
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        except IndexError as error:
            raise ValueError("INVALID_WRITER_JSON") from error
    try:
        return json.loads(clean, object_pairs_hook=unique_object)
    except RecursionError as error:
        raise ValueError("INVALID_WRITER_JSON") from error


def known_entities(traces: dict[str, Trace], scope: str | None = None,
                   task: str | None = None) -> frozenset[str]:
    """Names of more than one stored episodic statement in one context.

    Each name counts once per statement (conflict.statement_names).

    Hard-forgotten traces no longer exist for the memory, and statements of
    another scope or task are not the context the withdrawal is judged in.
    """
    counts = Counter(name for t in traces.values()
                     if t.kind != "semantic_card" and not t.hard_forgotten
                     and (scope is None or t.current.scope == scope)
                     and (task is None or t.current.task == task)
                     for name in statement_names(t.current.text, t.entity))
    return frozenset(name for name, n in counts.items() if n > 1)


def subject_frames(traces: dict[str, Trace], scope: str | None = None,
                   task: str | None = None) -> dict[str, frozenset[str]]:
    """For each subject of a stored episodic statement in one context, the
    words that follow it there (conflict.subject_frame); the same statements
    as known_entities."""
    frames: dict[str, set[str]] = {}
    for t in traces.values():
        if (t.kind != "semantic_card" and not t.hard_forgotten
                and (scope is None or t.current.scope == scope)
                and (task is None or t.current.task == task)):
            frame = subject_frame(t.current.text)
            if frame:
                frames.setdefault(frame[0], set()).add(frame[1])
    return {subject: frozenset(words) for subject, words in frames.items()}


def named_entities(traces: dict[str, Trace], scope: str | None = None,
                   task: str | None = None) -> frozenset[str]:
    """Words the same statements attest as names: written as a name where
    position cannot capitalise them (conflict.named_mentions), or declared by
    the writer for a trace (its about-metadata, its slot entity, or the first
    segment of its dotted key namespace: ``serava.scarf`` -> serava) and
    spelled with a capital in that trace's own text
    (conflict.capitalised_words).  A declaration alone is a storage label:
    key namespaces such as ``pref.`` and slot items such as "apple" are
    not names.  Repetition as a statement's first word is not evidence."""
    names: set[str] = set()
    for t in traces.values():
        if (t.kind != "semantic_card" and not t.hard_forgotten
                and (scope is None or t.current.scope == scope)
                and (task is None or t.current.task == task)):
            names |= named_mentions(t.current.text)
            declared: set[str] = set()
            for value in (*t.about, t.entity or ""):
                declared.update(word.casefold() for word in str(value).split())
            head, dot, _ = t.key.rpartition(":")[2].partition(".")
            if dot and head:
                declared.add(re.split(r"[_/-]", head)[0].casefold())
            names |= declared & capitalised_words(t.current.text)
    return frozenset(names)


def prepare(source: str, raw: str, traces: dict[str, Trace],
            known_metadata: set[str], withdrawal_support: bool = False) -> Admission:
    try:
        proposal = parse_writer_payload(raw)
    except (ValueError, TypeError, IndexError) as error:
        reason = "DUPLICATE_JSON_KEY" if str(error) == "DUPLICATE_JSON_KEY" else "INVALID_WRITER_JSON"
        return Admission(whole_error=reason)
    if type(proposal) is not dict or not {"upsert", "negate", "remove"} & set(proposal):
        return Admission(whole_error="WRITER_NOT_OBJECT")
    if any(type(proposal.get(name, [])) is not list for name in ("upsert", "negate", "remove", "evict")):
        return Admission(whole_error="WRITER_ACTIONS_NOT_ARRAYS")
    upserts, negates, dropped, pruned = {}, {}, [], []
    metadata = set(known_metadata)
    seen = set()

    def drop(action: str, key: Any, reason: str) -> None:
        dropped.append({"action": action, "key": key if type(key) is str else None, "reason": reason})

    for row in proposal.get("upsert", []):
        key = row.get("key") if type(row) is dict else None
        reason = None
        if type(row) is not dict or not {"key", "text", "quote"} <= set(row) or not set(row) <= FIELDS:
            reason = "ITEM_FIELDS_INVALID"
        elif type(key) is not str or not KEY.fullmatch(key):
            reason = "ITEM_KEY_INVALID"
        elif key in seen:
            reason = "ITEM_DUPLICATE_KEY"
        elif type(row["text"]) is not str or not row["text"].strip():
            reason = "ITEM_TEXT_INVALID"
        elif not supported(row["quote"], source):
            reason = "ITEM_QUOTE_NOT_FOUND"
        elif metadata_only(row["text"]):
            reason = "ITEM_METADATA_ONLY"
            metadata.add(key)
        elif (type(row.get("about", [])) is not list or len(row.get("about", [])) > 3
              or any(type(x) is not str or not x.strip() for x in row.get("about", []))):
            reason = "ITEM_ABOUT_INVALID"
        elif (row.get("when") is not None and
              (type(row["when"]) is not str or not row["when"].strip())):
            reason = "ITEM_WHEN_INVALID"
        elif type(row.get("scope", "default")) is not str or not row.get("scope", "default"):
            reason = "ITEM_SCOPE_INVALID"
        elif type(row.get("task", "default")) is not str or not row.get("task", "default"):
            reason = "ITEM_TASK_INVALID"
        elif (type(row.get("depends_on", [])) is not list
              or any(type(x) is not str for x in row.get("depends_on", []))
              or len(set(row.get("depends_on", []))) != len(row.get("depends_on", []))
              or key in row.get("depends_on", [])):
            reason = "ITEM_DEPENDENCY_INVALID"
        elif (any(row.get(n) is not None and (type(row[n]) is not str or not row[n].strip())
                  for n in ("entity", "attribute")) or type(row.get("exclusive", False)) is not bool):
            reason = "ITEM_SLOT_INVALID"
        if reason:
            drop("upsert", key, reason)
        else:
            seen.add(key)
            upserts[key] = dict(row)
    known_referents: dict[tuple[str, str], frozenset[str]] = {}
    frames_by_context: dict[tuple[str, str], dict[str, frozenset[str]]] = {}
    named_by_context: dict[tuple[str, str], frozenset[str]] = {}

    def known_for(trace: Trace) -> frozenset[str]:
        context = (trace.current.scope, trace.current.task)
        if context not in known_referents:
            known_referents[context] = known_entities(traces, *context)
        return known_referents[context]

    def named_for(trace: Trace) -> frozenset[str]:
        context = (trace.current.scope, trace.current.task)
        if context not in named_by_context:
            named_by_context[context] = named_entities(traces, *context)
        return named_by_context[context]

    def frames_for(trace: Trace) -> dict[str, frozenset[str]]:
        context = (trace.current.scope, trace.current.task)
        if context not in frames_by_context:
            frames_by_context[context] = subject_frames(traces, *context)
        return frames_by_context[context]

    for action in ("negate", "remove"):
        for row in proposal.get(action, []):
            key = row.get("key") if type(row) is dict else None
            if type(row) is not dict or not {"key", "quote"} <= set(row) or set(row) - {"key", "quote", "when", "scope", "task"}:
                reason = "ITEM_FIELDS_INVALID"
            elif type(key) is not str or not KEY.fullmatch(key):
                reason = "ITEM_KEY_INVALID"
            elif key in upserts:
                reason = "ITEM_CONFLICTING_ACTIONS"
            elif key in seen:
                reason = "ITEM_DUPLICATE_KEY"
            elif not supported(row["quote"], source):
                reason = "ITEM_QUOTE_NOT_FOUND"
            elif key not in traces or traces[key].current.action != "assert" or traces[key].hard_forgotten:
                reason = "NEGATE_UNKNOWN_OR_ALREADY_WITHDRAWN"
            elif traces[key].kind == "semantic_card":
                # A card is derived knowledge: it changes only when its
                # sources change. Withdrawing a source reconsolidates it.
                reason = "ITEM_DERIVED_CARD_READ_ONLY"
            elif (row.get("when") is not None and (type(row["when"]) is not str or not row["when"])):
                reason = "ITEM_WHEN_INVALID"
            elif (type(row.get("scope", traces[key].current.scope)) is not str
                  or row.get("scope", traces[key].current.scope) != traces[key].current.scope):
                reason = "ITEM_SCOPE_INVALID"
            elif (type(row.get("task", traces[key].current.task)) is not str
                  or row.get("task", traces[key].current.task) != traces[key].current.task):
                reason = "ITEM_TASK_INVALID"
            elif withdrawal_support and other_referent(
                    row["quote"], traces[key].current.text, known_for(traces[key]),
                    traces[key].entity, named_for(traces[key])) and shares_frame(
                    row["quote"], traces[key].current.text, traces[key].entity,
                    frames_for(traces[key])):
                # Support evaluation, the last per-item gate so that every
                # other drop reason is counted the same with the switch on or
                # off: the evidence must be about this trace's entity, not
                # merely present in the source.  The quote's subject must be
                # another entity of the same kind as the trace's (the store
                # shows both as subjects followed by the same word) and be
                # attested as a name (written as one, or writer about-metadata).
                reason = "WITHDRAWAL_OTHER_REFERENT"
            else:
                reason = None
            if reason:
                drop(action, key, reason)
            else:
                seen.add(key)
                negates[key] = dict(row)
    for key, row in upserts.items():
        vacuous = [d for d in row.get("depends_on", [])
                   if d in metadata and d not in traces and d not in upserts]
        if vacuous:
            row["depends_on"] = [d for d in row.get("depends_on", []) if d not in vacuous]
            pruned.append({"key": key, "dependencies": vacuous,
                           "reason": "DEPENDENCY_ON_METADATA_ONLY_PRUNED"})
    while True:
        available = {k for k, t in traces.items() if currently_valid(t) and k not in negates} | set(upserts)
        bad = sorted(k for k, row in upserts.items() if not set(row.get("depends_on", [])) <= available)
        if not bad:
            break
        for key in bad:
            del upserts[key]
            drop("upsert", key, "ITEM_DEPENDENCY_UNRESOLVED")
    pending, order = dict(upserts), []
    while pending:
        ready = sorted(k for k, row in pending.items() if not set(row.get("depends_on", [])) & set(pending))
        if not ready:
            for key in sorted(pending):
                drop("upsert", key, "ITEM_DEPENDENCY_CYCLE")
            break
        for key in ready:
            order.append(pending.pop(key))
    # A cycle can also run through existing nodes, so validate the final graph.
    candidate_deps = {k: t.depends_on for k, t in traces.items()}
    candidate_deps.update({row["key"]: tuple(row.get("depends_on", [])) for row in order})
    cyclic = set()
    for row in order:
        origin = row["key"]
        stack, visited = list(candidate_deps[origin]), set()
        while stack:
            key = stack.pop()
            if key == origin:
                cyclic.add(origin)
                break
            if key not in visited:
                visited.add(key)
                stack.extend(candidate_deps.get(key, ()))
    valid_order = []
    unavailable = set(cyclic)
    for row in order:
        key = row["key"]
        if key in cyclic or set(row.get("depends_on", [])) & unavailable:
            unavailable.add(key)
            drop("upsert", key, "ITEM_DEPENDENCY_CYCLE" if key in cyclic else "ITEM_DEPENDENCY_UNRESOLVED")
        else:
            valid_order.append(row)
    return Admission(tuple(valid_order), tuple(negates[k] for k in sorted(negates)),
                     tuple(dropped), tuple(pruned), frozenset(metadata),
                     ignored_evictions=len(proposal.get("evict", [])))
