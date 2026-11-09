"""Runtime data. Evidence identity, stance, accessibility and tier are separate."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import re
from typing import Any


class MemoryInvariantError(ValueError):
    pass


class Tier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    CONSOLIDATED = "consolidated"


@dataclass(frozen=True)
class SourceRef:
    key: str
    revision: int


@dataclass(frozen=True)
class Stance:
    operation: int
    action: str
    text: str
    quote: str
    source_id: str
    when: str | None = None
    scope: str = "default"
    task: str = "default"
    # Episode (encoding) date: when the source episode was experienced. It is
    # an envelope fact of the source, distinct from the stated event time.
    observed: str | None = None


@dataclass(frozen=True)
class SemanticFrame:
    """Shared relational context plus case bindings, with exact reconstruction.

    Segments remain in source order. Each row supplies the gaps before,
    between and after segments. No template slot implies exclusivity.
    """
    segments: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass
class Trace:
    key: str
    history: list[Stance]
    about: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    tier: Tier = Tier.HOT
    kind: str = "episodic"
    stale: bool = False
    hard_forgotten: bool = False
    merged_from: tuple[SourceRef, ...] = ()
    merged_into: str | None = None
    support_ids: set[str] = field(default_factory=set)
    access_count: int = 0
    replay_count: int = 0
    replay_score: float = 0.0
    stability: float = 0.0
    novelty: float = 1.0
    last_access: int | None = None
    last_replay: int | None = None
    entity: str | None = None
    attribute: str | None = None
    exclusive: bool = False
    frame: SemanticFrame | None = None
    processing_gain: float = 1.0
    context_error: float = 0.0
    suppressed: bool = False
    # Version of the vertical-gate criterion under which every member of a
    # semantic card was certified. 0 means unknown (cards restored from
    # snapshots written before the criterion was versioned); such members
    # earn no exemption when the card is extended (R4-01).
    gate_version: int = 0

    @property
    def current(self) -> Stance:
        return self.history[-1]

    @property
    def revision(self) -> int:
        # Re-observation adds provenance but does not fabricate a new content
        # version or invalidate a card that represents unchanged knowledge.
        last = None
        count = 0
        for stance in self.history:
            identity = (stance.action, stance.text, stance.scope, stance.task)
            if identity != last:
                count += 1
                last = identity
        return count

    @property
    def recurrence(self) -> int:
        return len(self.support_ids)

    @property
    def pool(self) -> str:
        return self.about[0].casefold() if self.about else re.split(r"[_.:/-]", self.key)[0].casefold()

    @property
    def concept(self) -> str | None:
        """Writer-declared identity of the thing this trace describes.

        An explicit entity is authoritative. Otherwise the dotted key
        namespace is the writer's own naming of the entity whose attribute
        this key holds (``project5.budget`` -> ``project5``). Only ``.`` is
        hierarchical: ``:`` marks a key type prefix (for example digest keys)
        and ``_``/``-`` join words. Nothing is inferred from the text.
        """
        if self.entity and self.entity.strip():
            return "entity:" + " ".join(self.entity.split()).casefold()
        head, dot, _ = self.key.rpartition(".")
        return "key:" + head.casefold() if dot and head else None


@dataclass(frozen=True)
class Query:
    text: str = ""
    view: str = "current"
    scope: str = "default"
    at_operation: int | None = None
    at_when: str | None = None
    task: str = "default"
    # A relative episodic window ("in the last 3 months", "this month"):
    # N days/weeks/months/years ending at the memory's present, or the
    # current calendar week/month/year to date. It is resolved against the
    # newest experienced episode, never against a wall clock.
    within: str | None = None

    def __post_init__(self) -> None:
        if self.view not in ("current", "history"):
            raise MemoryInvariantError("QUERY_VIEW_INVALID")
        if self.at_operation is not None and (type(self.at_operation) is not int
                                               or self.at_operation < 0):
            raise MemoryInvariantError("QUERY_OPERATION_INVALID")
        if self.at_when is not None and (type(self.at_when) is not str
                                         or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.at_when)):
            raise MemoryInvariantError("QUERY_DATE_INVALID")
        if self.at_when is not None:
            try:
                date.fromisoformat(self.at_when)
            except ValueError as error:
                raise MemoryInvariantError("QUERY_DATE_INVALID") from error
        if self.within is not None:
            match = (re.fullmatch(r"([1-9]\d{0,3})([dwmy])|this-([wmy])", self.within)
                     if type(self.within) is str else None)
            if match is None:
                raise MemoryInvariantError("QUERY_WINDOW_INVALID")
        if (not isinstance(self.text, str) or not isinstance(self.scope, str) or not self.scope
                or not isinstance(self.task, str) or not self.task):
            raise MemoryInvariantError("QUERY_INVALID")


@dataclass(frozen=True)
class Evidence:
    key: str
    text: str
    tier: str
    current: bool
    score: float
    ambiguous: bool = False
    operation: int = 0
    when: str | None = None


@dataclass(frozen=True)
class ConsolidationRequest:
    sources: tuple[SourceRef, ...]
    texts: tuple[str, ...]
    about: tuple[str, ...]


@dataclass(frozen=True)
class ConsolidationProposal:
    """A factored card plus lossless coverage, bound to exact source revisions.

    Free paraphrases cannot be certified by numeric/token checks. A proposal
    must preserve every source statement in `claims` and use the canonical
    factored representation. Adapters may propose one, but the core verifies
    it with no model call. This does not certify arbitrary semantic inference.
    """
    key: str
    sources: tuple[SourceRef, ...]
    claims: tuple[str, ...]
    text: str
    about: tuple[str, ...] = ()
    frame: SemanticFrame | None = None


@dataclass(frozen=True)
class BatchResult:
    status: str
    operation: int
    applied: tuple[str, ...] = ()
    negated: tuple[str, ...] = ()
    dropped: tuple[dict[str, Any], ...] = ()
    pruned: tuple[dict[str, Any], ...] = ()
    reason: str | None = None
