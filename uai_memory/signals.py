"""Derived signals. Lexical novelty and measured prediction residual differ."""
from __future__ import annotations

import math
import re
from functools import lru_cache
from .config import RuntimeConfig
from .types import Trace


@lru_cache(maxsize=1 << 17)
def terms(text: str) -> frozenset[str]:
    # Single-letter names and numeric values carry content, too. Dropping
    # them would make X/Y or cabinet 1/2 appear lexically identical.
    # Unspaced CJK text needs subword cues: a query for 茶 must be able to
    # retrieve 我喜欢茶. These are retrieval features only, not claim identity.
    cjk = r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+"
    words = set(re.findall(r"\w+", re.sub(cjk, " ", text.casefold())))
    for run in re.findall(cjk, text):
        words.update(run)
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return frozenset(words)


def overlap(left: str, right: str) -> float:
    a, b = terms(left), terms(right)
    return len(a & b) / len(a | b) if a or b else 1.0


# Event boundaries: sentence ends and line breaks. Retrieval features only;
# they never change the stored text, its identity or its stance.
_EVENT_BOUNDARY = re.compile(r"(?<=[.!?;\u3002\uff01\uff1f\uff1b])\s+|\n+")


@lru_cache(maxsize=1 << 16)
def events(text: str) -> tuple[str, ...]:
    return tuple(part for part in (p.strip() for p in _EVENT_BOUNDARY.split(text)) if part)


def cue_match(cue: str, text: str, segmented: bool = True) -> float:
    """Cue drive of a trace: its best-matching encoded event.

    An episode is indexed at its event boundaries, so a partial cue that
    matches one event of a long mixed episode completes that event as well
    as it would a trace holding the event alone. Whole-text Jaccard divides
    the match by every unrelated event in the trace, which made long mixed
    traces unreachable once they lost expression. The whole-text match is
    kept as a floor so a cue spanning several events is not weakened.
    """
    whole = overlap(cue, text)
    if not segmented:
        return whole
    parts = events(text)
    if len(parts) < 2:
        return whole
    return max(whole, max(overlap(cue, part) for part in parts))


def novelty(text: str, peers: list[Trace]) -> float:
    return 1.0 - max((overlap(text, p.current.text) for p in peers), default=0.0)


def priority(trace: Trace, operation: int, indegree: int, config: RuntimeConfig) -> float:
    age = max(0, operation - trace.current.operation)
    use = (math.log1p(trace.access_count) /
           (1 + max(0, operation - (trace.last_access or 0)) / config.recency_scale)
           if config.mechanisms.usage_priority else 0.0)
    reactivation = (math.log1p(trace.replay_count) * trace.stability
                    if config.mechanisms.replay else 0.0)
    return (math.log1p(trace.recurrence) + 0.5 * math.log1p(indegree) + use + reactivation
            + 1.0 / (1.0 + age / config.recency_scale))


# Accessibility modulates the cue drive within this band and no further.
ACCESSIBILITY_SPAN = 0.25


def retrieval_gain(accessibility: float) -> float:
    """Bounded gain of a trace's accessibility on its cue drive.

    The cue match is the drive of cued retrieval; accessibility (recurrence,
    use, reactivation, recency) is the excitability of the trace and scales
    that drive between 1 - ACCESSIBILITY_SPAN and 1. A saturating gain alone
    still let a strongly replayed partial match outscore a weak exact match
    (R5-02): 0.867 x 0.868 > 1.0 x 0.554. With the band, accessibility can
    reorder two traces only when the weaker cue match is within
    1 - ACCESSIBILITY_SPAN of the stronger one, so it orders traces the cue
    matches about equally and cannot overturn a clearly more specific match.
    Demotion order still uses the unbounded value.
    """
    saturated = accessibility / (1.0 + accessibility) if accessibility > 0 else 0.0
    return 1.0 - ACCESSIBILITY_SPAN + ACCESSIBILITY_SPAN * saturated


def replay_score(trace: Trace, operation: int, conflicts: int,
                 config: RuntimeConfig, prediction_error: float | None = None) -> float:
    support = trace.recurrence / (trace.recurrence + conflicts + 1)
    use = trace.access_count / (trace.access_count + 1) if config.mechanisms.usage_priority else 0
    fresh = 1 / (1 + max(0, operation - trace.current.operation) / config.recency_scale)
    new = trace.novelty * trace.processing_gain if config.mechanisms.novelty_priority else 0
    # Fixed arithmetic weights are not static neuromodulator biases.
    # Feedback is added only when an explicit prediction has an observation.
    value = (new + support + use + fresh + trace.stability) / 5
    if prediction_error is not None:
        value = (5 * value + prediction_error) / 6
    if config.mechanisms.context_prediction:
        value += (1 - value) * trace.context_error
    return value
