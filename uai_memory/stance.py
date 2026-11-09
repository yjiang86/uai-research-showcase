"""Temporal selection never derives withdrawal from a new positive fact."""
from __future__ import annotations

from datetime import date, datetime, timedelta
import re
from typing import Iterable
from .types import Query, Stance, Trace


def valid_day(value: object) -> bool:
    """True only for an exact ISO calendar day (YYYY-MM-DD)."""
    if type(value) is not str or len(value) != 10:
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def stated_day(stance: Stance) -> str | None:
    """Only an actual ISO date/time supports a calendar cutoff; unknown != old."""
    if stance.when is None:
        return None
    try:
        if len(stance.when) == 10:
            return date.fromisoformat(stance.when).isoformat()
        return datetime.fromisoformat(stance.when).date().isoformat()
    except (ValueError, TypeError):
        return None


def observed_day(stance: Stance) -> str | None:
    """The source episode date; it is never parsed from the claim text."""
    return stance.observed if valid_day(stance.observed) else None


def stance_day(stance: Stance) -> str | None:
    """Calendar position of a stance for an as-of query.

    A stated event time is used when the source supplied one. Otherwise a
    stance holds from the episode in which it was expressed: a present-tense
    assertion made on day D is evidence about D, not about an unknown time.
    """
    return stated_day(stance) or observed_day(stance)


def eligible_stances(trace: Trace, query: Query) -> list[Stance]:
    return [s for s in trace.history if s.scope == query.scope
                and s.task == query.task
                and (query.at_operation is None or s.operation <= query.at_operation)
                and (query.at_when is None or
                     (stance_day(s) is not None and stance_day(s) <= query.at_when))]


def selected_stance(trace: Trace, query: Query) -> Stance | None:
    eligible = eligible_stances(trace, query)
    return eligible[-1] if eligible else None


def present_day(traces: Iterable[Trace]) -> str | None:
    """The memory's own present: the newest day it has experienced.

    Relative time ("the last 3 months") is anchored to the latest episode
    the memory has lived through, as a temporal context drifts with
    experience. A future event date in a claim does not move that experience
    clock; event-time eligibility is still evaluated separately by
    :func:`stance_day`.  If a store has no episode envelope at all, fall back
    to stated dates as the only available calendar evidence.
    """
    experienced = [day for trace in traces for stance in trace.history
                   for day in (observed_day(stance),) if day is not None]
    if experienced:
        return max(experienced)
    return max((day for trace in traces for stance in trace.history
                for day in (stated_day(stance),) if day is not None), default=None)


def window_start(within: str, present: str) -> str:
    """First calendar day included in a relative window ending at present."""
    today = date.fromisoformat(present)
    calendar = re.fullmatch(r"this-([wmy])", within)
    if calendar:
        unit = calendar.group(1)
        if unit == "w":
            return (today - timedelta(days=today.weekday())).isoformat()
        return (today.replace(day=1) if unit == "m" else today.replace(month=1, day=1)).isoformat()
    amount, unit = int(within[:-1]), within[-1]
    if unit in "dw":
        days = amount * (7 if unit == "w" else 1)
        return (today - timedelta(days=days - 1)).isoformat()
    months = amount * (12 if unit == "y" else 1)
    index = today.year * 12 + today.month - 1 - months
    year, month = divmod(index, 12)
    month += 1
    # Clamp the day for shorter months, then start the day after it so a
    # window of N months ending on day D never includes day D N months ago.
    last = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)).day
    return (date(year, month, min(today.day, last)) + timedelta(days=1)).isoformat()


def in_window(stance: Stance, bounds: tuple[str, str] | None) -> bool:
    """A stance lies in the window only if it has a calendar position in it.

    The bounds are the inclusive (first day, present) pair: a relative window
    ends at the memory's present, so a claim stated for a later day is not
    one of the window's episodes.
    """
    if bounds is None:
        return True
    day = stance_day(stance)
    return day is not None and bounds[0] <= day <= bounds[1]


def currently_valid(trace: Trace) -> bool:
    return trace.current.action == "assert" and not trace.stale and not trace.hard_forgotten


def render_history(stance: Stance) -> str:
    stamp = f"op{stance.operation}"
    if stance.when:
        stamp += f"; stated time={stance.when}"
    if stance.observed:
        stamp += f"; observed={stance.observed}"
    return f"HISTORY [{stamp}; {stance.action}; scope={stance.scope}; task={stance.task}] {stance.text}"
