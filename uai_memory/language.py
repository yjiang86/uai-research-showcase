"""Explicit language controls and source-preserving claim identity.

The local parser handles single, direct assertions. Richer sentences remain the
provider writer's job. Word overlap is a retrieval signal, never authority to
withdraw another claim. Query controls require syntax, not isolated keywords.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
import unicodedata

from .types import MemoryInvariantError, Query, Trace


# Determiners and pronouns refer to a context; they never name one.  "In my
# job", "for this job" and "in that context" are everyday prose, so only a
# named label ("in planning task", "task=planning") may change the context.
_NOT_A_LABEL = (r"(?!(?:my|your|his|her|its|our|their|this|that|these|those|the|a|an|"
                r"same|each|every|any|some|one|other|another|which|what|whose|no)\b)")
_SCOPE = re.compile(
    r"\b(?:(?:in|within|for)\s+)?(?:scope|context)\s*[:=]\s*(?P<label>[\w./-]+)|"
    r"\b(?:in|within|for)\s+(?:the\s+)?(?P<english>" + _NOT_A_LABEL + r"[\w./-]+)\s+(?:context|scope)\b|"
    r"(?:在|于)\s*(?P<chinese>[\w./-]+?)\s*(?:情境|上下文|范围)", re.I)
_TASK = re.compile(
    r"\b(?:(?:in|within|for)\s+)?(?:task|job|purpose)\s*[:=]\s*(?P<label>[\w./-]+)|"
    r"\b(?:in|within|for)\s+(?:the\s+)?(?P<english>" + _NOT_A_LABEL + r"[\w./-]+)\s+(?:task|job|purpose)\b|"
    r"\b(?:in|within|for)\s+(?:task|job|purpose)\s+(?P<english2>" + _NOT_A_LABEL + r"[\w./-]+)\b|"
    r"(?:在|于)\s*(?P<chinese>[\w./-]+?)\s*(?:任务|工作)", re.I)
# Assignment syntax is the only context control that raw source text can carry
# by itself.  Phrases such as "for the given task" or "scope: specific" are
# ordinary prose unless a complete direct clause owns them.
_ASSIGNED_SCOPE = re.compile(r"(?<![\w-])(?:scope|context)\s*=\s*(?P<label>[\w./-]+)", re.I)
_ASSIGNED_TASK = re.compile(r"(?<![\w-])(?:task|job|purpose)\s*=\s*(?P<label>[\w./-]+)", re.I)
# A question may also open with a context frame: "current for research
# task: ...", "history in home context: ...", "当前在研究任务中，...".  The
# frame names its label without a determiner and is closed by a colon or
# comma in both languages.  Like a phrase inside the question, it selects a
# context only when its label is one the memory has learned (R5-03).
_FRAME_VIEW = (r"^(?:(?P<view_en>current(?:ly)?|now|present|latest|history|historical|past|timeline)\b"
               r"\s*:?\s*|(?P<view_zh>当前|现在|目前|最新|历史|过去|时间线|回顾)\s*:?\s*)?")
_FRAME_SCOPE = re.compile(
    _FRAME_VIEW + r"(?:(?:in|within|for)\s+(?P<english>" + _NOT_A_LABEL + r"[\w./-]+)\s+"
    r"(?:context|scope)\s*[:,]\s*|(?:在|于)\s*(?P<chinese>[\w./-]+?)\s*(?:情境|上下文|范围)\s*(?:中|里|下)?"
    r"\s*[:：,，]\s*)", re.I)
_FRAME_TASK = re.compile(
    _FRAME_VIEW + r"(?:(?:in|within|for)\s+(?P<english>" + _NOT_A_LABEL + r"[\w./-]+)\s+"
    r"(?:task|job|purpose)\s*[:,]\s*|(?:在|于)\s*(?P<chinese>[\w./-]+?)\s*(?:任务|工作)\s*(?:中|里|时)?"
    r"\s*[:：,，]\s*)", re.I)
# "op" inside a word ("co-op 2024") is content, not an operation number.
_OPERATION = re.compile(
    r"(?<![\w-])(?:(?:as\s+of|at)\s+)?(?:operation|op)\s*[:#]?\s*(\d+)(?![\d-])\b|"
    r"(?:截至|截止到|在)?(?:操作\s*|第\s*)(\d+)\s*(?:步|次|个操作)", re.I)
_CUTOFF = re.compile(r"\b(?:as\s+of|up\s+to)\s+(\d{4}-\d{2}-\d{2})\b|"
                     r"(?:截至|截止到)\s*(\d{4}-\d{2}-\d{2})", re.I)
# Relative episodic windows. "The last N months" ends at the memory's
# present; "this month" runs from the start of the current calendar month.
# A bare "last month" names the previous calendar month, a bounded period
# with an end, and is deliberately left to ordinary retrieval.
_COUNT = {"one": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
          "九": 9, "十": 10, "十二": 12}
_UNIT = {"day": "d", "week": "w", "month": "m", "year": "y", "天": "d", "日": "d", "周": "w",
         "星期": "w", "个月": "m", "月": "m", "年": "y"}
_WINDOW = re.compile(
    r"\b(?:(?:in|over|during|within|for)\s+)?the\s+(?:last|past)\s+"
    r"(?:(?P<n>\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+)?"
    r"(?P<unit>day|week|month|year)s?\b|"
    r"\b(?:(?:in|during|for|over)\s+)?this\s+(?P<this>week|month|year)\b|"
    r"(?:最近|过去|近)\s*(?P<zn>\d{1,4}|十二|[一两二三四五六七八九十])\s*(?P<zunit>天|日|个月|月|周|星期|年)(?:内|里|中|以来)?|"
    r"(?P<zthis>本周|这周|本月|这个月|今年)", re.I)
_ZH_THIS = {"本周": "w", "这周": "w", "本月": "m", "这个月": "m", "今年": "y"}


def _window(match: re.Match) -> str:
    if match["this"]:
        return "this-" + _UNIT[match["this"].casefold()]
    if match["zthis"]:
        return "this-" + _ZH_THIS[match["zthis"]]
    if match["zunit"]:
        number, unit = match["zn"], _UNIT[match["zunit"]]
    else:
        number, unit = match["n"] or "1", _UNIT[match["unit"].casefold()]
    count = int(number) if number.isdigit() else _COUNT[number.casefold()]
    if not 1 <= count <= 9999:
        raise MemoryInvariantError("NATURAL_TIME_AMBIGUOUS")
    return f"{count}{unit}"
_VIEW_PREFIX = re.compile(
    r"^(?P<en>current(?:ly)?|now|present|latest|history|historical|past|timeline)\b\s*:?\s*|"
    r"^(?P<zh>当前|现在|目前|最新|历史|过去|时间线|回顾)\s*:?\s*", re.I)
_HISTORY_PREFIX = {"history", "historical", "past", "timeline", "历史", "过去", "时间线", "回顾"}
_HISTORY_CUE = re.compile(r"\b(?:previously|used\s+to|no\s+longer)\b|"
                          r"\b(?:before|earlier)\s*[?.!]*$|曾经|以前|不再", re.I)
_CURRENT_CUE = re.compile(r"\b(?:currently|still)\b|\bnow\s*[?.!]*$", re.I)
# Reporting/position questions ask for the remembered series rather than the
# latest expression. This is deliberately a small, explicit grammar: it
# removes only interrogative/reporting scaffolding and keeps the named subject
# plus cue as retrieval text. It never turns reported speech in a writer
# statement into a fact and never infers a withdrawal.
_REPORT_HISTORY_EN = re.compile(
    r"^(?:what|which)\s+(?:did|does|do|has|have|had|is|was|were|can|could|will|would)\s+"
    r"(?P<subject>[\w.'-]+)\s+"
    r"(?:say|said|mention|mentioned|believe|believed|think|thought)"
    r"(?:\s+(?:about|on|regarding)\s+(?P<cue>.+))?$", re.I)
_POSITION_HISTORY_EN = re.compile(
    r"^what\s+(?:was|is)\s+(?P<subject>[\w.'-]+)'s\s+"
    r"(?:position|stance|view)\s+(?:about|on|regarding)\s+(?P<cue>.+)$", re.I)
# Chinese counterparts of the same two templates: "Alex 对 X 说过什么",
# "Alex 说过什么", "Alex 对 X 的立场是什么". The subject is a single token;
# an explicit present cue (现在/目前/当前) keeps the question on current state.
_REPORT_HISTORY_ZH = re.compile(
    r"^(?:请问|请告诉我|告诉我)?\s*(?P<subject>[\w.'-]+?)\s*"
    r"(?:(?:对|关于|就)\s*(?P<cue>.+?)\s*)?(?:都|曾)?"
    r"(?:说过|提过|讲过|表示过|认为过)(?:些)?什么$")
_POSITION_HISTORY_ZH = re.compile(
    r"^(?:请问|请告诉我|告诉我)?\s*(?P<subject>[\w.'-]+?)\s*(?:对|关于)\s*(?P<cue>.+?)\s*"
    r"的?(?:立场|态度|看法|观点)(?:是什么|是怎样的|如何|怎样)$")
_PRESENT_ZH = re.compile(r"现在|目前|当前")
_QUERY_LEAD = re.compile(r"^(?:what|when|where|which|who|how|why|tell\s+me|please)\b|"
                         r"^(?:请问|查询|查一下|告诉我|什么|哪些|何时|哪里|谁|如何|为什么)", re.I)
_QUESTION_EN = re.compile(
    r"^(?:what|which|where|when|who|whom|whose|how)\b\s*"
    r"(?:(?:does|did|do|is|are|was|were|has|have|had|can|could|will|would|"
    r"should|may|might)\b\s*)?|"
    # A yes/no question opens with its auxiliary.
    r"^(?:does|did|do|is|are|was|were|has|have|had)\b\s*", re.I)
_QUESTION_ZH = re.compile(r"^(?:请问|请告诉我|查询|查一下|告诉我|何时|哪里|谁|如何|为什么|哪些|什么)\s*")

# Normalize predicate inflections, never subjects or objects.
_VERBS = {form: lemma for lemma, forms in {
    "like": ("like", "likes", "liked"), "love": ("love", "loves", "loved"),
    "prefer": ("prefer", "prefers", "preferred"), "support": ("support", "supports", "supported"),
    "use": ("use", "uses", "used"), "keep": ("keep", "keeps", "kept"),
    "live": ("live", "lives", "lived"), "want": ("want", "wants", "wanted"),
    "work": ("work", "works", "worked"), "choose": ("choose", "chooses", "chose"),
    "move": ("move", "moves", "moved"), "take": ("take", "takes", "took", "taken"),
}.items() for form in forms}

# Only relations whose ordinary language form carries an explicit single-value
# constraint are eligible for automatic cross-key conflict detection.  This is
# deliberately a tiny, conservative set: "likes X" and "keeps X" do not imply
# that another value is impossible, while a direct "lives in X" statement does
# expose a current-location slot.  Structured writer metadata remains the
# authoritative way to mark any other relation as exclusive.
_NATURAL_EXCLUSIVE_PREDICATES = frozenset({"live", "move", "take", "住在"})
_EN_CLAIM = re.compile(
    r"(?P<subject>.+?)\s+(?:(?:currently|now|again)\s+)?"
    r"(?:(?P<negative>no\s+longer|do\s+not|does\s+not|did\s+not|don't|doesn't|didn't)\s+)?"
    r"(?P<verb>" + "|".join(sorted(_VERBS, key=len, reverse=True)) + r")\s+"
    r"(?P<object>.+?)(?:\s+(?P<again>again))?", re.I)
_ZH_CLAIM = re.compile(
    r"(?P<subject>.+?)(?:现在|目前|当前)?(?P<again>又|再次|重新)?"
    r"(?P<negative>不再|不)?(?P<verb>喜欢|支持|使用|保留|住在|想要|选择)(?P<object>.+)")
_COMPLEX = re.compile(
    r"\b(?:if|unless|but|and|or|although|whether|will|would|should|could|can|may|might|"
    r"perhaps|maybe|probably|said|says|say|claimed|claims|thinks|think|believes|believe|"
    r"heard|hears|asks|asked|never|not|dislike|dislikes)\b|"
    r"如果|假如|可能|也许|应该|有人说|听说|认为|据说|但是|不过|并且|还是|或者|[?;:！!，,。\n]", re.I)
_COMPOUND_EN = re.compile(r"\s+(?:but|and)\s+", re.I)
_COMPOUND_ZH = re.compile(r"(?:，|,)?\s*(?:并且|而且|但是|不过|也|还)\s*")
# A request to forget is an instruction about memory, never a claim about the
# world.  It names a proposition (or the one in focus) to be suppressed.
_FORGET_EN = re.compile(r"(?:(?:please|now|just)\s*,?\s+)?forget\s+(?:(?:that|about)\s+)?(?P<body>.+)", re.I)
_FORGET_ZH = re.compile(r"(?:请)?(?:忘掉|忘记)(?P<body>.+?)(?:吧)?")
# Idioms such as "forget it" mean "never mind"; only an explicit reference to
# what was said can name the proposition in focus.
_FOCUS_REFERENCE = frozenset({"what i just said", "what i said", "what i just told you",
                              "我刚才说的", "我刚说的", "刚才说的", "我刚才说的话"})


def forget_request(text: str) -> str | None:
    """Return the proposition named by an explicit request to forget, if any."""
    bare = normalize(text).strip(" .!。！")
    match = _FORGET_EN.fullmatch(bare) or _FORGET_ZH.fullmatch(bare)
    if match is None:
        return None
    body = normalize(match["body"]).strip(" .!。！")
    return body or None


# A request to stop mentioning something asks memory to stop expressing it,
# never to erase it: like retrieval suppression, the trace stays in history
# and a strong direct cue (or a new encounter) can still bring it back.
_SUPPRESS_EN = re.compile(
    r"(?:(?:please|now|just)\s*,?\s+)?(?:stop\s+(?:mentioning|bringing\s+up)|"
    r"(?:don't|do\s+not)\s+(?:mention|bring\s+up))\s+(?:(?:that|about)\s+)?"
    r"(?P<body>.+?)(?:\s+(?:again|anymore|any\s+more))?", re.I)
_SUPPRESS_ZH = re.compile(r"(?:请)?(?:别再|不要再)(?:提起|提)(?P<body>.+?)(?:了)?(?:吧)?")


def memory_request(text: str) -> tuple[str, str] | None:
    """Return (proposition, mode) for an explicit request about memory.

    "forget X" asks for hard forgetting; "stop mentioning X" asks only that X
    no longer be expressed (soft suppression). Anything else is None.
    """
    bare = normalize(text).strip(" .!。！")
    for mode, patterns in (("hard", (_FORGET_EN, _FORGET_ZH)),
                           ("soft", (_SUPPRESS_EN, _SUPPRESS_ZH))):
        match = patterns[0].fullmatch(bare) or patterns[1].fullmatch(bare)
        if match is not None:
            body = normalize(match["body"]).strip(" .!。！")
            return (body, mode) if body else None
    return None


def forget_reference_is_focus(body: str) -> bool:
    return body.casefold() in _FOCUS_REFERENCE


def normalize(text: str) -> str:
    if type(text) is not str:
        raise MemoryInvariantError("NATURAL_TEXT_INVALID")
    text = unicodedata.normalize("NFKC", text).replace("’", "'")
    return " ".join(text.split())


def _label(match: re.Match) -> str:
    return next(value for value in match.groupdict().values() if value).casefold()


def _is_assignment(match: re.Match) -> bool:
    return bool(match.groupdict().get("label")) and "=" in match.group(0)


def _context(pattern: re.Pattern, text: str, name: str) -> tuple[str | None, str]:
    matches = list(pattern.finditer(text))
    values = {_label(match) for match in matches}
    if len(values) > 1:
        assigned = {_label(match) for match in matches if _is_assignment(match)}
        # Conflicting assignments are a malformed control.  Any other
        # disagreement involves prose and is ambiguous evidence, not a control.
        raise MemoryInvariantError(f"NATURAL_{name}_AMBIGUOUS" if len(assigned) > 1
                                   else f"NATURAL_{name}_PROSE_AMBIGUOUS")
    cleaned = pattern.sub(" ", text).strip(" ,:;，。；")
    return next(iter(values), None), normalize(cleaned)


def _scope(text: str) -> tuple[str | None, str]:
    return _context(_SCOPE, text, "SCOPE")


def _task(text: str) -> tuple[str | None, str]:
    return _context(_TASK, text, "TASK")


def _assigned(pattern: re.Pattern, text: str, name: str) -> tuple[str | None, str]:
    """Assignment-only context; every other phrase stays in the text."""
    labels = {match["label"].casefold() for match in pattern.finditer(text)}
    if len(labels) > 1:
        raise MemoryInvariantError(f"NATURAL_{name}_AMBIGUOUS")
    return next(iter(labels), None), normalize(pattern.sub(" ", text))


def _query_context(text: str, name: str,
                   known: frozenset[str] | None = None) -> tuple[str | None, str, bool]:
    """Positive context grammar for questions (R4-03).

    A question selects a context only by an assignment ("task=planning") or
    by a context phrase whose label names a context this memory has learned,
    whether the phrase opens the question as a frame ("current for research
    task: ...") or sits inside it ("... in the planning task?").  An
    unlearned label ("In general context:", "在日常工作中，") names no context
    this memory holds, so it stays content (R5-03).  A definite description
    refers to a context the listener knows; "for the given task" or "for
    task completion" names none.  The third flag reports context phrases
    left in the cue as prose.
    """
    pattern, assigned, frame = ((_SCOPE, _ASSIGNED_SCOPE, _FRAME_SCOPE) if name == "SCOPE"
                                else (_TASK, _ASSIGNED_TASK, _FRAME_TASK))
    value, rest = _assigned(assigned, text, name)
    match = frame.match(rest)
    label = (match.group("english") or match.group("chinese")).casefold() if match else None
    if match and label in (known or frozenset()):
        if value is not None and value != label:
            raise MemoryInvariantError(f"NATURAL_{name}_AMBIGUOUS")
        view = match.group("view_en") or match.group("view_zh") or ""
        rest = normalize(view + " " + rest[match.end():])
        return label, rest, bool(pattern.search(rest))
    prose = list(pattern.finditer(rest))
    if value is not None or not prose:
        return value, rest, bool(prose)
    named = {_label(m) for m in prose} & (known or frozenset())
    if len(named) != 1:
        return None, rest, True
    label = next(iter(named))
    for m in reversed(prose):
        if _label(m) == label:
            rest = rest[:m.start()] + " " + rest[m.end():]
    rest = normalize(rest).strip(" ,:;，。；")
    return label, rest, bool(pattern.search(rest))


def _strip_question_frame(text: str) -> tuple[str, bool, str | None]:
    """Remove interrogative scaffolding while preserving content terms.

    The memory query is a cue, not a grammatical parse.  Removing the
    question operator prevents words such as ``what`` and ``does`` from
    blocking an otherwise valid current/history retrieval.  It never infers
    a key, object, scope or lifecycle action.
    """
    cue = normalize(text).strip(" ??.!！。？")
    match = _QUESTION_EN.match(cue) or _QUESTION_ZH.match(cue)
    if match is None:
        return cue, False, None
    auxiliary = None
    if match.re is _QUESTION_EN:
        prefix = cue[:match.end()]
        aux = re.search(r"\b(does|did|do|is|are|was|were|has|have|had|can|could|will|would|should|may|might)\b", prefix, re.I)
        remainder = cue[match.end():]
        if aux:
            auxiliary = aux.group(1).casefold()
        else:
            # Wh-questions may place the object between the wh-word and the
            # auxiliary ("what books did I read"). Keep the object as a cue,
            # while removing only the grammatical auxiliary.
            tokens = list(re.finditer(r"\b[\w'-]+\b", remainder))
            for token in tokens[:5]:
                if token.group(0).casefold() in {
                    "does", "did", "do", "is", "are", "was", "were",
                    "has", "have", "had", "can", "could", "will", "would",
                    "should", "may", "might"}:
                    auxiliary = token.group(0).casefold()
                    remainder = remainder[:token.start()] + remainder[token.end():]
                    break
        cue = normalize(remainder).strip(" ??.!！。？")
    else:
        cue = normalize(cue[match.end():]).strip(" ??.!！。？")
    return cue, True, auxiliary


def _reported_history_query(text: str) -> str | None:
    """Return a cue for a bounded report/position history question.

    Questions such as ``What has Alex said about X?`` are not yes/no
    questions about the current claim. They request the remembered series of
    statements, so they must select ``view=history``. Only the explicit
    templates above are accepted; broad reported speech, modal wording and
    arbitrary noun phrases remain ordinary retrieval text or fail closed.
    """
    bare = normalize(text).strip(" ??.!！。？")
    match = _REPORT_HISTORY_EN.fullmatch(bare) or _POSITION_HISTORY_EN.fullmatch(bare)
    if match is None and _PRESENT_ZH.search(bare) is None:
        match = _REPORT_HISTORY_ZH.fullmatch(bare) or _POSITION_HISTORY_ZH.fullmatch(bare)
    if match is None:
        return None
    subject = normalize(match["subject"])
    cue = normalize(match["cue"] or "")
    return normalize(f"{subject} {cue}") if cue else subject


@dataclass(frozen=True)
class QueryInterpretation:
    raw: str
    query: Query
    status: str
    markers: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"raw": self.raw, "query": asdict(self.query), "status": self.status,
                "markers": list(self.markers)}


_IGNORED_PROSE = frozenset({"context_prose_ignored", "scope_prose_ignored", "task_prose_ignored"})


def interpret_query(text: str, *, known_scopes: frozenset[str] | None = None,
                    known_tasks: frozenset[str] | None = None) -> QueryInterpretation:
    """Extract explicit controls, preserving ordinary content verbatim.

    known_scopes/known_tasks are the contexts the reading memory has learned;
    without them only assignments and opening frames select a context.
    """
    raw = normalize(text)
    scope, cue, scope_prose = _query_context(raw, "SCOPE", known_scopes)
    task, cue, task_prose = _query_context(cue, "TASK", known_tasks)
    markers = ["scope"] if scope else []
    if task:
        markers.append("task")
    if scope_prose or task_prose:
        # Disagreeing context phrases in a question are content; only an
        # assignment can still select a context.
        markers.append("context_prose_ignored")
    # Which dimension the ignored phrase talked about: a question that
    # mentions some task or scope it does not select still does not ask
    # about the focus's own one, so discourse does not fill it in.
    if scope_prose:
        markers.append("scope_prose_ignored")
    if task_prose:
        markers.append("task_prose_ignored")
    operations = {int(m.group(1) or m.group(2)) for m in _OPERATION.finditer(cue)}
    dates = {m.group(1) or m.group(2) for m in _CUTOFF.finditer(cue)}
    if len(operations) > 1 or len(dates) > 1:
        raise MemoryInvariantError("NATURAL_TIME_AMBIGUOUS")
    operation, at_when = next(iter(operations), None), next(iter(dates), None)
    windows = {_window(m) for m in _WINDOW.finditer(cue)}
    if len(windows) > 1:
        raise MemoryInvariantError("NATURAL_TIME_AMBIGUOUS")
    within = next(iter(windows), None)
    if within is not None:
        markers.append("window")
        cue = _WINDOW.sub(" ", cue)
    if operation is not None:
        markers.append("operation")
        cue = _OPERATION.sub(" ", cue)
    if at_when is not None:
        markers.append("date")
        cue = _CUTOFF.sub(" ", cue)
    prefix = _VIEW_PREFIX.match(cue)
    view = "current"
    if prefix:
        name = (prefix.group("en") or prefix.group("zh")).casefold()
        view = "history" if name in _HISTORY_PREFIX else "current"
        cue = cue[prefix.end():]
        markers.append(view)
    if _HISTORY_CUE.search(cue) or operation is not None or at_when is not None:
        view = "history"
        markers.append("history")
        cue = _HISTORY_CUE.sub(" ", cue)
    if _CURRENT_CUE.search(cue):
        markers.append("current")
        cue = _CURRENT_CUE.sub(" ", cue)
    report_cue = _reported_history_query(cue)
    if report_cue is not None and "current" not in markers:
        view = "history"
        markers.append("history")
        markers.append("reported_history")
        cue = report_cue
    question_cue, is_question, question_auxiliary = _strip_question_frame(cue)
    # A bare information question is ordinary retrieval text.  Strip its
    # grammatical frame only when another explicit control (current/history,
    # scope, operation, or date) has already established that this is a
    # lifecycle query.  This preserves content words for generic questions
    # while allowing e.g. ``What does Alex like now?`` to cue current state.
    past = question_auxiliary in {"did", "was", "were", "had"}
    # Ignored prose is a report, not a control: it neither makes a question
    # explicit nor strips its frame (R4-03).
    controls = [marker for marker in markers if marker not in _IGNORED_PROSE]
    if is_question and (controls or past):
        # A past-tense question asks about history unless it explicitly asks
        # about the present; a named task or scope does not change tense.  A
        # relative window already places the question in time ("did I spend
        # in the last 3 months"), so its tense asks about events in the window
        # rather than about revised beliefs.
        if past and view == "current" and "current" not in markers and within is None:
            view = "history"
            markers.append("history")
        markers.append("question")
        cue = question_cue
    controls = [marker for marker in markers if marker not in _IGNORED_PROSE]
    cue = normalize(cue).strip(" :：,，;；?？.!！。") if controls else raw
    return QueryInterpretation(raw, Query(cue, view=view, scope=scope or "default",
                                          at_operation=operation, at_when=at_when,
                                          task=task or "default", within=within),
                               "EXPLICIT" if controls else "DEFAULT_CURRENT",
                               tuple(dict.fromkeys(markers)))


@dataclass(frozen=True)
class LifecycleInterpretation:
    raw: str
    status: str
    action: str | None = None
    claim_text: str = ""
    scope: str | None = None
    when: str | None = None
    marker: str | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    task: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def identity(self) -> tuple[str, str, str] | None:
        if self.status != "PARSED":
            return None
        return (self.subject.casefold(), self.predicate, self.object.casefold())


@dataclass(frozen=True)
class StatementBatchInterpretation:
    """A bounded sequence of direct lifecycle clauses.

    This is deliberately narrower than general language understanding.  Each
    member must independently match the direct claim grammar; conjunctions
    only make explicit multiple operations that the caller already stated.
    """
    raw: str
    status: str
    parts: tuple[LifecycleInterpretation, ...] = ()
    scope: str | None = None
    when: str | None = None
    task: str | None = None
    marker: str | None = None

    def as_dict(self) -> dict:
        return {"raw": self.raw, "status": self.status,
                "parts": [part.as_dict() for part in self.parts],
                "scope": self.scope, "when": self.when, "task": self.task,
                "marker": self.marker}


def interpret_statement(text: str, *, scope: str | None = None,
                        when: str | None = None, task: str | None = None) -> LifecycleInterpretation:
    """Parse a complete direct clause; never infer a negation target by overlap."""
    raw = normalize(text)
    unknown = LifecycleInterpretation(raw, "UNPARSED")
    if not raw:
        return unknown
    try:
        detected_scope, content = _scope(raw)
        detected_task, content = _task(content)
    except MemoryInvariantError:
        return LifecycleInterpretation(raw, "AMBIGUOUS")
    if scope is not None and (not isinstance(scope, str) or not scope.strip()):
        raise MemoryInvariantError("NATURAL_SCOPE_INVALID")
    explicit_scope = normalize(scope).casefold() if scope is not None else None
    if explicit_scope and detected_scope and explicit_scope != detected_scope:
        return LifecycleInterpretation(raw, "AMBIGUOUS")
    final_scope = explicit_scope or detected_scope
    if task is not None and (not isinstance(task, str) or not task.strip()):
        raise MemoryInvariantError("NATURAL_TASK_INVALID")
    explicit_task = normalize(task).casefold() if task is not None else None
    if explicit_task and detected_task and explicit_task != detected_task:
        return LifecycleInterpretation(raw, "AMBIGUOUS")
    final_task = explicit_task or detected_task
    # Dates are metadata only with a statement-time prefix or suffix.
    dated = re.search(r"^(?:on\s+|在\s*)?(\d{4}-\d{2}-\d{2})\s*[:,，]?\s*|"
                      r"\s+(?:on|at)\s+(\d{4}-\d{2}-\d{2})[.]?$", content, re.I)
    final_when = when
    if dated:
        detected_when = dated.group(1) or dated.group(2)
        try:
            Query(at_when=detected_when)
        except MemoryInvariantError:
            # An impossible date mentioned in prose is source content, not a
            # control: keep the explicit context and leave the clause unparsed.
            if when is not None:
                raise
            return LifecycleInterpretation(raw, "UNPARSED", scope=final_scope, task=final_task)
        if when is not None and when != detected_when:
            return LifecycleInterpretation(raw, "AMBIGUOUS")
        final_when = detected_when
        content = content[:dated.start()] + content[dated.end():]
    if final_when is not None:
        Query(at_when=final_when)
    # Explicit context is independent of whether the proposition fits the
    # conservative local clause grammar. The provider writer still needs the
    # correct memory gate for compound, modal and reported source material.
    # Retaining these controls never creates a lifecycle action or identity.
    unknown = LifecycleInterpretation(raw, "UNPARSED", scope=final_scope,
                                      when=final_when, task=final_task)
    if _QUERY_LEAD.match(raw) or "?" in raw or memory_request(raw) is not None:
        return unknown
    content = content.strip(" .。")
    match = _EN_CLAIM.fullmatch(content)
    chinese = False
    if match is None:
        match = _ZH_CLAIM.fullmatch(content)
        chinese = True
    if match is None:
        return unknown
    subject, obj = normalize(match["subject"]), normalize(match["object"])
    if not subject or not obj or _COMPLEX.search(subject + " " + obj):
        return unknown
    if _EN_CLAIM.fullmatch(obj) or _ZH_CLAIM.fullmatch(obj):
        return unknown
    verb = match["verb"]
    predicate = verb if chinese else _VERBS[verb.casefold()]
    claim = subject + verb + obj if chinese else f"{subject} {verb} {obj}"
    marker = match["negative"] or match["again"]
    return LifecycleInterpretation(raw, "PARSED", "negate" if match["negative"] else "assert",
                                   claim, final_scope, final_when, marker,
                                   subject, predicate, obj, final_task)


def interpret_statements(text: str, *, scope: str | None = None,
                         when: str | None = None, task: str | None = None) -> StatementBatchInterpretation:
    """Parse a small explicit conjunction without inferring semantics.

    The single-clause parser remains the authority.  A compound is accepted
    only when every component is a direct assertion or withdrawal, and an
    omitted subject in a continuation is copied from the first explicit
    clause.  Conditionals, reported speech, questions, modals and ``or`` are
    rejected before any state-changing call.
    """
    raw = normalize(text)
    single = interpret_statement(raw, scope=scope, when=when, task=task)
    if single.status == "PARSED":
        return StatementBatchInterpretation(raw, "PARSED", (single,), single.scope,
                                            single.when, single.task, "single")
    try:
        detected_scope, content = _scope(raw)
        detected_task, content = _task(content)
    except MemoryInvariantError:
        return StatementBatchInterpretation(raw, "AMBIGUOUS")
    final_scope = normalize(scope).casefold() if scope is not None else detected_scope
    final_task = normalize(task).casefold() if task is not None else detected_task
    if scope is not None and detected_scope and final_scope != detected_scope:
        return StatementBatchInterpretation(raw, "AMBIGUOUS", scope=final_scope, task=final_task)
    if task is not None and detected_task and final_task != detected_task:
        return StatementBatchInterpretation(raw, "AMBIGUOUS", scope=final_scope, task=final_task)
    final_when = when
    # Keep the date prefix/suffix in the first component so the existing date
    # grammar validates it; subsequent components inherit the same date.
    splitter = _COMPOUND_EN if re.search(r"\b(?:but|and)\b", content, re.I) else _COMPOUND_ZH
    pieces = [piece.strip() for piece in splitter.split(content) if piece.strip()]
    if len(pieces) < 2 or len(pieces) > 4:
        return StatementBatchInterpretation(raw, single.status, (single,), final_scope,
                                            final_when, final_task, "unparsed")
    if any(re.search(r"\b(?:if|unless|or|whether|perhaps|maybe|probably|said|says|claims?|thinks?|believes?|heard)\b|"
                     r"如果|假如|可能|也许|应该|有人说|听说|认为|据说|还是|或者", piece, re.I)
           for piece in pieces):
        return StatementBatchInterpretation(raw, "UNPARSED", scope=final_scope,
                                            when=final_when, task=final_task, marker="unsafe")
    parsed: list[LifecycleInterpretation] = []
    first_subject = None
    for index, piece in enumerate(pieces):
        candidate = piece
        part = interpret_statement(candidate, scope=final_scope, when=final_when, task=final_task)
        if part.status != "PARSED" and first_subject:
            # English continuations such as ``likes Y`` and Chinese
            # continuations such as ``也喜欢咖啡`` omit only the subject.
            candidate = first_subject + " " + piece
            part = interpret_statement(candidate, scope=final_scope, when=final_when, task=final_task)
        if part.status != "PARSED":
            return StatementBatchInterpretation(raw, "UNPARSED", scope=final_scope,
                                                when=final_when, task=final_task, marker="component")
        if index == 0:
            first_subject = part.subject
            final_scope = part.scope
            final_task = part.task
            final_when = part.when
        parsed.append(part)
    if len({(part.scope, part.task, part.when) for part in parsed}) != 1:
        return StatementBatchInterpretation(raw, "AMBIGUOUS", scope=final_scope,
                                            when=final_when, task=final_task, marker="context")
    return StatementBatchInterpretation(raw, "PARSED", tuple(parsed), final_scope,
                                        final_when, final_task, "compound")


@dataclass(frozen=True)
class SourceContext:
    """The memory context that a raw source text controls by itself.

    status is CLAUSE when a complete direct clause owns its context,
    ASSIGNED when only scope=/task= assignments apply, and DEFAULT otherwise.
    ignored lists context phrases found in prose and deliberately not applied.
    """
    status: str
    scope: str | None = None
    task: str | None = None
    claims: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()


def source_context(text: str) -> SourceContext:
    """Context gate for raw source text (contract v0.4: prose is not a control).

    Conflicting assignments raise NATURAL_*_AMBIGUOUS; every other ambiguity
    is a no-op that falls back to the assignments or the default context.
    """
    raw = normalize(text)
    scope, rest = _assigned(_ASSIGNED_SCOPE, raw, "SCOPE")
    task, rest = _assigned(_ASSIGNED_TASK, rest, "TASK")
    batch = interpret_statements(raw)
    if batch.status == "PARSED":
        claims = tuple(part.claim_text for part in batch.parts)
        status = "CLAUSE" if batch.scope or batch.task else "DEFAULT"
        return SourceContext(status, batch.scope, batch.task, claims)
    ignored = tuple(match.group(0) for pattern in (_SCOPE, _TASK)
                    for match in pattern.finditer(rest))
    return SourceContext("ASSIGNED" if scope or task else "DEFAULT", scope, task, (), ignored)


# ---------------------------------------------------------------------------
# Discourse working memory
#
# Implicit language is resolved by one bounded register, the analogue of the
# prefrontal maintenance of "who is speaking" and "what proposition is in
# focus".  The register only rewrites a clause into an explicit clause that
# the direct grammar above already accepts.  It never searches memory, never
# guesses a withdrawal target by word overlap and never decides which stance
# is current: history and read-time context keep that authority.

_FIRST_PERSON = frozenset({"i", "me", "我"})
_OBJECT_PRONOUN = frozenset({"it", "that", "this", "them", "它", "它们", "那个", "这个"})
_THIRD_PERSON = {forms[0]: forms[1] for forms in (
    ("like", "likes"), ("love", "loves"), ("prefer", "prefers"), ("support", "supports"),
    ("use", "uses"), ("keep", "keeps"), ("live", "lives"), ("want", "wants"),
    ("work", "works"), ("choose", "chooses"), ("move", "moves"), ("take", "takes"))}
_ELLIPSIS_EN = re.compile(
    r"(?:(?:actually|well)\s*,?\s*)?(?:(?P<subject>[\w.'-]+)\s+)?"
    r"(?:not\s+any\s*more|no\s+longer|(?:do|does)\s*(?:n't|\s+not)(?:\s+any\s*more)?|"
    r"(?:don't|doesn't)(?:\s+any\s*more)?)", re.I)
_ELLIPSIS_ZH = re.compile(r"(?P<subject>[^\s不现已]+?)?(?:现在|已经)?(?:不再|不)了")
_ANYMORE = re.compile(r"\s+any\s*more$", re.I)
_INSTEAD_EN = re.compile(r"(?P<body>.+?)\s+instead(?:\s+of\s+(?P<old>.+))?", re.I)
_INSTEAD_OF_EN = re.compile(r"(?P<head>.+?)\s+instead\s+of\s+(?P<old>.+?)", re.I)
_REPLACE_ZH = re.compile(r"(?P<subject>.+?)(?:现在)?(?:改成|换成|改为|转而)(?P<rest>.+)")
_ZH_VERBS = ("喜欢", "支持", "使用", "保留", "住在", "想要", "选择")


@dataclass(frozen=True)
class Discourse:
    """Bounded session register: speaker and the proposition in focus."""
    speaker: str | None = None
    subject: str | None = None
    verb: str | None = None
    object: str | None = None
    chinese: bool = False
    # The focused proposition's own context.  Elliptic withdrawal and
    # replacement refer to that proposition, so they act in its context;
    # a plain new statement never inherits it.
    scope: str | None = None
    task: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def has_focus(self) -> bool:
        return bool(self.subject and self.verb and self.object)


@dataclass(frozen=True)
class DiscourseResolution:
    raw: str
    status: str
    statements: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()
    # Context bound from the focus (ellipsis/replacement) or from the
    # replacement clause itself; None means the utterance carries no
    # discourse-bound context.
    scope: str | None = None
    task: str | None = None

    def as_dict(self) -> dict:
        value = {"raw": self.raw, "status": self.status,
                 "statements": list(self.statements), "markers": list(self.markers)}
        if self.scope is not None or self.task is not None:
            value.update(scope=self.scope, task=self.task)
        return value


def _third_person(verb: str) -> str:
    return _THIRD_PERSON.get(verb.casefold(), verb)


def _clause(subject: str, verb: str, obj: str, *, chinese: bool, negative: bool = False) -> str:
    if chinese:
        return subject + ("不再" if negative else "") + verb + obj
    lemma = _VERBS.get(verb.casefold(), verb)
    return f"{subject} {'no longer ' if negative else ''}{_third_person(lemma)} {obj}"


def _resolve_person(text: str, speaker: str | None) -> tuple[str, bool]:
    """Bind a first-person subject to the session speaker, preserving the rest."""
    if not speaker:
        return text, False
    match = re.match(r"(?P<lead>(?:on\s+\d{4}-\d{2}-\d{2}\s*[:,]?\s*)?)(?P<pronoun>I|me)\b"
                     r"(?P<rest>.*)", text, re.I)
    if match:
        rest = re.sub(r"\b(?:don't|do\s+not)\b", "does not", match["rest"], count=1, flags=re.I)
        verb = re.match(r"(\s+(?:(?:currently|now|again|no\s+longer)\s+)*)(\w+)", rest, re.I)
        if verb and verb.group(2).casefold() in _THIRD_PERSON:
            rest = rest[:verb.start(2)] + _third_person(verb.group(2)) + rest[verb.end(2):]
        return match["lead"] + speaker + rest, True
    if text.startswith("我"):
        return speaker + text[1:], True
    return text, False


def resolve_discourse(text: str, discourse: Discourse | None = None) -> DiscourseResolution:
    """Rewrite one utterance into explicit direct clauses using the register.

    Supported implicit forms are deliberately few and share one mechanism:
    first-person subjects bind to the speaker; object pronouns and elliptic
    withdrawals bind to the focused proposition; an explicit replacement
    marker ("instead", "instead of", "改成", "换成") expands into the
    withdrawal of the replaced value plus the new assertion.  Any implicit
    form without a focus is UNRESOLVED and mutates nothing.
    """
    discourse = discourse or Discourse()
    raw = normalize(text)
    markers: list[str] = []
    content, bound = _resolve_person(raw, discourse.speaker)
    if bound:
        markers.append("speaker")
    if not content:
        return DiscourseResolution(raw, "UNRESOLVED")
    bare = content.strip(" .!。！")
    # 1. Elliptic withdrawal of the focused proposition.
    ellipsis = _ELLIPSIS_EN.fullmatch(bare) or _ELLIPSIS_ZH.fullmatch(bare)
    if ellipsis:
        if not discourse.has_focus:
            return DiscourseResolution(raw, "UNRESOLVED", markers=("ellipsis",))
        subject = ellipsis["subject"]
        if subject:
            if subject.casefold() not in {discourse.subject.casefold(), "he", "she", "they", "他", "她"}:
                return DiscourseResolution(raw, "UNRESOLVED", markers=("ellipsis",))
        markers.append("ellipsis")
        return DiscourseResolution(raw, "RESOLVED", (_clause(
            discourse.subject, discourse.verb, discourse.object,
            chinese=discourse.chinese, negative=True),), tuple(markers),
            *_focus_context(discourse))
    # 2. Explicit replacement markers.  A context phrase may sit anywhere in
    # the utterance; it is separated first so "instead" is never stored as
    # part of a claim.
    try:
        said_scope, unframed = _scope(bare)
        said_task, unframed = _task(unframed)
    except MemoryInvariantError:
        return DiscourseResolution(raw, "UNRESOLVED", markers=("context",))
    replacement = _INSTEAD_EN.fullmatch(unframed.strip(" .!"))
    if replacement:
        body, old = replacement["body"], replacement["old"]
        new = interpret_statement(_bind_object(body, discourse), scope=said_scope, task=said_task)
        if new.status != "PARSED" or new.action != "assert":
            # "Y instead" may omit subject and verb entirely.
            if discourse.has_focus and old is None and not discourse.chinese and body.strip():
                new = interpret_statement(_clause(discourse.subject, discourse.verb,
                                                  body.strip(), chinese=False),
                                          scope=said_scope, task=said_task)
            if new.status != "PARSED" or new.action != "assert":
                return DiscourseResolution(raw, "UNRESOLVED", markers=("replacement",))
        if old is None:
            if not (discourse.has_focus and not discourse.chinese
                    and discourse.subject.casefold() == new.subject.casefold()
                    and _VERBS.get(discourse.verb.casefold(), discourse.verb) == new.predicate):
                return DiscourseResolution(raw, "UNRESOLVED", markers=("replacement",))
            old = discourse.object
            context = _focus_context(discourse)
            if ((new.scope is not None and new.scope != context[0])
                    or (new.task is not None and new.task != context[1])):
                # "Y instead" in a different context is not a replacement of
                # the focused proposition.
                return DiscourseResolution(raw, "UNRESOLVED", markers=("replacement",))
        else:
            context = (new.scope or "default", new.task or "default")
        old = normalize(old).strip(" .!")
        if not old or old.casefold() == new.object.casefold():
            return DiscourseResolution(raw, "UNRESOLVED", markers=("replacement",))
        markers.append("replacement")
        return DiscourseResolution(raw, "RESOLVED", (
            _clause(new.subject, new.predicate, old, chinese=False, negative=True),
            _clause(new.subject, new.predicate, new.object, chinese=False)), tuple(markers),
            *context)
    zh = _REPLACE_ZH.fullmatch(bare)
    if zh:
        subject, rest = zh["subject"].strip(), zh["rest"].strip()
        verb = next((v for v in _ZH_VERBS if rest.startswith(v)), None)
        if verb is None:
            verb, obj = discourse.verb, rest
        else:
            obj = rest[len(verb):]
        if not (discourse.has_focus and discourse.chinese
                and discourse.subject == subject and discourse.verb == verb
                and obj and discourse.object != obj):
            return DiscourseResolution(raw, "UNRESOLVED", markers=("replacement",))
        markers.append("replacement")
        return DiscourseResolution(raw, "RESOLVED", (
            _clause(subject, verb, discourse.object, chinese=True, negative=True),
            _clause(subject, verb, obj, chinese=True)), tuple(markers),
            *_focus_context(discourse))
    # 3. Object pronoun bound to the focused object.
    bound_text = _bind_object(content, discourse)
    if bound_text != content:
        markers.append("object")
    elif _pronoun_object(content):
        return DiscourseResolution(raw, "UNRESOLVED", markers=("object",))
    if re.search(r"\binstead\b", bound_text, re.I):
        # A replacement marker the register could not bind must not be
        # written into a literal claim.
        return DiscourseResolution(raw, "UNRESOLVED", markers=("replacement",))
    if re.search(r"\b(?:not|n't|no\s+longer)\b", bound_text, re.I):
        bound_text = _ANYMORE.sub("", bound_text.rstrip(" .!"))
    return DiscourseResolution(raw, "RESOLVED", (bound_text,), tuple(markers))


def _focus_context(discourse: Discourse) -> tuple[str, str]:
    return discourse.scope or "default", discourse.task or "default"


def _object_pronoun(text: str):
    """Return (stripped clause, parsed, pronoun) when the object leads with a pronoun."""
    stripped = _ANYMORE.sub("", text.rstrip(" .!"))
    parsed = interpret_statement(stripped)
    if parsed.status != "PARSED":
        return stripped, parsed, None
    head = parsed.object.split()[0] if " " in parsed.object else parsed.object
    for pronoun in sorted(_OBJECT_PRONOUN, key=len, reverse=True):
        if head.casefold() == pronoun or (parsed.object.startswith(pronoun)
                                          and re.search(r"[\u4e00-\u9fff]", pronoun)):
            return stripped, parsed, pronoun
    return stripped, parsed, None


def _pronoun_object(text: str) -> bool:
    return _object_pronoun(text)[2] is not None


def _bind_object(text: str, discourse: Discourse) -> str:
    stripped, parsed, pronoun = _object_pronoun(text)
    if pronoun is None or not discourse.has_focus:
        return text
    start = len(stripped) - len(parsed.object)
    if stripped[start:].casefold() != parsed.object.casefold():
        return text
    return stripped[:start] + discourse.object + parsed.object[len(pronoun):]


def focus_after(parts, discourse: Discourse) -> Discourse:
    """Advance the register to the last parsed proposition."""
    focus = discourse
    for part in parts:
        if part.status != "PARSED":
            continue
        chinese = part.predicate in _ZH_VERBS
        focus = Discourse(discourse.speaker, part.subject, part.predicate, part.object, chinese,
                          part.scope or "default", part.task or "default")
    return focus


def resolve_query_person(text: str, discourse: Discourse | None) -> str:
    """Replace first-person query cues with the bound speaker."""
    if discourse is None or not discourse.speaker:
        return text
    text = re.sub(r"\b(?:I|me|my)\b", discourse.speaker, text)
    return text.replace("我", discourse.speaker)


_QUERY_SUBJECT_ANAPHOR = re.compile(r"\b(?:he|she|they)\b|[他她](?![们])", re.I)
_QUERY_OBJECT_ANAPHOR = re.compile(r"\b(?:it|them)\b|它(?![们])", re.I)


def interpret_query_in_discourse(text: str, discourse: Discourse | None,
                                 bind_focus: bool = True, *,
                                 known_scopes: frozenset[str] | None = None,
                                 known_tasks: frozenset[str] | None = None,
                                 ) -> tuple[QueryInterpretation, dict | None]:
    """Interpret a question inside the session's discourse register.

    One mechanism serves statements and questions: an anaphor refers to the
    proposition in focus, and that proposition acts in its own context.  A
    question that binds the focus (``Does he still like it?``) is therefore
    asked in the focus's scope and task unless it names its own.  A question
    without an anaphor never inherits the focus context, exactly as a plain
    new statement does not.  Only third-person subjects other than the
    speaker and object pronouns are bound; nothing is inferred from overlap.
    """
    text = resolve_query_person(text, discourse)
    known = {"known_scopes": known_scopes, "known_tasks": known_tasks}
    if not bind_focus or discourse is None or not discourse.has_focus:
        return interpret_query(text, **known), None
    bound = []
    speaker = (discourse.speaker or "").casefold()
    if discourse.subject.casefold() != speaker:
        text, count = _QUERY_SUBJECT_ANAPHOR.subn(discourse.subject, text, count=1)
        if count:
            bound.append("subject")
    text, count = _QUERY_OBJECT_ANAPHOR.subn(discourse.object, text, count=1)
    if count:
        bound.append("object")
    interpretation = interpret_query(text, **known)
    if not bound:
        return interpretation, None
    query = interpretation.query
    inherited = []
    if ("scope" not in interpretation.markers and "scope_prose_ignored" not in interpretation.markers
            and (discourse.scope or "default") != "default"):
        query = replace(query, scope=discourse.scope)
        inherited.append("scope")
    if ("task" not in interpretation.markers and "task_prose_ignored" not in interpretation.markers
            and (discourse.task or "default") != "default"):
        query = replace(query, task=discourse.task)
        inherited.append("task")
    markers = interpretation.markers + ("discourse",)
    binding = {"bound": bound, "inherited": inherited, "resolved_text": text}
    return QueryInterpretation(interpretation.raw, query, "EXPLICIT", markers), binding


def focus_after_query(interpretation: QueryInterpretation,
                      discourse: Discourse | None = None,
                      *, speaker: str | None = None) -> Discourse | None:
    """Derive a bounded discourse focus from a direct natural-language query.

    A read is allowed to establish the proposition that a caller is asking
    about, but only when the query reduces to the same single-clause grammar
    used for writes.  Broad questions (``What does Alex like?``), reported
    speech, and modal/compound wording therefore leave the register alone.
    This keeps query interpretation from becoming an unbounded parser or a
    source of inferred claims.
    """
    if not isinstance(interpretation, QueryInterpretation):
        return None
    query = interpretation.query
    candidates: list[str] = []
    # DEFAULT_CURRENT intentionally preserves a bare yes/no question as its
    # retrieval cue.  Strip its grammatical frame only for this bounded
    # focus probe; the public Query text remains unchanged.  A question is
    # read only without its frame: parsed whole, "Does Alex like tea" would
    # make "Does Alex" the subject later pronouns resolve to.  This changes
    # the focus a journal replays to; no journal version marker is added, so
    # a journal written before it fails closed on replay with
    # CORE_JOURNAL_REPLAY_CHANGED rather than silently rebinding (r52 F9,
    # disclosed).
    for source in (query.text, interpretation.raw):
        cue, is_question, _ = _strip_question_frame(source)
        if is_question:
            candidates.append(cue)
            continue
        candidates.append(source)
        stripped = normalize(source).strip(" ??.!！。？")
        stripped = re.sub(r"(?:吗|呢)$", "", stripped).strip()
        if stripped != normalize(source).strip(" ??.!！。？"):
            candidates.append(stripped)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = normalize(candidate).strip(" :：,，;；?？.!！。")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        # ``interpret_query`` removes English current cues when they are
        # explicit controls.  The fallback raw candidate needs the same
        # conservative cleanup, including the common Chinese ``still`` cue.
        candidate = re.sub(r"\b(?:still|currently|now|again)\b", " ", candidate, flags=re.I)
        candidate = re.sub(r"(?:仍然|仍旧|还|现在|目前|当前|再次|重新|又)(?=\S)", " ", candidate)
        candidate = re.sub(r"(?:吗|呢)$", "", candidate).strip(" :：,，;；?？.!！。")
        try:
            parsed = interpret_statement(candidate, scope=query.scope, task=query.task)
        except MemoryInvariantError:
            continue
        if parsed.status != "PARSED" or parsed.identity is None:
            continue
        # A pronoun subject the register could not resolve names nobody; it
        # must not become the subject later pronouns resolve to.
        if _QUERY_SUBJECT_ANAPHOR.fullmatch(parsed.subject or ""):
            continue
        focused_speaker = discourse.speaker if speaker is None and discourse is not None else speaker
        return Discourse(
            speaker=focused_speaker,
            subject=parsed.subject,
            verb=parsed.predicate,
            object=parsed.object,
            chinese=parsed.predicate in _ZH_VERBS,
            scope=parsed.scope or query.scope or "default",
            task=parsed.task or query.task or "default",
        )
    return None


def statement_key(claim_text: str, scope: str = "default", task: str = "default") -> str:
    interpreted = interpret_statement(claim_text)
    identity = interpreted.identity or normalize(claim_text).casefold()
    payload = json.dumps([scope, task, identity], ensure_ascii=False, separators=(",", ":"))
    return "nl:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:28]


def candidate_score(claim_text: str, trace: Trace) -> float:
    """Compatibility helper: only complete identity supplies a target now."""
    left, right = interpret_statement(claim_text), interpret_statement(trace.current.text)
    return float(left.identity is not None and left.identity == right.identity)


def cue_identity_match(cue_text: str, claim_text: str) -> bool | None:
    """Return an exact claim-identity match for a cue when one is parseable.

    Lexical overlap remains the broad retrieval signal.  A direct lifecycle
    cue, however, carries a stronger subject/predicate/object identity and
    must be able to distinguish ``Alex likes tea`` from ``Alex likes cocoa``.
    Yes/no questions are reduced to their declarative cue by the same bounded
    question-frame parser used by query interpretation.  ``None`` means the
    cue or claim is outside the conservative grammar and callers may fall
    back to lexical retrieval.
    """
    if not isinstance(cue_text, str) or not isinstance(claim_text, str):
        return None

    def identity(text: str) -> tuple[str, str, str] | None:
        parsed = interpret_statement(text)
        if parsed.identity is not None:
            return parsed.identity
        cue, is_question, _ = _strip_question_frame(text)
        if not is_question:
            # Chinese yes/no questions use a trailing particle rather than an
            # English auxiliary; retry the same direct-claim grammar after
            # removing only that particle.
            stripped = normalize(text).strip(" ??.!！。？")
            stripped = re.sub(r"(?:吗|呢)$", "", stripped).strip()
            parsed = interpret_statement(stripped)
            return parsed.identity
        parsed = interpret_statement(cue)
        return parsed.identity

    left, right = identity(cue_text), identity(claim_text)
    if left is None or right is None:
        return None
    return left == right


# A sentence break is terminal punctuation followed by a new capitalised
# sentence, or CJK terminal punctuation followed by more text.  A short
# capitalised word before the period ("St. Louis", "Dr. Lee") is an
# abbreviation, not a break.
_SENTENCE_BREAK = re.compile(r"(\S+)[.!?]\s+(?=[A-Z0-9\"'])|[。！？]\s*(?=\S)")


def _sentence_breaks(text: str) -> int:
    breaks = 0
    for match in _SENTENCE_BREAK.finditer(text):
        word = match.group(1)
        if word is not None and len(word) <= 3 and word[:1].isupper():
            continue
        breaks += 1
    return breaks


def natural_slot(text: str) -> tuple[str, str, bool] | None:
    """Return a conservative ``(entity, attribute, exclusive)`` slot.

    This helper is intentionally narrower than :func:`interpret_statement`.
    The lifecycle parser can recognize many direct clauses for exact
    history/withdrawal identity, but recognition alone is not evidence that a
    relation is mutually exclusive.  Only the explicitly enumerated
    single-value predicates receive an automatic conflict slot; all other
    clauses return ``None`` and must use writer-supplied metadata.
    """
    parsed = interpret_statement(text)
    if parsed.status != "PARSED" or parsed.action != "assert":
        return None
    if parsed.predicate not in _NATURAL_EXCLUSIVE_PREDICATES:
        return None
    # A slot belongs to one clause.  A multi-sentence summary ("Joralen moved
    # to the bedroom. Serava journeyed to the bedroom.") would otherwise give
    # its first clause's subject to the whole trace.
    if _sentence_breaks(normalize(text)):
        return None
    # "lives with ...", "lives a quiet life" are not a residence. Only a
    # residence relation ("lives in/at <place>", 住在<place>) is single-valued.
    if parsed.predicate == "live" and not re.match(r"(?:in|at)\s+\S", parsed.object or "", re.I):
        return None
    entity = normalize(parsed.subject or "").casefold()
    if not entity:
        return None
    if parsed.predicate == "move":
        # A move only states a current location when it names a destination.
        # Bare/transitive uses stay ordinary event prose and do not create an
        # exclusive slot without structured writer metadata.
        if not re.match(r"(?:to|into|onto)\s+(?:the\s+)?\S", parsed.object or "", re.I):
            return None
        return entity, "location", True
    if parsed.predicate == "take":
        # Possession language is intentionally narrow.  Requiring ``the``
        # avoids treating idioms such as "took a break" as a holder claim;
        # common event complements remain explicitly non-exclusive.
        match = re.match(r"the\s+(?P<item>\S.+?)$", parsed.object or "", re.I)
        if match is None:
            return None
        item = normalize(match["item"]).casefold()
        if item in {"break", "place", "stage", "floor"}:
            return None
        return item, "holder", True
    if parsed.predicate in {"live", "住在"}:
        # Residence and explicit movement are two surface forms for the same
        # single-valued location relation.  Keeping ``live`` as a separate
        # attribute would let "Alex lives in Austin" coexist with
        # "Alex moved to Boston" without entering the cross-key conflict
        # graph, even though both statements claim the current location.
        return entity, "location", True
    return entity, parsed.predicate, True



# Explicit revision language names the prior proposition it revises: "used to
# like X", "no longer enjoys X", "以前喜欢X", or a labelled value "Budget: B
# (changed from A)".  The new statement itself is the evidence that the earlier
# proposition no longer holds, so it can be bound to an earlier trace that
# states exactly that proposition.  Nothing here infers a withdrawal from a new
# positive value alone ("likes Y" never revises "likes X").
_REVISION_VERBS_POSITIVE = ("like", "likes", "liked", "love", "loves", "loved",
                            "enjoy", "enjoys", "enjoyed", "prefer", "prefers", "preferred")
_REVISION_VERBS_NEGATIVE = ("dislike", "dislikes", "disliked", "hate", "hates", "hated")
_REVISION_STANCE_EN = re.compile(
    r"\b(?:used\s+to|previously|formerly|no\s+longer)\s+(?:really\s+|truly\s+)?"
    r"(?P<verb>" + "|".join(sorted(_REVISION_VERBS_POSITIVE + _REVISION_VERBS_NEGATIVE,
                                    key=len, reverse=True)) + r")\s+"
    r"(?P<object>[^,;.!?()\"]+?)"
    r"(?=\s+(?:but|anymore|any\s+more|now|after|since|because|as|when|then)\b|[,;.!?()\"]|$)",
    re.I)
_REVISION_STANCE_ZH = re.compile(
    r"(?:以前|曾经|过去|不再)(?P<verb>喜欢|爱|讨厌|支持|使用)(?P<object>[^，,。；;！!？?、]+?)"
    r"(?=了|，|,|。|；|;|！|!|？|\?|、|但|现在|$)")
_REVISION_VALUE = re.compile(
    r"^(?P<label>[^:\n()]{1,48}):\s*(?P<value>[^()\n]+?)\s*"
    r"\((?:changed|updated|revised)\s+from\s+(?P<old>[^()\n]+?)\)\s*\.?$", re.I)
_REVISION_REFERENT = frozenset({
    "it", "them", "this", "that", "these", "those", "him", "her", "his", "its", "their",
    "one", "ones", "things", "stuff", "anything", "everything", "something", "much", "more"})
# Articles are dropped; a possessive ("his films") is anaphoric and names no
# single prior proposition, so it is kept and rejected as a referent.
_REVISION_DETERMINER = re.compile(r"^(?:the|a|an)\s+", re.I)
_REVISION_QUOTES = re.compile(r"[\"\u201c\u201d]")
_REVISION_QUOTES_ZH = re.compile(r"[\u300c\u300d\u300e\u300f\u2018\u2019]")
_REVISION_TOKEN = re.compile(r"[\w$][\w$,'.-]*\w|[\w$]", re.U)


@dataclass(frozen=True)
class RevisionReference:
    """A prior proposition that a statement explicitly says no longer holds."""

    kind: str          # "stance" or "value"
    polarity: str      # "positive"/"negative" for stance, "value" for value
    target: str        # object phrase (stance) or old value (value)
    label: str = ""    # attribute label for a value revision
    # The revising statement's own words before the revision marker in its
    # sentence (English) or clause (Chinese): "Bob" in "Bob no longer likes
    # tea after meeting Alice".  Only the subject read from these words may
    # be compared with the prior trace's subject; a name mentioned elsewhere
    # in the revising text is not the entity whose proposition is revised.
    subject: str = ""


def revision_tokens(text: str) -> list[str]:
    return [token.casefold() for token in _REVISION_TOKEN.findall(normalize(text))]


# A revision marker is the statement's own assertion only when it stands
# outside quotation marks, its sentence is not a question, and no listed
# hedge, condition or modal stands right before it ("might have finished",
# "if Alice no longer likes tea", "will have completed").  These lists are a
# first filter only and are not complete: a hedge or modal separated from the
# marker ("will soon no longer like", "probably finished", "expects to have
# finished") is caught structurally instead, because the words before the
# marker must then be exactly the prior trace's subject phrase
# (conflict._subject_phrase), and a subject phrase never contains them.
_SENTENCE_START = re.compile(r"[.!?;]\s+|[\u3002\uff1b\uff01\uff1f\n]")
_SENTENCE_END = re.compile(r"[.!?;](?=\s|$)|[\u3002\uff1b\uff01\uff1f\n]")
_CLAUSE_START_ZH = re.compile(r"[\uff0c,\u3002\uff1b;\uff01!\uff1f?\u3001\n]")
_MODAL_BEFORE = re.compile(
    r"\b(?:will|would|shall|should|could|can|may|might|must)\s+(?:not\s+)?(?:have\s+)?$", re.I)
_HEDGED_ZH = re.compile(
    r"(?:\u53ef\u80fd|\u4e5f\u8bb8|\u6216\u8bb8|\u5927\u6982|\u5982\u679c|\u5047\u5982|\u8981\u662f|\u662f\u5426)[^\u3002\uff1b;\uff01!\uff1f?]*$")
_QUOTE_MARKS = ('"', "\u201c", "\u201d")
_QUOTE_MARKS_ZH = ("\u300c", "\u300d", "\u300e", "\u300f")


def _prefix(text: str, start: int, boundary: re.Pattern) -> str:
    """TEXT from the last BOUNDARY before START up to START."""
    cut = 0
    for match in boundary.finditer(text, 0, start):
        cut = match.end()
    return text[cut:start]


def _quoted(text: str, start: int, marks: tuple[str, ...]) -> bool:
    return sum(text.count(mark, 0, start) for mark in marks) % 2 == 1


def _question(text: str, start: int) -> bool:
    """The sentence holding START ends with a question mark."""
    end = _SENTENCE_END.search(text, start)
    return end is not None and end.group() in ("?", "\uff1f")


def _asserted(clean: str, start: int) -> str | None:
    """The sentence prefix before an English revision marker at START, or
    None when the marker is quoted, asked, hedged, conditional or under a
    modal."""
    if _quoted(clean, start, _QUOTE_MARKS) or _question(clean, start):
        return None
    before = _prefix(clean, start, _SENTENCE_START)
    if _DROP_HEDGED.search(before) or _MODAL_BEFORE.search(before):
        return None
    return before


def revision_references(text: str) -> tuple[RevisionReference, ...]:
    """Return the prior propositions this statement explicitly revises.

    Only explicit revision syntax counts.  A pronoun or overly long object
    cannot identify one prior proposition and is ignored, and so is a marker
    that the statement does not itself assert (quoted, hedged, conditional or
    modal).  Each reference carries the words before its marker, from which
    the revising subject is read.
    """
    if type(text) is not str:
        return ()
    clean = normalize(text)
    refs: list[RevisionReference] = []
    value = _REVISION_VALUE.match(clean)
    if value is not None:
        label = normalize(value["label"]).casefold()
        old = normalize(value["old"]).strip(" .").casefold()
        if label and old and old != normalize(value["value"]).strip(" .").casefold():
            refs.append(RevisionReference("value", "value", old, label, label))
    # Quotation marks delimit a named object ('used to like "prestige
    # pictures"'); they are not clause boundaries.  They are blanked rather
    # than removed so that positions in UNQUOTED are positions in CLEAN.
    unquoted = _REVISION_QUOTES.sub(" ", clean)
    for match in _REVISION_STANCE_EN.finditer(unquoted):
        subject = _asserted(clean, match.start())
        if subject is None:
            continue
        obj = _REVISION_DETERMINER.sub("", normalize(match["object"]).strip())
        tokens = revision_tokens(obj)
        # "crime and investigation books" is one coordinated object; only a
        # dangling coordinator ("X and now ...") is trimmed.
        while tokens and tokens[-1] in ("and", "or"):
            tokens.pop()
        if (not tokens or len(tokens) > 6 or tokens[0] in _REVISION_REFERENT
                or not any(len(t) >= 3 and t not in _REVISION_REFERENT for t in tokens)):
            continue
        polarity = ("negative" if match["verb"].casefold() in _REVISION_VERBS_NEGATIVE
                    else "positive")
        refs.append(RevisionReference("stance", polarity, " ".join(tokens),
                                      subject=normalize(subject)))
    unquoted_zh = _REVISION_QUOTES_ZH.sub(" ", unquoted)
    for match in _REVISION_STANCE_ZH.finditer(unquoted_zh):
        if (_quoted(clean, match.start(), _QUOTE_MARKS + _QUOTE_MARKS_ZH)
                or _question(clean, match.start())):
            continue
        if _HEDGED_ZH.search(_prefix(clean, match.start(), _SENTENCE_START)):
            continue
        obj = normalize(match["object"]).strip()
        if not obj or obj in {"它", "它们", "他", "她", "这个", "那个"}:
            continue
        polarity = "negative" if match["verb"] == "讨厌" else "positive"
        subject = normalize(_prefix(unquoted_zh, match.start(), _CLAUSE_START_ZH))
        refs.append(RevisionReference("stance", polarity, obj.casefold(), subject=subject))
    # A fulfilled intention ("finished reviewing the board report") ends the
    # prior intention ("needs to review the board report"): prospective
    # memory deactivates on completion, while the history keeps both.
    for match in _COMPLETION_EN.finditer(unquoted):
        if _COMPLETION_NEGATED.search(unquoted[max(0, match.start() - 24):match.start()]):
            continue
        subject = _asserted(clean, match.start())
        if subject is None:
            continue
        action = intention_action(match["object"])
        if action is not None:
            refs.append(RevisionReference("intention", "completed", action,
                                          subject=normalize(subject)))
    # Dropping an intention ('removed "Plan date night" from their to-do
    # list', 'no longer needs to track "X"', 'no longer plans to attend X')
    # ends it as well, without asserting that it was done.  A hedged or
    # negated drop ("might remove", "didn't remove") ends nothing.
    for pattern, text_form in ((_DROP_LISTED_EN, clean), (_DROP_TRACKED_EN, clean),
                               (_DROP_PLANNED_EN, unquoted)):
        for match in pattern.finditer(text_form):
            before = text_form[max(0, match.start() - 40):match.start()]
            if _COMPLETION_NEGATED.search(before):
                continue
            subject = _asserted(clean, match.start())
            if subject is None:
                continue
            action = intention_action(match["object"])
            if action is not None:
                refs.append(RevisionReference("intention", "dropped", action,
                                              subject=normalize(subject)))
    unique: list[RevisionReference] = []
    for ref in refs:
        if ref not in unique:
            unique.append(ref)
    return tuple(unique)


# Completion and intention phrases name the same action in different verb
# forms ("reviewing" / "review"), so both are reduced to the same canonical
# action: content words, crudely stemmed, without time, determiners or
# possessives.  Equality of the whole action is required; a shared word is not
# enough.
_ACTION_END = (r"(?=\s+(?:and|but|so|as|because|before|after|yesterday|today|tonight|earlier|"
               r"already|reported|noted|mentioned|which|that|while|ahead)\b|[,;.!?()\"]|$)")
_COMPLETION_EN = re.compile(
    r"\b(?:has\s+|have\s+|had\s+)?(?:completed|finished|(?:is|was|were|are)\s+done\s+with|"
    r"done\s+with)\s+(?:(?:the|their|his|her|my)\s+(?:work\s+|personal\s+)?"
    r"(?:task|to-?do(?:\s+item)?)\s+(?:to\s+|of\s+)?)?"
    r"(?P<object>[^,;.!?()\"]+?)" + _ACTION_END, re.I)
_COMPLETION_NEGATED = re.compile(r"(?:n't|\bnot|\bnever|\byet)\s*(?:\w+\s+){0,2}$", re.I)
_DROP_QUOTED = r"[\"\u201c'](?P<object>[^\"\u201d']{3,80})[\"\u201d']"
_DROP_LISTED_EN = re.compile(
    r"\b(?:remov(?:e|es|ed|ing)|delet(?:e|es|ed|ing)|took|tak(?:e|es|ing)|"
    r"cross(?:es|ed|ing)?|drop(?:s|ped|ping)?|cancel(?:s|led|ed|ling|ing)?)\s+"
    # Only removal from the user's own task list ends an intention; removing
    # a recipient or a key point from a document does not.
    + _DROP_QUOTED + r"\s+(?:off|from)\s+(?:(?:their|his|her|my|your|the)\s+)?"
    r"(?:(?:work|personal)\s+)?(?:to-?do|tasks?\b|list\b|calendar|plate)", re.I)
_DROP_TRACKED_EN = re.compile(
    r"\b(?:no\s+longer|stopped|won'?t\s+be|will\s+not\s+be)\s+(?:(?:needs?|wants?)\s+to\s+)?"
    r"track(?:s|ing)?\s+" + _DROP_QUOTED, re.I)
_DROP_PLANNED_EN = re.compile(
    r"\b(?:no\s+longer\s+(?:needs?|plans?|planning|wants?|intends?|going)\s+to|"
    r"(?:decided|decides|chose|chooses|choosing)\s+not\s+to)\s+"
    # "no longer needs to track X" names X as the dropped item (above).
    r"(?!track(?:s|ing)?\b)(?P<object>[^,;.!?()\"]+?)(?=\s+(?:anymore|any\s+more)\b|"
    + _ACTION_END[3:], re.I)
_DROP_HEDGED = re.compile(
    r"\b(?:might|may|maybe|perhaps|consider\w*|thinking\s+about|whether|if|not\s+sure)\b[^.;]*$",
    re.I)
_INTENTION_EN = re.compile(
    r"\b(?:still\s+)?(?:needs?|has|have|had|plans?|planning|wants?|intends?)\s+to\s+"
    r"(?P<object>[^,;.!?()\"]+?)" + _ACTION_END, re.I)
_INTENTION_LISTED = re.compile(
    r"\badded\s+\"(?P<object>[^\"]+)\"\s+to\s+(?:their|his|her|my|the)\s+"
    r"(?:work\s+|personal\s+)?(?:to-?do|task|work)", re.I)
_ACTION_FILLER = frozenset({
    "the", "a", "an",
    "today", "tonight", "tomorrow", "soon", "later", "now", "this", "next", "week", "day",
    "eventually", "still", "also", "really", "work", "task", "tasks", "item", "list",
    "to-do", "todo", "of", "on", "end"})
# Prepositions that mark an argument's role ("email the report to Bob" is not
# "... for Bob").  They are kept when an argument follows them; one left with
# no argument after the filler words ("for work", "by end of day") marks none.
_ACTION_ROLE = frozenset({"to", "for", "by", "with", "from"})
# Words that can stand before the action verb but never head the phrase:
# articles, a demonstrative, temporal adverbs and nouns, and the two
# prepositions of the filler set.  The remaining filler words ("end",
# "work", "list", ...) are fillers only after the verb; as the first
# word they are the action itself ("end the planning session").
_ACTION_PRE_HEAD = frozenset({
    "the", "a", "an", "this", "today", "tonight", "tomorrow", "soon", "later",
    "now", "next", "week", "day", "eventually", "still", "also", "really",
    "of", "on"})
assert _ACTION_PRE_HEAD <= _ACTION_FILLER
# The two prepositions of the filler set, and the filler nouns that are
# fillers only inside an adjunct (intention_action).
_ACTION_PREPOSITION = frozenset({"of", "on"})
_ACTION_ADJUNCT_NOUN = _ACTION_FILLER - _ACTION_PRE_HEAD
assert _ACTION_PREPOSITION <= _ACTION_PRE_HEAD
# Determiners that say whose or how much of an argument is meant.  They stay
# in the canonical action and are compared like roles.  Two different
# possessives ("their parents", "his parents"; "my draft", "her draft") or
# two different quantifiers never name the same argument; a possessive stated
# by one form only is compatible.  A partial quantifier ("some", "any") limits
# what was done, so a completion that states one closes only an intention
# that states the same one ("answered some client emails" does not close
# "answer the client emails"), while a completion without one closes an
# intention stated with one ("sign off on some major contracts" is closed by
# "signed off on major contracts").
_ACTION_QUANTIFIER = frozenset({"some", "any", "all"})
_ACTION_PARTIAL = frozenset({"some", "any"})
_ACTION_POSSESSIVE = frozenset({"my", "our", "your", "their", "his", "her", "its"})
_ACTION_DETERMINER = _ACTION_QUANTIFIER | _ACTION_POSSESSIVE
# A noun right after a determiner heads a noun phrase (intention_action).
# The determiners are a closed class: articles, demonstratives, the
# possessives and quantifiers above, and the distributive and negative
# determiners; a genitive name ("Bob's") is a determiner too
# (_action_noun_marker).  This differs from conflict._noun_phrase_ends,
# which requires a capitalised name and also accepts a bare trailing
# apostrophe (r57 F3): the tokens here are lower-cased, so a contraction
# ("it's", "that's", "let's") also matches and only keeps a filler in the
# action (narrower, never a new equality), and a plural genitive ("James'")
# is never seen because the tokenizer drops its apostrophe (a disclosed
# known limit, test_language_checker_absent_shapes_are_disclosed_known_limits).
# Only the articles and "this" are themselves fillers; the others stay in
# the action as written.
_ACTION_NOUN_MARKER = (frozenset({"the", "a", "an", "this", "that", "these", "those",
                                  "each", "every", "no", "another", "either", "neither",
                                  "both"})
                       | _ACTION_DETERMINER)
assert _ACTION_NOUN_MARKER & _ACTION_FILLER <= _ACTION_PRE_HEAD
_ACTION_GENITIVE = re.compile(r"\w(?:'s|\u2019s)$")
# "end of" opens a time label only when the noun it governs (articles,
# determiners, genitives and time modifiers aside) is a time word that ends
# the noun phrase: "by the end of the day", "by the end of next week's
# sprint", "by end of day next week".  Any other noun keeps "end" as the
# preposition's object, so "working on the end of the report" is not
# "working on the report" (r68 Claude F1), and so does a time word that
# modifies a noun after it: "the end of the year report" is part of the
# report (r69 Claude F1), as is "the end of the day list".  Time words and
# modifiers in a row, a time genitive among them ("day next week", "next
# week's sprint"), form one label, and the word after the last of them
# decides (r71 Claude F1: "the end of the night shift report" is part of the
# report): the label may be followed only by a role preposition, a word that
# can stand before an action (articles, time adverbs and nouns), or a word
# that opens another phrase or clause (_ACTION_TIME_LABEL_NEXT).
_ACTION_TIME_NOUN = frozenset({
    "day", "days", "week", "weeks", "month", "months", "quarter", "quarters", "year", "years",
    "semester", "term", "season", "weekend", "morning", "afternoon", "evening", "night",
    "shift", "sprint", "today", "tonight", "tomorrow"})
_ACTION_TIME_MODIFIER = frozenset({"next", "last", "coming", "current", "following", "same"})
_ACTION_TIME_LABEL_NEXT = frozenset({"in", "at", "before", "after", "until", "and", "or",
                                     "so", "then", "because", "when", "if", "as"})


def _action_noun_marker(word: str) -> bool:
    return word in _ACTION_NOUN_MARKER or bool(_ACTION_GENITIVE.search(word))


def _action_time_word(word: str) -> bool:
    """WORD is a closed time word, or its genitive ("week's")."""
    return word in _ACTION_TIME_NOUN or (
        bool(_ACTION_GENITIVE.search(word)) and word[:-2] in _ACTION_TIME_NOUN)


def _action_time_label(rest: list[str], index: int) -> bool:
    """REST[INDEX] is "end" opening a time label ("end of the day")."""
    if rest[index] != "end" or rest[index + 1:index + 2] != ["of"]:
        return False
    position, seen = index + 2, False
    while position < len(rest):
        w = rest[position]
        if _action_time_word(w):
            seen = True
        elif not (w in _ACTION_TIME_MODIFIER or (not seen and _action_noun_marker(w))):
            break
        position += 1
    if not seen:
        return False
    after = rest[position:position + 1]
    return not after or (after[0] in _ACTION_ROLE or after[0] in _ACTION_PREPOSITION
                         or after[0] in _ACTION_PRE_HEAD or after[0] in _ACTION_TIME_LABEL_NEXT)


def _action_stem(word: str) -> str:
    word = word.casefold()
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    else:
        for suffix in ("ing", "ed", "es", "s"):
            if len(word) > len(suffix) + 2 and word.endswith(suffix) and not word.endswith("ss"):
                word = word[:-len(suffix)]
                break
    word = word.rstrip("e") or word
    if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def intention_action(phrase: str) -> str | None:
    """Canonical action named by a completion or intention phrase, else None.

    The content words keep their order (and repeats) and the role
    prepositions that link them, so the action's arguments stay distinct:
    "send Bob to Carol" is not "send Carol to Bob", "send it to Bob" is not
    "send it for Bob".  Only the action verb (the first word) is reduced to
    its stem; an argument is compared as written, so "James" never becomes
    "Jam".  Possessive and quantifier determiners stay in place
    (same_intention_action compares them)."""
    tokens = revision_tokens(_REVISION_QUOTES.sub("", phrase))
    head = next((i for i, w in enumerate(tokens) if w not in _ACTION_PRE_HEAD), None)
    if head is None:
        return None
    # The verb is identified before filler removal so a filler word that
    # heads the phrase stays the action.  A filler noun ("work", "list",
    # "end", ...) outside an adjunct is the object itself when it is the
    # first word after the verb (determiners aside) or follows an article or
    # determiner: "send the list to Bob" is not "send the work to Bob",
    # "hand in the list" is not "hand in the work", "send Bob the list" is
    # not "send Bob the work", and "review the end of the contract" is not
    # "review the contract".  After a content word it labels the object ("the
    # Review SEO performance to-do item"), and inside an adjunct ("by end of
    # day", "for work") it is a filler.  A preposition opens one phrase and
    # licenses no deletion by itself: inside an adjunct a filler noun after
    # an article or determiner is still the preposition's object ("work on
    # the list for Bob" is not "work on the task for Bob", r67 Pro F1),
    # except "end" opening a time label ("by the end of the day"; the noun
    # after "of" must be a time word, _action_time_label, r68 Claude F1).
    # When "end of" is kept as an object, the time words of its noun phrase
    # stay too, so "the end of the day report" is not "the end of the
    # report" (r71 Claude F4).
    words, adjunct, previous = [tokens[head]], False, tokens[head]
    end_object = False
    rest = tokens[head + 1:]
    for index, w in enumerate(rest):
        adjunct = adjunct or w in _ACTION_ROLE or w in _ACTION_PREPOSITION
        time_label = _action_time_label(rest, index)
        heads_object = w in _ACTION_ADJUNCT_NOUN and (
            (not adjunct and (_action_noun_marker(previous)
                              or all(x in _ACTION_DETERMINER for x in words[1:])))
            or (adjunct and _action_noun_marker(previous) and not time_label))
        kept_time = end_object and (_action_time_word(w) or w in _ACTION_TIME_MODIFIER)
        if w not in _ACTION_FILLER or heads_object or kept_time:
            words.append(w)
        if w == "end" and heads_object and rest[index + 1:index + 2] == ["of"]:
            end_object = True
        elif end_object and not (w == "of" or kept_time or _action_noun_marker(w)):
            end_object = False
        previous = w
    words = [w for i, w in enumerate(words)
             if w not in _ACTION_ROLE
             or (i + 1 < len(words) and words[i + 1] not in _ACTION_ROLE)]
    if (len(words) < 2 or words[0] in _REVISION_REFERENT or words[0] in _ACTION_ROLE
            or words[0] in _ACTION_DETERMINER
            or len([w for w in words if w not in _ACTION_DETERMINER]) > 8):
        return None
    return " ".join([_action_stem(words[0])] + words[1:])


def same_intention_action(first: str, second: str) -> bool:
    """Whether completion action SECOND closes intention action FIRST (both
    canonical, intention_action).

    The content words must agree in order.  A role preposition, a quantifier
    and a possessive before a word are compared only where both forms state
    one: "send the report to Bob" is not "send the report for Bob", but
    "schedule car service" and "schedule car for service" name one action.
    A partial quantifier the completion states must also be stated by the
    intention.
    A possessive or quantifier with no content word after it before a role
    or the end ("call her") is the argument itself."""
    def parts(action: str) -> tuple[list[str], list[dict[str, str]]]:
        tokens = action.split()
        words: list[str] = []
        slots: list[dict[str, str]] = []
        slot: dict[str, str] = {}
        for index, word in enumerate(tokens):
            following = tokens[index + 1:]
            if word in _ACTION_ROLE:
                slot["role"] = word
                continue
            if (word in _ACTION_DETERMINER and following
                    and any(item not in _ACTION_DETERMINER for item in following[:2])
                    and following[0] not in _ACTION_ROLE):
                slot["quantifier" if word in _ACTION_QUANTIFIER else "possessive"] = word
                continue
            words.append(word)
            slots.append(slot)
            slot = {}
        return words, slots

    def compatible(intended: dict[str, str], done: dict[str, str]) -> bool:
        for name in ("role", "quantifier", "possessive"):
            if name in intended and name in done and intended[name] != done[name]:
                return False
        return done.get("quantifier") not in _ACTION_PARTIAL or "quantifier" in intended

    first_words, first_slots = parts(first)
    second_words, second_slots = parts(second)
    return first_words == second_words and all(
        compatible(a, b) for a, b in zip(first_slots, second_slots))


# Coordinating conjunctions are a closed grammatical class; "as well as",
# "along with" and "together with" coordinate the same way.  An action or
# object followed by one of them has a second part.
_COORDINATED = re.compile(
    r"\s*,?\s*(?:and|or|but|nor|yet|plus|so|as\s+well\s+as|along\s+with|together\s+with)\b",
    re.I)


def intention_matches(text: str) -> list[tuple[str, str, str, int]]:
    """(canonical action, sentence words before the intention verb, the
    normalised text matched, the verb's offset in it) for each action a plain
    statement presents as still to be done."""
    found: list[tuple[str | None, str, str, int]] = []
    clean = _REVISION_QUOTES.sub("", normalize(text))
    # "needs to review the terms and sign them" or "... but also call Bob"
    # is one intention with two parts; completing one part does not close it.
    found += [(intention_action(m["object"]), _prefix(clean, m.start(), _SENTENCE_START),
               clean, m.start())
              for m in _INTENTION_EN.finditer(clean)
              if not _COORDINATED.match(clean[m.end():])]
    plain = normalize(text)
    found += [(intention_action(m["object"]), _prefix(plain, m.start(), _SENTENCE_START),
               plain, m.start())
              for m in _INTENTION_LISTED.finditer(plain)]
    return [match for match in found if match[0] is not None]


def intention_actions(text: str) -> list[str]:
    """Canonical actions a plain statement presents as still to be done."""
    return [match[0] for match in intention_matches(text)]
