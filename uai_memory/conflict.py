"""Explicit exclusive slots and explicit revisions; lexical opposition is a hint."""
from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from functools import lru_cache

from .language import (_COORDINATED, _INTENTION_EN, _INTENTION_LISTED, RevisionReference,
                       _REVISION_QUOTES, _REVISION_TOKEN, _REVISION_VERBS_NEGATIVE,
                       _REVISION_VERBS_POSITIVE,
                       intention_actions, intention_matches, normalize, revision_tokens,
                       same_intention_action)
from .stance import currently_valid
from .types import Trace


def incompatible(left: Trace, right: Trace) -> bool:
    return bool(left.exclusive and right.exclusive and left.entity and left.attribute
                and (left.entity, left.attribute, left.current.scope, left.current.task)
                == (right.entity, right.attribute, right.current.scope, right.current.task)
                and left.current.text != right.current.text
                and currently_valid(left) and currently_valid(right))


def opposition_hint(left: str, right: str) -> bool:
    # A hint must never be promoted to an automatic semantic contradiction.
    negators = {"not", "never", "no", "neither", "without"}
    return bool(set(left.casefold().split()) & negators) != bool(set(right.casefold().split()) & negators)


# A trace that itself carries negation, revision or hedging is not a plain
# statement of the prior proposition and is never bound by a revision.
_NOT_PLAIN_EN = re.compile(
    r"n't\b|\b(?:no\s+longer|used\s+to|previously|formerly|not|never|anymore|"
    r"changed|updated|revised|shifted|revers\w*|replac\w*|instead|maybe|might|perhaps|if|"
    r"over|than|rather)\b", re.I)
_NOT_PLAIN_ZH = re.compile(r"不|没|以前|曾经|过去|改成|换成")
_DEICTIC_ZH = frozenset({"我", "你", "您", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
                         "这", "那", "这个", "那个"})
_DEICTIC_ZH_HEAD = frozenset(word[0] for word in _DEICTIC_ZH)
_POSITIVE_ZH = ("喜欢", "爱", "支持", "使用")
_NEGATIVE_ZH = ("讨厌",)
_FAN_OF = ("fan", "of")
# The revised object must be the verb's own object: at most this many tokens
# ("director", "movies by", "movies that are") may separate verb and object.
_OBJECT_GAP = 3
# Verb and object must share one clause: "loves jazz; Ella is a favourite"
# does not state "loves Ella".  The full stop after a single capital is an
# initial ("Cecil B. DeMille"), and the one after a title is an abbreviation
# ("Mr. Smith"); neither is a sentence end (_clause_breaks).
_CLAUSE = re.compile(r"[;,!?]|(?<!\b[A-Z])\.(?:\s|$)")
# The revised proposition must be the trace's own main clause.  Only a time or
# session prefix ("In session 12 (2025-06-01)," / "On 2025-08-31,") may precede
# it; a multi-topic summary is never withdrawn because one of its parts is.
_PREFIX_WORDS = frozenset({"in", "on", "at", "as", "of", "session", "sessions", "date"})
_PREFIX_NUMBER = re.compile(r"^[\d][\d/.,-]*$")


def _prefix_clause(words: list[str]) -> bool:
    return all(word in _PREFIX_WORDS or _PREFIX_NUMBER.match(word) for word in words)


def _contains(tokens: list[str], phrase: list[str]) -> int:
    """Index of the first contiguous occurrence of PHRASE in TOKENS, else -1."""
    for start in range(len(tokens) - len(phrase) + 1):
        if tokens[start:start + len(phrase)] == phrase:
            return start
    return -1


# An intention already reported as fulfilled, dropped or negated is not a
# pending intention and cannot be closed again.
_NOT_PENDING_EN = re.compile(
    r"n't\b|\b(?:not|never|no\s+longer|completed|finished|done|removed|deleted|cancel\w*|"
    r"already|anymore|if|maybe|might|perhaps)\b", re.I)


def _subject(words: list[str]) -> str | None:
    for word in words:
        if word not in _PREFIX_WORDS and not _PREFIX_NUMBER.match(word):
            return word
    return None


# Articles, deictic pronouns and the indefinite compounds (some/any/no/every +
# one/body/thing, also hyphenated) point into a context or at no one in
# particular; they never name an entity, so they cannot make evidence "about
# another entity".
_ARTICLES = frozenset({"the", "a", "an"})
_DEICTIC = frozenset({"i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
                      "she", "her", "it", "its", "they", "them", "their", "this", "that",
                      "these", "those"}) | frozenset(
    a + sep + b for a in ("some", "any", "no", "every") for sep in ("", "-")
    for b in ("one", "body", "thing"))
# Personal, demonstrative and indefinite pronouns anywhere in a quote may
# refer to the withdrawn statement's entity ("Serava took it from her",
# "Serava took that"); without coreference such a quote cannot be shown to be
# about another entity.  "that" is often a complementiser, which only leaves
# more quotes to the writer.
_PRONOUN = _DEICTIC
# Calendar names, seasons, times of day, time determiners/units and their
# abbreviations stand in a sentence's time prefix ("Last Friday", "On Monday
# morning", "3 pm").  They are skipped like the session prefix, but see
# _from_head: after such a word the subject must be written as a name.  Of the
# other time words, "date" is a session prefix word (_PREFIX_WORDS: skipped
# without the name check), today/tonight/yesterday/tomorrow are sentence
# adverbials (_ADVERBIAL), and dated/time/timestamp are not skipped.
_TIME_UNITS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
               "morning", "afternoon", "evening", "night", "day", "week", "weekend",
               "month", "year", "spring", "summer", "autumn", "fall", "winter")
_TIME_WORDS = frozenset(_TIME_UNITS) | frozenset(unit + "s" for unit in _TIME_UNITS) | frozenset(
    {"january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
     "sept", "oct", "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
     "sat", "sun", "am", "pm", "utc", "last", "next", "past", "early", "late", "noon",
     "midnight"})
# Clitics attached by the shared tokenizer ("i've", "they're", "don't") and
# commas inside a token ("friday,i") are split off here, so a pronoun or a
# name is seen as itself.
_CLITIC = re.compile(r"(?:n't|['](?:ve|m|re|d|ll))$", re.I)
# Quantifying determiners and free-relative pronouns (a closed class): at the
# head of a subject they make the sentence about a set or about no one in
# particular ("Every guest left", "No one took it", "Most of us quit"), so the
# subject is unresolved and the writer keeps the call.  Inside an object noun
# phrase they are ordinary words ("took the most valuable gem").
_QUANTIFIER = frozenset({"no", "none", "one", "each", "every", "all", "both", "either", "neither",
                         "some", "any", "few", "many", "several", "most", "much", "half",
                         "another", "other", "such", "enough", "whoever", "whatever",
                         "whichever"})
_POSSESSIVE = re.compile(r"(?:'s|\u2019s|')$")
# Sentence adverbials and coordinators modify the whole main clause and stand
# before its subject; they name no entity ("Then I quit tea", "Lately the lamp
# was sold").  Subordinators (after/when/while ...) are excluded: the word
# after them is the subject of a subordinate clause, so the referent stays
# unknown and the writer keeps the call.  Focus particles ("only") change the
# claim and are excluded too.  The list is closed: a suffix rule would also
# skip names such as Emily or Holly.  Frequency and negation adverbs stand in
# the same place ("Never went back to pilates", "Always skipped breakfast"):
# they open subject-less statements and are followed by a verb, so read as a
# subject they would look like a name to the frame rule.
_ADVERBIAL = frozenset({"then", "now", "later", "afterwards", "so", "but", "and", "also", "still",
                        "anyway", "meanwhile", "today", "tonight", "yesterday", "tomorrow",
                        "again", "well", "yes", "oh", "ok", "okay", "lately",
                        "recently", "finally", "actually", "honestly", "really", "usually",
                        "eventually", "basically", "clearly", "apparently", "obviously",
                        "fortunately", "unfortunately", "originally", "currently",
                        "initially", "previously", "suddenly", "nowadays",
                        "never", "always", "often", "sometimes", "rarely", "seldom",
                        "ever", "not", "frequently", "occasionally", "normally", "generally"})
# "Label: text" -- a speaker label, a heading or a value slot ("Melanie: I
# quit", "Note: ...", "Favourite tea: sencha").  Which one it is cannot be
# read from the words, so a labelled text has no resolvable subject.
_LABELLED = re.compile(r"^\s*[^\W\d][\w'.-]*(?:\s+[^\W\d][\w'.-]*)?\s*:\s+\S", re.S)
# A title stands before a name and is not itself the entity ("Dr Okafor").
_HONORIFIC = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "sir", "madam"})


def _clause_breaks(text: str) -> list[re.Match]:
    """_CLAUSE matches of TEXT, without the full stop of a title ("Mr.
    Smith"): a capitalised title word right before it and a capitalised word
    right after it ("hard to miss. Bob ..." ends a sentence)."""
    def title_stop(match: re.Match) -> bool:
        before = re.search(r"\b([A-Za-z]+)$", text[:match.start()])
        return bool(before and before[1][:1].isupper() and before[1].casefold() in _HONORIFIC
                    and re.match(r"\s*[A-Z]", text[match.end():]))
    return [match for match in _CLAUSE.finditer(text)
            if not (match.group()[0] == "." and title_stop(match))]


# A sentence that opens with a subordinate clause ("When Serava arrived, ...")
# or a focus particle ("Only Serava went ...") does not put the main-clause
# subject first; its referent is unknown and the writer keeps the call.  The
# check runs before the time-prefix skip, so "As Serava arrived" and "As of
# May, ..." both stay unresolved.
_UNRESOLVED_LEAD = frozenset({"after", "as", "before", "when", "while", "since", "once", "if",
                              "because", "although", "though", "until", "unless",
                              "whenever", "only", "even", "just"})


def _skippable(word: str) -> bool:
    """A word that stands before a sentence's subject without naming it."""
    return (word in _PREFIX_WORDS or bool(_PREFIX_NUMBER.match(word)) or word in _ARTICLES
            or word in _TIME_WORDS or word in _ADVERBIAL or word.rstrip(".") in _HONORIFIC)


def _modifier(word: str) -> bool:
    """A skippable word that may also open a sentence with no subject
    ("Today gave up yoga", "June went home", "Last Friday stopped yoga")."""
    return word in _TIME_WORDS or word in _ADVERBIAL or word.rstrip(".") in _HONORIFIC


def _cased_tokens(text: str) -> list[str]:
    """The shared tokenizer's tokens with their case, in-token commas split
    and clitics removed ("I've" -> I, "Friday,I" -> Friday I)."""
    return [_CLITIC.sub("", part) for token in _REVISION_TOKEN.findall(normalize(text))
            for part in token.split(",") if _CLITIC.sub("", part)]


def _tokens(text: str) -> list[str]:
    """revision_tokens with in-token commas split and clitics removed."""
    return [word.casefold() for word in _cased_tokens(text)]


def _from_head(words: list[str], cased: list[str]) -> list[str] | None:
    """WORDS from the first word that can name an entity, skipping the closed
    session/date prefix words and numerals, time words, listed sentence
    adverbials and a title (CASED holds the same words with their case); None
    when a subordinate lead, focus particle, quantifier or article comes
    first.  An article opens a description ("The 3 keys", "The old lamp",
    "The user's tea") whose head noun is not its first word.  A time word,
    adverbial or title may also open a sentence with no subject, or be a name
    itself ("Today gave up yoga", "June went home"); after one, the next word
    is the subject only when it is written as a name (capitalised: "Last
    Friday Marinth", "Today I").  In these cases no entity is named and the
    writer keeps the call."""
    modified = False
    for i, word in enumerate(words):
        if word in _UNRESOLVED_LEAD or word in _QUANTIFIER or word in _ARTICLES:
            return None
        if _skippable(word):
            modified = modified or _modifier(word)
            continue
        if modified and not cased[i][:1].isupper():
            return None
        return words[i:]
    return None


def _subject_word(text: str) -> str | None:
    cased = _cased_tokens(text)
    words = _from_head([word.casefold() for word in cased], cased)
    return words[0] if words else None


def referent(text: str) -> str | None:
    """The entity a statement is about: its first word after a time/session
    prefix, sentence adverbials and a title, without a possessive ("User's
    tea ..." -> user).  A description ("The user's tea", "The 3 keys"), a
    quantified subject and a labelled text ("Melanie: I quit hiking",
    "Favourite tea: sencha") have none."""
    if isinstance(text, str) and _LABELLED.match(text):
        return None
    word = _subject_word(text)
    return (_POSSESSIVE.sub("", word) or None) if word else None


def _plain_name(text: str) -> bool:
    """The subject is a bare name, not a possessive phrase ("Serava's brother",
    "The Smiths' dog"; the tokenizer drops a trailing apostrophe, so the text
    itself is checked)."""
    word = _subject_word(text)
    if word is None or _POSSESSIVE.search(word):
        return False
    first = re.search(r"(?<![\w'])" + re.escape(word) + r"(?!\w)", normalize(text).casefold())
    # A trailing "'" or "'s" is possessive; a clitic ("Serava'll") is not.  A
    # subject that cannot be found as a word ("'James' brother") is not known
    # to be plain.
    return first is not None and not re.match(r"'(?:s\b|(?!\w))", first.string[first.end():])


def _words(text: str) -> set[str]:
    return {_POSSESSIVE.sub("", word) for word in _tokens(text)}


def _slot_name(entity: str | None, text: str = "") -> str | None:
    """A slot entity as a name.  A slot that is the leading subject phrase of
    TEXT is read in place, together with the text words before it, exactly
    like the text subject (_from_head: "Dr. Okafor" -> okafor, "Then Marinth"
    -> marinth; "Only Serava", "The user", "the still camera" -> none), so
    the writer's spelling of the slot does not change its name.  Any other
    slot is an object noun phrase that only loses a leading article or title
    and keeps every other word ("the still camera" -> still camera, "the 3
    keys" -> 3 keys, "the most valuable gem" -> most valuable gem), so its
    noun always reaches the mention guard.  A subject phrase with a deictic
    or indefinite head ("my sister", "we", "someone") names no entity."""
    raw = _tokens(entity or "")
    words = [_POSSESSIVE.sub("", word) for word in raw]
    cased = _cased_tokens(text)
    text_words = [word.casefold() for word in cased]
    start = _leads(raw, text_words) if raw else None
    if start is not None:
        words = _from_head([_POSSESSIVE.sub("", word) for word in text_words[:start]] + words,
                           cased[:start + len(raw)])
        if not words or words[0] in _DEICTIC:
            return None
    else:
        while words and (words[0] in _ARTICLES or words[0].rstrip(".") in _HONORIFIC):
            words = words[1:]
    return " ".join(word for word in words if word) or None


def _leads(raw: list[str], text_words: list[str]) -> int | None:
    """Where RAW opens TEXT_WORDS' subject phrase: it starts the text or
    follows only words that may stand before a subject ("Today only Serava
    ...", "The still camera ..."); None for any other slot."""
    for i in range(len(text_words) - len(raw) + 1):
        if text_words[i:i + len(raw)] == raw:
            return i
        if not _skippable(text_words[i]):
            return None
    return None


def statement_names(text: str, entity: str | None = None) -> frozenset[str]:
    """The entities one statement names.

    Without a slot entity this is the text referent.  A slot entity (given by
    the writer or derived by natural_slot) is named too; when it is another
    constituent of the same text ("Marinth took the apple", entity apple), the
    plain text subject is named as well.  An entity the text does not contain
    stands in for the subject ("Favourite instrument is the cello", entity
    Casey, names only casey).  Entity and text words are normalised the same
    way (_slot_name, referent).  A deictic subject ("I", "my sister") names
    no entity, as on the evidence side.
    """
    slot = _slot_name(entity, text)
    subject = referent(text)
    if subject in _DEICTIC:
        subject = None
    if slot is None:
        return frozenset({subject} - {None})
    names = {slot}
    if (subject and subject not in slot.split() and _plain_name(text)
            and set(slot.split()) <= _words(text)):
        names.add(subject)
    return frozenset(names)


def other_referent(evidence: str, prior_text: str, known: frozenset[str],
                   prior_entity: str | None = None,
                   named: frozenset[str] | None = None) -> bool:
    """True when withdrawal evidence is about a different known entity.

    Withdrawing a proposition is a claim about that proposition's entity.  A
    quote whose own subject is another entity the memory knows ("Serava
    travelled to the office" offered to withdraw "Marinth went back to the
    office"), when neither statement names the other's entities, supports no
    stance change for it.  An entity is known when more than one stored
    statement names it, so a one-off headline word is not treated as an
    entity.  The prior's entities are statement_names(PRIOR_TEXT,
    PRIOR_ENTITY), the same names that make entities known.  Unknown
    subjects, labelled texts, deictic or indefinite pronouns ("I used to like
    X" for "User likes X") in either statement, quotes with a personal or
    indefinite pronoun anywhere ("Last Friday I stopped yoga", "Serava took
    it from her": without coreference the pronoun may be the prior's entity),
    and quotes that name any of the prior's entities, are left to the writer,
    whether or not a slot entity was stored for the prior.

    Being known (repeated as a statement subject) does not show that a word
    is an entity: an unlisted sentence adverb opens statements the same way
    ("Hardly went out.").  When NAMED is given, the quote's subject must also
    be attested as a name there (named_mentions or writer about-metadata);
    otherwise its subject is unresolved and the writer keeps the call.
    """
    prior = statement_names(prior_text, prior_entity)
    about = referent(evidence)
    prior_words = {word for name in prior for word in name.split()}
    if (not prior or not about or about in prior_words or about in _DEICTIC
            or about not in known or not prior & known or _words(evidence) & _PRONOUN
            or (named is not None and about not in named)):
        return False
    # A possessive subject has a different head ("Serava's brother left"); a
    # deictic subject ("She took the apple") points into the context; a
    # labelled or unresolved prior ("Note: she took the apple") has no subject
    # to compare.  In each case the writer keeps the call.
    prior_subject = referent(prior_text)
    if (not _plain_name(evidence) or not _plain_name(prior_text)
            or prior_subject is None or prior_subject in _DEICTIC):
        return False
    return not prior_words & _words(evidence) and about not in _words(prior_text)


# A word is written as a name where its position cannot capitalise it: not
# at the start of a sentence, quotation or parenthesis.
_NAME_POSITION_START = re.compile(r"[.!?;:]\s+|[\"\u201c\u201d(\[]")


def named_mentions(text: str) -> frozenset[str]:
    """Words TEXT writes as names: capitalised words that do not open a
    sentence, quotation or parenthesis ("Marinth gave Serava the key" ->
    serava).  A capital forced by position ("Hardly went out.") is no
    evidence that a word names an entity, and neither is title case: a text
    that capitalises every word ("User Likes Green Tea") distinguishes no
    word from another."""
    if not isinstance(text, str):
        return frozenset()
    if _title_case(text):
        return frozenset()
    names = set()
    for part in _NAME_POSITION_START.split(normalize(text)):
        for word in _cased_tokens(part)[1:]:
            if word[:1].isupper():
                names.add(_POSSESSIVE.sub("", word.casefold()))
    return frozenset(names - {""})


def _title_case(text: str) -> bool:
    """Every lettered token of TEXT starts with a capital (two or more)."""
    lettered = [word for word in _cased_tokens(text) if word[:1].isalpha()]
    return len(lettered) > 1 and all(word[:1].isupper() for word in lettered)


def capitalised_words(text: str) -> frozenset[str]:
    """Words TEXT writes with a capital in any position, possessive removed.

    A writer declaration (about-metadata, slot entity, key namespace) names
    a word; the statement corroborates it only by spelling that word as a
    name.  A title-case text spells every word that way and corroborates
    none."""
    if not isinstance(text, str) or _title_case(text):
        return frozenset()
    return frozenset(_POSSESSIVE.sub("", word.casefold()) for word in _cased_tokens(text)
                     if word[:1].isupper()) - {""}


def subject_frame(text: str) -> tuple[str, str] | None:
    """A statement's subject and the word that follows it ("Marinth went
    back ..." -> (marinth, went)), read like referent; None without a
    subject or a following word.  A possessive subject ("User's tea is
    sencha") is followed by the head of its phrase, and a subject followed by
    a coordinator or sentence adverbial ("User and Marinth went", "Marinth
    then went") by no verb of its own, so neither has a frame."""
    if not isinstance(text, str) or _LABELLED.match(text) or not _plain_name(text):
        return None
    cased = _cased_tokens(text)
    words = _from_head([word.casefold() for word in cased], cased)
    if not words or len(words) < 2 or words[1] in _ADVERBIAL:
        return None
    return (words[0], words[1])


def subject_names(text: str, entity: str | None = None) -> frozenset[str]:
    """The names statement_names gives TEXT's subject: the slot entity when it
    is the subject or stands in for one ("Favourite instrument is the cello",
    entity Casey), the plain text subject when the slot is another
    constituent ("Marinth took the apple", entity apple -> marinth only)."""
    slot = _slot_name(entity, text)
    names = statement_names(text, entity)
    subject = referent(text)
    if slot and subject not in slot.split() and set(slot.split()) <= _words(text):
        return names - {slot}
    return names


def shares_frame(evidence: str, prior_text: str, prior_entity: str | None,
                 frames: dict[str, frozenset[str]]) -> bool:
    """True when the store shows the quote's subject and the prior's subject
    as the same kind of entity: each is the subject of a stored statement
    followed by the same word ("Serava went ..." and "Marinth went ...").  A
    word that opens subject-less statements ("Gave a talk.", "Gave blood.")
    is followed by its objects, so it shares no frame with a name and a
    withdrawal quoting it is left to the writer.  Only the prior's subject
    names (subject_names) are compared: an object of the prior ("the apple")
    is not the entity whose stance is withdrawn.  FRAMES maps each stored
    subject to the words that follow it (subject_frame)."""
    about = referent(evidence)
    if not about:
        return False
    following = frames.get(about, frozenset())
    return any(following & frames.get(name, frozenset())
               for name in subject_names(prior_text, prior_entity))


# Auxiliaries and the intensifiers the revision syntax itself allows ("User
# has finished", "User really likes") stand between a subject and its
# predicate.  Any other word there (a reporting verb, a modal, a hedge, a
# second name) means the words before the predicate are not one subject.
_AUXILIARY = frozenset({"is", "are", "was", "were", "has", "have", "had", "does", "did", "do"})
# Intensifiers the revision syntax itself allows ("really likes") and
# aspectual or additive adverbs ("just finished", "still needs", "also
# likes"); none of them hedges the predicate.
_INTENSIFIER = frozenset({"really", "truly", "just", "already", "now", "finally", "recently",
                          "still", "also"})
_SUBJECT_PHRASE_MAX = 4
_COORDINATOR = frozenset({"and", "or", "but", "nor", "yet", "plus"})
_COORDINATOR_PHRASES = (("as", "well", "as"), ("along", "with"), ("together", "with"))
# Closed grammatical classes decide what may follow a revised object.  A
# preposition or a focus adverb opens a phrase ("as an actress", "for its
# energy", ", especially her elegance"); a subordinator or relative word opens
# a clause.  Words that are both a preposition and a subordinator (after,
# before, since, until) are subordinators here; as, like, than and "for" head
# either a phrase or a clause.  The positive phrase grammar below (_phrases)
# reads them.
_PREPOSITION = frozenset({"in", "on", "at", "as", "for", "from", "with", "by", "about", "during",
                          "than", "to", "like", "including", "despite", "without", "through",
                          "into", "toward", "towards"})
_CLAUSE_PREPOSITION = frozenset({"as", "like", "than"})
_FOCUS = frozenset({"especially", "particularly", "mainly", "mostly"})
_SUBORDINATOR = frozenset({"after", "before", "since", "until", "till", "because", "when",
                           "whenever", "while", "whereas", "where", "though", "although",
                           "once", "unless", "if", "which", "who", "whom", "whose", "that",
                           "how", "why", "what"})
# Subject pronouns.  The nominative ones can only be a subject; "it" and
# "you" have one form for both cases.  With a contracted verb ("it's",
# "you're", "she'd") either is the subject of a finite clause
# (_contracted_subject).
_NOMINATIVE = frozenset({"i", "we", "he", "she", "they"})
_CASE_AMBIGUOUS = frozenset({"it", "you"})
# Modal verbs (a closed class) are finite like the auxiliaries, so neither is
# a noun word (_noun_word).
_MODAL = frozenset({"will", "would", "shall", "should", "can", "could", "may", "might", "must"})
# Of the subject pronouns only a third-person animate one can corefer with
# a named reporter: "Alice said she/they ..." may be Alice's own statement;
# "Alice said I/we/you/it ..." never is.
_REPORTER_PRONOUN = frozenset({"he", "she", "they"})
# The indefinite compounds ("nothing", "everyone") are noun phrases that can
# be a subject; "nothing" is no participle although it ends in -ing.
_INDEFINITE = frozenset(word for word in _DEICTIC if word[-3:] in ("one", "ody", "ing"))


# A closed list of common wordings that set up something still to be done,
# read without the filters of intention_matches (a coordinated action, a
# one-word or long action, a verb outside _INTENTION_EN).  It is a second
# line of defence only: the grounds grammar below (_grounds_list), which
# admits no verb phrase after "for" + a possessive or in a continuing grounds
# clause, is what keeps a plan out of a stance binding (r58/r59/r60 Claude
# F1).  The list is not complete, and its known misses are pinned in the
# tests.  "going", "about" and "set" count only after a form of "be"
# ("loves going to concerts" is not a plan), with up to two words of any
# kind between ("is kinda going to", r62 Claude F3); the verbs take up to two
# adverbs before "to".  "due to" is not a cue, since the corpus uses it for
# a cause ("too abstract due to dense language"), so "is due to see them"
# is a known miss (r62 Claude F5).  "have to say" is refused too, which
# costs recall only.
_CUE_ADVERBS = r"(?:\s+(?:still|really|just|also|definitely|finally|\w+ly)){0,2}"
_BE_GAP = r"(?:\s+[\w'-]+){0,2}"
_INTENTION_CUE_EN = re.compile(
    r"\b(?:(?:needs?|needed|has|have|had|plans?|planning|planned|wants?|wanted|wanting|"
    r"intends?|intended|intending|hopes?|hoped|hoping|aims?|aiming|wish(?:es)?|wishing|"
    r"promis(?:e|es|ed|ing)|agree(?:s|d|ing)?|offer(?:s|ed|ing)?|decid(?:e|es|ed|ing)|"
    r"eager|keen|ready|determined|committed|time|"
    r"would\s+(?:like|love)|'d\s+(?:like|love))" + _CUE_ADVERBS + r"|"
    r"(?:is|are|am|was|were|be|'s|'re|'m)" + _BE_GAP + r"\s+(?:going|about|set)|gonna)"
    r"\s+to\s+\w"
    # A future or conditional modal, and the nouns of a list of things to do.
    r"|\b(?:will|shall|gonna|would)\b|'ll\b|'d\b|\b(?:to-?do|reminders?|agenda)\b", re.I)
# A quoted span written in title case is taken for a title ("I Want to Hold
# Your Hand", "There Will Be Blood"), not a plan of the subject (r60 Claude
# F5): every lowercase word in it is one of the small words of a title.
# Capitalisation alone cannot tell a title from words written in title case,
# so a span that a verb of saying or writing leads (_SPEECH_LEAD: ", saying
# "We Will See Them In June"") is the subject's own words and is read like
# any other quoted span, such as ", saying "I am going to see them"", or a
# stray or inch mark (12" vinyl): with its quote marks removed (r61 Claude
# F2, r62 Claude F2).  A verb of saying leads every quote later in its
# sentence, whatever stands between them (r63 Claude F2, r64 Claude F1:
# ", saying excitedly", ", saying again", ", saying to her best friend",
# ", saying in a text", ", saying, as ever,").  A sentence ends at a break
# of the clause grammar (_clause_breaks: ";", "!", "?", or a full stop
# before a space that is no initial or title, r65 Claude F1: "Dr. Lee",
# "the U.S.", "7.30" and "..." end none), and marks inside a quote end none
# ("they rock." and "..."); words inside a quote are no verb of saying.
# The guard does not copy the stance grammar: it masks quotes, and it ends
# a sentence at ";", which a grounds chain reads through, so ", saying they
# rock; her words "We Will See Them In June"" binds (r66 Claude F4).  A
# stop inside a quote, after a single capital or an initial ("the U.S.",
# "an A.") or a bare line break ends no sentence, so a verb of saying
# before it leads a title-case quote after it, at a recall cost (r66
# Claude F2).  Neither shape changes a verdict over the saved writer texts.
# The verbs are a closed list, so a quote after "describing them as" stays a
# title; a real title in a sentence with a verb of saying (", saying she
# loves "There Will Be Blood"", ", saying that "I Want to Hold Your Hand" is
# their best song"), or with a noun or name spelt like one ("a note",
# "State Champs"), is read as words, at a recall cost.
_QUOTED_SPAN = re.compile(r"\"[^\"]*\"|\u201c[^\u201d]*\u201d")
_QUOTE_MARK = re.compile(r"[\"\u201c\u201d]")
_TITLE_SMALL = frozenset({"a", "an", "the", "and", "or", "nor", "but", "of", "to", "in", "on", "at",
                          "for", "by", "with", "from", "as", "into", "vs"})
_SPEECH_LEAD = re.compile(
    r"\b(?:say|says|said|saying|tell|tells|told|telling|write|writes|wrote|writing|"
    r"add|adds|added|adding|reply|replies|replied|replying|ask|asks|asked|asking|"
    r"declare|declares|declared|declaring|exclaim|exclaims|exclaimed|exclaiming|"
    r"mention|mentions|mentioned|mentioning|note|notes|noted|noting|state|states|stated|"
    r"stating|announce|announces|announced|announcing|insist|insists|insisted|insisting|"
    r"promise|promises|promised|promising|text|texts|texted|texting)\b", re.I)


def _speech_led(plain: str, start: int) -> bool:
    """A verb of saying stands, outside any quote, in the sentence of PLAIN
    before offset START, where a quote opens (see _QUOTED_SPAN)."""
    masked = _QUOTED_SPAN.sub(lambda match: '"' + "x" * (len(match.group(0)) - 2) + '"', plain)
    ends = [match.end() for match in _clause_breaks(masked)
            if match.group(0)[0] in ".;!?" and match.end() <= start]
    return bool(_SPEECH_LEAD.search(masked[ends[-1] if ends else 0:start]))


def _unquoted(plain: str) -> str:
    """PLAIN with each title-case quoted span that no verb of saying
    leads blanked and every other quote mark removed (see _QUOTED_SPAN)."""
    def title(match: re.Match) -> str:
        words = re.findall(r"[^\W\d_][\w'-]*", match.group(0))
        if _speech_led(plain, match.start()):
            return match.group(0)
        return " " if words and words[0][0].isupper() and all(
            word[0].isupper() or word in _TITLE_SMALL for word in words) else match.group(0)
    return _QUOTE_MARK.sub(" ", _QUOTED_SPAN.sub(title, plain))


def _states_intention(text: str) -> bool:
    """TEXT holds one of the common wordings of something still to be done
    (_INTENTION_CUE_EN, outside quoted titles) or a listed task.  A closed
    list, not a complete test: see _INTENTION_CUE_EN."""
    plain = normalize(text)
    return bool(_INTENTION_CUE_EN.search(_unquoted(plain))
                or _INTENTION_LISTED.search(plain))


def _participle(cased: str) -> bool:
    """CASED, a word as written, may be a present participle or gerund: a
    lowercase word of more than four letters ending in -ing that is not an
    indefinite pronoun or a time word ("morning", "evening").  A capitalised
    -ing word is a name ("Sterling", "Beijing").  This is a suffix test, not a
    closed class: the phrase grammar (_phrases) reads such a word only after
    "when"/"whenever", and only with the narrow objects of _after_participle.
    Whether a participle evaluates the object or states an event of its own
    (", finding them too long" / ", leaving her too tired") depends on the
    verb's meaning, which spelling does not show, so the phrase grammar never
    reads a participle at the head of a comma clause.  Only a stance trace's
    grounds read one: _grounds_clause from closed verb classes, and
    _grounds_continuation only from the closed effect class
    (_GROUNDS_EFFECT) or after "after"."""
    word = cased.casefold()
    return (cased.islower() and len(word) > 4 and word.endswith("ing")
            and word not in _INDEFINITE and word not in _TIME_WORDS)


def _phrase_word(word: str) -> bool:
    """WORD, casefolded, may follow a noun phrase inside a phrase without
    being that noun phrase's predicate: a word of the closed classes above
    (a preposition, coordinator, focus or sentence adverbial, time word,
    quantifier, article, subordinator) or a number."""
    return (word in _PREPOSITION or word in _COORDINATOR or word in _FOCUS or word in _ADVERBIAL
            or word in _TIME_WORDS or word in _QUANTIFIER or word in _ARTICLES
            or word in _SUBORDINATOR or bool(_PREFIX_NUMBER.match(word)))


def _open_word(word: str) -> bool:
    """WORD, casefolded and without a possessive, is an open-class word: a
    word of letters that belongs to none of the closed classes
    (_phrase_word, pronouns, session-prefix words)."""
    return (word[:1].isalpha() and not _phrase_word(word) and word not in _DEICTIC
            and word not in _PREFIX_WORDS)


# What may follow a revised object is read by a positive phrase grammar
# (_phrases): the words after the object in its clause, a comma clause led by
# a preposition, focus adverb or subordinator, and the tail of a
# pending intention's own clause elaborate the revised proposition only when
# they parse completely as the phrases below.  Any word the grammar does not
# place -- an open-class word where a noun phrase has ended, a coordinator, a
# pronoun that can only be a subject, a contracted verb, a function word
# written in capitals, an all-caps word, a word in another script or with
# mixed case ("iPhone") -- keeps the trace.  Spelling cannot tell a clause
# from a noun phrase ("coffee helps" / "iconic bands"), so the grammar admits
# only shapes no predicate can follow and refuses the rest, at a disclosed
# recall cost.
#
# Prepositions that cannot open a finite clause govern a noun phrase of one
# word after an optional determiner ("in movies", "at the end of the day").
# A second word may be the first one's predicate ("with coffee helps", "with
# her mother away", "with the kids asleep"), which spelling does not show, so
# a two-word noun phrase ("in old movies") is not read.
_GOVERNING = (_PREPOSITION - _CLAUSE_PREPOSITION - {"for"}) | {"of"}
# "with" and "without" may also head an absolute clause, a noun phrase with a
# predicate of its own of any kind -- a word, a phrase or a time ("with her
# mother away", "with the kids in bed", "with the meeting today").  So
# nothing after their noun phrase is read, and a word that may be a noun
# phrase by itself ("her", "that") is that noun phrase there, never a
# determiner ("with her away", "with that settled").
_ABSOLUTE = frozenset({"with", "without"})
# "as", "like", "than" and the conjunction-or-preposition "for" head a phrase
# ("as an actress", "for its energy") or a clause ("as Bob says", "for coffee
# keeps her awake"), as do the words that are both a preposition and a
# subordinator ("before the meeting" / "before the clinic closes").  After
# them only a noun phrase no predicate can follow is read: one word, a
# determiner with one word, a name or an object pronoun.
_CLAUSE_CAPABLE = _CLAUSE_PREPOSITION | {"for"}
_TEMPORAL = frozenset({"before", "after", "until", "till", "since"})
# "when"/"whenever" before a participle open a reduced clause that shares
# the main subject and states the condition of the stance ("when needing a
# break", "especially when travelling").  The other subordinators assert the
# participle's event ("since moving to Japan", "while living in Austin") or
# take a gerund subject ("because hearing it calms her"), so they open no
# reduced clause the grammar reads.
_REDUCED_LEAD = frozenset({"when", "whenever"})
# A demonstrative is a determiner after a governing preposition ("in that
# movie") but may be a subject pronoun after a word that can open a clause
# ("as this helps", "since that happened"), so a narrow noun phrase does not
# read it as a determiner.
_DEMONSTRATIVE = frozenset({"this", "that", "these", "those"})
_DETERMINER = _ARTICLES | frozenset({"my", "your", "his", "her", "its", "our",
                                     "their"}) | _DEMONSTRATIVE
_OBJECT_PRONOUN = frozenset({"me", "us", "him", "her", "them", "it", "you"})
_DEGREE = frozenset({"too", "very", "so", "quite", "rather"})
# Adverbs that may stand between phrases: the sentence adverbials without
# the coordinators, "so", negation and interjections; the intensifiers; and
# "earlier" ("as noted earlier").
_PHRASE_ADVERB = ((_ADVERBIAL - _COORDINATOR
                   - {"so", "not", "never", "well", "yes", "oh", "ok", "okay"})
                  | _INTENSIFIER | frozenset({"earlier"}))
_SESSION_WORDS = frozenset({"session", "sessions", "date"})
# The deictic time adverbs that are also noun phrases ("today mostly
# matters", "tomorrow morning").
_DEICTIC_TIME = frozenset({"today", "tonight", "yesterday", "tomorrow"})
# Time words that are also verbs or modals ("as leaves fall", "as Bob may",
# "as long as fears last", "as Bob summers in Lyon") stand alone only after a
# determiner reading (_phrases: "last summer"); every plural time unit is
# such a word.  Of the singular time units, the ones English also uses as
# verbs (_VERB_UNIT_WORDS: "storks winter in Spain", "you summer in Maine",
# "flowers spring from the rocks", "they weekend in Lyon") may be the verb of
# a noun phrase right before them, so _phrases reads them as time words only
# where no noun phrase precedes them ("in winter", "that summer").  The other
# units ("day", "week", "morning", "monday") are never verbs.
_UNIT_WORDS = frozenset(_TIME_UNITS)
_VERB_UNIT_WORDS = frozenset({"spring", "summer", "fall", "winter", "weekend"})
_VERB_TIME_WORDS = frozenset({"fall", "may", "march", "mar", "last", "past", "sat", "sun", "wed",
                              "am"}) | frozenset(unit + "s" for unit in _TIME_UNITS)
# A determiner that is also a relative pronoun ("that") may be the subject
# of the word after it when it follows a noun phrase ("the storks that
# winter in Spain"), and so may a plural demonstrative, which is also a
# pronoun ("as these winter in Spain").  A singular demonstrative pronoun
# takes an "-s" verb ("as that winters"), so a bare singular unit after it is
# a time word ("since that summer").
_RELATIVE_DETERMINER = _DETERMINER & _SUBORDINATOR
_PLURAL_DEMONSTRATIVE = frozenset({"these", "those"})


def _verb_subject(word: str, after_noun: bool) -> bool:
    """WORD may be the subject of the word right after it: a plural
    demonstrative anywhere ("these winter in Spain"), a relative determiner
    after a noun phrase ("the storks that winter in Spain")."""
    return word in _PLURAL_DEMONSTRATIVE or (after_noun and word in _RELATIVE_DETERMINER)


def _verb_time(word: str, after_noun: bool) -> bool:
    """WORD, a time word, may be a verb where it stands: a plural time unit
    or a time word that is also a verb or modal (_VERB_TIME_WORDS) anywhere,
    a singular time unit that English uses as a verb (_VERB_UNIT_WORDS)
    right after a noun phrase or a word that may be its subject
    (_verb_subject: "as storks winter", "storks winter year after year", "the
    storks that winter in Spain", "as these winter in Spain"), never "day" or
    "morning" ("later that day").  Such a word is never read as a time word
    there, alone, in a pair or after a demonstrative.  Every reading of a time
    word as a verb goes through this predicate."""
    return word in _VERB_TIME_WORDS or (after_noun and word in _VERB_UNIT_WORDS)
_LOWER_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*\Z")
_NAME_WORD = re.compile(r"[A-Z][A-Za-z]*(?:[-'.][A-Za-z]+)*\Z")


def _noun_word(cased: str) -> bool:
    """CASED, a word as written, may stand in a noun phrase: a number, or a
    lowercase open-class word, time word or session word that is no verb of
    the closed classes (auxiliary, modal) and no intensifier or degree word."""
    if _PREFIX_NUMBER.match(cased):
        return True
    return (bool(_LOWER_WORD.match(cased)) and cased not in _AUXILIARY and cased not in _MODAL
            and cased not in _INTENSIFIER and cased not in _DEGREE
            and (_open_word(cased) or cased in _TIME_WORDS or cased in _SESSION_WORDS))


def _name_word(cased: str) -> bool:
    """CASED, a word as written, may be (part of) a name: it starts with a
    capital, is not written in capitals throughout, and is no closed-class
    word ("As", "The"); a possessive ("Bob's") is allowed."""
    bare = _POSSESSIVE.sub("", cased)
    letters = [char for char in bare if char.isalpha()]
    word = bare.casefold()
    return (bool(_NAME_WORD.match(bare)) and not (len(letters) > 1 and bare.isupper())
            and (_open_word(word) or word in _TIME_WORDS))


def _noun_phrase_ends(cased: list[str], i: int, governed: bool, bare: bool = True,
                      absolute: bool = False) -> list[int]:
    """The indexes at which a noun phrase that starts at CASED[I] may end:
    an object pronoun ("it", "them"), a name of up to three words with
    numbers after its first ("Wes Anderson", "Player 3"), or one noun after
    an optional determiner (article, possessive or demonstrative determiner,
    possessive name: "the end", "Bob's sake").  A demonstrative is a
    determiner after a GOVERNED (governing) preposition ("in that movie"),
    elsewhere only before a time word that is no verb ("for this week", not
    "as this helps"); BARE false requires the noun to follow a determiner.
    After a word that may head an absolute clause (ABSOLUTE: _ABSOLUTE) an
    object pronoun or a demonstrative is the whole noun phrase."""
    n, ends = len(cased), []
    if i >= n:
        return ends
    word = cased[i]
    if word in _OBJECT_PRONOUN or (absolute and word in _DEMONSTRATIVE):
        ends.append(i + 1)
    j = i
    while (j < n and j - i < 3 and not _POSSESSIVE.search(cased[j])
           and (_name_word(cased[j]) if j == i else
                (_name_word(cased[j]) or bool(_PREFIX_NUMBER.match(cased[j]))))):
        j += 1
        ends.append(j)
    if absolute and (word in _OBJECT_PRONOUN or word in _DEMONSTRATIVE):
        return ends
    if ((word in _DETERMINER and (governed or word not in _DEMONSTRATIVE or (
            i + 1 < n and cased[i + 1] in _TIME_WORDS
            and not _verb_time(cased[i + 1], _verb_subject(word, False)))))
            or (_POSSESSIVE.search(word) and _name_word(word))):
        start = i + 1
    elif bare:
        start = i
    else:
        return ends
    # A bare participle is a gerund that may head a clause of its own
    # ("since moving to Japan"), so it opens no bare noun phrase.
    if start < n and _noun_word(cased[start]) and not (start == i and _participle(cased[start])):
        ends.append(start + 1)
    return ends


def _after_participle(cased: list[str], i: int) -> bool:
    """CASED[I:], the words after a participle that "when"/"whenever" make
    the stance's condition, close the clause without stating anything of
    their own: the participle has no object, an object pronoun, a name of up
    to three words or a determiner-led noun, and phrases follow ("when
    travelling", "when hearing it", "when visiting Bob", "when needing a
    break from reality").  A bare word after the participle may be its
    gerund's predicate ("when cooking helps", "when cooking mostly helps")
    and is not read: the words after it are read after a noun phrase."""
    return any(_phrases(cased, end, after_noun=True)
               for end in [i] + _noun_phrase_ends(cased, i, False, bare=False))


def _session_note(cased: list[str], i: int) -> int:
    """Index past a session or date note that starts at CASED[I] ("session
    974", "Session 1158", "date 2025-07-17"), else -1."""
    if i + 1 < len(cased) and cased[i].casefold() in _SESSION_WORDS:
        j = i + 1
        while j < len(cased) and _PREFIX_NUMBER.match(cased[j]):
            j += 1
        if j > i + 1:
            return j
    return -1


def _note(cased: list[str], i: int) -> bool:
    """CASED[I:], the words after a verb of saying, only place the statement:
    adverbs, time words, numbers and session or date notes, each also after
    "in", "on", "at" or "during" ("as noted earlier", "mentioned in session
    18", "noted 2025-07-16").  Anything else makes the verb a predicate of
    its own ("added to her playlist", "shared with her sister", "shared by
    Bob")."""
    n = len(cased)
    while i < n:
        j = i + 1 if cased[i] in ("in", "on", "at", "during") and i + 1 < n else i
        note = _session_note(cased, j)
        if note > 0:
            i = note
        elif (cased[j] in _PHRASE_ADVERB or _PREFIX_NUMBER.match(cased[j])
              or (cased[j] in _TIME_WORDS and not _verb_time(cased[j], False))):
            i = j + 1
        else:
            return False
    return True


def _phrases(cased: list[str], i: int = 0, after_noun: bool = False) -> bool:
    """CASED[I:], words as written, parse completely as phrases that state
    nothing of their own: adverbs (_PHRASE_ADVERB); time words and numbers
    ("last summer", "that day"); a governing preposition with its noun
    phrase ("in movies", "of the day"); "with"/"without" with a noun phrase
    that ends the words ("with milk", _ABSOLUTE); as/like/than/for or a temporal
    preposition with a noun phrase no predicate can follow ("as an actress",
    "for Bob's sake", "before the end"); a session or date note ("(Session
    1158)", "as of session 118"); a source note ("as already noted earlier",
    "as mentioned in session 18", "(noted 2025-07-16"), whose verb never
    directly follows a noun phrase (AFTER_NOUN: "as Bob noted" is Bob's
    statement) and is followed only by a note (_note); "when" with a
    participle ("when needing a break", _after_participle); a focus adverb
    with a phrase or a noun phrase (", especially her elegance"), after a
    noun phrase with a phrase only, elsewhere with no bare word.  A session
    note, time word, number, deictic time word ("today") or determiner or
    time word with a time word ("early morning") is a noun phrase too
    (AFTER_NOUN after it).  A bare
    time unit right after a noun phrase may be its verb ("as storks winter")
    and is not skipped.  Function words count only in lowercase."""
    with _grammar_readings():
        return _GRAMMAR.phrases_at(tuple(cased), i, after_noun)


# The phrase grammar tries every end of a noun phrase ("Player 3 4" may
# end after each of its words) and backtracks, so read naively the same
# words are parsed again for every way of reaching them, and a short
# repeated shape costs time exponential in its length (r81 Claude F1).  The
# readings are pure functions of the words, the index and AFTER_NOUN, so
# each is computed at most once per word sequence while one revises() call
# runs: a sequence of N words has at most 2(N + 1) phrase states and N + 1
# states of each grounds reading.  A state's own work still grows with N (a
# grounds item may end at O(N) places, a note reads the words after it and
# every lookup hashes the word sequence), so a call costs a polynomial
# amount of work in N, roughly N^2 to N^3, never an exponential one.
# The tables belong to one revises() call on one thread (_grammar_readings):
# OAKS runs one memory per document on a thread pool, and a call on another
# thread neither reads nor empties them (r82 Claude F1); they hold every
# state the call reaches and are dropped when it returns.  The tables only
# save work; a reading never depends on what they hold.
_GRAMMAR = threading.local()
_READINGS = ("phrases_at", "grounds_list_at", "grounds_item_ends_at",
             "grounds_postmodifier_ends_at")


@contextmanager
def _grammar_readings():
    """Give the calling thread fresh memoised grammar readings until the
    block ends, unless it already has them (a reading inside revises())."""
    if getattr(_GRAMMAR, "phrases_at", None) is not None:
        yield
        return
    for name in _READINGS:
        setattr(_GRAMMAR, name, lru_cache(maxsize=None)(globals()["_" + name]))
    try:
        yield
    finally:
        for name in _READINGS:
            setattr(_GRAMMAR, name, None)


def _phrases_at(cased: tuple[str, ...], i: int, after_noun: bool) -> bool:
    n = len(cased)
    if i >= n:
        return True
    word = cased[i]
    # Adverbs, time words, numbers and notes between a noun phrase and a verb
    # of saying ("as Mom often said", "like Player 3 said") leave it after
    # the noun phrase.
    # A session note, a time word, a number, a deictic time word and a
    # determiner or time word with a time word are noun phrases that may be
    # a subject ("winter mostly drags", "2024 mostly sucked", "today mostly
    # matters"), so the words after them are read after a noun phrase; any
    # other adverb leaves the state as it is.
    note = _session_note(cased, i)
    if note > 0:
        return _GRAMMAR.phrases_at(cased, note, True)
    if word == "as" and i + 1 < n and cased[i + 1] == "of":
        return _GRAMMAR.phrases_at(cased, i + 2, after_noun)
    if (not after_noun and word in _SAYING and not word.endswith("s")
            and _note(cased, i + 1)):
        return True
    # A determiner, a time word or a deictic time word before a time word
    # makes one noun phrase with it ("that day", "early morning", "tomorrow
    # night"), unless either word may be a verb where it stands (_verb_time:
    # "as the leaves last autumn", "storks winter year after year"); after a
    # noun phrase the word after a relative determiner, and anywhere the word
    # after a plural demonstrative, stands after its subject (_verb_subject:
    # "the storks that winter in Spain", "as these winter in Spain").
    if ((word in _DETERMINER or word in _TIME_WORDS or word in _DEICTIC_TIME) and i + 1 < n
            and not (word in _TIME_WORDS and _verb_time(word, after_noun)
                     and (after_noun or word in _UNIT_WORDS))
            and cased[i + 1] in _TIME_WORDS
            and not _verb_time(cased[i + 1], _verb_subject(word, after_noun))):
        return _GRAMMAR.phrases_at(cased, i + 2, True)
    # A deictic time word is a noun phrase as well as an adverb; any other
    # adverb leaves the state as it is.
    if word in _PHRASE_ADVERB:
        return _GRAMMAR.phrases_at(cased, i + 1, after_noun or word in _DEICTIC_TIME)
    if (_PREFIX_NUMBER.match(word)
            or (word in _TIME_WORDS and not _verb_time(word, after_noun))):
        return _GRAMMAR.phrases_at(cased, i + 1, True)
    if word in _ABSOLUTE:
        return n in _noun_phrase_ends(cased, i + 1, True, absolute=True)
    # "to" before a bare lowercase word may be an infinitive marker ("on her
    # list to buy tomorrow", ", to rewatch tomorrow"), which spelling cannot
    # tell from a bare noun ("to work"), so after "to" only a name, an object
    # pronoun, a number, a time word that is no verb there (_verb_time: "to
    # spring for tickets", "to weekend in London" are infinitives, r63 Claude
    # F1) or a determiner with its noun is read (r62 Claude F1, at a recall
    # cost: "on the way to work" and "from spring to summer" are refused).
    if word in _GOVERNING:
        bare = word != "to" or (i + 1 < n and (bool(_PREFIX_NUMBER.match(cased[i + 1]))
                                               or (cased[i + 1] in _TIME_WORDS
                                                   and not _verb_time(cased[i + 1], True))))
        return any(_GRAMMAR.phrases_at(cased, end, True)
                   for end in _noun_phrase_ends(cased, i + 1, True, bare=bare))
    if word == "as":
        j = i + 1
        while j < n and cased[j] in _PHRASE_ADVERB:
            j += 1
        if j < n and cased[j] in _SAYING and not cased[j].endswith("s") and _note(cased, j + 1):
            return True
    if (word in _CLAUSE_CAPABLE or word in _TEMPORAL) and any(
            _GRAMMAR.phrases_at(cased, end, True)
            for end in _noun_phrase_ends(cased, i + 1, False)):
        return True
    if word in _REDUCED_LEAD:
        j = i + 1
        while j < n and cased[j] in _PHRASE_ADVERB:
            j += 1
        return j < n and _participle(cased[j]) and _after_participle(cased, j + 1)
    # After a noun phrase a focus adverb may open that noun phrase's
    # predicate ("as Bob mostly prefers", "as storks mostly winter"), so only
    # a phrase is read after it there, still after the noun phrase.
    if word in _FOCUS and after_noun:
        return i + 1 < n and _GRAMMAR.phrases_at(cased, i + 1, True)
    # Elsewhere a bare word after it may be a predicate without a subject
    # (", mostly lives in Paris"), so only a phrase, a name, an object
    # pronoun or a determiner with its noun follows (", especially her
    # elegance").
    if word in _FOCUS:
        return i + 1 < n and (_GRAMMAR.phrases_at(cased, i + 1, False) or any(
            _GRAMMAR.phrases_at(cased, end, True)
            for end in _noun_phrase_ends(cased, i + 1, False, bare=False)))
    return False


def _contracted_subject(text: str) -> bool:
    """TEXT holds a subject pronoun with a contracted verb ("you're",
    "it's"), which _cased_tokens splits off; such a pronoun is always the
    subject of a finite clause."""
    for token in revision_tokens(text):
        bare = _CLITIC.sub("", _POSSESSIVE.sub("", token))
        if bare != token and (bare in _NOMINATIVE or bare in _CASE_AMBIGUOUS):
            return True
    return False


def _context_bound(phrase: tuple[str, ...]) -> bool:
    """PHRASE contains a deictic or possessive pronoun ("his wife", "my
    sister"): it names an entity only in its speaker's context, so two equal
    phrases written by different speakers need not be the same entity."""
    return any(word in _DEICTIC for word in phrase)
# Verbs of saying assert their complement ("User said they no longer like
# X" is the user's own statement).  A belief, hope or expectation verb does
# not, so it is not in this closed class and its frame stays unresolved.
_SAYING = frozenset({"said", "says", "mentioned", "mentions", "noted", "notes", "stated",
                     "states", "reported", "reports", "confirmed", "confirms", "shared",
                     "shares", "explained", "explains", "indicated", "indicates",
                     "reiterated", "reiterates", "added", "adds", "clarified", "clarifies"})


def _subject_phrase(words: list[str]) -> tuple[str, ...] | None:
    """The subject phrase of a predicate whose sentence opens with WORDS:
    WORDS without the leading time/session prefix, articles, titles and
    sentence adverbials (_skippable) and without trailing auxiliaries or
    intensifiers.  "In session 12, the user really" -> (user,); "Alice said
    Bob" -> (alice, said, bob); "Alice will soon" -> (alice, will, soon).
    None when nothing is left, the phrase is long, opens with a subordinator
    or quantifier, or is a bare deictic pronoun ("She", "They")."""
    start = 0
    while start < len(words) and _skippable(words[start]):
        start += 1
    # Intensifiers and an inserted session note (a prefix word with the
    # numbers after it: "user in session 819") name no entity wherever they
    # stand.  A number right after a word is part of the name ("Player 7").
    phrase: list[str] = []
    note = False
    for i, word in enumerate(words[start:]):
        note = i > 0 and (word in _PREFIX_WORDS or (note and bool(_PREFIX_NUMBER.match(word))))
        if not note and (i == 0 or word not in _INTENSIFIER):
            phrase.append(word)
    while phrase and (phrase[-1] in _AUXILIARY or phrase[-1] in _INTENSIFIER):
        phrase.pop()
    if (not phrase or len(phrase) > _SUBJECT_PHRASE_MAX or phrase[0] in _UNRESOLVED_LEAD
            or phrase[0] in _QUANTIFIER or (len(phrase) == 1 and phrase[0] in _DEICTIC)):
        return None
    return tuple(phrase)


def _speaker(phrase: tuple[str, ...]) -> str | None:
    """The entity a subject phrase names: a bare name ("user"), or the
    speaker of a self-report ("user said they", "Alice mentioned she",
    "user mentioned" before a gerund), where the pronoun subject of a verb of
    saying, or its absent subject, is read as the speaker.  A named
    complement subject ("Alice said Bob") is another entity."""
    if len(phrase) == 1:
        return phrase[0]
    if phrase[0] in _DEICTIC or _POSSESSIVE.search(phrase[0]) or phrase[1] not in _SAYING:
        return None
    # The complementiser of a verb of saying ("mentioned that they") adds no
    # entity.
    if len(phrase) > 2 and phrase[2] == "that":
        phrase = phrase[:2] + phrase[3:]
    if len(phrase) == 2:
        return phrase[0]
    if len(phrase) == 3 and phrase[2] in _REPORTER_PRONOUN:
        return phrase[0]
    return None


def _same_subject(revising_before: str, prior_before: str) -> bool:
    """The revising predicate and the prior predicate have the same subject.
    Each side is the sentence's words before its own predicate, read by
    _subject_phrase; the two phrases must be equal or name the same speaker
    (_speaker).  So "Bob no longer likes" (bob) is not "Alice likes"
    (alice), and neither are "Alice said Bob no longer likes" (alice said
    bob), "Alice's brother no longer likes" (alice's brother) or "Alice will
    soon no longer like" (alice will soon); "In session 30, user said they
    used to like" is "User likes".  Equal phrases that hold a deictic or
    possessive pronoun ("His wife", "My sister") may come from different
    speakers, so only a speaker form resolves them.  An unresolved phrase on
    either side binds nothing."""
    new = _subject_phrase(_tokens(revising_before))
    old = _subject_phrase(_tokens(prior_before))
    if new is None or old is None:
        return False
    speaker = _speaker(new)
    if speaker is not None and speaker == _speaker(old):
        return True
    return new == old and not _context_bound(new)


def _clause_spans(text: str) -> list[tuple[int, int, int, str]]:
    """(start, end, sentence start, delimiter before it) of each
    _CLAUSE-delimited clause; the first clause's delimiter is "."."""
    spans, start, sentence, delimiter = [], 0, 0, "."
    for match in _clause_breaks(text):
        spans.append((start, match.start(), sentence, delimiter))
        start = match.end()
        delimiter = match.group()[0]
        if match.group()[0] in ".;!?":
            sentence = start
    spans.append((start, len(text), sentence, delimiter))
    return spans


def _coordinated(words: list[str]) -> bool:
    """WORDS contain a coordinator or coordinating phrase ("and", "plus",
    "as well as", "along with")."""
    return bool(_COORDINATOR & set(words)) or any(
        _contains(words, list(phrase)) >= 0 for phrase in _COORDINATOR_PHRASES)


def _adjunct(cased: list[str], after_noun: bool = False) -> bool:
    """CASED, the rest of a clause after its object as written, elaborate the
    predicate: nothing beyond a time prefix, or phrases that parse completely
    under the positive phrase grammar (_phrases: "as an actress", "in old
    movies", "for its energy", "when needing a break", "lately").  Anything
    the grammar does not place -- a coordinated part ("and coffee"), a
    clause ("while Bob prefers coffee", "for coffee keeps her awake"), a
    longer head ("Pixies album") -- is not.  AFTER_NOUN: the words follow a
    noun phrase that may be their subject ("loves the way storks winter",
    "fix the bug Bob reported")."""
    if _prefix_clause([word.casefold() for word in cased]):
        return True
    # A verb of saying at its head is read as a source note only with a
    # note after it (_note), so "the album shared by Bob" is not one.
    return _phrases(cased, after_noun=after_noun)


def _separate_proposition(clause: str) -> bool:
    """True when CLAUSE may state a proposition of its own rather than
    elaborate the trace's main clause.

    A time/session prefix or a source note ("Mentioned in session 1897",
    whose verb of saying is no "-s" form, as in _phrases) elaborates, and so
    does a clause that parses completely under the positive phrase grammar
    (_phrases: ", especially her elegance", ", as an actress", ", when
    traveling").  Every other clause may state its own
    proposition: one with a contracted subject pronoun (", as you're
    aware"), a clause led by a subordinator, coordinator, relative word or
    possible subject (", who won four Oscars", ", for coffee keeps her
    awake", "and lives in Austin", "; it always puts them ..."), a predicate
    without a subject (", lives in Austin"), a participle with its own event
    or judgement (", living in Austin", ", finding them too long") and a
    bare noun phrase (", innovation and drive"), which spelling cannot tell
    from a subject with its predicate (", coffee helps").  A stance trace
    first reads such clauses as the stance's grounds (_one_stance), so there
    ", finding them too long" binds."""
    cased = _cased_tokens(clause)
    words = [word.casefold() for word in cased]
    if _prefix_clause(words) or (words[0] in _SAYING and not words[0].endswith("s")
                                 and _prefix_clause(words[1:])):
        return False
    if _contracted_subject(clause):
        return True
    # The head of a clause, after any adverbs, may be a predicate whose
    # subject is elided (", lives in Austin", ", mostly winter in Spain"), so
    # a time word there is read as it is right after a noun phrase: one that
    # may be a verb (_verb_time) opens a predicate (", winter in Spain", ";
    # summer in Maine"), as it does in the tail after the object.
    head = 0
    while head < len(cased) and (cased[head] in _PHRASE_ADVERB or cased[head] in _FOCUS):
        head += 1
    if head < len(cased) and cased[head] in _TIME_WORDS and _verb_time(cased[head], True):
        return True
    # A clause follows the words before it, so a relative determiner opening
    # it may be a relative pronoun whose antecedent ends the clause before
    # (", that winter in Spain"); the clause is then read after a noun phrase.
    return not _phrases(cased, after_noun=cased[0] in _RELATIVE_DETERMINER)


def _one_proposition(prior: str, at_clause: int) -> bool:
    """No clause of PRIOR but AT_CLAUSE may carry a proposition of its own,
    so withdrawing the whole trace withdraws only the revised proposition."""
    return not any(_separate_proposition(prior[start:end])
                   for i, (start, end, _, _) in enumerate(_clause_spans(prior))
                   if i != at_clause and prior[start:end].strip())


# The grounds of a stance.  A stance trace often goes on to say why the
# subject holds it (", finding them too heavy", ", praising his range",
# ", especially her elegance", ", his stories are so immersive", "for its
# energy, innovation, and iconic bands").  Those words state no proposition
# that outlives the stance: withdrawing "User likes X" withdraws the reasons
# given for liking X with it.  Spelling alone cannot tell grounds from a
# second event (", finding them too long" / ", leaving her too tired"), so
# grounds are read only from closed classes: a verb of evaluation or
# attribution at the head of a comma clause whose words point back to the
# object; a focus adverb before a possessive that points back to it
# (", especially her elegance and dancing"); a clause led by "it" or such a
# possessive that evaluates (a copula: ", his stories are so immersive") or
# reports an effect on the subject (", her music brings them joy"); or "for"
# with such a possessive right after the object.  A possessive-led clause
# with any other predicate may be about another entity ("; her brother
# lives in Austin").  A verb of liking opens grounds only under a positive
# stance ("dislikes X, loves Y" is a second stance), and no grounds hold a
# word that sets up another stance or a contrast ("prefers", "instead",
# "but").
_GROUNDS_EVALUATIVE = frozenset({"finding", "finds", "find", "citing", "cites", "cite",
                                 "describing", "describes", "describe", "noting", "notes", "note",
                                 "saying", "says", "drawn"})
_GROUNDS_LIKING = frozenset({"loving", "loves", "love", "enjoying", "enjoys", "enjoy", "liking",
                             "likes", "like", "admiring", "admires", "admire", "appreciating",
                             "appreciates", "appreciate", "praising", "praises", "praise",
                             "valuing", "values", "value"})
_ANAPHOR = frozenset({"it", "its", "they", "them", "their", "he", "him", "his", "she", "her"})
_THING_ANAPHOR = frozenset({"it", "its", "they", "them", "their"})
_PERSONAL_ANAPHOR = _ANAPHOR - _THING_ANAPHOR
# Only a possessive or "it" opens grounds: "they" may be the subject itself
# ("they prefer ..."), and "he"/"she" may open any event of the object's own
# (", he lives in Austin").
_ANAPHOR_LEAD = frozenset({"it", "its", "his", "her", "their"})
_POSSESSIVE_ANAPHOR = _ANAPHOR_LEAD - {"it"}
_EXPERIENCER = frozenset({"them"})
_OTHER_STANCE = frozenset({"prefer", "prefers", "preferring", "preferred", "rather", "instead",
                           "but", "although", "though", "whereas", "while", "unlike", "except",
                           "want", "wants", "wanting", "avoid", "avoids", "avoiding",
                           "dislike", "dislikes", "disliking", "hate", "hates", "hating"})
# A head noun after the revised object names the same thing as a category
# ("disco music", "Preston Sturges movies", "Oliver Sacks' work").
_CATEGORY_HEAD = frozenset({"music", "movies", "films", "books", "work", "writing"})


def _no_other_stance(cased: list[str]) -> bool:
    return not any(word.casefold() in _OTHER_STANCE for word in cased)


# Grounds listed after "for" + a possessive or in a clause continuing them
# name attributes of the object: noun phrases ("its energy", "raw, honest
# portrayal of female friendships", "Freddie Mercury's incredible voice"),
# joined by "and"/"or", each with prepositional phrases after it ("skill in
# literary adaptation", "depth to her characters"), and notes or times at
# the end ("(Session 1273)").  A plan or a second event needs a verb phrase
# ("and is about to see them", "and set a reminder to buy tickets", "and
# lives in London", ", eager to see them in June"), which this grammar does
# not admit, except where spelling cannot tell a verb from a noun (a bare or
# past-tense verb with its object: "and put tickets on her list", "and
# booked tickets"; pinned as known limits): an auxiliary or modal is no noun
# word, a word after a noun phrase
# must be a coordinator, a preposition or a closing note, "to" before a bare
# word is an infinitive, and an item opening with a bare lowercase "-s" word
# or participle that has words after it may be a predicate without a
# subject, so it is that word alone (", lives in Austin" / ", technologies").
# The one infinitive read is a "tough" infinitive: a closed class of tough
# adjectives (_TOUGH) with the stance object itself left as a gap at the end
# ("hard to get into"); any other word before "to" + a verb ("promised to
# drop by", "eager to drop by", "time to drop in") is a plan (r61 Claude
# F1).  Of the participles, only a closed class of effect verbs opens a
# continuing clause (", bringing depth to her characters").
_GROUNDS_PREPOSITION = frozenset({"of", "in", "with", "from", "by", "about", "on", "to", "for",
                                  "into", "through"})
_GROUNDS_EFFECT = frozenset({"bringing", "sparking", "showing", "making", "giving"})
# The corpus heads are hard, easier, interesting and disturbing; in a grounds
# list a noun head (", a compelling topic to read about") is refused, since
# spelling cannot tell it from "time to drop in" (a recall cost only).
_TOUGH = frozenset({"hard", "harder", "easy", "easier", "difficult", "impossible", "tough",
                    "tricky", "simple", "fun", "boring", "tiring", "painful", "pleasant",
                    "awkward", "interesting", "disturbing"})
_GROUNDS_NP_MAX = 4
_LIST_COORDINATOR = frozenset({"and", "or", "plus"})


def _grounds_word(cased: str) -> bool:
    return _noun_word(cased) or _name_word(cased)


def _grounds_determiner(cased: list[str], i: int) -> int:
    """Index past a determiner at CASED[I] (an article, a possessive or
    demonstrative determiner, a possessive name of up to three words:
    "Freddie Mercury's"), else I."""
    n = len(cased)
    if i < n and cased[i] in _DETERMINER:
        return i + 1
    j = i
    while j < n and j - i < 3 and _name_word(cased[j]):
        if _POSSESSIVE.search(cased[j]):
            return j + 1
        j += 1
    return i


def _bare_predicate(cased: list[str], i: int) -> bool:
    """CASED[I], with no determiner before it, is a lowercase "-s" word or a
    participle that may head a predicate ("lives", "seeing"); a hyphenated
    compound ("boundary-pushing") is a modifier."""
    first = cased[i]
    return (first.islower() and "-" not in first
            and ((first.endswith("s") and not first.endswith("ss")) or _participle(first)))


def _grounds_item_ends_at(cased: tuple[str, ...], i: int) -> tuple[int, ...]:
    """The indexes at which a grounds item starting at CASED[I] may end
    (see above); an item that is a bare predicate word ends there and takes
    nothing after it but a coordinator."""
    n = len(cased)
    while i < n and (cased[i] in _FOCUS or cased[i] in _PHRASE_ADVERB):
        i += 1
    start = _grounds_determiner(cased, i)
    if start >= n or not _grounds_word(cased[start]):
        return ()
    if start == i and _bare_predicate(cased, start):
        return (start + 1,)
    ends, j = [], start
    while j < n and j - start < _GROUNDS_NP_MAX and _grounds_word(cased[j]) and (
            j == start or not _POSSESSIVE.search(cased[j])):
        j += 1
        ends.append(j)
    out = []
    for end in ends:
        out.append(end)
        out.extend(_GRAMMAR.grounds_postmodifier_ends_at(cased, end))
    return tuple(dict.fromkeys(out))


def _grounds_postmodifier_ends_at(cased: tuple[str, ...], k: int) -> tuple[int, ...]:
    """The indexes at which prepositional phrases opening at CASED[K] after
    a grounds noun phrase may end."""
    n = len(cased)
    if k >= n or cased[k] not in _GROUNDS_PREPOSITION:
        return ()
    if cased[k] == "to" and k + 1 < n and _noun_word(cased[k + 1]):
        # An infinitive: only a tough infinitive, after a tough adjective
        # and with its object gapped at the end of the clause ("hard to get
        # into"), is read.
        return (n,) if (k > 0 and cased[k - 1] in _TOUGH and k + 3 == n
                       and cased[k + 2] in _PREPOSITION) else ()
    out = []
    for end in _GRAMMAR.grounds_item_ends_at(cased, k + 1):
        if end > k + 1:
            out.append(end)
    return tuple(out)


def _grounds_list(cased: list[str], i: int) -> bool:
    """CASED[I:] parse completely as a list of grounds items (see above)."""
    with _grammar_readings():
        return _GRAMMAR.grounds_list_at(tuple(cased), i)


def _grounds_list_at(cased: tuple[str, ...], i: int) -> bool:
    n = len(cased)
    j = i
    while j < n and (cased[j] in _FOCUS or cased[j] in _PHRASE_ADVERB):
        j += 1
    bare = j < n and _grounds_determiner(cased, j) == j and _bare_predicate(cased, j)
    for end in _GRAMMAR.grounds_item_ends_at(cased, i):
        if end == n:
            return True
        if cased[end] in _LIST_COORDINATOR and end + 1 < n and _GRAMMAR.grounds_list_at(cased, end + 1):
            return True
        # Closing notes and times; "to" before a bare word is an infinitive.
        if not bare and not (cased[end] == "to" and end + 1 < n and _noun_word(cased[end + 1])) \
                and _GRAMMAR.phrases_at(cased, end, True):
            return True
    return False


# A form of "be" or a linking verb ("remains", "seems"), finite or not
# ("has been", "would be"), whose complement may be an infinitive plan.
_COPULA = frozenset({"is", "are", "was", "were", "be", "been", "being", "remain", "remains",
                     "remained", "seem", "seems", "seemed"})
# Words that may stand between the copula and "to": any adverbs or adverbial
# phrases ("simply", "of course", "in fact", "long"), a "for" + noun phrase
# subject ("for her little brother Sam"), or an extraposed noun phrase ("it
# is her goal for the coming year to ...").  Spelling cannot tell these
# apart, so the gap is any words up to the end of the comma clause (r71
# Claude F5).  What decides is the word after "to": an infinitive "to" is
# followed by a verb, which may be hyphenated ("to re-watch", r71 Claude F3)
# and is never a determiner or an object pronoun ("devoted to the fans",
# "entirely to her taste" are prepositional, r71 Claude F7).  A
# prepositional "to" does not end the search: a later "to" may still open
# the plan ("it is important to her to see them live", "it is up to her to
# book", r73 Claude F1).  A tough adjective before "to" makes an attribute
# only when the infinitive's object is the gap at the end of the words read
# ("hard to follow", "hard to get into"), as in _grounds_postmodifier_ends_at.
# A colon followed by a space or the end of the text joins two parts, and the
# words are grounds only when the part before it is and the part after it
# goes on listing them (_grounds_continuation): "her reason is simple: the
# rhythm" is grounds, while "her plan is simple: to book the studio", "her
# goal is easy to explain: to see them live" and "its rhythms are hard to
# follow: she quit the band" state something else (r71 Claude F2, r73
# Claude F2, r75 Claude F1).  The shared tokenizer drops colons, so the
# grounds readings (_grounds_clause, _grounds_continuation, _reason_phrase),
# the revised object's own reason phrase included (r75 Claude F2), keep each
# as the word _COLON (_stance_tokens); _adjunct, _contracted_subject and
# _separate_proposition read the tokens without it (r77 Claude F4).  A colon
# inside a word ("7:30", "https://") is no boundary (r75 Claude F5); a title
# holding a colon before a space ("Batman: The Animated Series") is split
# too, so its left part is read alone, a disclosed recall cost (r77 Claude
# F5).  Disclosed known limit (r77 Claude F1, restated as one class by r79
# Claude F1): the tokenizer drops every mark that is not part of a word
# except a colon before a space, and _CLAUSE breaks clauses only at ";",
# ",", "!", "?" and ". ", so any such mark between a tough adjective and a
# plan (a dash of any kind, a spaced hyphen, a slash, parentheses, brackets
# or braces, an arrow, a tilde, a bullet: "her goal is simple — to
# improve", "her plan is easy [to drop by]") leaves the plan read as a tough
# gap and the clause as grounds.  No corpus text has the shape (a tough
# adjective or a copula, then any dropped mark, then "to": 0 of 17,408
# writer texts).  The one measured fix, reading every such mark like a
# colon, unbinds a true corpus pair whose grounds hold a dash-enclosed
# aside; a narrower boundary (a mark directly before "to" + a verb) is
# unmeasured (r79 Claude F2).  The same dropped mark costs recall: a noun
# phrase aside after it, following a tough infinitive or a "for" list
# ("hard to follow — odd meters", "for its energy (the grooves)"), is read
# inside the infinitive or list and is no grounds (r79 Claude F3).  A word
# ending in
# "'s" counts as a contracted copula ("her plan's to see them live"); a
# possessive ("their singer's devotion to younger fans") is read the same
# way, so an attribute "to" + a word that may be a verb after it is read as a
# plan, a recall cost only (r73 Claude F3).
_COLON = ":"
_COLON_BOUNDARY = re.compile(r":(?=\s|$)")


def _stance_tokens(text: str) -> list[str]:
    """_cased_tokens with each colon before a space or the end kept as the
    word _COLON."""
    words: list[str] = []
    for i, part in enumerate(_COLON_BOUNDARY.split(normalize(text))):
        if i:
            words.append(_COLON)
        words.extend(_cased_tokens(part))
    return words


def _infinitive_verb(word: str) -> bool:
    return (word.replace("-", "").isalpha() and word not in _DETERMINER
            and word not in _OBJECT_PRONOUN)


def _tough_gap(content: list[str], k: int) -> bool:
    """CONTENT[K] is "to" of a tough infinitive with its object gapped
    before the end of CONTENT."""
    n = len(content)
    return k > 0 and content[k - 1] in _TOUGH and (
        k + 2 == n or (k + 3 == n and content[k + 2] in _PREPOSITION))


def _copula_infinitive(content: list[str]) -> bool:
    """CONTENT (lower-cased) holds a copula, written out or contracted onto
    the word before it ("plan's"), whose first later "to" that is no
    preposition opens an infinitive plan (see above)."""
    n = len(content)
    for index, w in enumerate(content):
        if not (w in _COPULA or ((w.endswith("'s") or w.endswith("\u2019s")) and len(w) > 2)):
            continue
        for k in range(index + 1, n):
            if content[k] != "to" or k + 1 == n:
                continue
            after = content[k + 1]
            if _infinitive_verb(after):
                if not _tough_gap(content, k):
                    return True
                break
            if after not in _DETERMINER and after not in _OBJECT_PRONOUN:
                break
    return False


def _colon_parts(cased: list[str]) -> list[list[str]]:
    """CASED split at every _COLON.  The colon readings below are written
    over these parts in one pass, so a long chain of colons is read without
    nesting a call per colon (r80 Pro F1)."""
    parts: list[list[str]] = [[]]
    for word in cased:
        if word == _COLON:
            parts.append([])
        else:
            parts[-1].append(word)
    return parts


def _grounds_clause(cased: list[str], polarity: str) -> bool:
    """CASED, a comma clause (any colon kept as _COLON), opens the grounds of
    the stance (see above): the part before its first colon does, and every
    part after a colon goes on listing them (_grounds_continuation)."""
    if _COLON in cased:
        first, *rest = _colon_parts(cased)
        return (_grounds_clause(first, polarity)
                and all(_grounds_continuation(part, polarity) for part in rest))
    head = 0
    while head < len(cased) and cased[head] in _PHRASE_ADVERB:
        head += 1
    if head >= len(cased) or not _no_other_stance(cased):
        return False
    word, content = cased[head].casefold(), [w.casefold() for w in cased[head + 1:]]
    if word in _FOCUS:
        return bool(content) and content[0] in _POSSESSIVE_ANAPHOR and len(content) > 1
    if word in _ANAPHOR_LEAD:
        # A copula whose complement is an infinitive states a plan with its
        # own lifecycle ("her next task is to book the studio", "her plan
        # was to ..."), not grounds of the stance (r67 Pro F2); the check is
        # the complement's structure, not a list of plan nouns.  Any words
        # may stand between the copula and "to" ("is simply to", "is of
        # course to", "is for her little brother Sam to", "it is her goal to"),
        # the copula may be contracted ("her plan's to see them") or
        # non-finite ("has long been to", "remains to"); r68 Claude F2, r69
        # Claude F2-F4, r71 Claude F2-F5.  A "to" + a word that may be a verb
        # after a copula also reads a prepositional attribute ("is mostly to
        # younger fans", "are drawn to more modern sounds") as a plan, a
        # recall cost only.
        if _copula_infinitive(content):
            return False
        return any(w in _AUXILIARY - {"has", "have", "had", "does", "did", "do"}
                   or w in _EXPERIENCER for w in content)
    if polarity == "positive" and word in _GROUNDS_LIKING:
        # A liking verb names what it likes about the object: a possessive
        # pointing back to it (", enjoying his charm"), "how", a word for a
        # thing pointing back to it, or a "the"-led attribute.  A "the"-led
        # phrase holding a personal pronoun may instead be a second thing
        # liked, owned by someone else (", enjoying the records her brother
        # plays", where "her" is the subject).
        if not content:
            return False
        if content[0] in _ANAPHOR_LEAD or "how" in content or _THING_ANAPHOR & set(content):
            return True
        return content[0] == "the" and not (_PERSONAL_ANAPHOR & set(content))
    if word in _GROUNDS_EVALUATIVE:
        # A reported judgement about the object is grounds whether it is a
        # small clause (", finding it too dramatic") or a clause with a
        # pronoun subject pointing back to it (", finds they make you
        # think", ", saying they're so good"): both are corpus shapes.
        return bool(content) and (content[0] == "the" or "how" in content
                                  or bool(_ANAPHOR & set(content)))
    return False


def _grounds_continuation(cased: list[str], polarity: str) -> bool:
    """CASED, a comma clause right after grounds, goes on listing them
    (", innovation", ", and world music", ", distracting and hard to get
    into", ", bringing depth to her characters"): it opens grounds itself,
    or it holds no auxiliary, modal, contracted subject or word setting up
    another stance, and, after any coordinator and adverbs, is a note or
    time (_phrases) or a list of grounds items (_grounds_list), after an
    effect participle or alone.  A subject pronoun (", they prefer ..."), a
    predicate (", lives in Austin", ", eager to see them in June") or any
    other verb phrase is no grounds item.  Around colons every part goes on
    listing them."""
    if _COLON in cased:
        return all(_grounds_continuation(part, polarity) for part in _colon_parts(cased))
    if _grounds_clause(cased, polarity):
        return True
    words = [word.casefold() for word in cased]
    if (not words or not _no_other_stance(cased) or _contracted_subject(" ".join(cased))
            or any(word in _AUXILIARY or word in _MODAL for word in words)):
        return False
    head = 0
    while head < len(cased) and (words[head] in _COORDINATOR or cased[head] in _PHRASE_ADVERB):
        head += 1
    if head >= len(cased):
        return False
    if _phrases(cased, head):
        return True
    # An "after" clause states a completed event that occasioned the stance
    # (", after rewatching "Black Panther""), never something still to be
    # done; "before"/"since"/"while" clauses are not read.
    if (words[head] == "after" and head + 2 < len(cased) and _participle(cased[head + 1])
            and _grounds_list(cased, head + 2)):
        return True
    if cased[head] in _GROUNDS_EFFECT:
        return _grounds_list(cased, head + 1)
    return _grounds_list(cased, head)


def _reason_phrase(cased: list[str], polarity: str) -> bool:
    """CASED, the words after the revised object in its clause, give the
    grounds of the stance: "for" with a list of grounds items led by a
    possessive pointing back to the object (_grounds_list: "for its
    energy", "for their intricate plots and suspense (Session 1273)"), or
    "and" before a grounds clause ("and their sound", "and finds their
    music incredible", "and love how it makes you think"); colons are kept
    as _COLON (_stance_tokens) and read as in _grounds_clause."""
    if len(cased) > 2 and cased[0] == "for" and cased[1] in _POSSESSIVE_ANAPHOR:
        if _COLON in cased:
            first, *rest = _colon_parts(cased)
            return (_reason_phrase(first, polarity)
                    and all(_grounds_continuation(part, polarity) for part in rest))
        return _no_other_stance(cased) and _grounds_list(cased, 1)
    return len(cased) > 1 and cased[0] == "and" and _grounds_clause(cased[1:], polarity)


def _one_stance(prior: str, at_clause: int, grounds: bool, polarity: str) -> bool:
    """_one_proposition for a stance trace, whose clauses after AT_CLAUSE in
    the same sentence may give the stance's grounds (see above).  GROUNDS:
    the object's own clause ended in a reason phrase."""
    for i, (start, end, _, delimiter) in enumerate(_clause_spans(prior)):
        clause = prior[start:end]
        if i == at_clause or not clause.strip():
            continue
        if i > at_clause:
            cased = _stance_tokens(clause)
            inner = delimiter in ",;"
            grounds = grounds and inner
            if inner and (_grounds_clause(cased, polarity) or (
                    grounds and _grounds_continuation(cased, polarity))):
                grounds = True
                continue
        if _separate_proposition(clause):
            return False
    return True


# The phrase grammar (_phrases, _grounds_list and the readings built on them)
# nests a call for about every word it reads, so the stack it needs grows
# with the trace.  A stance or intention trace longer than this many words
# (_stance_tokens) is not read and binds no revision, before any word is
# parsed: the longest writer text in the Memora corpus has 263, and a refused
# trace keeps its stance (r80 Pro F1).  The refused trace itself only loses
# recall, but it also leaves the candidate set the runtime counts against
# _REVISION_MAX_TARGETS, so it can lift an ambiguity refusal and let the
# other candidates bind (r83 Pro R83-01; pinned as a known limit in
# test_bounded_recursion.py, no corpus trace reaches the limit).
_REVISES_MAX_WORDS = 320


def revises(ref: RevisionReference, revising_text: str, prior_text: str) -> bool:
    """True when PRIOR_TEXT plainly states the proposition REF revises, and
    nothing else.

    The subject phrase of the prior predicate must be the subject phrase of
    the revising predicate (_same_subject: the words right before each
    predicate, so a name the revising text merely mentions, a reporter, a
    possessor or a modal between subject and marker does not count).  The
    trace must state the revised object verbatim after a verb of the same
    polarity, carry no negation, revision or hedge of its own, run the object
    (or the pending action) to the end of its clause with nothing coordinated
    with it, and carry no other proposition, since the negate stance applies
    to the whole trace: _one_proposition for an intention, and for a stance
    _one_stance, which also accepts the stance's grounds (the shapes it
    binds wrongly are pinned as known limits in the tests).  Grounds are
    noun phrases (_grounds_list): the only infinitive read follows a closed
    tough adjective, and a verb is read as a noun only where spelling cannot
    tell them apart (known limits); a stance trace holding a common plan
    wording (_states_intention, a closed list, quoted titles aside) is
    refused as well.
    A value revision binds only a trace whose labelled value is exactly the
    stated old value.  REVISING_TEXT is kept for callers; the subject comes
    from REF.  A stance or intention trace longer than _REVISES_MAX_WORDS
    words is not read.  The grammar readings are memoised for this call
    only (_grammar_readings).
    """
    with _grammar_readings():
        return _revises(ref, prior_text)


def _revises(ref: RevisionReference, prior_text: str) -> bool:
    prior = normalize(prior_text)
    if ref.kind in ("intention", "stance") and len(_stance_tokens(prior)) > _REVISES_MAX_WORDS:
        return False
    # Clause structure is read from case (a capitalised word may be a name,
    # a lowercase one a predicate); a title-case or all-caps trace carries
    # no such information, so it is never shown to state one proposition
    # only, as it corroborates no name either (capitalised_words).
    english = not re.search(r"[\u4e00-\u9fff]", ref.target)
    if ref.kind in ("intention", "stance") and english and _title_case(prior):
        return False
    if ref.kind == "intention":
        if _NOT_PENDING_EN.search(prior):
            return False
        pending = intention_matches(prior)
        if len(pending) != 1 or not same_intention_action(pending[0][0], ref.target):
            return False
        _, before, text, start = pending[0]
        if not _same_subject(ref.subject, before):
            return False
        at_clause = sum(1 for match in _clause_breaks(text) if match.end() <= start)
        # The intention's own clause may go on past the action ("needs to
        # call the dentist because Bob is sick", "... today and pick up the
        # kids", "... so her sister drives the kids"); the rest of the clause
        # is read like a stance trace's tail (_adjunct) and closes the
        # intention only when it parses as phrases ("today", "before the end
        # of the day", "later that day").  A contracted subject in it ("like
        # you'd") opens a clause.  A listed task's pattern ends inside its
        # list phrase ('added "X" to their work to-do'), whose head noun
        # ("list") may come first; no other word completes the phrase.
        clause_end = _clause_spans(text)[at_clause][1]
        action = _INTENTION_EN.match(text, start)
        listed = None if action is not None else _INTENTION_LISTED.match(text, start)
        matched = action or listed
        if matched is not None:
            rest = text[matched.end():clause_end]
            cased = _cased_tokens(rest)
            if _contracted_subject(rest):
                return False
            # The action ends on its object, which may be the subject of the
            # words after it ("fix the bug Bob reported", "fix the bug
            # (reported yesterday)"), so they are read after a noun phrase.
            if not (_adjunct(cased, after_noun=True)
                    or (listed is not None and cased and cased[0] in ("list", "lists")
                        and _adjunct(cased[1:], after_noun=True))):
                return False
        return _one_proposition(text, at_clause)
    if ref.kind == "value":
        # The only parenthesis skipped is an earlier revision note ("$800
        # (changed from $1,100)", "(updated from $1,050 in session 668)"): one
        # earlier value of the same field, so no more words than the value
        # itself, optionally its session.  Any other one ("$100 (Alice paid the
        # deposit)", "(changed from $120; Alice paid the deposit)") may state a
        # separate fact the negate would withdraw.
        match = re.match(r"^(?P<label>[^:\n()]{1,48}):\s*(?P<value>[^()\n]+?)\s*"
                         r"(?:\((?:changed|updated|revised)\s+from\s+(?P<earlier>[^()\n]+?)"
                         r"(?:\s+in\s+session\s+\d+)?\))?\s*\.?$",
                         prior, re.I)
        return bool(match and normalize(match["label"]).casefold() == ref.label
                    and normalize(match["value"]).strip(" .").casefold() == ref.target
                    and (match["earlier"] is None
                         or len(match["earlier"].split()) <= len(match["value"].split())))
    if ref.kind != "stance":
        return False
    # A trace that also states a pending intention ("loves Coldplay for their
    # energy and plans to see them in June") carries a proposition with its
    # own lifecycle: a completion closes it only while the trace is current,
    # so withdrawing the stance would silently withdraw the intention too
    # (r58 Claude F1).  The grounds grammar admits no verb phrase (r60 Claude
    # F1); this closed cue list also catches the plans intention_matches
    # leaves out in a main clause (r59 Claude F1).
    if _states_intention(prior):
        return False
    if re.search(r"[\u4e00-\u9fff]", ref.target):
        if _NOT_PLAIN_ZH.search(prior):
            return False
        verbs = _NEGATIVE_ZH if ref.polarity == "negative" else _POSITIVE_ZH
        compact = prior.replace(" ", "").casefold()
        # One clause only: "用户喜欢爵士乐，也喜欢摇滚" carries two stances.
        if len([part for part in re.split(r"[\uff0c,\u3002\uff1b;\uff01!\uff1f?\u3001]", compact)
                if part]) != 1:
            return False
        for verb in verbs:
            at = compact.find(verb + ref.target)
            if at < 0:
                continue
            # The subject is what stands before the stance verb itself, so a
            # name containing a verb character ("爱丽丝") stays whole.  A
            # subject that is or opens with a deictic or personal pronoun
            # ("他", "我们", "我的妈妈", "这位老师") points into a context; as in
            # English it names no entity whose stance could be withdrawn.
            subject = compact[:at].strip()
            if (not subject or subject[:1] in _DEICTIC_ZH_HEAD
                    or subject != ref.subject.replace(" ", "").casefold()):
                return False
            rest = compact[at + len(verb + ref.target):]
            # The object must end its clause: "喜欢爵士乐和摇滚" (a coordinated
            # object), "喜欢茶并且住在北京" (a second predicate) and "喜欢茶叶蛋"
            # (a longer word) all carry more than the revised stance.
            return not rest.strip("\u3002.!?\uff01\uff1f")
        return False
    if _NOT_PLAIN_EN.search(prior):
        return False
    tokens = revision_tokens(prior)
    phrase = ref.target.split()
    if not tokens:
        return False
    positive, negative = set(_REVISION_VERBS_POSITIVE), set(_REVISION_VERBS_NEGATIVE)
    if ref.polarity == "positive" and negative & set(tokens):
        return False
    for index, (start, end, sentence, _) in enumerate(_clause_spans(prior)):
        words = revision_tokens(prior[start:end])
        at = _contains(words, phrase)
        if at < 1:
            continue
        window = words[max(0, at - _OBJECT_GAP - 1):at]
        # "dislikes Pat Metheny's jazz style" is about someone's style, so a
        # possessive between verb and object names a different head.
        if any(word.endswith(("'s", "\u2019s")) for word in window):
            return False
        # "likes action movies and Jason Statham" / "likes tea and coffee" /
        # "likes tea as well as coffee" / "likes tea but lives in Austin":
        # nothing may be coordinated with the revised object, and after it
        # its clause may hold only one adjunct ("as an actress", "lately"),
        # so withdrawing the trace withdraws nothing else.
        # The tail is read from the shared tokenizer's words (_REVISION_TOKEN);
        # only the grounds readings keep colons (_stance_tokens, r79 Claude F4).
        # A contracted subject in the tail ("as you'd expect") opens a clause.
        spans = list(_REVISION_TOKEN.finditer(normalize(prior[start:end])))
        rest = " ".join(match.group() for match in spans[at + len(phrase):])
        if _coordinated(window) or _contracted_subject(rest):
            return False
        offset = at - len(window)
        verbs = negative if ref.polarity == "negative" else positive
        verb = max((offset + i for i, word in enumerate(window) if word in verbs), default=-1)
        if verb < 0 and ref.polarity == "positive":
            fan = _contains(window, list(_FAN_OF))
            if fan >= 0:
                # "User is a big fan of X": the predicate opens at the copula.
                verb = max((i for i, word in enumerate(words[:offset + fan])
                            if word in _AUXILIARY), default=offset + fan)
        if verb < 0:
            return False
        # A gerund, participle or infinitive between the stance verb and the
        # object makes the object belong to that verb phrase ("likes avoiding
        # coffee", "likes to avoid coffee"): the stance is toward the action,
        # so withdrawing the stance toward the object names another
        # proposition.
        cased = _REVISION_TOKEN.findall(normalize(prior[start:end]))
        between = (cased if len(cased) == len(words) else words)[verb + 1:at]
        if any(word.casefold() == "to" or _participle(word) for word in between):
            return False
        # The object may be the subject of the words after it, whether a
        # subordinator ("loves it when Bob ...", "loves how storks ..."), a
        # contact clause ("loves the way storks winter ...") or a perception
        # complement ("loves watching storks winter ...") puts it there, and
        # no closed class marks the last two; so the tail is always read
        # after a noun phrase, an aside in parentheses too ("storks (winter
        # in Spain)").
        tail = _cased_tokens(rest)
        # The reason phrase keeps its colons (_stance_tokens, r75 Claude F2).
        marked = _stance_tokens(normalize(prior[start:end])[spans[at + len(phrase) - 1].end():])
        if tail and tail[0].casefold() in _CATEGORY_HEAD:
            tail = tail[1:]
        if marked and marked[0].casefold() in _CATEGORY_HEAD:
            marked = marked[1:]
        grounds = _reason_phrase(marked, ref.polarity)
        if not (grounds or _adjunct(tail, after_noun=True)):
            return False
        before = prior[sentence:start] + " " + " ".join(words[:verb])
        return (_same_subject(ref.subject, before)
                and _one_stance(prior, index, grounds, ref.polarity))
    return False
