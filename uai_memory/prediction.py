"""Online conditional event distributions, independent of stored trace access.

This finite categorical predictor is an engineering hypothesis, not a model of
SOM cells. It learns only admitted source events. Entropy, finite-sample strength
and pre-update surprise remain separate from lexical novelty and truth support.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re

from .admission import normalize
from .config import Mechanisms


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def event_for(row: dict, action: str, text: str, scope: str, pool: str,
              task: str = "default") -> dict:
    content = normalize(text).casefold()
    relation = content
    # A source-grounded entity slot can be factored out for relation transfer.
    # No free semantic conclusion, exclusivity or value is inferred here.
    if row.get("entity") and row.get("attribute"):
        entity = normalize(row["entity"]).casefold()
        relation = re.sub(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", "{entity}", content)
    return {"scope": scope, "task": task,
            "channel": normalize(row.get("attribute") or pool).casefold(),
            "event": action + ":" + relation, "content": identity([action, content]),
            # A semantic frame is an internal schema cue.  It is deliberately
            # separate from source exposure and may be empty for an episodic
            # event that has not yet matched a learned frame.
            "schema": normalize(row.get("schema") or "").casefold()}


def schema_identity(segments: tuple[str, ...] | list[str]) -> str:
    """Stable identity for a lossless semantic frame's shared structure."""
    return identity({"segments": list(segments)})


def _scoped_schema_identity(schema: str, scope: str, channel: str,
                            task: str = "default") -> str:
    """Separate support for one structural frame across learned contexts."""
    return identity({"schema": schema, "scope": scope, "task": task, "channel": channel})


def _frame_structure(prototype: str) -> str:
    """Return a frame's shared relational form without its example fillers.

    A card prototype renders the frame as ''{1} stores {2} sample ... :: rows''.
    The example rows are instances, not the schema.  Comparing a new event
    with them penalizes exactly the novel fillers a schema must generalize
    over, so only the literal segments are the predictive cue.
    """
    structure = prototype.split(" :: ", 1)[0]
    return re.sub(r"\{\d+\}", " ", structure)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize(text).casefold()))


def _overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


class ContextPredictor:
    def __init__(self):
        self.exposures: dict[str, dict] = {}
        self.contexts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
        self.background: dict[str, Counter] = defaultdict(Counter)
        self.familiarity: Counter = Counter()
        self._content_sources: set[tuple[str, str]] = set()
        self._context_sources: set[tuple[str, str, str, str, str]] = set()
        self._background_sources: set[tuple[str, str, str]] = set()
        # Schema support is learned from accepted local cards, not from a new
        # environmental exposure.  It therefore cannot inflate recurrence or
        # context_observations and remains safe across replay/reopen.
        self.schemas: dict[str, dict] = {}

    def learn_schema(self, schema: str, *, scope: str, channel: str,
                     card_id: str, prototype: str, task: str = "default") -> bool:
        if (type(schema) is not str or not schema or type(scope) is not str
                or not scope or type(channel) is not str or not channel
                or type(task) is not str or not task
                or type(card_id) is not str or not card_id
                or type(prototype) is not str or not prototype):
            raise ValueError("SCHEMA_OBSERVATION_INVALID")
        storage_schema = _scoped_schema_identity(schema, scope, channel, task)
        row = self.schemas.setdefault(storage_schema, {"scope": scope, "task": task,
                                               "channel": channel,
                                               "prototype": prototype,
                                               "prototypes": [prototype], "card_ids": []})
        # A structural schema can have many concrete examples.  Retaining
        # each admitted prototype keeps matching evidence-backed without
        # mistaking different slot values for a schema collision.
        prototypes = row.setdefault("prototypes", [row["prototype"]])
        if prototype not in prototypes:
            prototypes.append(prototype)
            prototypes.sort()
        if card_id in row["card_ids"]:
            return False
        row["card_ids"].append(card_id)
        row["card_ids"].sort()
        return True

    def forget_schema_card(self, card_id: str, *, prototypes: dict[str, str]) -> None:
        """Remove a stale derived card from contextual schema support."""
        empty = []
        for schema, row in self.schemas.items():
            if card_id not in row['card_ids']:
                continue
            row["card_ids"] = [value for value in row["card_ids"] if value != card_id]
            if not row["card_ids"]:
                empty.append(schema)
            else:
                # Card text is already held by the runtime. Reuse that source
                # instead of retaining a prototype whose support was retired.
                row['prototypes'] = sorted({prototypes[key] for key in row['card_ids']})
                if row['prototype'] not in row['prototypes']:
                    row['prototype'] = row['prototypes'][0]
        for schema in empty:
            del self.schemas[schema]

    def _match_schema(self, event: dict) -> tuple[str, dict] | None:
        best: tuple[float, str, dict] | None = None
        for schema, row in self.schemas.items():
            if (row["scope"] != event["scope"] or row.get("task", "default") != event["task"]
                    or row["channel"] != event["channel"]):
                continue
            prototypes = row.get("prototypes", [row["prototype"]])
            score = max(_overlap(event["event"], _frame_structure(prototype))
                        for prototype in prototypes)
            candidate = (score, schema, row)
            if score >= 0.5 and (best is None or candidate[:2] > best[:2]):
                best = candidate
        return (best[1], best[2]) if best is not None else None

    def preview(self, event: dict, source_id: str, switches: Mechanisms) -> dict:
        counts = self.contexts.get((event["scope"], event["task"], event["channel"]), Counter())
        background = self.background.get(event["channel"], Counter())
        n, k = sum(counts.values()), len(counts)
        # Unit Dirichlet smoothing includes a reserved unseen-outcome bucket.
        p = (counts[event["event"]] + 1) / (n + k + 1)
        bg = (background[event["event"]] + 1) / (sum(background.values()) + len(background) + 1)
        probabilities = [(count + 1) / (n + k + 1) for count in counts.values()]
        probabilities.append(1 / (n + k + 1))
        entropy = -sum(value * math.log(value) for value in probabilities)
        uncertainty = entropy / math.log(k + 1) if k else 1.0
        precision = max(0.0, min(1.0, 1 - uncertainty)) if n else 0.0
        strength = n / (n + k + 1)
        prior = self.familiarity[event["content"]]
        schema = event.get("schema", "")
        schema_row = self.schemas.get(schema) if schema else None
        if schema_row is None:
            matched = self._match_schema(event)
            if matched is not None:
                schema, schema_row = matched
        schema_samples = len(schema_row["card_ids"]) if schema_row else 0
        # A learned frame makes a matching relational form more expected.  The
        # support is bounded and is not a truth signal; it only reduces the
        # contextual deviance term when the frame is already known.
        schema_strength = (schema_samples / (schema_samples + 1)
                           if schema_samples else 0.0)
        base_context_error = precision * strength * (1 - p)
        return {"status": "LEARNED" if n else "UNLEARNED", "context_samples": n,
                "prior_exposures": prior, "probability": p, "background_probability": bg,
                "surprisal": -math.log(p), "context_log_ratio": math.log(bg) - math.log(p),
                "uncertainty": uncertainty, "precision": precision, "sample_strength": strength,
                "schema": schema, "schema_samples": schema_samples,
                "schema_strength": schema_strength,
                "processing_gain": 1 / (1 + prior) if switches.repetition_adaptation else 1.0,
                # Low probability in an uncertain, diffuse context is distinct
                # from violation of a well-supported concentrated expectation.
                "context_error": (base_context_error * (1 - schema_strength)
                                  if switches.context_prediction else 0.0),
                "independent": identity({**event, "source_id": source_id}) not in self.exposures}

    def learn(self, event: dict, source_id: str) -> bool:
        row = {**event, "source_id": source_id}
        key = identity(row)
        if key in self.exposures:
            return False
        self.exposures[key] = row
        context = (source_id, event["scope"], event["task"], event["channel"], event["event"])
        background = (source_id, event["channel"], event["event"])
        content = (source_id, event["content"])
        # Aliased keys/entities/scopes in one source do not create extra
        # independent support for the same content or conditional event.
        if context not in self._context_sources:
            self._context_sources.add(context)
            self.contexts[event["scope"], event["task"], event["channel"]][event["event"]] += 1
        if background not in self._background_sources:
            self._background_sources.add(background)
            self.background[event["channel"]][event["event"]] += 1
        if content not in self._content_sources:
            self._content_sources.add(content)
            self.familiarity[event["content"]] += 1
        return True

    def snapshot(self) -> dict:
        return {"schema": "uai-core-v2.context-prediction.v1",
                "exposures": [self.exposures[key] for key in sorted(self.exposures)],
                "schemas": {key: {"scope": value["scope"], "task": value.get("task", "default"),
                                   "channel": value["channel"],
                                   "prototype": value["prototype"],
                                   "prototypes": list(value.get("prototypes", [value["prototype"]])),
                                   "card_ids": list(value["card_ids"])}
                            for key, value in sorted(self.schemas.items())}}

    @classmethod
    def restore(cls, state: dict, source_hashes: dict):
        if (type(state) is not dict or set(state) not in ({"schema", "exposures"},
                                                           {"schema", "exposures", "schemas"})
                or state["schema"] != "uai-core-v2.context-prediction.v1"
                or type(state["exposures"]) is not list):
            raise ValueError("CONTEXT_STATE_INVALID")
        model = cls()
        for raw_row in state["exposures"]:
            if type(raw_row) is not dict:
                raise ValueError("CONTEXT_EXPOSURE_INVALID")
            row = dict(raw_row)
            row.setdefault("task", "default")
            required = {"scope", "task", "channel", "event", "content", "source_id"}
            allowed = required | {"schema"}
            if (type(row) is not dict or not required.issubset(row)
                    or set(row) - allowed
                    or any(type(row[key]) is not str or not row[key] for key in required)
                    or ("schema" in row and (type(row["schema"]) is not str))
                    or not re.fullmatch(r"[0-9a-f]{64}", row["content"])
                    or row["source_id"] not in source_hashes
                    or not row["event"].startswith(("assert:", "negate:"))):
                raise ValueError("CONTEXT_EXPOSURE_INVALID")
            event = {key: value for key, value in row.items() if key != "source_id"}
            event.setdefault("schema", "")
            if not model.learn(event, row["source_id"]):
                raise ValueError("CONTEXT_EXPOSURE_DUPLICATE")
        for schema, raw_row in state.get("schemas", {}).items():
            if type(raw_row) is not dict:
                raise ValueError("CONTEXT_SCHEMA_INVALID")
            row = dict(raw_row)
            row.setdefault("task", "default")
            if (type(schema) is not str or not schema or type(row) is not dict
                    or set(row) not in ({"scope", "task", "channel", "card_ids"},
                                         {"scope", "task", "channel", "prototype", "card_ids"},
                                         {"scope", "task", "channel", "prototype", "prototypes", "card_ids"})
                    or type(row["scope"]) is not str or not row["scope"]
                    or type(row["task"]) is not str or not row["task"]
                    or type(row["channel"]) is not str or not row["channel"]
                    or ("prototype" in row and (type(row["prototype"]) is not str
                                                  or not row["prototype"]))
                    or ("prototypes" in row and (type(row["prototypes"]) is not list
                                                  or not row["prototypes"]
                                                  or any(type(value) is not str or not value
                                                         for value in row["prototypes"])
                                                  or row["prototypes"] != sorted(set(row["prototypes"]))))
                    or type(row["card_ids"]) is not list
                    or any(type(card_id) is not str or not card_id for card_id in row["card_ids"])
                    or row["card_ids"] != sorted(set(row["card_ids"]))):
                raise ValueError("CONTEXT_SCHEMA_INVALID")
            prototype = row.get("prototype") or row.get("prototypes", [schema])[0]
            prototypes = list(row.get("prototypes", [prototype]))
            if prototype not in prototypes:
                prototypes.append(prototype)
                prototypes.sort()
            model.schemas[schema] = {"scope": row["scope"], "task": row["task"],
                                     "channel": row["channel"],
                                     "prototype": prototype, "prototypes": prototypes,
                                     "card_ids": list(row["card_ids"])}
        return model
