"""Existing synthetic fixtures, extracted without external adapters or calls."""
from dataclasses import replace
import hashlib
import json
from uai_memory import Mechanisms, Query, RuntimeConfig, UAIMemoryRuntimeV2
from uai_memory.types import Tier

def _write(runtime, key, text, **fields):
    return runtime.encode(text, json.dumps({"upsert": [{"key": key, "text": text, "quote": text,
                           "about": ["laboratory"], **fields}]}), source_id=f"source-{runtime.operation}-{key}")


def _withdraw(runtime, key):
    text = f"The previous assertion about {key} is withdrawn."
    return runtime.encode(text, json.dumps({"negate": [{"key": key, "quote": text}]}))


def _config(switch=None, **bounds):
    mechanisms = replace(Mechanisms(), **{switch: False}) if switch else Mechanisms()
    return RuntimeConfig(mechanisms=mechanisms, **bounds)


def replay_ranking(switch=None):
    runtime = UAIMemoryRuntimeV2(_config(switch, replay_limit=2))
    for i in range(12):
        _write(runtime, f"specimen{i}", f"Specimen {i} is kept in a separate temperature controlled laboratory cabinet")
    for _ in range(8):
        runtime.maintenance()
    tiers = [t.tier for t in runtime._traces.values()]
    return {"warm": sum(t == Tier.WARM for t in tiers),
            "replays": runtime.telemetry.summary()["replay_triggered"],
            "requests": runtime.telemetry.summary()["consolidations.requested"],
            "independent_support": sum(t.recurrence for t in runtime._traces.values()),
            "covered_traces": sum(t.replay_count > 0 for t in runtime._traces.values()),
            "replay_signature": hashlib.sha256(json.dumps(
                [(e["operation"], e["ranked"]) for e in runtime.telemetry.events if e["action"] == "replay"]
            ).encode()).hexdigest()}


def consolidation_gate(switch=None):
    runtime = UAIMemoryRuntimeV2(_config(switch))
    for i, name in enumerate(("Alex", "Blair", "Casey", "Drew")):
        _write(runtime, f"sample{i}", f"{name} stores specimen {i} in the temperature controlled laboratory cabinet")
    runtime.maintenance()
    request = runtime.pending_consolidation
    source_words = sum(len(t.split()) for t in request.texts) if request else 0
    key = runtime.consolidate()
    card_words = len(runtime.trace(key).current.text.split()) if key else 0
    if key:
        _withdraw(runtime, request.sources[0].key)
    selected = runtime.peek("temperature controlled laboratory cabinet")
    return {"accepted_cards": int(key is not None), "source_words": source_words, "card_words": card_words,
            "stale_card_expressed": sum(e.key == key and e.current for e in selected),
            "card_destabilized": int(bool(key and runtime.trace(key).stale)),
            "unwithdrawn_sources_reexpressed": sum(e.current and e.key.startswith("sample") for e in selected),
            "requests": runtime.telemetry.summary()["consolidations.requested"],
            "late_allowed": runtime.telemetry.summary()["vertical_consolidation_allowed"]}


def spread_competition(switch=None):
    runtime = UAIMemoryRuntimeV2(_config(switch, reader_records=3, reader_words=200, reader_bytes=1000))
    _write(runtime, "target", "The sample audit confirmed a successful independent validation")
    _write(runtime, "support", "Independent validation confirmed the sample measurement")
    _write(runtime, "distractor", "The sample audit recorded an unverified pending claim")
    # Repeated joint use supplies a relational signal beyond isolated overlap.
    runtime.graph.coactivate(["support", "target"])
    runtime.graph.coactivate(["support", "target"])
    found = runtime.peek("sample audit independent validation")
    score_map = {e.key: e.score for e in found}
    return {"target_selected": int(any(e.key == "target" for e in found)),
            "distractor_selected": int(any(e.key == "distractor" for e in found)),
            "distractor_score": score_map.get("distractor", 0.0),
            "selection": [e.key for e in found], "scores": score_map}


def conflict_update(switch=None):
    runtime = UAIMemoryRuntimeV2(_config(switch))
    slot = {"entity": "Alex", "attribute": "current_residence", "exclusive": True}
    _write(runtime, "paris", "Alex currently lives in Paris", **slot)
    _write(runtime, "rome", "Alex currently lives in Rome", **slot)
    unresolved = runtime.peek("Alex currently lives")
    _withdraw(runtime, "paris")
    current = runtime.peek("Alex currently lives")
    _write(runtime, "music", "Alex likes music")
    _write(runtime, "painting", "Alex likes painting")
    return {"unresolved_claims": sum(e.ambiguous for e in unresolved),
            "resolved_current_rome": int(any(e.key == "rome" and not e.ambiguous for e in current)),
            "withdrawn_paris_current": int(any(e.key == "paris" for e in current)),
            "history_preserved": len(runtime.trace("paris").history),
            "compatible_preferences_preserved": int(all(runtime.trace(k).current.action == "assert"
                                                       for k in ("music", "painting")))}


def main():
    result = {
        "scope": "Synthetic engineering diagnostics, not a benchmark",
        "external_model_calls": 0,
        "replay": replay_ranking(),
        "consolidation": consolidation_gate(),
        "competition": spread_competition(),
        "conflict_and_withdrawal": conflict_update(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
