from dataclasses import replace
import copy
import json
import unittest
from uai_memory import ConsolidationProposal, Mechanisms, Query, RuntimeConfig, UAIMemoryRuntimeV2
from uai_memory.consolidation import proposal_for
from uai_memory.prediction import event_for
from uai_memory.types import MemoryInvariantError, SourceRef, Tier
from tests.helpers import write, withdraw, clustered

def test_positive_preference_does_not_erase_previous_preference():
    runtime = UAIMemoryRuntimeV2()
    write(runtime, "x", "Alex likes X")
    write(runtime, "y", "Alex likes Y")
    assert {e.key for e in runtime.peek("Alex likes")} == {"x", "y"}
    assert runtime.trace("x").current.action == "assert"


def test_stance_flips_remain_in_history_and_current_read_excludes_withdrawn():
    runtime = UAIMemoryRuntimeV2()
    write(runtime, "x", "Alex likes X", when="2025-01-01")
    withdraw(runtime, "x", "Alex no longer likes X", when="2025-02-01")
    assert "Alex likes X" not in runtime.read("Alex", touch=False)
    assert "negate" in runtime.read(Query("Alex", view="history"), touch=False)
    write(runtime, "x", "Alex likes X", when="2025-03-01")
    assert [s.action for s in runtime.trace("x").history] == ["assert", "negate", "assert"]
    assert "2025-02-01" in runtime.read(Query("Alex", view="history"), touch=False)
    assert runtime.trace("x").current.action == "assert"


def test_as_of_query_preserves_stated_past_without_claiming_current_fact():
    runtime = UAIMemoryRuntimeV2()
    write(runtime, "city", "Alex lives in Paris")
    first = runtime.operation
    write(runtime, "city", "Alex lives in Rome")
    text = runtime.read(Query("Alex", at_operation=first), touch=False)
    assert "Paris" in text and "Rome" not in text and "HISTORY" in text
    assert "Paris" not in runtime.read("Alex", touch=False)


def test_scope_gate_and_scope_key_collision():
    runtime = UAIMemoryRuntimeV2()
    write(runtime, "home", "Alex keeps X at home", scope="home")
    assert runtime.peek(Query("Alex", scope="work")) == []
    assert runtime.peek(Query("Alex", scope="home"))
    before = runtime.trace("home")
    result = write(runtime, "home", "Alex keeps Y at work", scope="work")
    assert result.dropped[0]["reason"] == "ITEM_SCOPE_KEY_COLLISION"
    assert runtime.trace("home").history == before.history


def test_replay_and_usage_do_not_create_independent_evidence():
    runtime = UAIMemoryRuntimeV2()
    write(runtime, "x", "Alex likes X", source_id="source1")
    support = runtime.trace("x").recurrence
    runtime.replay()
    runtime.retrieve("Alex")
    assert runtime.trace("x").recurrence == support == 1
    assert runtime.trace("x").replay_count >= 2
    assert runtime.trace("x").access_count == 1
    assert runtime.trace("x").tier == Tier.WARM
    write(runtime, "x", "Alex likes X", source_id="source2")
    assert runtime.trace("x").recurrence == 2
    assert runtime.trace("x").revision == 1


def test_item_rejection_and_metadata_dependency_pruning_leave_valid_fact():
    runtime = UAIMemoryRuntimeV2()
    source = "Session 1 took place on Monday. Alex stores the blue sample."
    rows = [{"key": "session1", "text": "Session 1 took place on Monday", "quote": "Session 1"},
            {"key": "good", "text": "Alex stores the blue sample", "quote": "Alex stores the blue sample",
             "depends_on": ["session1"]},
            {"key": "bad", "text": "invented", "quote": "not in source"}]
    result = runtime.encode(source, json.dumps({"upsert": rows}))
    assert result.status == "APPLIED" and result.applied == ("good",)
    assert runtime.trace("good").depends_on == ()
    assert {d["reason"] for d in result.dropped} == {"ITEM_METADATA_ONLY", "ITEM_QUOTE_NOT_FOUND"}
    assert result.pruned[0]["dependencies"] == ["session1"]


def test_writer_eviction_is_counted_and_never_executed():
    runtime = UAIMemoryRuntimeV2()
    source = "Alex stores the blue sample."
    runtime.encode(source, json.dumps({"upsert": [{"key": "blue", "text": "Alex stores the blue sample",
                                                   "quote": "Alex stores the blue sample"}]}))
    result = runtime.encode("Alex keeps notes.", json.dumps(
        {"upsert": [], "evict": [{"key": "blue"}, {"key": "unknown"}]}))
    assert result.status == "APPLIED" and result.dropped == ()
    assert runtime.trace("blue").current.text == "Alex stores the blue sample"
    summary = runtime.telemetry.summary()
    assert summary["writer_evictions_ignored"] == 2 and summary["writes_proposed"] == 1
    assert runtime.telemetry.events[-1]["ignored_evictions"] == 2


def test_dependency_change_invalidates_child_until_reasserted():
    runtime = UAIMemoryRuntimeV2()
    write(runtime, "parent", "The procedure uses buffer X")
    write(runtime, "child", "Use buffer X before incubation", depends_on=["parent"])
    write(runtime, "parent", "The procedure uses buffer Y")
    assert runtime.trace("child").stale
    assert "child" not in {e.key for e in runtime.peek("buffer")}
    write(runtime, "child", "Use buffer Y before incubation", depends_on=["parent"])
    assert not runtime.trace("child").stale


def test_existing_node_dependency_cycle_is_rejected_without_mutation():
    runtime = UAIMemoryRuntimeV2()
    write(runtime, "a", "A is a fact")
    write(runtime, "b", "B depends on A", depends_on=["a"])
    before = runtime.trace("a")
    result = write(runtime, "a", "A now depends on B", depends_on=["b"])
    assert result.dropped[0]["reason"] == "ITEM_DEPENDENCY_CYCLE"
    assert runtime.trace("a").history == before.history


def test_dependency_closure_includes_shared_ancestor_once():
    runtime = UAIMemoryRuntimeV2(RuntimeConfig(mechanisms=Mechanisms(competition=False)))
    write(runtime, "a", "Alex uses A")
    write(runtime, "b", "Alex uses B", depends_on=["a"])
    write(runtime, "c", "Alex uses C", depends_on=["a"])
    write(runtime, "d", "Alex uses D", depends_on=["b", "c"])
    found = [e.key for e in runtime.peek("Alex")]
    assert set(found) == {"a", "b", "c", "d"} and len(found) == 4


def test_capacity_never_rejects_and_cold_fact_remains_cue_accessible():
    runtime = UAIMemoryRuntimeV2(RuntimeConfig(reader_words=8, reader_bytes=55, reader_records=1))
    for i in range(4):
        assert write(runtime, f"fact_{i}", f"Alex remembers unique{i} sample").status == "APPLIED"
    assert len(runtime.traces()) == 4
    assert runtime.telemetry.summary()["demotions"] >= 3  # the budget bound expression, not storage
    assert runtime.window_usage()["records"] <= 1
    cold = next(t for t in runtime.traces().values() if t.tier == Tier.COLD)
    assert cold.key in {e.key for e in runtime.peek(cold.current.text)}


def test_hard_forgetting_excludes_cues_but_preserves_provenance():
    runtime = UAIMemoryRuntimeV2()
    write(runtime, "x", "Alex likes X")
    runtime.forget("x", mode="hard")
    before = runtime.trace("x")
    assert runtime.peek(Query("Alex", view="history")) == []
    result = write(runtime, "x", "Alex likes Y")
    assert result.dropped[0]["reason"] == "ITEM_HARD_FORGOTTEN"
    assert runtime.trace("x") == before


def test_exclusive_slot_conflict_is_marked_not_auto_resolved():
    runtime = UAIMemoryRuntimeV2()
    slot = {"entity": "Alex", "attribute": "current_city", "exclusive": True}
    write(runtime, "paris", "Alex lives in Paris", **slot)
    write(runtime, "rome", "Alex lives in Rome", **slot)
    items = runtime.peek("Alex")
    assert len(items) == 2 and all(e.ambiguous for e in items)
    # Both claims stay expressed; the warning carries their experienced order.
    labels = {e.key: e.text for e in items}
    assert labels["paris"].startswith("COMPETING EXCLUSIVE-SLOT CLAIM (follows 0, followed by 1 ")
    assert labels["rome"].startswith("COMPETING EXCLUSIVE-SLOT CLAIM (follows 1, followed by 0 ")
    assert all("UNRESOLVED" not in text and "LATEST" not in text for text in labels.values())
    withdraw(runtime, "paris", "Alex no longer lives in Paris")
    assert not runtime.peek("Alex")[0].ambiguous


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value, description=name)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )
