from dataclasses import replace
import copy
import json
import unittest
from uai_memory import ConsolidationProposal, Mechanisms, Query, RuntimeConfig, UAIMemoryRuntimeV2
from uai_memory.consolidation import proposal_for
from uai_memory.prediction import event_for
from uai_memory.types import MemoryInvariantError, SourceRef, Tier
from tests.helpers import write, withdraw, clustered

def stream(runtime):
    for i in range(6):
        source = f"Alex keeps the laboratory sample in cabinet {i}"
        runtime.encode(source, json.dumps({"upsert": [{"key": f"sample{i}", "text": source,
                                  "quote": source, "about": ["sample"]}]}), source_id=f"source{i}")
    runtime.replay()
    runtime.check_invariants()


def test_deterministic_full_sequence_and_snapshot():
    left, right = UAIMemoryRuntimeV2(), UAIMemoryRuntimeV2()
    stream(left)
    stream(right)
    assert left.snapshot() == right.snapshot()


def test_readonly_peek_and_returned_trace_cannot_mutate_runtime():
    runtime = UAIMemoryRuntimeV2()
    stream(runtime)
    frozen = runtime.snapshot()
    runtime.peek("Alex")
    trace = runtime.trace("sample0")
    trace.history.clear()
    assert frozen == runtime.snapshot()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value, description=name)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )
