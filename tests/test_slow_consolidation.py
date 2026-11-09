from dataclasses import replace
import copy
import json
import unittest
from uai_memory import ConsolidationProposal, Mechanisms, Query, RuntimeConfig, UAIMemoryRuntimeV2
from uai_memory.consolidation import proposal_for
from uai_memory.prediction import event_for
from uai_memory.types import MemoryInvariantError, SourceRef, Tier
from tests.helpers import write, withdraw, clustered

def test_slow_runtime_requires_an_independent_maintenance_epoch():
    # Reuse the existing coactivation fixture so the request is produced by
    # the real cluster gate rather than by a test-only shortcut.
    runtime = clustered(RuntimeConfig(slow_consolidation=True))
    assert runtime.pending_consolidation is not None
    assert not any(t.kind == "semantic_card" for t in runtime.traces().values())
    assert runtime.consolidate() is None
    assert runtime.telemetry.summary()["consolidations.deferred"] == 1
    runtime.maintenance()
    card = runtime.consolidate()
    assert card is not None
    assert runtime.maintenance_epoch == 1
    runtime.check_invariants()


def test_slow_runtime_direct_call_cannot_consume_a_new_request():
    runtime = clustered(RuntimeConfig(slow_consolidation=True))
    runtime.pending_consolidation = None
    assert runtime.consolidate() is None
    assert runtime.telemetry.summary()["consolidations.deferred"] == 1
    runtime.maintenance()
    assert runtime.consolidate() is not None


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value, description=name)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )
