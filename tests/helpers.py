from dataclasses import replace
import copy
import json
import unittest
from uai_memory import ConsolidationProposal, Mechanisms, Query, RuntimeConfig, UAIMemoryRuntimeV2
from uai_memory.consolidation import proposal_for
from uai_memory.prediction import event_for
from uai_memory.types import MemoryInvariantError, SourceRef, Tier

def write(runtime, key, text, *, source=None, source_id=None, **fields):
    item = {"key": key, "text": text, "quote": text, "about": ["topic"], **fields}
    return runtime.encode(source or text, json.dumps({"upsert": [item], "negate": []}),
                          source_id=source_id)


def withdraw(runtime, key, quote, **fields):
    return runtime.encode(quote, json.dumps({"upsert": [], "negate": [{"key": key, "quote": quote, **fields}]}))


def clustered(config=None):
    runtime = UAIMemoryRuntimeV2(config)
    for i in range(3):
        write(runtime, f"note_{i}", f"Alex keeps the laboratory sample in cabinet {i}")
    runtime.replay()
    assert runtime.pending_consolidation is not None
    return runtime
