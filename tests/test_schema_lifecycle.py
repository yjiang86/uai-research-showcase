from dataclasses import replace
import copy
import json
import unittest
from uai_memory import ConsolidationProposal, Mechanisms, Query, RuntimeConfig, UAIMemoryRuntimeV2
from uai_memory.consolidation import proposal_for
from uai_memory.prediction import event_for
from uai_memory.types import MemoryInvariantError, SourceRef, Tier
from tests.helpers import write, withdraw, clustered

def learned_runtime(**mechanisms):
    runtime = UAIMemoryRuntimeV2(RuntimeConfig(mechanisms=Mechanisms(**mechanisms)))
    rows = []
    for key, name, color in [('a', 'Alex', 'blue'), ('b', 'Blair', 'red'), ('c', 'Casey', 'green')]:
        text = f'{name} stores {color} sample in the temperature controlled laboratory cabinet'
        rows.append({'key': key, 'text': text, 'quote': text, 'about': ['laboratory']})
    runtime.encode('\n'.join(row['text'] for row in rows), json.dumps({'upsert': rows}), source_id='samples')
    runtime.replay()
    card = runtime.consolidate()
    assert card is not None and runtime.trace(card).frame is not None
    for i in range(8):
        text = 'Routine calibrations follow an unrelated protocol every morning'
        runtime.encode(text, json.dumps({'upsert': [{'key': 'routine', 'text': text,
            'quote': text, 'about': ['laboratory']}]}), source_id=f'routine-{i}')
    return runtime, card


def preview(runtime):
    text = 'Alex stores blue sample in temperature controlled laboratory cabinet'
    event = event_for({}, 'assert', text, 'default', 'laboratory')
    return runtime.context_prediction.preview(event, 'unseen-probe', runtime.config.mechanisms)


def test_hard_forgetting_card_removes_predictive_support_and_releases_sources():
    runtime, card = learned_runtime()
    before = runtime.snapshot()
    soft = UAIMemoryRuntimeV2.restore(before)
    hard = UAIMemoryRuntimeV2.restore(before)
    soft.forget(card, mode='soft')
    hard.forget(card, mode='hard')
    soft_observation, hard_observation = preview(soft), preview(hard)
    assert soft_observation['schema_samples'] == 1
    assert soft_observation['context_error'] > 0
    assert hard_observation['schema_samples'] == 0
    assert hard_observation['context_error'] > soft_observation['context_error']
    assert hard.context_prediction.exposures == runtime.context_prediction.exposures
    assert hard.trace(card).hard_forgotten
    assert card not in {item.key for item in hard.peek('laboratory cabinet')}
    assert {'a', 'b', 'c'} <= {item.key for item in hard.peek('laboratory cabinet')}
    for key in 'abc':
        assert hard.trace(key).merged_into is None
        assert hard.trace(key).history == runtime.trace(key).history
        assert hard.trace(key).recurrence == 1
    assert soft.trace('a').merged_into == card
    assert UAIMemoryRuntimeV2.restore(hard.snapshot()).snapshot() == hard.snapshot()
    hard.check_invariants()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value, description=name)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )
