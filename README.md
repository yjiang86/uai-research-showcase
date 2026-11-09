# UAI Memory Research — selected review source

**Brain-inspired persistent memory, with explicit admission, lifecycle and
consolidation mechanisms.** March 2026–present.

This curated snapshot makes one implemented part of the larger UAI research
program inspectable and runnable. Experience is represented as versioned
Engram-like traces with source-bound writes, conflicts, withdrawal, replay,
separate memory tiers and controlled consolidation. Retrieval is one consumer
of those states. The software is a research prototype, not a biological model
or a claim of benchmark superiority.

## Run locally

Use Python 3.12 from the repository root. No pip packages, model accounts,
network services or downloaded datasets are needed.

```sh
python -m examples.mechanisms
python -m unittest discover -s tests -t . -v
```

The first command prints actual results from four existing synthetic fixtures:
replay, consolidation, competition, and conflict/withdrawal. All people,
locations and samples in the examples are fictional. The second runs selected
existing behavioral assertions through the standard-library unittest runner.
See [validation](VALIDATION.md) for the recorded execution and its limits.

## Navigate the implementation

| Component | Role |
|---|---|
| `uai_memory/admission.py` | Parse and validate source-bound proposed updates |
| `uai_memory/types.py` | Keep source identity, stance, revision and tier separate |
| `uai_memory/runtime.py` | Encoding, lifecycle transitions, replay and snapshots |
| `uai_memory/conflict.py` | Handle incompatible claims and explicit revisions |
| `uai_memory/consolidation.py` | Build source-bound, reconstructable semantic cards |
| `uai_memory/integration.py` | Select bounded evidence for a query |
| `uai_memory/prediction.py` | Track contextual prediction signals and their support |

[Architecture](ARCHITECTURE.md) explains the data flow.
[Contributions](CONTRIBUTIONS.md) separates research leadership from
AI-assisted implementation. [Research status](RESEARCH_STATUS.md) keeps the
larger program's results and unfinished directions distinct from this snapshot.

## Delivery scope

This repository starts with a clean, independent history. It contains selected
implementation modules and synthetic checks, not the original research
repository's history, datasets, model artifacts or operating environment.
The library modules retain their original content; their package directory is
relocated to `uai_memory`. Selected test and fixture functions retain their
original bodies; only their import context and small command-line wrappers
are adapted.

See [included/excluded material](INCLUDE_EXCLUDE.md), [references](REFERENCES.md)
and [rights and attribution](NOTICE.md). This is a private evaluation copy;
no open-source license is granted for the original project code.
