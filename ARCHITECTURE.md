# Architecture

The central question is how experience should change persistent state across
different timescales. The implemented path is:

```text
source text + proposed structured update
    -> source/quote and dependency admission
    -> trace identity + stance history + revision
    -> conflict and validity propagation
    -> bounded retrieval / replay selection
    -> gated, source-revision-bound consolidation
    -> semantic cards that retire when their support becomes invalid
```

A `Trace` separates what was asserted, where its support came from, whether it
is currently expressible, and its memory tier. Re-observing identical content
can add source support without fabricating a new content revision. Replay and
retrieval usage do not become independent source evidence.

Conflicting exclusive-slot claims remain visible as competing evidence until
supported lifecycle operations resolve them. Updating or withdrawing a parent
invalidates dependent content. Historical queries can inspect prior stances
without treating them as current facts.

Consolidation verifies source revisions and reconstructable claims. Optional
slow consolidation requires a separate maintenance epoch, separating immediate
encoding from later semantic integration. It is state-level consolidation;
this snapshot does not train neural weights.

Snapshots preserve and validate state rather than merely dumping text.
Mechanism switches support controlled engineering comparisons. The natural-
language helpers are bounded heuristics; they do not establish reliable
interpretation of arbitrary real-world language.

The runtime uses only the Python standard library. It has no model client,
training loop, scheduler, hardware controller or network transport. The
structured update input is explicit in the examples so their outcomes cannot
be mistaken for an autonomous language-model extraction experiment.
