# UAI — Persistent Memory and Controlled Evaluation

March 2026–present

## CV bullets

- Led a neuroscience-inspired investigation of persistent AI memory, specifying admission, provenance, revision, withdrawal, replay, and consolidation, and directing AI-assisted implementation of a deterministic runtime with durable recovery.
- Designed controlled evaluations with shared candidate evidence, fixed readers, and matched context budgets; isolated conditional lifecycle-policy gains in synthetic drift and rollback tasks while retaining phase-specific losses and withdrawing confounded findings.
- Directed mechanism ablations and public-data local evaluations, using negative retrieval results, invalid assays, and failed adoption criteria to refine hypotheses and distinguish engineering correctness from demonstrated task benefit.

## Project paragraph

UAI investigates how experience should change an AI system’s persistent memory while preserving evidence, handling conflicting updates, and limiting harmful carryover. I lead the research framing, mechanism requirements, experimental criteria, and supervision of AI-assisted engineering.

The prototypes represent memories as source-linked, versioned objects and implement bounded admission, expression, withdrawal, replay, consolidation, and recovery operations. To separate lifecycle policy from retrieval and prompt assistance, I designed comparisons with shared candidates, fixed readers, neutral labels, and matched context budgets. In an accepted study of 128 synthetic worlds, governance added approximately 17.9–19.1 percentage points of lifecycle accuracy within three retrieval surfaces, chiefly through drift and rollback, with losses on ordinary correction and harm probes. This result assumes oracle lifecycle labels and is not a public-leaderboard comparison.

Public-data evaluations provided essential counterevidence: the standalone lexical retrieval path underperformed stronger local baselines, and a later memory-policy extension failed its frozen retrieval and forgetting criteria and was not adopted. Earlier confounded positive interpretations were withdrawn rather than retained as headline results.

The engineering evidence includes saved-trajectory replay, mechanism controls, and durable reconstruction checks. Consolidation changes external state and semantic structure; it does not demonstrate model-weight learning. General natural-language lifecycle detection, end-to-end benefits from the latest mechanisms, and biological equivalence remain unproven.
