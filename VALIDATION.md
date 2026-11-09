# Validation

This source subset was actually executed with CPython 3.12.10
on Windows. No third-party packages were installed.

| Check | Actual result |
|---|---|
| Closed file set | 37 intended files (one public documentation supplement added); missing 0, extra 0 |
| Runtime extraction | 18 original module contents match their source digests |
| Selected existing test/fixture function bodies | AST-equivalent; import context and wrappers adapted |
| `python -m unittest discover -s tests -t . -v` | 18 tests passed |
| `python -m examples.mechanisms` | Exit 0; four existing synthetic diagnostics executed |
| Sensitive-content scan of intended files | No matches in the configured path/host/account/secret checks |

The actual example reported replay coverage 12,
accepted consolidation cards 1,
stale cards expressed 0, and
withdrawn claims expressed as current 0.
These are synthetic fixture observations, not an accuracy score or a claim
about arbitrary natural-language data.

This is a selected mechanism suite, not the full original research suite.
Linux/macOS execution, external model adapters, external benchmarks, training,
large-model continual learning and biological validation were not run here.
The content scan is a bounded audit, not proof that every possible secret or
privacy issue is detectable. Original history and raw operating evidence are
excluded by construction and are audited separately before publication.

The UAI_1 documentation-only successor preserves every validated library,
test and example byte. The 18 tests and four fixtures were not rerun for this
text correction. The corrected 37-file closed set and its reachable history
receive a new content scan before the successor is pushed.
