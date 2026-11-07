# Verification status

Phases 1-2 tests cover atomic memory activation/rejection/rollback, stale or tampered proposals, call limits including failed calls, evaluation-label isolation, group leakage checks, baseline failure-to-rule reuse, benign bypass, fallback retrieval, deduplication/capacity behavior, read-only execution, output truncation, and the HTTP adapter through a local stub server.

The HTTP integration test needs localhost socket access; if an execution sandbox blocks loopback, run it in a normal local development shell. This is a software test, not a remote-model run. No GPU inference or model-quality result is established by the suite.

## Later controller checks

Phases 3-4 now additionally test admission regressions, bounded/deferred replay, cache separation, strict historical selection, revision/retirement, shared budgets, matched variant ordering, frozen holdout, seeded feedback noise, missing-outcome denominators, review integrity, deterministic source splitting, and notebook syntax. Real-model hardware execution remains a separate checkpoint.

Add meaningful tests alongside implementation for:

- Rejected changes preserve active memory.
- Atomic commits and stale-parent rejection.
- Replay never learns or accesses future examples.
- Evaluation labels stay outside runtime inputs.
- Rollback restores exact memory membership and versions.
- Cache keys distinguish memory states.
- Capacity replacement is tested as one change.
- Every model call is accounted for in the correct budget.

There are no independently scored research results yet. Additional tests will accompany SHREK behavioral validation, replay pools, cache keys, and maintenance when implemented.
