# Planned controller checks

Add meaningful tests alongside implementation for:

- Rejected changes preserve active memory.
- Atomic commits and stale-parent rejection.
- Replay never learns or accesses future examples.
- Evaluation labels stay outside runtime inputs.
- Rollback restores exact memory membership and versions.
- Cache keys distinguish memory states.
- Capacity replacement is tested as one change.
- Every model call is accounted for in the correct budget.

There are no implemented research tests yet.
