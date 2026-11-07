# Phase 3: validated admission and maintenance

## Status

Implemented and tested with fictional controller fixtures. No GPU outputs were available in this workspace when this phase was implemented. Real-model calibration remains pending; the demo establishes mechanics, not effectiveness.

```powershell
python -m shrek shrek-demo
python -m pytest tests/test_controller.py -q
```

The demo verifies narrow admission, rejection of an overly broad rule, deferral under exhausted replay budget, retirement of unused protection, retention of needed protection, and a supported revision. Every output is labeled mock. It uses four probes, minimum two per class, for compact software checks only.

## Actual experimental defaults

- Target 12 probes: six harmful and six legitimate, approximately four related plus two old/diverse per class.
- Require at least four independent behavior groups per class. Below this minimum, defer without replay; never fabricate evidence.
- Exclude the trigger's behavior group from supporting probes. A trigger is additional evidence and cannot satisfy independent support.
- Replay contains only earlier interactions with explicitly noisy judge-derived labels. Low harmfulness alone never creates a legitimate replay label.
- Development benchmark labels initialize anchors and are declared extra supervision. Their presence does not establish that a human reviewed them.
- Zero observed new harmful regressions and zero legitimate utility regressions by default. Every required judgment must be known.
- Admission must reproduce/fix the trigger and improve another independent harmful example.
- Revision must improve supported behavior without tested regressions. Archival must reduce active size and preserve tested outcomes.
- Maintenance every 10 interactions, at most one nomination per interval, four maintenance attempts per run.
- At most four eligible admission attempts per run, one per interaction. Attempts include induction that later skips/deduplicates; this ceiling applies to B0-B3 for controlled comparisons.
- Update-role budget 120 calls; replay budget 240 calls. Both also consume the shared run ceiling. After local learning budget exhaustion, inference continues until the shared ceiling is reached. Exhausted replay defers changes.
- Run-local replay pool of 100 records; LRU validation-outcome cache of 256 entries. Cache keys include memory digest, request identity/text, model, output cap, and prompt hash. Cache never crosses variants or runs; uncertain outcomes are not cached.

All values are provisional development settings. Small probe sets are not statistical safety certificates. The validator uses a prompted online assessor, potentially the same model as the host, so correlated judgment mistakes remain possible.

## Real execution

Use an audited JSONL file with development anchors and adaptation examples, following `examples/dataset-schema.md`:

```powershell
python -m shrek run-system data/evaluation/study.jsonl --backend transformers --model YOUR_MODEL --device cuda --config configs/combined.toml --max-calls 2000 --output runs/combined-01
```

`configs/baseline.toml`, `admission_only.toml`, `maintenance_only.toml`, and `combined.toml` are B0-B3 under the same controlled memory limits. Real controlled runs require the model's tokenizer. `run-baseline` remains the legacy B0-only command; use `run-system` for the combined controller.

For a frozen holdout, use `--split holdout --no-learning --initial-memory PATH_TO_ADAPTATION/final_memory.json`. No online feedback, proposal, replay, or maintenance calls occur during frozen inference.

## New trace files

`validation.jsonl` records each accepted/rejected/deferred candidate, before/after digests, probe IDs and provenance, outcomes, regressions, improvements, call cost, cache hits, and actual activation. SQLite records pending and resolved proposals; candidate memory is never activated for replay. Stale activation fails rather than overwriting newer memory. `memory_timeline.json` records active size/version after each interaction. Size is host tokens with an actual tokenizer and explicitly serialized bytes otherwise.

`calls.jsonl` assigns each attempt to inference, learning, or validation. `answers.jsonl` preserves original responses before feedback or changes. Final evaluation labels appear only in evaluation files, never model-role payloads. Supported false-refusal nominations require online evidence of legitimate intent, not a refusal alone.

## Scope and operational limits

Maintenance nominates supported false-refusal rules first, exact redundant signatures next, then low-use/old rules. This is a cheap nomination heuristic; every edit still needs behavioral acceptance. There is no automatic merging or learned maintenance scheduler. At most one maintenance candidate is considered even if the configuration ceiling is larger.

The policy prompt now permits explicit safe-exclusion IDs on revised rules; IDs without stored safe exclusions are rejected. This inference extension is shared by every variant and documented as an adaptation. Safe exclusions cannot override the host's own policy. Historical SQLite snapshots retain archived rules. Operator rollback exists; automatic production rollback based on later incidents is not claimed.

The nominal global cap of 20 rules/2,000 serialized-rule tokens is experimental. Preserve a paper-capacity B0 reference separately. Validation replay and archived history are additional resources, reported rather than hidden. There is no monetary/token-total hard cap: the enforced resource limits are calls and output ceilings, with measured token accounting where supported.
