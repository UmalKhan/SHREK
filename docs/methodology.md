# SHREK: Regression-Tested Defensive Memory for Frozen LLMs

Implementation methodology, 6 November 2026. This is a proposed study, not a tested system or an established novelty claim. Initial numbers below are engineering defaults to tune on development data, then freeze before final evaluation.

## 1. Objective and research questions

Keep the protected language model's weights unchanged. Learn reusable defensive rules from observed failures, but evaluate proposed memory changes before activating them. Reassess existing rules periodically so that unnecessary restrictions do not persist indefinitely.

Primary question: Under fixed active-memory and update-testing limits, does validating both rule admission and rule retirement improve the security-utility tradeoff over either mechanism alone?

Secondary questions:
- Does validation prevent overly broad new rules from increasing false refusals?
- Can retirement reduce active memory without losing protection against returning attacks?
- How much adaptation delay and inference expense do these mechanisms add?

The contribution being investigated is one behavioral acceptance gate for additions, revisions, and retirements, evaluated under resource constraints. Validation, memory editing, and auditing are individually established ideas. Do not claim their invention or claim a first unified lifecycle without further related-work comparison.

## 2. External code and data: required versus optional

No particular research repository is a technical prerequisite. A language-model endpoint, reproducible benchmark examples, and our own implementation of the baseline and experimental controller are required.

### Baseline implementation

Hu and Hooi: https://arxiv.org/html/2608.26008v1

The targeted paper/site search did not verify an official author repository. This is not proof none exists. Implement its operators and consult Appendix A.6 for prompts. Record every deviation in a baseline-fidelity document. Call the result a paper-based reimplementation, not official code. Preserve the original classifier, retrieval, policy selection, failure detection, deduplication, and per-label capacity before adding SHREK components.

The paper uses label-first retrieval with fallbacks, retrieves at most two rules, and caps each label at four rules in its main design. Preserve these in the faithful baseline; additional global caps belong to explicitly identified resource-controlled experiments.

### Optional implementation reference

SED official code: https://github.com/Infini-AI-Lab/SED

The inspected repository has an MIT license, a `sed` package, benchmark runners, configuration, and small smoke-run instructions. It can supply a second implemented baseline or reference code; it is not needed to implement Hu and Hooi. Do not combine pieces from two papers and label that mixture as either original baseline. Inspect dependencies, pin a commit, and retain license notices if reusing code. Repository availability was checked; its runtime was not tested here.

MEMBRANE official code: https://github.com/brightjade/membrane

Optional comparison/reference. Its documented setup includes local model and embedding servers and an additional reviewer for test-time adaptation. Avoid making that entire stack a dependency of the initial prototype.

### Recommended data sources

- JailbreakBench framework: https://github.com/JailbreakBench/jailbreakbench
- Published attack artifacts: https://github.com/JailbreakBench/artifacts
- XSTest official data: https://github.com/paul-rottger/xstest

Use published artifact inputs instead of building an expensive attack generator. Artifact success on another model does not establish success on our host. Test all selected examples and report their effectiveness on the selected host. Check overlap between datasets by behavior as well as exact text. Retain dataset-specific attribution and usage conditions. Using these datasets changes the evaluation setup from the original baseline paper; describe it explicitly.

## 3. Scope and threat model

Initial scope: single-turn text jailbreaks and legitimate text requests, with memory accumulating across independent requests. No fine-tuning, reinforcement learning, live external actions, generated executable guardrails, or multi-agent communication attacks are needed.

Attackers control request text. They cannot directly edit the rule store or system prompts. The online judge can make natural errors. A separate feedback-noise experiment flips selected update labels under a seeded protocol; this is a controlled robustness experiment, not a demonstration of a realistic poisoning attack. Keep final evaluation labels unmodified.

The host's own safety instructions remain in force. A safe exception means a particular defensive rule does not apply; it never grants permission to override the host's policy.

This setup evaluates memory governance for a prompted defense. It does not establish protection for tool-using agents or long conversations.

## 4. System components and contracts

Use plain Python modules and structured outputs. Separate prompted roles do not require separate large models or a multi-agent framework.

1. `ModelClient`: generate structured or textual completions; record model ID, parameters, input/output tokens, latency, retries, and errors. Never log credentials.
2. `BaselineDefender`: classify attack pattern, retrieve relevant active rules, select allow/soft-refuse/hard-refuse policy, and produce a host response. Returns response, retrieved rule IDs, policy, and memory version.
3. `FeedbackJudge`: inspect the request and actual response; return a harmful-compliance assessment with evidence and a separate refusal assessment. An ambiguous or invalid judgment does not authorize a memory mutation.
4. `RuleProposer`: propose ADD after confirmed harmful compliance, REVISE after supported false refusal, or ARCHIVE after evidence of redundant or harmful coverage. Proposals do not modify active memory.
5. `MemoryStore`: manage immutable rule versions, active membership, archived records, and atomic memory snapshots. SQLite is sufficient.
6. `ProbeSelector`: select a bounded set of relevant and diverse past/development examples available at that time.
7. `ChangeValidator`: compare identical probe inputs using current memory and a candidate memory snapshot. No learning during these runs.
8. `MaintenanceScheduler`: periodically consider a bounded number of existing rules for revision/retirement.
9. `EvaluationRunner`: keep ground-truth labels and final evaluation judgments separate from the adaptive defense.

The online judge and proposed-rule writer can initially share a model endpoint. Use an independent evaluation judge when feasible, plus manual checks; shared-model agreement is not ground truth.

## 5. Persistent records

`RuleVersion`: rule_id, version, label, mechanism, applicability_conditions, defensive_action, safe_exclusions, source_event_ids, created_at, token_length.

`MemoryVersion`: version_id, parent_version_id, active_rule_versions, config_hash, change_id.

`RuleStatistics`: retrieved_count, relevant_probe_count, confirmed_false_refusal_events, last_retrieved_event, last_validated_event. Retrieval frequency is exposure, not causal effectiveness.

`Interaction`: event_id, sequence_index, input_hash, dataset_source, behavior_group, request, host_response, memory_version, retrieved_rule_ids, selected_policy, online_feedback, call_usage.

`Proposal`: change_id, operation, base_memory_version, target_rule_version, candidate_rule, evidence_ids, status, rejection_reason. Status is pending/accepted/rejected/deferred.

`ValidationReport`: change_id, probe_ids, origin_of_labels, before_and_after_results, security_regressions, benign_regressions, improvements, memory_token_delta, calls_and_tokens, gate_decision.

`BenchmarkExample`: example_id, source, behavior_group, attack_family, input, evaluation_label, split. Keep evaluation-only labels out of the runtime payload and candidate-memory prompts.

## 6. One interaction, end to end

1. Load the active memory snapshot.
2. Produce the response through the baseline defender.
3. Log and score this original response. Do not retrospectively replace it after learning.
4. Ask the online judge for update feedback, without access to evaluation labels.
5. If justified, create one rule-change proposal.
6. Evaluate the proposal against frozen current memory using available probes.
7. Accept atomically, reject, or defer the proposal. Activation can affect only later interactions.
8. At the scheduled maintenance interval, consider one existing rule for revision or archival using the same gate.

Maintain one sequential writer initially. If the active memory version changed after a proposal was created, reject its stale validation or revalidate it against the new version before committing.

## 7. Probe pools and label provenance

Maintain two runtime-available probe sources:

- A small fixed development anchor pool, with manually reviewed labels and diverse legitimate/harmful cases. This pool is explicitly extra supervision and is available equally to controlled variants. Its size and annotation effort must be reported.
- A bounded replay pool of previous interactions with online judgments. Mark these labels as noisy; use reliable evidence and audit a sample. Never silently replace online labels with final benchmark labels.

For a proposal, initially select up to 20 probes: about 8 related harmful cases, 8 related legitimate cases, and 4 older diverse cases split across both classes. The triggering event can be an additional training probe. Counts are defaults, not requirements to fabricate examples when the pool is too small.

If relevant probes of either class are missing, defer the change or use the fixed anchors and mark low support. Never construct a probe from a future evaluation example. Do not call deferred changes validated.

Use label matching and a simple local text-similarity implementation initially. An embedding index is optional. Include diverse old cases so a highly local update cannot trivially sacrifice unrelated protection. A small pool provides evidence, not a guarantee.

False-refusal correction needs feedback supporting legitimate intent as well as evidence of refusal. A refusal by itself is not proof of an error. In final evaluation, only the online mechanism supplies adaptive feedback; human benchmark labels score the outcome externally.

## 8. Behavioral acceptance gate

Create candidate memory M' from current memory M without modifying M. Run the entire defender, including retrieval, on each probe under both snapshots. Freeze both snapshots and disable proposal generation and maintenance. Compare the actual host responses, not just rule wording or guard decisions.

Define a harmful regression as a harmful probe changing from non-compliance to harmful compliance. Define a benign regression as a legitimate probe changing from an acceptable useful answer to refusal or materially unhelpful behavior. Define improvements in the opposite directions. Judge uncertainty is separate from success or failure.

Initial conservative gate:

- No newly observed harmful regression on selected probes.
- No newly observed benign regression on selected probes.
- ADD: correct the triggering failure and improve at least one additional related probe if one exists; otherwise defer pending more support.
- REVISE: improve at least one supported error and preserve tested previous protection.
- ARCHIVE: reduce active memory tokens while preserving all tested outcomes.
- Respect the configured active-memory and validation budgets.

This gate is deliberately simple for the first implementation. Tune only on development data; later compare modest tolerance settings if needed. Zero observed regressions on a small set does not prove zero risk. Independent held-out results determine effectiveness. Do not interpret the validation set as a statistical safety certificate.

If before/after judgments are unstable or ambiguous, defer; allow a bounded repeat only for disputed pairs and charge its cost. Record temperature, seeds where supported, and model revision. Low temperature reduces variation but does not guarantee determinism.

## 9. Maintenance and memory constraints

Every 25 processed interactions, consider at most one revision or archival candidate. Prioritize supported false-refusal events, then redundant coverage, then old low-exposure rules. Do not treat old age alone as evidence a rule is unsafe or useless.

Do not auto-merge rules in the first prototype. Revisions and archiving are enough to test the main hypothesis. Candidate ranking can be heuristic; acceptance must depend on behavioral replay.

Keep archived versions and their supporting evidence outside active retrieval. Restoration is another tested proposal; never automatically reactivate an old rule solely because its label appears. Limit archive/replay storage separately and report these costs so reduced active memory is not mistaken for reduced total storage.

Initial controlled active budget: at most 20 rules and 2,000 tokenizer-measured rule tokens, plus the original per-label constraint where applicable. Compare alternative caps on development data. These are experimental limits, not values claimed from the baseline paper.

If ADD exceeds capacity, propose an atomic replacement: archive one rule and add the new one, then test the combined memory change. If no replacement passes, defer the addition. Do not prune first and assume the later addition makes the change safe.

Use a simple deterministic capacity-replacement policy in capacity-controlled comparators and document it. Keep a separate faithful baseline run with the paper's capacity design. This distinguishes improvements in the controller from effects of artificially constraining the baseline.

Rollback restores a versioned memory state after a supported harmful update. It cannot undo previously generated harmful responses. Include reversal of retired rules and later revalidation in the audit trail.

## 10. Data preparation and experiments

Split by underlying behavior/scenario before making attack variants: the same behavior, paraphrase, or template instance must not leak between development anchors and final held-out evaluation. Exact-text deduplication alone is insufficient. Keep attack-family holdouts where data allows.

Start with a pilot: roughly 100 harmful and 100 legitimate distinct examples if available after grouping, plus separate development anchors. This is a feasibility sample, not evidence for small percentage-point improvements. Add more distinct behaviors or another benchmark if the final hypothesis needs greater statistical power; do not increase apparent sample size by treating repeated variants as independent.

Primary four-way experiment, using the same defender and comparable capacity rules:

- B0: baseline updates, no behavioral gate or maintenance.
- B1: behavioral admission validation, no periodic maintenance.
- B2: baseline admission plus behaviorally tested periodic maintenance.
- B3: combined validation and maintenance.

Also report the faithful baseline, no-defense host, and a frozen-memory control if resources permit. The B2 maintenance gate remains enabled: the intended factorial factors are admission validation and periodic maintenance, not validation everywhere versus nowhere.

Use identical ordered inputs and anchor information; memory states can naturally diverge between methods. Start every independent run with the prescribed empty memory and same development anchors. Reset mutable memory, counters, and replay state across runs.

Evaluate:
1. Mixed clean streams with balanced attacks and legitimate requests.
2. Family-shift streams: A, then B/C, then fresh examples from A. Returning A examples must not be exact repeats of examples used to validate memory changes.
3. Feedback-noise stress streams, with a predeclared seeded error protocol such as 0%, 5%, and 10%; inject errors in runtime update feedback only, never final labels. Report both natural judge error and injected error.

Two complementary protocols:
- Online test-before-update: final examples are scored before their feedback can affect later examples. This is continual adaptation, not a claim that all test examples remain unseen forever.
- Frozen holdout: after adaptation, freeze the library and evaluate independent unseen examples with no writes. This measures generalization beyond the adaptation stream.

Use at least three stream orders if affordable. Judges evaluating a method should not see its name, proposal decision, or preferred outcome.

## 11. Metrics, costs, and evidence

- ASR: harmful-compliance outcomes / all attack attempts, using actual host responses. A guard's allow decision alone is not attack success.
- FRR: unnecessary refusals / legitimate requests.
- Benign task success: acceptable useful completions / legitimate requests, using a predefined rubric; non-refusal alone is insufficient.
- Active rule count and token size over time; archive and replay storage separately.
- Newly admitted damaging changes, rejected/deferred changes, and updates later reversed.
- Adaptation delay: interactions between initial failure and accepted effective protection, with a precise operational definition.
- Returning-family ASR and retention of older coverage.
- End-to-end latency: host path and synchronous update delay separately. Initial implementation is synchronous; do not claim asynchronous serving benefits without implementing and measuring them.
- All calls, tokens, retry costs, and billed or estimated cost: inference, judging, proposals, replay, revisions, maintenance, evaluation. Separate deployment cost from research evaluation cost.

Initially allow at most one immediate proposal per interaction, one maintenance candidate per interval, and a fixed global update-testing token/call budget. Once exhausted, freeze further changes and continue scoring requests. Do not make emergency unvalidated updates.

One 20-probe comparison can require up to 40 full defended runs plus response judging. It is not 40 cheap classifier calls. Each defended run can have multiple model calls. Cache current-snapshot outputs by complete input, memory hash, configuration, model revision, and seed where meaningful. Never reuse current outputs after memory changes. Pilot this cost before selecting the final candidate budget.

Report practical equal-workload results with costs, plus a predeclared matched-total-inference-budget sensitivity run where feasible. Equal update-candidate limits alone do not mean equal total compute. Count independent behavior groups when computing uncertainty; use paired/group-aware bootstrap summaries where appropriate and report across-order variability separately.

Manually inspect a stratified sample of harmful outcomes, false refusals, judge disagreements, and accepted/rejected updates. If the host/learning judge are the same model, use a different evaluator family where feasible and disclose remaining correlated-error risk.

## 12. Suggested code layout and initial configuration

```text
shrek/
  config.py
  schemas.py
  model_client.py
  baseline.py
  feedback.py
  proposals.py
  memory.py
  probes.py
  validation.py
  maintenance.py
  runner.py
  metrics.py
prompts/
  baseline/
  feedback/
  proposal/
  evaluation/
configs/
  baseline.yaml
  admission_only.yaml
  maintenance_only.yaml
  combined.yaml
data/
  manifests/
  development/
  evaluation/
runs/
tests/
```

Minimal stack: Python, a provider client or HTTP library, Pydantic for output schemas, SQLite for persistence, a tokenizer appropriate to the host, YAML configuration, and pytest. Use a single CLI initially. No LangChain, CrewAI, vector database service, dashboard, or Docker is required for this text-only prototype. Pick and pin package versions when implementation begins.

```yaml
retrieval_k: 2
per_label_capacity: 4
active_rule_limit: 20          # controlled-budget runs only
active_rule_token_limit: 2000 # controlled-budget runs only
maintenance_every: 25
maintenance_candidates: 1
max_proposals_per_interaction: 1
validation_probe_limit: 20
max_repair_attempts: 1
max_new_harmful_regressions: 0
max_new_benign_regressions: 0
```

Model IDs, output limits, context limits, and budget ceilings must be selected from an actual pilot; do not embed assumed vendor models/prices in the controller. Treat model outputs as data, validate JSON, and make malformed output defer an update rather than mutate memory.

```python
# Algorithm sketch, not runnable code.
for event in ordered_stream:
    current = store.active_snapshot()
    result = defender.respond(event.request, current)
    evaluation.record_original(event, result)  # labels stay outside runtime
    feedback = online_judge.assess(event.request, result.response)
    runtime_log.append(event, result, feedback)

    proposal = proposer.from_feedback(event, result, feedback, current)
    if proposal is not None and budget.can_start_validation():
        probes = selector.available_at(event.sequence_index, proposal)
        candidate = store.preview(current, proposal)  # includes any replacement
        report = validator.compare(current, candidate, probes, learning=False)
        if gate.accepts(proposal, report):
            store.commit_if_parent_unchanged(candidate, report)

    if scheduler.due(event.sequence_index):
        # Reload memory: an immediate proposal may already have been committed.
        maintenance.try_one_validated_change(store.active_snapshot(), budget)
```

## 13. Implementation order and completion criteria

1. Build a logging/model interface, data manifests, and read-only baseline inference.
2. Add the paper-based learning loop, deduplication, and capacity behavior. Verify cold-start chronological updates.
3. Run the baseline pilot. Inspect whether naturally learned rules damage later legitimate use and whether simple capacity controls already suffice.
4. Add snapshot storage, proposal generation, and behavioral admission tests.
5. Add revision/archive proposals, periodic scheduling, and atomic replacements.
6. Run B0-B3, then family-return and feedback-noise experiments. Expand only if a worthwhile effect survives costs and uncertainty.

Necessary implementation tests: rejected proposals leave active state unchanged; accepted proposals commit atomically; stale proposals cannot overwrite newer state; replay cannot learn; evaluation labels cannot enter runtime payloads; rollback restores exact membership/version; cache keys distinguish memory states; capacity replacement is tested as one transaction; every model call is charged to the correct budget.

The proposal is supported if the combined method offers a repeatable useful tradeoff over both single-component variants, with total costs disclosed. A smaller memory with unchanged protection can be a valid outcome. If only injected errors produce gains, narrow the conclusion to noisy-feedback robustness. If the baseline barely fails or validation costs outweigh benefits, report that result and simplify instead of claiming universal necessity.

## 14. Closest related work to check before final novelty claims

- Hu and Hooi: https://arxiv.org/abs/2608.26008
- SED: https://arxiv.org/abs/2609.36603
- AgentAntibody, persistent-update validation: https://arxiv.org/abs/2608.04053
- MEMBRANE, contrastive memory and memory actions: https://arxiv.org/abs/2606.05743
- SafeHarbor, structured memory evolution: https://arxiv.org/abs/2605.05704
- EVOLVE, audited guardrail-tool admission and repair: https://arxiv.org/abs/2611.02516
- A-MemGuard, validation and corrective memory: https://arxiv.org/abs/2510.02373

The implementation specification above is our proposed design. It should not be attributed wholesale to any one of these papers. A code-available secondary baseline is useful, but a difference in artifact type or a smaller implementation alone is not proof of research novelty.
