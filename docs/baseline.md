# Baseline and external sources

Primary methodological baseline: Hu and Hooi, *A Self-Evolving Multi-Agent Framework Defense against LLM Jailbreak Attacks*.

- Paper and prompt appendix: https://arxiv.org/html/2608.26008v1
- An official public author repository was not verified during project preparation. This does not establish that none exists.
- Reimplementation must preserve the paper's operators before introducing SHREK changes. Record deviations and never describe this scaffold as official baseline code.

Optional reference implementation:

- SED: https://github.com/Infini-AI-Lab/SED
- It is a different method. Reusing its architecture changes the baseline choice; document that explicitly.

Recommended data sources:

- https://github.com/JailbreakBench/jailbreakbench
- https://github.com/JailbreakBench/artifacts
- https://github.com/paul-rottger/xstest

Before importing code or data, record source version, license, and attribution. Pin commits and dataset revisions for experiments. No upstream code or datasets have been downloaded into this scaffold.

## Implemented fidelity and declared differences (Phase 2)

The implementation is paper-based, not author code, and does not reproduce reported paper scores. The model, task data, prompts, and independent evaluation have not been calibrated on the group's GPU hardware yet.

Preserved concepts: classify structural wrapper into benign/known/other; bypass rule triggering and induction for benign classifications; label-first retrieval followed by LLM relevance and lexical fallback; select up to K=2 rules; policy selection and explicit refusal contract; score actual harmful compliance on 1-10, learn at >=7; failure-driven method-level induction; semantic deduplication; C=4 rules per label; chronological test-before-update; frozen host weights.

Declared implementation choices/differences:

- Prompts are newly worded adaptations of Section 3 and Appendix A.6, not copied author templates. JSON schemas replace line-based outputs to support strict parsing. See packaged `src/shrek/prompt_templates/`.
- The host receives a JSON user message containing the actual request and selected constraints; this differs from a plain-text chat wrapper and must remain fixed across comparisons.
- Induction uses bounded text fields and a severity action field. Known-label induction must preserve the classified label; a novel wrapper can create a new label.
- Where the paper leaves tie-breaking/eviction unspecified, retrieve the newest K matching-label rules and evict oldest same-label rules beyond C. An update keeps the same rule ID and creates a new memory snapshot.
- Semantic deduplication uses a separate model call on failed interactions with existing memory; its cost is counted. Unknown IDs and malformed output fail rather than silently editing memory.
- Selected stored action constraints are resolved in HARD_REFUSE > SOFT_REFUSE > ALLOW order, conservatively combined with the policy agent. This explicit severity representation is an adaptation and can affect over-refusal.
- Reflection/judging is called after each answered event when learning is enabled, including benign inputs. This makes all learning-signal cost explicit; benign classification still prevents induction. A no-learning run omits feedback calls.
- Baseline updates are admitted immediately; there is no SHREK behavioral acceptance test, maintenance, merging, or safe-exclusion learning in Phase 2.
- Paper-capacity runs have no experimental global memory limit. Controlled B0 adds a deterministic oldest-first global replacement under actual tokenizer-measured serialized-rule size. It is a separate experimental condition.
- Structured output is greedy and capped; invalid/uncertain judgments are recorded without mutation. Truncated host answers are preserved and flagged but cannot induce rules. Prompt overflow is an error, never silent truncation.
- Online judge feedback is not independent evaluation. Model ID/revision, prompts, data, decoding, and environment must be recorded for comparisons. The same local model initially serves host and learning roles, so correlated mistakes remain a limitation.

Software integrity tests establish control flow, not baseline score fidelity. Benign real-model smoke integration and attack/judge calibration remain required on the GPU machine.
