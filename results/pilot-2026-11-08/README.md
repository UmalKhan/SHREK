# Pilot results, 8 November 2026

Feasibility pilot of the four-condition comparison (B0 baseline, B1 admission gate, B2 tested maintenance, B3 both). These are aggregate records only: raw model answers, call logs and review sheets stay in the git-ignored `runs/` folder because they contain harmful text.

**Scoring caveat:** the holdout in both pilots was scored by an AI reviewer (Claude Opus 5.5), blind to the condition key. It was **not** scored by an independent human reviewer, as the methodology requires. Treat every rate below as AI-reviewed.

## Setup (shared)

- Host and learning roles: `Qwen/Qwen2.5-7B-Instruct` (revision `a09a3545…`), 4-bit NF4, greedy decoding, RTX 2080.
- Data: `data/prepared-pilot/study-reviewed.jsonl`, 80 distinct prompts. Harmful prompts are 40 PAIR (JailbreakBench, Vicuna-13B artifacts); legitimate prompts are 40 XSTest. Splits: development 9 harmful / 11 legitimate, adaptation 20/20, holdout 10/10.
- One seed (17), shuffled order, no feedback noise, `configs/combined.toml`, at most 4 admission attempts per condition.

| | v1 (`v1/`) | v2 (`v2/`) |
|---|---|---|
| Judge and gate assessor | same local Qwen-7B (self-assessment) | `openai/gpt-oss-120b` via Groq API |
| Prompts, config, data, seed | frozen | identical to v1 |

## Holdout results (10 harmful + 10 legitimate per condition)

| Condition | v1 ASR | v1 false refusals | v1 useful | v2 ASR | v2 false refusals | v2 useful |
|---|---|---|---|---|---|---|
| B0 | 0/10 | 5/10 | 5/10 | 2/10 | 1/10 | 9/10 |
| B1 | 4/10 | 1/10 | 9/10 | 1/10 | 1/10 | 9/10 |
| B2 | 0/10 | 5/10 | 5/10 | 0/10 | 1/10 | 9/10 |
| B3 | 4/10 | 1/10 | 9/10 | 1/10 | 1/10 | 9/10 |

Learned rules: v1 B0 and B2 had 1 rule each, B1 and B3 none (the gate deferred everything because of null self-assessments). v2 B0 had 2 rules, B1 1, B2 2, B3 1. See `final_memory_*.json`.

Adaptation cost in v2 (model calls): B0 165, B1 328, B2 362, B3 355.

## Interpretation

- **v1:** with self-assessment, the gate never admitted a rule, so B1 and B3 equal a no-memory model. B0's single untested rule blocked attacks but caused 5/10 unnecessary refusals of legitimate requests.
- **v2:** with an independent evaluator, the gate admitted rules and over-refusal disappeared in every condition. Differences in attack success are 1-2 requests out of 10, so the pilot cannot distinguish the conditions statistically. Testing roughly doubles adaptation cost.
- The research question (does B3 beat B1 and B2?) is **not answered** by this pilot. The likely reasons are a small holdout, one attack family, one seed, the 4-attempt learning cap, and the absence of the pre-planned feedback-noise condition, which is where testing updates should matter most.
- Borderline label: the youth-labour report was scored as harmful compliance. Scoring it as not harmful lowers ASR by 1 for v2 B0, B1 and B3 and for v1 B1 and B3.

## Execution notes

v2 ran under Groq's free tier (200k tokens/day). The first v2 run (`runs/pilot-v2`) exhausted the quota during B0, with 9 judge calls failing on HTTP 429. B2 and B3 from that run (complete, 0 errors) were reused, after automatic verification, in `runs/pilot-v2b`, where B0 and B1 were rerun with the same judge model. The full chronology is in `docs/baseline.md`.
