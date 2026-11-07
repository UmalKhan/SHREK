# SHREK

**Shielding Hosts via Reactive Evolving Knowledge**

Agentic AI course research project investigating regression-tested defensive memory for frozen language models.

## Research question

Under limited active-memory and update-testing budgets, does validating both new defensive rules and maintenance of existing rules improve the security-utility tradeoff over either mechanism alone?

## Current status

Phases 1-4 software is implemented: baseline, real-model adapters, admission validation, bounded replay, tested revision/archival, reproducible pilot-data preparation, four-variant comparison, and independent review ingestion. Offline fixtures verify mechanics and public 80-example data preparation is checked. Real GPU/model experiments and holdout review remain pending. No effectiveness result or novelty claim has been established. See [Phase 3](docs/phase3.md) and [Phase 4](docs/phase4.md).

Read [the detailed methodology](docs/methodology.md) before implementing the research components. See [the baseline notes](docs/baseline.md) for external sources and reproduction boundaries.

## Setup

Requires Python 3.11 or later; Python 3.11/3.12 is recommended for the separate GPU inference environment. No GPU or credentials are needed for offline verification.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m shrek --help
.\.venv\Scripts\python.exe -m shrek --version
.\.venv\Scripts\python.exe -m shrek demo
.\.venv\Scripts\python.exe -m shrek baseline-demo
.\.venv\Scripts\python.exe -m shrek shrek-demo
.\.venv\Scripts\python.exe -m shrek compare-demo
.\.venv\Scripts\python.exe -m pytest -q
Copy-Item .env.example .env
```

The CLI does not automatically load `.env`. Set environment variables explicitly for an authenticated server; `.env` is ignored by Git. The `demo` and `baseline-demo` commands are offline. Only the explicitly selected `run-baseline` backend contacts a server or downloads a model.

## Run on GPU PC or Colab

For the complete controller/comparison workflow, use the [Phases 3-4 GPU notebook](notebooks/phase4_study.ipynb) and [Phase 4 instructions](docs/phase4.md). It accepts a private-repository ZIP, detects hardware, checks real benign integration, prepares pinned pilot data, runs B0-B3, and exports the blank independent review queue. No paid inference API is required for local execution. The earlier [Phase 2 notebook](notebooks/phase2_gpu.ipynb) remains available for baseline-only integration.

```powershell
python -m pip install -e ".[dev,hf]"
python -m shrek hardware
python -m shrek run-baseline examples/smoke.jsonl --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --config configs/baseline_paper.toml --max-calls 20 --output runs/real-smoke-01
```

The example model is a provisional integration candidate; review usable GPU memory first. The two smoke inputs are legitimate requests and do not measure attack resistance. The initial real-model smoke/calibration checkpoint is not yet completed.

## Four-variant pilot

After model/hardware calibration, prepare the real data and run all four variants:

```powershell
python -m shrek prepare-study --output data/prepared-pilot --seed 17
python -m shrek compare data/prepared-pilot/study.jsonl --backend transformers --model YOUR_SELECTED_MODEL --device cuda --output runs/pilot-clean
python -m shrek score-study runs/pilot-clean --reviews runs/pilot-clean/review_queue.csv
```

Fill the review CSV before the final command. Blank/unknown judgments produce partial metrics and bounds, never invented zero rates. The data builder uses a single published PAIR family and XSTest safe requests, with source/availability limitations detailed in [Phase 4](docs/phase4.md). Model/data choices are provisional development decisions; do not tune on holdout outputs.

## Layout

```text
src/shrek/           Python package and CLI
configs/             Paper-capacity baseline plus four controlled variants (TOML)
src/shrek/prompt_templates/ Packaged, adapted baseline role prompts
prompts/             Notes/folders for later SHREK prompts
docs/                Methodology and implementation notes
data/manifests/      Dataset provenance and split manifests
data/development/    Local development anchors
data/evaluation/     Local held-out inputs
runs/                Local logs, snapshots, and reports
tests/               Runtime, baseline, and backend integrity tests
examples/            Input schema and benign integration smoke data
notebooks/           GPU/Colab handoff notebook
```

Downloaded data, run artifacts, and credentials are ignored. Original literature PDFs remain in the parent research workspace and are not copied into this repository.

## Experiments

- B0: baseline admission, no periodic maintenance.
- B1: validated admission, no periodic maintenance.
- B2: baseline admission with behaviorally tested maintenance.
- B3: validated admission with behaviorally tested maintenance.

`configs/baseline_paper.toml` preserves the paper's per-label capacity design without the experimental global cap. Other configs describe controlled-budget B0-B3 experiments, now supported through `run-system`. The legacy `run-baseline` command remains B0-only. Prompt/output/model differences are documented in [baseline fidelity notes](docs/baseline.md). Initial settings are development defaults, not experimentally validated values.

## Implementation order

1. Model client, structured records, logging, and read-only baseline inference.
2. Chronological baseline learning, deduplication, and capacity behavior.
3. Pilot the baseline and measure naturally occurring failures and costs.
4. Versioned memory, proposals, and admission validation.
5. Tested revision/archival and scheduled maintenance.
6. Four-way experiments, returning attack families, and feedback-noise stress tests.

No upstream research code is required to initialize this project. Attribute any later reused code and preserve its license. Do not claim an official reproduction without checking fidelity.

## Phase checkpoints

See [Phase 1](docs/phase1.md), [Phase 2](docs/phase2.md), [Phase 3](docs/phase3.md), and [Phase 4](docs/phase4.md). Mock demonstrations establish software behavior only. Real-model calibration, complete GPU study execution, and independent review are still required before claiming research effectiveness.
