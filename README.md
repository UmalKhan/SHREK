# SHREK

**Shielding Hosts via Reactive Evolving Knowledge**

Agentic AI course research project investigating regression-tested defensive memory for frozen language models.

## Research question

Under limited active-memory and update-testing budgets, does validating both new defensive rules and maintenance of existing rules improve the security-utility tradeoff over either mechanism alone?

## Current status

Phase 1 infrastructure is implemented: configuration, model-call interface/accounting, versioned SQLite memory, split auditing, and a scripted offline demo. Baseline model reasoning and SHREK behavioral validation/maintenance are not implemented yet. No research results or novelty claims have been established.

Read [the detailed methodology](docs/methodology.md) before implementing the research components. See [the baseline notes](docs/baseline.md) for external sources and reproduction boundaries.

## Setup

Requires Python 3.11 or later. No GPU or model credentials are needed to inspect this scaffold.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m shrek --help
.\.venv\Scripts\python.exe -m shrek --version
.\.venv\Scripts\python.exe -m shrek demo
.\.venv\Scripts\python.exe -m pytest -q
Copy-Item .env.example .env
```

The scaffold does not load `.env` or contact any endpoint yet. Model/provider dependencies will be selected during implementation. `.env` is ignored by Git.

## Layout

```text
src/shrek/           Python package and CLI
configs/             Four proposed experimental variants (TOML)
prompts/             Baseline, feedback, proposal, and evaluation templates
docs/                Methodology and implementation notes
data/manifests/      Dataset provenance and split manifests
data/development/    Local development anchors
data/evaluation/     Local held-out inputs
runs/                Local logs, snapshots, and reports
tests/               Future controller-integrity tests
```

Downloaded data, run artifacts, and credentials are ignored. Original literature PDFs remain in the parent research workspace and are not copied into this repository.

## Experiments

- B0: baseline admission, no periodic maintenance.
- B1: validated admission, no periodic maintenance.
- B2: baseline admission with behaviorally tested maintenance.
- B3: validated admission with behaviorally tested maintenance.

The supplied configs describe controlled-budget experiments, not an exact reproduction of the paper's full setup. A separate faithful-baseline configuration must be added and documented during baseline implementation. Initial settings are development defaults, not experimentally validated values.

## Implementation order

1. Model client, structured records, logging, and read-only baseline inference.
2. Chronological baseline learning, deduplication, and capacity behavior.
3. Pilot the baseline and measure naturally occurring failures and costs.
4. Versioned memory, proposals, and admission validation.
5. Tested revision/archival and scheduled maintenance.
6. Four-way experiments, returning attack families, and feedback-noise stress tests.

No upstream research code is required to initialize this project. Attribute any later reused code and preserve its license. Do not claim an official reproduction without checking fidelity.

## Phase checkpoints

See [Phase 1 verification](docs/phase1.md). The demo is explicitly synthetic and cannot support research performance claims.
