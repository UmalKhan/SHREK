# Phase 1: infrastructure verification

Implemented: strict experiment configuration, immutable runtime records, SQLite memory versions, atomic activation with stale-parent checks, logged proposals/rejections/operator rollback, model backend interface, call-budget accounting, JSON output parsing, and behavior-group split checks.

## Run

```powershell
python -m pip install -e ".[dev]"
python -m shrek demo
python -m shrek check-config configs/combined.toml
python -m pytest -q
```

Each demo creates a fresh `runs/demo-*` directory with `summary.json`, `calls.jsonl`, and `memory.sqlite3`. Existing output directories are refused to prevent accidental mixing of experiments. The demo is a scripted engineering fixture, never research evidence. Its accepted/rejected proposals exercise storage decisions, not an implemented behavioral gate.

The call limit is enforced before a backend call; failures and malformed completions still consume attempts. Token usage is recorded when provided, and missing usage is explicitly counted. No monetary or total-token cap is claimed in this phase. Output limits are part of the backend contract. Provider exceptions are logged by type only; prompts/responses remain local ignored research artifacts and may contain sensitive text.

`RuntimeRequest` carries only ID and request text. Evaluation labels remain in separate `Example` records. Group-based checks require correct provenance/group assignments; they do not automatically discover paraphrases.

Still unresolved without hardware/course data: model IDs, usable VRAM, latency, dataset selection and annotation, scoring calibration, and expense/runtime ceiling. No model-quality conclusions can be drawn from these checks.
