# Phase 2: baseline implementation and GPU handoff

Historical Phase 2 checkpoint. The B0-only command and notebook remain supported, but controller/comparison software now exists in [Phase 3](phase3.md) and [Phase 4](phase4.md). Statements below about later components being unimplemented describe the original Phase 2 boundary, not current repository status. Real-model calibration is still pending.

## Status and limits

Implemented: the paper-based baseline control flow, role prompts, local frozen-model adapter, compatible chat-completions adapter, chronological runner, read-only/frozen execution, initial-memory import, dataset manifests, review export, and hardware diagnostics.

Locally verified: deterministic control-flow fixtures, a local HTTP stub integration, and automated integrity tests. These are engineering checks. No GPU/model-backed research run has been performed on the development computer. The real-model verification checkpoint remains pending execution on the group's GPU PC or Colab. The provided two-input smoke file contains only legitimate examples and cannot establish jailbreak resistance or failure-driven learning quality.

SHREK admission validation and periodic maintenance are not implemented in this phase. The runner refuses B1/B2/B3 instead of substituting B0.

## Offline verification

```powershell
python -m pip install -e ".[dev]"
python -m shrek hardware
python -m shrek baseline-demo
python -m pytest -q
```

The baseline demo includes a fictional failed request, immediate learned rule, subsequent retrieval/refusal, and benign bypass. All fixture outputs are marked `is_mock: true`. The fixture is not a jailbreak dataset or model-quality result.

## GPU PC setup

Use Python 3.11 or 3.12 in a separate environment. CUDA-enabled Torch depends on the hardware/driver installation. First inspect GPU availability; do not assume that installing a package creates CUDA support.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,hf]"
.\.venv\Scripts\python.exe -m shrek hardware
.\.venv\Scripts\python.exe -m pytest -q
```

Record the hardware report. Confirm the GPU has usable free VRAM before loading a model. A 1.5B instruction model is a provisional smoke-test candidate, not the final model selection or a claim that it fits every GPU. Model/activation memory depends on context, dtype, and available VRAM. There is no silent quantization or CPU offloading.

Official example model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

```powershell
.\.venv\Scripts\python.exe -m shrek run-baseline examples/smoke.jsonl --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --config configs/baseline_paper.toml --max-calls 20 --max-output-tokens 384 --output runs/real-smoke-01
```

This downloads the selected model and produces a real two-request integration run. It does not use a paid inference API. Downloads/storage/runtime still depend on the machine and network. `--device cpu` is explicit and potentially slow. `--revision` accepts a model commit for later reproducibility; the resolved model revision is recorded when available. Package versions are recorded; the optional dependency ranges are compatibility constraints, not a hardware-tested lockfile.

For Colab, open `notebooks/phase2_gpu.ipynb`. Because the GitHub repository is private, the notebook accepts a repository ZIP uploaded from GitHub rather than embedding a token or saving credentials in a clone URL. It installs local-model extras, prints hardware, runs offline checks, then loads a user-reviewed model and exports the run artifacts.

## Existing local model server

```powershell
$env:MODEL_BASE_URL = "http://127.0.0.1:8000/v1"
$env:HOST_MODEL = "your-served-model-id"
python -m shrek run-baseline examples/smoke.jsonl --backend http --config configs/baseline_paper.toml --output runs/server-smoke-01
```

For an authenticated endpoint set `MODEL_API_KEY` in the environment, not in command arguments. The CLI does not automatically load `.env`. HTTPS is required except for localhost. The HTTP backend has no verified local host tokenizer; controlled token-cap runs are refused. Use the paper-capacity config or implement a verified tokenizer adapter later.

## Research input and holdout

Use the JSONL contract in `examples/dataset-schema.md`. Audit the complete multi-split file before selecting one split. Do not learn on holdout examples.

```powershell
python -m shrek audit-data data/evaluation/study.jsonl
python -m shrek run-baseline data/evaluation/study.jsonl --backend transformers --model YOUR_MODEL --split adaptation --config configs/baseline_paper.toml --output runs/adaptation-01
python -m shrek run-baseline data/evaluation/study.jsonl --backend transformers --model YOUR_MODEL --split holdout --no-learning --initial-memory runs/adaptation-01/final_memory.json --output runs/holdout-01
```

The model, prompts, and config must be kept consistent across adaptation and holdout unless a transfer experiment is explicitly intended. With no initial-memory file every run starts empty. Imported memory is identified by its content hash, but automatic cross-run provenance/leakage checking is not implemented yet; the operator must verify its training split/manifest.

## Inspect a run

- `manifest.json`: config/hash, ordered example IDs, dataset hash, backend settings, model revision where available, code commit/dirty state, and environment.
- `calls.jsonl`: model-role inputs, raw completions, usage, errors, latency, and mock provenance. Never receives benchmark evaluation labels.
- `answers.jsonl`: original responses flushed before feedback or memory changes.
- `interactions.jsonl`: chronological feedback, update outcome, errors, and before/after memory versions.
- `memory.sqlite3`: immutable memory history and proposal decisions.
- `final_memory.json`: portable active rules.
- `manual_review.csv`: original requests/answers plus evaluation labels and blank reviewer columns. This is an evaluation artifact, not a model input.
- `summary.json`: counts, missing answers, error events, budget stop, active rules, and usage. It deliberately does not claim independently measured ASR or FRR.

Check that event 1's answer used memory version 0 and that any new rule affects only later events. Inspect every error. A malformed classifier cannot produce a fabricated answer; a malformed judge cannot activate a rule. Budget exhaustion stops additional calls while preserving already emitted answers. Exit code 2 indicates event errors/incomplete execution. Do not compare only completed successful examples and silently ignore errors.

## Phase completion check on real hardware

1. Hardware and resolved-model report saved.
2. Benign smoke run completes with no schema errors or missing answers.
3. Judge/induction reliability checked on a small development stream with published attacks.
4. Actual latency, calls, and available memory inspected.
5. At least one naturally supported failure-to-rule trace inspected if such failures occur. If none occur, report that; the synthetic trace does not substitute for research evidence.

Items 3-5 are the resource/calibration checkpoint before scaling experiments or claiming effective adaptation. Hardware-specific model fitness, judge reliability, and natural failure observations remain unresolved until those runs.
