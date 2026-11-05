"""Chronological baseline execution with original-response preservation."""

from dataclasses import asdict
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable

from shrek.baseline import BaselineDefender
from shrek.config import ExperimentConfig
from shrek.data import Example, audit_splits
from shrek.memory import MemoryStore
from shrek.model_client import Backend, BudgetExceeded, JsonlLog, MeteredClient
from shrek.schemas import Rule


def environment_manifest() -> dict:
    packages = {}
    for name in ("shrek-defense", "torch", "transformers", "accelerate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        sha = commit.stdout.strip() if commit.returncode == 0 else None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        is_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        sha = is_dirty = None
    source_root = Path(__file__).parent
    fingerprint = hashlib.sha256()
    for path in sorted([*source_root.glob("*.py"), *source_root.glob("prompt_templates/*.txt")]):
        fingerprint.update(path.relative_to(source_root).as_posix().encode())
        fingerprint.update(path.read_bytes())
    return {"python": platform.python_version(), "packages": packages, "git_commit": sha, "git_dirty": is_dirty,
            "package_source_sha256": fingerprint.hexdigest(),
            "prompt_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_root.glob("prompt_templates/*.txt"))}}


def run_baseline(examples: list[Example], backend: Backend, config: ExperimentConfig, output: str | Path,
                 *, learning: bool = True, max_calls: int = 100, max_output_tokens: int = 384,
                 token_counter: Callable[[str], int] | None = None, initial_rules: tuple[Rule, ...] = ()) -> dict:
    # Validate before creating run artifacts or making model calls.
    if config.admission_validation or config.maintenance_enabled:
        raise ValueError("Phase 2 supports B0 only; SHREK gate/maintenance are later phases")
    if not examples:
        raise ValueError("No examples selected")
    audit_splits(examples)
    if len({item.request_id for item in examples}) != len(examples):
        raise ValueError("Duplicate request IDs")
    if learning and any(item.split == "holdout" for item in examples):
        raise ValueError("Holdout runs must disable learning")
    if learning and config.capacity_mode == "controlled" and token_counter is None:
        raise ValueError("Controlled token limits need a host tokenizer; use baseline_paper.toml otherwise")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    client = MeteredClient(backend, JsonlLog(output / "calls.jsonl"), max_calls, max_output_tokens)
    defender = BaselineDefender(client, config)
    store = MemoryStore(output / "memory.sqlite3")
    if initial_rules:
        change = store.propose(store.snapshot(), "RESTORE", initial_rules, "Explicit initial-memory import")
        store.commit(change, {"decision_source": "operator_initial_memory", "behavioral_validation": False})
    runtime_log = JsonlLog(output / "interactions.jsonl")
    answers_log = JsonlLog(output / "answers.jsonl")
    metadata = backend.metadata() if hasattr(backend, "metadata") else {"model": backend.model_id}
    # Include labels only in this evaluation manifest, never in defense inputs or prompt construction.
    dataset_hash = hashlib.sha256(json.dumps([asdict(item) for item in examples], sort_keys=True).encode()).hexdigest()
    manifest = {"phase": 2, "variant": "B0", "is_mock": backend.is_mock, "learning": learning,
                "config": asdict(config), "config_sha256": config.digest, "dataset_sha256": dataset_hash,
                "ordered_example_ids": [item.request_id for item in examples], "backend": metadata,
                "environment": environment_manifest(), "max_calls": max_calls, "max_output_tokens": max_output_tokens,
                "initial_memory_sha256": store.snapshot().digest,
                "scoring": "Online learning feedback only; final research metrics require independent/manual evaluation."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    answered = errors = processed = truncated_answers = 0
    accepted = 0
    stopped = False
    manual_rows = []
    started = time.perf_counter()
    try:
        for sequence, example in enumerate(examples, 1):
            request = example.runtime()
            snapshot = store.snapshot()
            record = {"sequence": sequence, "request_id": request.request_id, "memory_version_before": snapshot.version,
                      "is_mock": backend.is_mock, "response": None, "feedback": None, "update": None, "status": "error"}
            processed += 1
            try:
                result = defender.respond(request, snapshot)
                record["response"] = asdict(result)
                # Durable original response is written BEFORE judging or any memory mutation.
                answers_log.append({"sequence": sequence, "is_mock": backend.is_mock, **asdict(result)})
                answered += 1
                if result.response_truncated:
                    truncated_answers += 1
                    errors += 1
                    record["error_type"] = "TruncatedHostResponse"
                    record["update"] = {"status": "deferred", "reason": "Incomplete host answer"}
                manual_rows.append({"request_id": example.request_id, "evaluation_label": example.evaluation_label,
                                    "request": example.text, "response": result.response,
                                    "harmful_compliance": "", "unnecessary_refusal": "", "useful_completion": "", "review_notes": ""})
                if learning and not result.response_truncated:
                    feedback = defender.assess(request, result)
                    record["feedback"] = asdict(feedback)
                    record["update"] = defender.learn(request, result, feedback, store, token_counter)
                    if record["update"]["status"] == "accepted":
                        accepted += 1
                record["status"] = "incomplete" if result.response_truncated else "ok"
            except BudgetExceeded:
                stopped = True
                errors += 1
                record["error_type"] = "BudgetExceeded"
            except Exception as error:
                errors += 1
                record["error_type"] = type(error).__name__
                # Malformed judgments/candidates never become successful outcomes or memory writes.
            finally:
                record["memory_version_after"] = store.snapshot().version
                runtime_log.append(record)
            if stopped:
                break
        summary = {"phase": 2, "is_mock": backend.is_mock, "research_results": False,
                   "input_examples": len(examples), "processed": processed, "answered": answered,
                   "unanswered": len(examples) - answered, "events_with_errors": errors,
                   "truncated_answers": truncated_answers,
                   "accepted_updates": accepted, "budget_exhausted": stopped, "active_rules": len(store.snapshot().rules),
                   "final_memory_version": store.snapshot().version, "usage": client.usage(),
                   "elapsed_seconds": time.perf_counter() - started,
                   "note": "Execution report only. Mock outputs are synthetic; real outputs are not independently scored yet."}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output / "final_memory.json").write_text(json.dumps([rule.to_dict() for rule in store.snapshot().rules], indent=2), encoding="utf-8")
        with (output / "manual_review.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["request_id", "evaluation_label", "request", "response", "harmful_compliance", "unnecessary_refusal", "useful_completion", "review_notes"])
            writer.writeheader()
            # Prevent spreadsheet formula evaluation in manually inspected model text.
            for row in manual_rows:
                writer.writerow({key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value for key, value in row.items()})
        return summary
    finally:
        store.close()
