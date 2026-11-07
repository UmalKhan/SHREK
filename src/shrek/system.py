"""Chronological B0-B3 execution with disjoint inference/update/replay accounting."""

from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import time

from shrek.baseline import BaselineDefender
from shrek.config import ExperimentConfig
from shrek.controller import ShrekController
from shrek.data import Example, audit_splits
from shrek.memory import MemoryStore
from shrek.model_client import BudgetExceeded, CallBudget, JsonlLog, MeteredClient
from shrek.probes import ProbePool
from shrek.runner import environment_manifest
from shrek.schemas import Rule
from shrek.validation import ChangeValidator, memory_size, within_capacity


def dataset_digest(examples) -> str:
    return hashlib.sha256(json.dumps([asdict(item) for item in examples], sort_keys=True).encode()).hexdigest()


def noisy_feedback(feedback, request_id: str, label: str, rate: float, seed: int):
    eligible = feedback.score is not None and label != "benign"
    selection = int.from_bytes(hashlib.sha256(f"{seed}:{request_id}".encode()).digest(), "big") / 2**256
    if not eligible or selection >= rate:
        return feedback, None, eligible
    direction = "false_negative" if feedback.violation else "false_positive"
    return replace(feedback, score=1 if feedback.violation else 10,
                   rationale="CONTROLLED_FEEDBACK_NOISE: " + feedback.rationale), direction, eligible


def run_system(examples: list[Example], backend, config: ExperimentConfig, output: str | Path, *,
               anchors: list[Example] | None = None, learning: bool = True, max_calls: int = 2000,
               max_output_tokens: int = 384, initial_rules: tuple[Rule, ...] = (),
               noise_rate: float = 0.0, seed: int = 17) -> dict:
    anchors = anchors or []
    if not examples or len({item.request_id for item in [*anchors, *examples]}) != len(anchors) + len(examples):
        raise ValueError("Need nonempty examples with distinct anchor/runtime IDs")
    audit_splits([*anchors, *examples])
    if learning and any(item.split == "holdout" for item in examples):
        raise ValueError("Holdout cannot adapt")
    if not 0 <= noise_rate <= 1:
        raise ValueError("Noise rate must be between 0 and 1")
    token_counter = getattr(backend, "count_tokens", None)
    if config.capacity_mode == "controlled" and token_counter is None:
        raise ValueError("Controlled memory caps require a verified host tokenizer")
    if not within_capacity(initial_rules, config, token_counter):
        raise ValueError("Initial memory exceeds the experiment's capacity")
    pool = ProbePool(anchors, config.replay_limit)
    if learning and (config.admission_validation or config.maintenance_enabled):
        counts = {kind: sum(item.evaluation_label == kind for item in anchors) for kind in ("harmful", "legitimate")}
        if any(count < config.validation_min_per_class for count in counts.values()):
            raise ValueError("Provide sufficient development anchors of BOTH classes before running SHREK")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    budget = CallBudget(max_calls)
    calls = JsonlLog(output / "calls.jsonl")
    inference = MeteredClient(backend, calls, max_calls, max_output_tokens, budget=budget, purpose="inference")
    updates = MeteredClient(backend, calls, config.update_call_budget, max_output_tokens, budget=budget, purpose="learning")
    replay = MeteredClient(backend, calls, config.validation_call_budget, max_output_tokens, budget=budget, purpose="validation")
    defender = BaselineDefender(inference, config)
    learner = BaselineDefender(updates, config)
    validator = ChangeValidator(BaselineDefender(replay, config), config, token_counter)
    store = MemoryStore(output / "memory.sqlite3")
    if initial_rules:
        imported = store.propose(store.snapshot(), "RESTORE", initial_rules, "Explicit initial memory")
        store.commit(imported, {"decision_source": "operator_initial_memory"})
    controller = ShrekController(store, learner, validator, pool, config, JsonlLog(output / "validation.jsonl"), token_counter)
    interactions, answers = JsonlLog(output / "interactions.jsonl"), JsonlLog(output / "answers.jsonl")
    manifest = {"phase": 3, "variant": config.variant, "is_mock": backend.is_mock, "config": asdict(config),
                "config_sha256": config.digest, "learning": learning, "seed": seed, "noise_rate": noise_rate,
                "dataset_sha256": dataset_digest(examples), "anchors_sha256": dataset_digest(anchors),
                "ordered_example_ids": [item.request_id for item in examples],
                "development_anchor_ids": [item.request_id for item in anchors],
                "initial_memory_sha256": store.snapshot().digest, "max_total_calls": max_calls,
                "max_output_tokens": max_output_tokens, "environment": environment_manifest(),
                "backend": backend.metadata() if hasattr(backend, "metadata") else {"model": backend.model_id},
                "anchor_provenance": "Development benchmark labels; manual review is a separate operator responsibility",
                "noise_protocol": "Seeded request-ID selection; only known-score, non-benign-classified online verdicts are eligible. Final labels unchanged."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, timeline = [], []
    answered = errors = processed = truncated = feedback_missing = 0
    noise_counts = {"eligible": 0, "false_positive": 0, "false_negative": 0}
    stopped = updates_frozen = False
    started = time.perf_counter()
    try:
        for sequence, example in enumerate(examples, 1):
            request = example.runtime()
            before = store.snapshot()
            processed += 1
            feedback = None
            result = None
            record = {"sequence": sequence, "request_id": request.request_id, "is_mock": backend.is_mock,
                      "memory_version_before": before.version, "status": "error", "update": None, "maintenance": None}
            try:
                result = defender.respond(request, before)
                answers.append({"sequence": sequence, "is_mock": backend.is_mock, **asdict(result)})
                record["response"] = asdict(result)
                answered += 1
                rows.append({"request_id": request.request_id, "evaluation_label": example.evaluation_label,
                             "request": request.text, "response": result.response, "response_truncated": result.response_truncated,
                             "harmful_compliance": "", "unnecessary_refusal": "", "useful_completion": "", "review_notes": ""})
                if result.response_truncated:
                    truncated += 1
                    errors += 1
                    record.update(status="incomplete", error_type="TruncatedHostResponse")
                elif learning:
                    if updates_frozen:
                        feedback_missing += 1
                        record["update"] = {"status": "deferred", "reason": "Learning budget exhausted; inference continues"}
                    else:
                        try:
                            natural = learner.assess(request, result)
                            feedback, direction, eligible = noisy_feedback(natural, request.request_id, result.attack_label, noise_rate, seed)
                            noise_counts["eligible"] += int(eligible)
                            if direction:
                                noise_counts[direction] += 1
                            record["natural_feedback"] = asdict(natural)
                            record["feedback"] = asdict(feedback)
                            record["injected_error"] = direction
                            record["update"] = controller.learn(request, result, feedback, sequence, example.behavior_group)
                        except BudgetExceeded:
                            updates_frozen = True
                            feedback_missing += int(feedback is None)
                            record["update"] = {"status": "deferred", "reason": "Learning call budget exhausted"}
                    controller.observe(request, result, feedback, sequence, example.behavior_group)
                    if not updates_frozen:
                        try:
                            record["maintenance"] = controller.maintain(sequence)
                        except BudgetExceeded:
                            updates_frozen = True
                            record["maintenance"] = {"status": "deferred", "reason": "Learning call budget exhausted"}
                if not result.response_truncated:
                    record["status"] = "ok"
            except BudgetExceeded:
                stopped = True
                errors += 1
                record["error_type"] = "BudgetExceeded"
            except Exception as error:
                errors += 1
                record["error_type"] = type(error).__name__
            finally:
                current = store.snapshot()
                size, unit = memory_size(current.rules, token_counter)
                record["memory_version_after"] = current.version
                interactions.append(record)
                timeline.append({"sequence": sequence, "active_rules": len(current.rules), "active_memory_size": size,
                                 "memory_size_unit": unit, "memory_version": current.version})
            if stopped:
                break
        usage = {"total_calls": budget.used, "max_total_calls": budget.maximum,
                 "inference": inference.usage(), "learning": updates.usage(), "validation": replay.usage()}
        summary = {"phase": 3, "variant": config.variant, "is_mock": backend.is_mock, "research_results": False,
                   "input_examples": len(examples), "processed": processed, "answered": answered,
                   "unanswered": len(examples) - answered, "truncated_answers": truncated, "events_with_errors": errors,
                   "feedback_missing": feedback_missing, "budget_exhausted": stopped, "learning_frozen": updates_frozen,
                   "active_rules": len(store.snapshot().rules), "final_memory_version": store.snapshot().version,
                   "decisions": dict(controller.decisions), "admission_attempts": controller.admission_attempts,
                   "maintenance_attempts": controller.maintenance_attempts, "validation_cache_hits": validator.cache_hits,
                   "replay_entries": len(pool.history), "feedback_noise": noise_counts, "usage": usage,
                   "elapsed_seconds": time.perf_counter() - started,
                   "note": "Execution/online validation only. Final rates require independent manual/evaluation review; fixtures are not research evidence."}
        for name, value in (("summary.json", summary), ("memory_timeline.json", timeline),
                            ("final_memory.json", [rule.to_dict() for rule in store.snapshot().rules])):
            (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
        with (output / "manual_review.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["request_id", "evaluation_label", "request", "response", "response_truncated", "harmful_compliance", "unnecessary_refusal", "useful_completion", "review_notes"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value for key, value in row.items()})
        return summary
    finally:
        store.close()
