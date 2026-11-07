"""Matched-input B0-B3 adaptation and frozen holdout; no invented performance scores."""

from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import random

from shrek.config import ExperimentConfig
from shrek.data import Example, audit_splits
from shrek.runner import environment_manifest
from shrek.schemas import Rule
from shrek.system import dataset_digest, run_system


FLAGS = {"B0": (False, False), "B1": (True, False), "B2": (False, True), "B3": (True, True)}


def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def csv_safe(value):
    return "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def run_study(examples: list[Example], backend, template: ExperimentConfig, output, *, seeds=(17,),
              max_calls=2000, holdout_calls=500, max_output_tokens=384, noise_rate=0.0, order="shuffle", progress=None) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("Provide distinct integer seeds")
    if order not in {"shuffle", "preserve"} or not 0 <= noise_rate <= 1:
        raise ValueError("Invalid ordering or feedback-noise condition")
    if type(max_calls) is not int or max_calls < 1 or type(holdout_calls) is not int or holdout_calls < 1:
        raise ValueError("Positive stage call budgets required")
    audit = audit_splits(examples)
    if len({item.request_id for item in examples}) != len(examples):
        raise ValueError("Study IDs must be unique")
    partitions = {split: [item for item in examples if item.split == split] for split in ("development", "adaptation", "holdout")}
    for split, selected in partitions.items():
        counts = {kind: len({item.behavior_group for item in selected if item.evaluation_label == kind}) for kind in ("harmful", "legitimate")}
        minimum = template.validation_min_per_class if split == "development" else 1
        if any(count < minimum for count in counts.values()):
            raise ValueError(f"{split} needs at least {minimum} examples of each class")
    if template.capacity_mode == "controlled" and not callable(getattr(backend, "count_tokens", None)):
        raise ValueError("Controlled study requires the actual host tokenizer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = environment_manifest()
    plan = {"phase": 4, "is_mock": backend.is_mock, "dataset_sha256": dataset_digest(examples), "audit": audit,
            "evaluation_only_input_snapshot": "study_input.jsonl",
            "seeds": list(seeds), "order": order, "noise_rate": noise_rate, "template": asdict(template),
            "max_calls_per_adaptation": max_calls, "max_calls_per_holdout": holdout_calls,
            "max_calls_entire_study": len(seeds) * 4 * (max_calls + holdout_calls),
            "max_output_tokens": max_output_tokens, "environment": environment,
            "variant_flags": {name: {"admission_validation": flags[0], "maintenance_enabled": flags[1]} for name, flags in FLAGS.items()},
            "stage_note": "Holdout uses the final memory of the matching variant/seed; no feedback, noise, validation, or maintenance occurs in holdout.",
            "budget_note": "Identical stage ceilings, not a claim of identical consumed compute. Report actual call/token usage."}
    (output / "study_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (output / "study_input.jsonl").write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in examples), encoding="utf-8")
    runs, key, queue = [], {}, []
    for seed in seeds:
        adaptation = list(partitions["adaptation"])
        if order == "shuffle":
            random.Random(seed).shuffle(adaptation)
        variant_order = list(FLAGS)
        random.Random(seed + 1).shuffle(variant_order)
        for variant in variant_order:
            config = replace(template, variant=variant, admission_validation=FLAGS[variant][0], maintenance_enabled=FLAGS[variant][1])
            if callable(getattr(backend, "reset_seed", None)):
                backend.reset_seed(seed)
            run_dir = output / f"seed-{seed}" / variant
            adapt_progress = (lambda state: progress({"seed": seed, "variant": variant, "stage": "adaptation", **state})) if progress else None
            hold_progress = (lambda state: progress({"seed": seed, "variant": variant, "stage": "holdout", **state})) if progress else None
            adaptation_summary = run_system(adaptation, backend, config, run_dir / "adaptation", anchors=partitions["development"],
                max_calls=max_calls, max_output_tokens=max_output_tokens, noise_rate=noise_rate, seed=seed, progress=adapt_progress)
            learned = tuple(Rule.from_dict(item) for item in json.loads((run_dir / "adaptation/final_memory.json").read_text(encoding="utf-8")))
            holdout_summary = run_system(partitions["holdout"], backend, config, run_dir / "holdout", anchors=partitions["development"],
                learning=False, max_calls=holdout_calls, max_output_tokens=max_output_tokens, initial_rules=learned, seed=seed, progress=hold_progress)
            job = {"seed": seed, "variant": variant, "adaptation": adaptation_summary, "holdout": holdout_summary,
                   "asr": None, "false_refusal_rate": None, "benign_task_success": None,
                   "scoring_status": "awaiting independent holdout review"}
            runs.append(job)
            answered = {item["request_id"]: item for item in read_jsonl(run_dir / "holdout/answers.jsonl")}
            for example in partitions["holdout"]:
                review_id = content_hash(f"{seed}:{variant}:holdout:{example.request_id}")[:20]
                answer = answered.get(example.request_id)
                key[review_id] = {"seed": seed, "variant": variant, "stage": "holdout", "request_id": example.request_id,
                                  "behavior_group": example.behavior_group, "evaluation_label": example.evaluation_label,
                                  "request_sha256": content_hash(example.text), "answered": answer is not None,
                                  "export_request_sha256": content_hash(csv_safe(example.text)),
                                  "response_sha256": content_hash(answer["response"]) if answer else None,
                                  "export_response_sha256": content_hash(csv_safe(answer["response"])) if answer else None,
                                  "response_truncated": answer["response_truncated"] if answer else None}
                if answer:
                    queue.append({"review_id": review_id, "evaluation_label": example.evaluation_label,
                                  "request": example.text, "response": answer["response"],
                                  "response_truncated": answer["response_truncated"], "harmful_compliance": "",
                                  "unnecessary_refusal": "", "useful_completion": "", "review_notes": ""})
            # Checkpoint progress after each complete variant; interruption leaves an inspectable plan/log.
            (output / "execution_summary.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    random.Random(seeds[0] + 2).shuffle(queue)
    (output / "review_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    with (output / "review_queue.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["review_id", "evaluation_label", "request", "response", "response_truncated", "harmful_compliance", "unnecessary_refusal", "useful_completion", "review_notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in queue:
            writer.writerow({name: csv_safe(value) for name, value in row.items()})
    report = {"phase": 4, "is_mock": backend.is_mock, "research_results": False, "runs": runs,
              "review_rows": len(queue), "expected_holdout_outcomes": len(key),
              "unanswered_holdout": sum(not item["answered"] for item in key.values()),
              "total_model_calls": sum(job[stage]["usage"]["total_calls"] for job in runs for stage in ("adaptation", "holdout")),
              "execution_complete": all(job[stage]["unanswered"] == 0 and job[stage]["events_with_errors"] == 0 for job in runs for stage in ("adaptation", "holdout")),
              "note": "No performance rates are inferred from online learning feedback. Review the blinded queue before scoring. Synthetic runs remain mock evidence."}
    (output / "study_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
