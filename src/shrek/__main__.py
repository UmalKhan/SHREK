"""Small, offline entry point for the project scaffold."""

import argparse
import json
from datetime import datetime, timezone
import os

from shrek import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shrek",
        description="SHREK: regression-tested defensive memory for frozen LLMs.",
        epilog="Use an offline demo to verify infrastructure before model-backed experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    demo = commands.add_parser("demo", help="Run a clearly labeled offline engineering fixture")
    demo.add_argument("--output", default=None)
    check = commands.add_parser("check-config", help="Validate an experiment TOML file")
    check.add_argument("path")
    audit = commands.add_parser("audit-data", help="Check dataset schema and cross-split overlap")
    audit.add_argument("path")
    commands.add_parser("hardware", help="Report GPU and optional inference dependencies")
    baseline_demo = commands.add_parser("baseline-demo", help="Exercise chronological learning with synthetic fixtures")
    baseline_demo.add_argument("--output", default=None)
    baseline_demo.add_argument("--max-calls", type=int, default=100)
    shrek_demo = commands.add_parser("shrek-demo", help="Verify admission, deferral, retirement, and revision with fictional fixtures")
    shrek_demo.add_argument("--output", default=None)
    system = commands.add_parser("run-system", help="Run B0-B3 with development anchors and bounded validation")
    system.add_argument("input", help="Audited JSONL with development/adaptation/holdout splits")
    system.add_argument("--config", default="configs/combined.toml")
    system.add_argument("--backend", choices=["transformers", "http"], required=True)
    system.add_argument("--model", default=os.environ.get("HOST_MODEL"))
    system.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8000/v1"))
    system.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    system.add_argument("--revision", default="main")
    system.add_argument("--context-limit", type=int, default=4096)
    system.add_argument("--load-in-4bit", action="store_true", help="Load the local model with 4-bit NF4 quantization (needs bitsandbytes)")
    system.add_argument("--seed", type=int, default=17)
    system.add_argument("--split", choices=["adaptation", "holdout"], default="adaptation")
    system.add_argument("--no-learning", action="store_true")
    system.add_argument("--initial-memory")
    system.add_argument("--noise-rate", type=float, default=0.0)
    system.add_argument("--max-calls", type=int, default=2000)
    system.add_argument("--max-output-tokens", type=int, default=384)
    system.add_argument("--output", default=None)
    prepare = commands.add_parser("prepare-study", help="Prepare a pinned 80-example PAIR/XSTest pilot")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--source-dir", default=None, help="Optional local snapshots named pair.json, xstest.csv, and license files")
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--development-per-class", type=int, default=10)
    prepare.add_argument("--adaptation-per-class", type=int, default=20)
    prepare.add_argument("--holdout-per-class", type=int, default=10)
    comparison_demo = commands.add_parser("compare-demo", help="Run the four-variant pipeline on clearly fictional inputs")
    comparison_demo.add_argument("--output", default=None)
    compare = commands.add_parser("compare", help="Run matched B0-B3 adaptation and frozen holdout")
    compare.add_argument("input")
    compare.add_argument("--config", default="configs/combined.toml", help="Shared settings; variant flags are assigned by the study")
    compare.add_argument("--backend", choices=["transformers", "http"], required=True)
    compare.add_argument("--model", default=os.environ.get("HOST_MODEL"))
    compare.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8000/v1"))
    compare.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    compare.add_argument("--revision", default="main")
    compare.add_argument("--context-limit", type=int, default=4096)
    compare.add_argument("--load-in-4bit", action="store_true", help="Load the local model with 4-bit NF4 quantization (needs bitsandbytes)")
    compare.add_argument("--seeds", type=int, nargs="+", default=[17])
    compare.add_argument("--noise-rate", type=float, default=0.0)
    compare.add_argument("--order", choices=["shuffle", "preserve"], default="shuffle")
    compare.add_argument("--max-calls", type=int, default=2000)
    compare.add_argument("--holdout-calls", type=int, default=500)
    compare.add_argument("--max-output-tokens", type=int, default=384)
    compare.add_argument("--output", default=None)
    score = commands.add_parser("score-study", help="Compute rates/bounds from independent review, preserving missing outcomes")
    score.add_argument("study_dir")
    score.add_argument("--reviews", required=True)
    run = commands.add_parser("run-baseline", help="Run the paper-based B0 adaptation on a real model")
    run.add_argument("input", help="JSONL input; use audit-data before research runs")
    run.add_argument("--config", default="configs/baseline_paper.toml")
    run.add_argument("--backend", choices=["transformers", "http"], required=True)
    run.add_argument("--model", default=os.environ.get("HOST_MODEL"))
    run.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:8000/v1"))
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--revision", default="main")
    run.add_argument("--context-limit", type=int, default=4096)
    run.add_argument("--load-in-4bit", action="store_true", help="Load the local model with 4-bit NF4 quantization (needs bitsandbytes)")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--split", choices=["development", "adaptation", "holdout"], default="adaptation")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--no-learning", action="store_true")
    run.add_argument("--initial-memory", help="Optional final_memory.json from a prior run; never loads evaluation labels")
    run.add_argument("--max-calls", type=int, default=100)
    run.add_argument("--max-output-tokens", type=int, default=384)
    run.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.command == "demo":
        from shrek.demo import run_demo
        output = args.output or "runs/demo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        print(json.dumps(run_demo(output), indent=2))
    elif args.command == "check-config":
        from dataclasses import asdict
        from shrek.config import load_config
        config = load_config(args.path)
        print(json.dumps({"config": asdict(config), "sha256": config.digest}, indent=2))
    elif args.command == "audit-data":
        from shrek.data import audit_splits, file_digest, load_examples
        print(json.dumps({**audit_splits(load_examples(args.path)), "file_sha256": file_digest(args.path)}, indent=2))
    elif args.command == "hardware":
        from shrek.hardware import hardware_report
        print(json.dumps(hardware_report(), indent=2))
    elif args.command == "baseline-demo":
        from shrek.config import ExperimentConfig
        from shrek.fixtures import BaselineFixtureBackend, baseline_examples
        from shrek.runner import run_baseline
        output = args.output or "runs/baseline-demo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        summary = run_baseline(baseline_examples(), BaselineFixtureBackend(), ExperimentConfig(), output, max_calls=args.max_calls)
        print(json.dumps(summary, indent=2))
        if summary["events_with_errors"]:
            raise SystemExit(2)
    elif args.command == "shrek-demo":
        from shrek.controller_demo import run_controller_demo
        output = args.output or "runs/shrek-demo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report = run_controller_demo(output)
        print(json.dumps({"passed": report["passed"], "is_mock": True, "checks": report["checks"], "output": output}, indent=2))
        if not report["passed"]:
            raise SystemExit(2)
    elif args.command == "prepare-study":
        from shrek.prepare import prepare_study
        report = prepare_study(args.output, seed=args.seed, source_dir=args.source_dir,
                               development_per_class=args.development_per_class, adaptation_per_class=args.adaptation_per_class,
                               holdout_per_class=args.holdout_per_class)
        print(json.dumps({"output": args.output, "audit": report["audit"], "review_required": report["review_required"], "study_sha256": report["study_sha256"]}, indent=2))
    elif args.command == "compare-demo":
        from shrek.config import ExperimentConfig
        from shrek.shrek_fixtures import ControllerFixtureBackend, fixture_study
        from shrek.study import run_study
        output = args.output or "runs/comparison-demo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        template = ExperimentConfig(capacity_mode="controlled", validation_probe_limit=4, validation_min_per_class=2,
                                    maintenance_every=2, validation_call_budget=400)
        report = run_study(fixture_study(), ControllerFixtureBackend(), template, output, max_calls=1000, holdout_calls=100)
        print(json.dumps({"is_mock": report["is_mock"], "execution_complete": report["execution_complete"],
                          "review_rows": report["review_rows"], "total_model_calls": report["total_model_calls"], "output": output,
                          "note": "Pipeline verification only; review queue is blank and no effectiveness rates are claimed."}, indent=2))
        if not report["execution_complete"]:
            raise SystemExit(2)
    elif args.command == "compare":
        from pathlib import Path
        from shrek.backends import CompatibleHTTPBackend, TransformersBackend
        from shrek.config import load_config
        from shrek.data import load_examples, audit_splits
        from shrek.study import run_study
        if not args.model or args.max_calls < 1 or args.holdout_calls < 1 or args.max_output_tokens < 1:
            parser.error("Provide an explicit model and positive stage/output budgets")
        if not 0 <= args.noise_rate <= 1 or len(set(args.seeds)) != len(args.seeds):
            parser.error("Use distinct seeds and a noise rate between 0 and 1")
        examples = load_examples(args.input)
        audit_splits(examples)
        config = load_config(args.config)
        if args.backend == "http" and config.capacity_mode == "controlled":
            parser.error("Controlled study requires the host tokenizer; use paper capacity for HTTP")
        output = args.output or "runs/study-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if Path(output).exists():
            parser.error("Choose a fresh study output directory")
        backend = (TransformersBackend(args.model, args.revision, args.device, args.context_limit, args.seeds[0], load_in_4bit=args.load_in_4bit)
                   if args.backend == "transformers" else CompatibleHTTPBackend(args.base_url, args.model, os.environ.get("MODEL_API_KEY")))
        report = run_study(examples, backend, config, output, seeds=tuple(args.seeds), max_calls=args.max_calls,
                           holdout_calls=args.holdout_calls, max_output_tokens=args.max_output_tokens,
                           noise_rate=args.noise_rate, order=args.order, progress=lambda state: print(json.dumps({"progress": state}), flush=True))
        print(json.dumps({"output": output, "is_mock": report["is_mock"], "execution_complete": report["execution_complete"],
                          "review_rows": report["review_rows"], "unanswered_holdout": report["unanswered_holdout"],
                          "total_model_calls": report["total_model_calls"], "note": report["note"]}, indent=2))
        if not report["execution_complete"]:
            raise SystemExit(2)
    elif args.command == "score-study":
        from shrek.scoring import score_study
        print(json.dumps(score_study(args.study_dir, args.reviews), indent=2))
    elif args.command == "run-system":
        from pathlib import Path
        from shrek.backends import CompatibleHTTPBackend, TransformersBackend
        from shrek.config import load_config
        from shrek.data import audit_splits, load_examples
        from shrek.schemas import Rule
        from shrek.system import run_system
        if not args.model or args.max_calls < 1 or args.max_output_tokens < 1:
            parser.error("Provide a model and positive call/output limits")
        if args.split == "holdout" and not args.no_learning:
            parser.error("Holdout must use --no-learning")
        if not 0 <= args.noise_rate <= 1:
            parser.error("Noise rate must be between 0 and 1")
        config = load_config(args.config)
        if args.backend == "http" and config.capacity_mode == "controlled":
            parser.error("Controlled token caps require the local host tokenizer; use a paper-capacity config for HTTP")
        examples = load_examples(args.input)
        audit_splits(examples)
        anchors = [item for item in examples if item.split == "development"]
        selected = [item for item in examples if item.split == args.split]
        if not selected:
            parser.error("No examples for the selected split")
        initial = ()
        if args.initial_memory:
            initial = tuple(Rule.from_dict(item) for item in json.loads(Path(args.initial_memory).read_text(encoding="utf-8")))
        output = args.output or "runs/system-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if Path(output).exists():
            parser.error("Choose a fresh output directory")
        backend = (TransformersBackend(args.model, args.revision, args.device, args.context_limit, args.seed, load_in_4bit=args.load_in_4bit)
                   if args.backend == "transformers" else CompatibleHTTPBackend(args.base_url, args.model, os.environ.get("MODEL_API_KEY")))
        summary = run_system(selected, backend, config, output, anchors=anchors, learning=not args.no_learning,
                             max_calls=args.max_calls, max_output_tokens=args.max_output_tokens, initial_rules=initial,
                             noise_rate=args.noise_rate, seed=args.seed)
        print(json.dumps(summary, indent=2))
        if summary["events_with_errors"]:
            raise SystemExit(2)
    elif args.command == "run-baseline":
        from pathlib import Path
        from shrek.backends import CompatibleHTTPBackend, TransformersBackend
        from shrek.config import load_config
        from shrek.data import audit_splits, load_examples
        from shrek.runner import run_baseline
        from shrek.schemas import Rule
        if not args.model:
            parser.error("Provide --model or set HOST_MODEL; model selection is explicit")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.max_calls < 1 or args.max_output_tokens < 1:
            parser.error("Call and output limits must be positive")
        config = load_config(args.config)
        if config.admission_validation or config.maintenance_enabled:
            parser.error("Only B0 is implemented in Phase 2; do not run SHREK variants yet")
        if args.split == "holdout" and not args.no_learning:
            parser.error("Use --no-learning for holdout evaluation")
        if args.backend == "http" and config.capacity_mode == "controlled":
            parser.error("HTTP backend has no verified tokenizer; use baseline_paper.toml")
        examples = load_examples(args.input)
        audit_splits(examples)
        selected = [item for item in examples if item.split == args.split]
        if args.limit is not None:
            selected = selected[:args.limit]
        if not selected:
            parser.error("No examples match the selected split")
        initial = ()
        if args.initial_memory:
            records = json.loads(Path(args.initial_memory).read_text(encoding="utf-8"))
            if not isinstance(records, list):
                parser.error("Initial memory must be a JSON list of rules")
            initial = tuple(Rule.from_dict(item) for item in records)
        output = args.output or "runs/baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if Path(output).exists():
            parser.error("Output directory already exists; choose a fresh run directory")
        if args.backend == "transformers":
            backend = TransformersBackend(args.model, args.revision, args.device, args.context_limit, args.seed, load_in_4bit=args.load_in_4bit)
        else:
            backend = CompatibleHTTPBackend(args.base_url, args.model, os.environ.get("MODEL_API_KEY"))
        summary = run_baseline(selected, backend, config, output, learning=not args.no_learning,
                               max_calls=args.max_calls, max_output_tokens=args.max_output_tokens,
                               token_counter=getattr(backend, "count_tokens", None), initial_rules=initial)
        print(json.dumps(summary, indent=2))
        if summary["events_with_errors"]:
            raise SystemExit(2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
