"""Small, offline entry point for the project scaffold."""

import argparse
import json
from datetime import datetime, timezone

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
