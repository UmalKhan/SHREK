"""Phase 1 offline demonstration: plumbing, not behavioral validation."""

import json
from pathlib import Path

from shrek.memory import MemoryStore
from shrek.model_client import BudgetExceeded, JsonlLog, MeteredClient, ScriptedBackend
from shrek.schemas import Rule, RuntimeRequest


def run_demo(output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    client = MeteredClient(ScriptedBackend([
        ("demo", {"rule_id": "fixture-rule", "label": "fixture-format", "pattern": "Synthetic forbidden marker", "text": "Decline the synthetic forbidden marker."}),
        ("demo", {"decision": "reject", "reason": "Synthetic review rejects an overbroad candidate"}),
    ]), JsonlLog(output / "calls.jsonl"), max_calls=2)
    store = MemoryStore(output / "memory.sqlite3")
    try:
        initial = store.snapshot()
        request = RuntimeRequest("fixture-1", "Synthetic engineering request")
        rule = Rule.from_dict(client.ask("demo", "Return a synthetic fixture rule.", request.payload()))
        accepted = store.propose(initial, "ADD", (rule,), "Scripted storage acceptance demonstration")
        current = store.commit(accepted, {"is_mock": True, "decision_source": "scripted_demo_not_validation"})
        rejected = store.propose(current, "ARCHIVE", (), "Scripted rejection demonstration")
        review = client.ask("demo", "Return a synthetic review.", request.payload())
        store.reject(rejected, review["reason"])
        stopped = False
        try:
            client.ask("demo", "Must not reach backend", {})
        except BudgetExceeded:
            stopped = True
        report = {"phase": 1, "is_mock": True, "research_results": False,
                  "initial_memory_version": initial.version, "active_memory_version": store.snapshot().version,
                  "active_rules": len(store.snapshot().rules), "rejected_proposal_preserved_memory": store.snapshot() == current,
                  "budget_stopped_extra_call": stopped, "usage": client.usage(),
                  "note": "Storage and call-accounting demonstration; SHREK behavioral gate is not implemented."}
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        store.close()
