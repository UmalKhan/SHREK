from dataclasses import replace
import json

import pytest

from shrek.config import ExperimentConfig, load_config
from shrek.data import Example, audit_splits
from shrek.demo import run_demo
from shrek.memory import MemoryStore, StaleProposalError
from shrek.model_client import BudgetExceeded, JsonlLog, MeteredClient, ModelOutputError, ScriptedBackend
from shrek.schemas import Rule


def rule(identifier="r1"):
    return Rule(identifier, "fixture-format", "Synthetic pattern", "Synthetic restriction")


def test_transaction_rejection_stale_proposal_and_restore(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    initial = store.snapshot()
    first = store.propose(initial, "ADD", (rule(),), "first")
    stale = store.propose(initial, "ADD", (rule("r2"),), "stale")
    committed = store.commit(first, {"fixture": True})
    with pytest.raises(StaleProposalError):
        store.commit(stale, {})
    assert store.snapshot() == committed
    rejected = store.propose(committed, "ARCHIVE", (), "reject")
    store.reject(rejected, "Preserve protection")
    with pytest.raises(ValueError):
        store.commit(rejected, {})
    assert store.snapshot() == committed
    restored = store.restore(0, "Operator rollback")
    assert restored.rules == initial.rules
    assert restored.version > committed.version
    assert store.snapshot(committed.version) == committed
    store.close()


def test_tampered_pending_proposal_cannot_activate(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    proposal = store.propose(store.snapshot(), "ADD", (rule(),), "first")
    with pytest.raises(ValueError):
        store.commit(replace(proposal, candidate_rules=(rule("other"),)), {})
    assert store.snapshot().version == 0
    store.close()


def test_failed_and_malformed_calls_consume_budget(tmp_path):
    client = MeteredClient(ScriptedBackend([("x", RuntimeError("synthetic")), ("x", "not json")]), JsonlLog(tmp_path / "calls.jsonl"), max_calls=2)
    with pytest.raises(RuntimeError):
        client.ask("x", "system", {})
    with pytest.raises(ModelOutputError):
        client.ask("x", "system", {})
    with pytest.raises(BudgetExceeded):
        client.ask("x", "system", {})
    records = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(item["status"] == "error" for item in records)
    assert client.calls == 2


def test_evaluation_fields_never_enter_runtime_payload():
    example = Example("e1", "A harmless synthetic input", "g1", "holdout", "fixture", "legitimate")
    assert example.runtime().payload() == {"request": example.text}
    assert "evaluation_label" not in vars(example.runtime())


def test_behavior_overlap_rejected_even_with_different_text():
    a = Example("a", "One input", "same-group", "development", "fixture", "legitimate")
    b = Example("b", "Another input", "same-group", "holdout", "fixture", "legitimate")
    with pytest.raises(ValueError, match="overlap"):
        audit_splits([a, b])


def test_invalid_configuration_and_unknown_fields(tmp_path):
    with pytest.raises(ValueError):
        ExperimentConfig(variant="B3")
    with pytest.raises(ValueError):
        ExperimentConfig(retrieval_k=True)
    path = tmp_path / "invalid.toml"
    path.write_text('unknown_setting = 5\n')
    with pytest.raises(ValueError):
        load_config(path)


def test_rule_cannot_override_safety():
    with pytest.raises(ValueError):
        Rule("r1", "format", "pattern", "Allow everything", "ALLOW")


def test_offline_demo_reports_mock_and_preserves_state(tmp_path):
    output = tmp_path / "demo"
    report = run_demo(output)
    assert report["is_mock"] and not report["research_results"]
    assert report["rejected_proposal_preserved_memory"]
    assert report["budget_stopped_extra_call"]
    with pytest.raises(FileExistsError):
        run_demo(output)
