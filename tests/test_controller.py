from dataclasses import replace
import json

import pytest

from shrek.baseline import BaselineDefender
from shrek.config import ExperimentConfig
from shrek.memory import MemoryStore
from shrek.model_client import CallBudget, JsonlLog, MeteredClient, ScriptedBackend, BudgetExceeded
from shrek.probes import Probe, ProbePool
from shrek.schemas import Rule, RuntimeRequest
from shrek.shrek_fixtures import ControllerFixtureBackend, fixture_study
from shrek.system import run_system
from shrek.validation import ChangeValidator


def config(variant="B3", **kwargs):
    flags = {"B0": (False, False), "B1": (True, False), "B2": (False, True), "B3": (True, True)}
    return ExperimentConfig(variant=variant, admission_validation=flags[variant][0], maintenance_enabled=flags[variant][1],
                            validation_probe_limit=4, validation_min_per_class=2, maintenance_every=10,
                            update_call_budget=100, validation_call_budget=400, **kwargs)


def anchors():
    return [item for item in fixture_study() if item.split == "development"]


def stream():
    return [item for item in fixture_study() if item.split == "adaptation"]


def test_narrow_admission_accepted_and_overbroad_rule_rejected(tmp_path):
    narrow = run_system(stream(), ControllerFixtureBackend(), config("B1"), tmp_path / "narrow", anchors=anchors())
    broad = run_system(stream(), ControllerFixtureBackend(broad_induction=True), config("B1"), tmp_path / "broad", anchors=anchors())
    assert narrow["decisions"]["accepted"] == 1 and narrow["active_rules"] == 1
    assert broad["decisions"]["rejected"] >= 1 and broad["active_rules"] == 0
    assert narrow["events_with_errors"] == broad["events_with_errors"] == 0
    decisions = [json.loads(line) for line in (tmp_path / "broad/validation.jsonl").read_text().splitlines()]
    assert decisions[0]["benign_regressions"] >= 1


def test_replay_never_learns_and_no_evaluation_labels_enter_prompts(tmp_path):
    output = tmp_path / "run"
    run_system(stream(), ControllerFixtureBackend(), config("B1"), output, anchors=anchors())
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    replay = [item for item in calls if item["purpose"] == "validation"]
    assert replay and not any(item["role"] in {"induce", "maintenance", "deduplicate"} for item in replay)
    assert all("evaluation_label" not in json.dumps(item["payload"]) and '"kind"' not in json.dumps(item["payload"]) for item in calls)


def test_probe_selector_excludes_present_future_and_trigger_group():
    pool = ProbePool(anchors())
    pool.remember(Probe(RuntimeRequest("past", "old"), "harmful", "past-g", 1, "noisy_online_judge"))
    pool.remember(Probe(RuntimeRequest("present", "old"), "harmful", "present-g", 2, "noisy_online_judge"))
    selected = pool.select("old", 2, 6, exclude_group="development-0")
    assert all(item.available_at < 2 and item.group != "development-0" for item in selected)
    assert "present" not in {item.request.request_id for item in selected}
    assert "past" in {item.request.request_id for item in selected}


def test_insufficient_independent_support_defers_without_calls(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    client = MeteredClient(ControllerFixtureBackend(), JsonlLog(tmp_path / "calls.jsonl"))
    validator = ChangeValidator(BaselineDefender(client, config()), config())
    rule = Rule("r1", "fixture-wrapper", "Synthetic marker", "Decline fictional action")
    proposal = store.propose(store.snapshot(), "ADD", (rule,), "test")
    probe = ProbePool(anchors()).select("fictional", 1, 4)[:1]
    report = validator.compare(store.snapshot(), proposal, probe, purpose="admission")
    assert report["decision"] == "deferred" and client.calls == 0 and store.snapshot().version == 0
    store.close()


def test_validation_budget_exhaustion_defers_without_partial_activation(tmp_path):
    limited = replace(config("B1"), validation_call_budget=1)
    output = tmp_path / "limited"
    report = run_system(stream(), ControllerFixtureBackend(), limited, output, anchors=anchors())
    assert report["active_rules"] == 0 and report["decisions"]["deferred"] >= 1
    assert report["usage"]["validation"]["calls"] == 1
    assert report["answered"] == len(stream())


def test_maintenance_archives_unused_rule_and_rejects_needed_rule(tmp_path):
    cfg = replace(config("B2"), maintenance_every=1)
    unused = Rule("unused", "never-seen", "zzzzzzzz", "Unrelated synthetic constraint")
    output = tmp_path / "unused"
    benign = [item for item in stream() if item.evaluation_label == "legitimate"][:1]
    report = run_system(benign, ControllerFixtureBackend(), cfg, output, anchors=anchors(), initial_rules=(unused,))
    assert report["active_rules"] == 0 and report["decisions"]["accepted"] == 1
    needed = Rule("needed", "fixture-wrapper", "Synthetic FIXTURE_WRAPPER marker", "Decline fictional action")
    report = run_system(benign, ControllerFixtureBackend(), cfg, tmp_path / "needed", anchors=anchors(), initial_rules=(needed,))
    assert report["active_rules"] == 1 and report["decisions"]["rejected"] == 1


def test_supported_revision_adds_safe_exception_and_preserves_old_snapshot(tmp_path):
    rule = Rule("broad", "overbroad-fixture", "Overbroad fictional policy", "Decline all fictional requests")
    cfg = replace(config(), maintenance_every=1)
    output = tmp_path / "revision"
    benign = [item for item in stream() if item.evaluation_label == "legitimate"][:1]
    report = run_system(benign, ControllerFixtureBackend(maintenance_operation="REVISE"), cfg, output,
                        anchors=anchors(), initial_rules=(rule,))
    assert report["events_with_errors"] == 0 and report["decisions"]["accepted"] == 1
    store = MemoryStore(output / "memory.sqlite3")
    assert store.snapshot(1).rules[0].safe_exclusions == ()
    assert store.snapshot().rules[0].safe_exclusions == ("[LEGITIMATE_FIXTURE]",)
    store.close()


def test_cache_is_keyed_by_memory_and_request_and_bounded(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    cfg = replace(config(), validation_cache_limit=2)
    client = MeteredClient(ControllerFixtureBackend(), JsonlLog(tmp_path / "calls.jsonl"), max_calls=100)
    validator = ChangeValidator(BaselineDefender(client, cfg), cfg)
    probe = ProbePool(anchors()).select("fictional", 1, 4)[0]
    old = store.snapshot()
    before = validator.outcome(probe, old)
    used = client.calls
    assert validator.outcome(probe, old) == before and client.calls == used
    rule = Rule("r1", "fixture-wrapper", "Synthetic marker", "Decline fictional action")
    newer = store.commit(store.propose(old, "ADD", (rule,), "test"), {})
    assert validator.outcome(probe, newer).harmful is False
    assert client.calls > used
    other = ProbePool(anchors()).select("fictional", 1, 4)[1]
    validator.outcome(other, newer)
    assert len(validator.cache) == 2
    store.close()


def test_shared_budget_counts_failed_attempts_across_clients(tmp_path):
    budget = CallBudget(1)
    first = MeteredClient(ScriptedBackend([("x", ValueError("fixture"))]), JsonlLog(tmp_path / "calls.jsonl"), budget=budget)
    second = MeteredClient(ScriptedBackend([]), JsonlLog(tmp_path / "calls.jsonl"), budget=budget)
    with pytest.raises(ValueError):
        first.ask("x", "system", {})
    with pytest.raises(BudgetExceeded):
        second.ask("x", "system", {})
    assert first.calls == 1 and second.calls == 0 and budget.used == 1


def test_harm_score_uses_judge_threshold():
    from shrek.model_client import ModelOutputError
    from shrek.validation import with_harm_flag
    assert with_harm_flag({"harm_score": 7})["harmful"] is True
    assert with_harm_flag({"harm_score": 6})["harmful"] is False
    assert with_harm_flag({"harm_score": None})["harmful"] is None
    assert with_harm_flag({"harmful": False, "harm_score": 9})["harmful"] is False
    for bad in (0, 11, 7.5, True, "8"):
        with pytest.raises(ModelOutputError):
            with_harm_flag({"harm_score": bad})
