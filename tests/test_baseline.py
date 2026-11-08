import json

import pytest

from shrek.baseline import BaselineDefender, Feedback
from shrek.config import ExperimentConfig
from shrek.data import Example
from shrek.fixtures import BaselineFixtureBackend, baseline_examples
from shrek.memory import MemoryStore
from shrek.model_client import JsonlLog, MeteredClient, ModelOutputError, ScriptedBackend
from shrek.runner import run_baseline
from shrek.schemas import Rule, RuntimeRequest


def client(tmp_path, steps):
    return MeteredClient(ScriptedBackend(steps), JsonlLog(tmp_path / "calls.jsonl"), max_calls=100)


def populate(store, rules):
    return store.commit(store.propose(store.snapshot(), "ADD", tuple(rules), "Test setup"), {"fixture": True})


def test_failure_learn_reuse_and_benign_bypass(tmp_path):
    output = tmp_path / "baseline"
    report = run_baseline(baseline_examples(), BaselineFixtureBackend(), ExperimentConfig(), output)
    events = [json.loads(line) for line in (output / "interactions.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    assert report["events_with_errors"] == 0 and report["accepted_updates"] == 1
    assert report["is_mock"] and not report["research_results"]
    assert events[0]["response"]["memory_version"] == 0
    assert events[0]["response"]["response"] == "[SIMULATED_UNSAFE_RESPONSE]"
    assert events[0]["memory_version_after"] == 1
    assert events[1]["response"]["policy"] == "HARD_REFUSE"
    assert len(events[1]["response"]["retrieved_rule_ids"]) == 1
    assert events[2]["response"]["retrieved_rule_ids"] == []
    assert events[2]["memory_version_after"] == 1
    assert all("evaluation_label" not in json.dumps(call["payload"]) for call in calls)


def test_no_learning_means_no_judging_or_writes(tmp_path):
    report = run_baseline(baseline_examples(), BaselineFixtureBackend(), ExperimentConfig(), tmp_path / "frozen", learning=False)
    assert report["accepted_updates"] == 0 and report["final_memory_version"] == 0
    assert report["usage"]["calls"] == 6  # classifier + host for each request


def test_budget_exhaustion_keeps_original_answer(tmp_path):
    output = tmp_path / "budget"
    report = run_baseline(baseline_examples(), BaselineFixtureBackend(), ExperimentConfig(), output, max_calls=2)
    assert report["answered"] == 1 and report["budget_exhausted"]
    assert report["final_memory_version"] == 0
    answers = [json.loads(line) for line in (output / "answers.jsonl").read_text().splitlines()]
    assert answers[0]["response"] == "[SIMULATED_UNSAFE_RESPONSE]"


def test_invalid_classifier_never_fabricates_answer(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    defender = BaselineDefender(client(tmp_path, [("classify", {"label": "invented"})]), ExperimentConfig())
    with pytest.raises(ModelOutputError):
        defender.respond(RuntimeRequest("a", "Synthetic request"), store.snapshot())
    assert store.snapshot().version == 0
    store.close()


def test_malformed_judge_causes_no_memory_change(tmp_path):
    backend = ScriptedBackend([("classify", {"label": "other"}), ("host", "An answer"), ("judge", {"score": True, "refused": False, "rationale": "bad"})])
    report = run_baseline(baseline_examples()[:1], backend, ExperimentConfig(), tmp_path / "invalid")
    assert report["events_with_errors"] == 1 and report["answered"] == 1
    assert report["final_memory_version"] == 0


def test_label_then_semantic_then_lexical_fallback(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    rule = Rule("r1", "known-label", "fictional wrapper", "Decline fictional action")
    snapshot = populate(store, [rule])
    steps = [("classify", {"label": "other"}), ("retrieve", {"rule_ids": []}),
             ("policy", {"policy": "ALLOW"}), ("host", "Safe fixture refusal")]
    defender = BaselineDefender(client(tmp_path, steps), ExperimentConfig())
    result = defender.respond(RuntimeRequest("a", "fictional wrapper for a test"), snapshot)
    assert result.retrieved_rule_ids == ("r1",) and result.policy == "HARD_REFUSE"
    store.close()


def test_lexical_fallback_ignores_function_words(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    rule = Rule("r1", "known-label", "describing a method to clone cards and exploit a system", "Decline card cloning", "HARD_REFUSE")
    snapshot = populate(store, [rule])
    steps = [("classify", {"label": "other"}), ("retrieve", {"rule_ids": []}), ("host", "Add yogurt to reduce heat")]
    defender = BaselineDefender(client(tmp_path, steps), ExperimentConfig())
    result = defender.respond(RuntimeRequest("a", "How do I overpower the spiciness in a dish?"), snapshot)
    assert result.retrieved_rule_ids == () and result.policy == "ALLOW"
    store.close()


def test_unknown_retrieval_ids_rejected(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    snapshot = populate(store, [Rule("r1", "known", "pattern", "constraint")])
    defender = BaselineDefender(client(tmp_path, [("classify", {"label": "other"}), ("retrieve", {"rule_ids": ["invented"]})]), ExperimentConfig())
    with pytest.raises(ModelOutputError):
        defender.respond(RuntimeRequest("a", "Synthetic request"), snapshot)
    store.close()


def test_per_label_capacity_and_atomic_eviction(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    original = Rule("old", "known", "old pattern", "old restriction")
    snapshot = populate(store, [original])
    steps = [("classify", {"label": "known"}), ("policy", {"policy": "HARD_REFUSE"}), ("host", "Fixture response"),
             ("induce", {"learn": True, "label": "known", "pattern": "new pattern", "text": "new restriction"}),
             ("deduplicate", {"duplicate_rule_id": None})]
    defender = BaselineDefender(client(tmp_path, steps), ExperimentConfig(per_label_capacity=1))
    request = RuntimeRequest("a", "Synthetic request")
    result = defender.respond(request, snapshot)
    update = defender.learn(request, result, Feedback(9, False, "fixture"), store)
    assert update["evicted_rule_ids"] == ["old"]
    assert len(store.snapshot().rules) == 1 and store.snapshot().rules[0].pattern == "new pattern"
    assert store.snapshot(snapshot.version).rules == (original,)
    store.close()


def test_semantic_duplicate_does_not_write(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    snapshot = populate(store, [Rule("old", "known", "pattern", "restriction")])
    steps = [("classify", {"label": "known"}), ("policy", {"policy": "HARD_REFUSE"}), ("host", "Fixture response"),
             ("induce", {"learn": True, "label": "known", "pattern": "pattern", "text": "restriction"}),
             ("deduplicate", {"duplicate_rule_id": "old"})]
    defender = BaselineDefender(client(tmp_path, steps), ExperimentConfig())
    request = RuntimeRequest("a", "Synthetic request")
    update = defender.learn(request, defender.respond(request, snapshot), Feedback(9, False, "fixture"), store)
    assert update["status"] == "duplicate" and store.snapshot() == snapshot
    store.close()


def test_holdout_cannot_learn_and_shrek_not_silently_baseline(tmp_path):
    example = Example("a", "Safe example", "g1", "holdout", "fixture", "legitimate")
    with pytest.raises(ValueError, match="Holdout"):
        run_baseline([example], BaselineFixtureBackend(), ExperimentConfig(), tmp_path / "bad")
    config = ExperimentConfig(variant="B3", admission_validation=True, maintenance_enabled=True)
    with pytest.raises(ValueError, match="Phase 2"):
        run_baseline(baseline_examples(), BaselineFixtureBackend(), config, tmp_path / "bad2")
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad2").exists()


def test_imported_memory_available_in_frozen_run(tmp_path):
    rule = Rule("r1", "fixture-wrapper", "Synthetic FIXTURE_WRAPPER marker", "Decline fictional action")
    report = run_baseline(baseline_examples()[:1], BaselineFixtureBackend(), ExperimentConfig(), tmp_path / "imported", learning=False, initial_rules=(rule,))
    assert report["active_rules"] == 1 and report["accepted_updates"] == 0
    answer = json.loads((tmp_path / "imported" / "answers.jsonl").read_text())
    assert answer["policy"] == "HARD_REFUSE"


def test_benign_classifier_never_induces_even_if_judge_flags_harm(tmp_path):
    steps = [("classify", {"label": "benign"}), ("host", "Fixture answer"),
             ("judge", {"score": 9, "refused": False, "rationale": "Contradictory fixture"})]
    report = run_baseline(baseline_examples()[:1], ScriptedBackend(steps), ExperimentConfig(), tmp_path / "benign-gate")
    assert report["accepted_updates"] == 0 and report["usage"]["calls"] == 3


def test_unknown_judgment_is_not_safe_or_failure(tmp_path):
    steps = [("classify", {"label": "other"}), ("host", "Fixture answer"),
             ("judge", {"score": None, "refused": None, "rationale": "Uncertain fixture"})]
    report = run_baseline(baseline_examples()[:1], ScriptedBackend(steps), ExperimentConfig(), tmp_path / "uncertain")
    assert report["accepted_updates"] == 0
    event = json.loads((tmp_path / "uncertain" / "interactions.jsonl").read_text())
    assert event["feedback"]["score"] is None


def test_controlled_capacity_defers_oversize_candidate_without_partial_eviction(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    snapshot = populate(store, [Rule("old", "known", "old pattern", "old restriction")])
    steps = [("classify", {"label": "known"}), ("policy", {"policy": "HARD_REFUSE"}), ("host", "Fixture answer"),
             ("induce", {"learn": True, "label": "known", "pattern": "new pattern", "text": "new restriction"}),
             ("deduplicate", {"duplicate_rule_id": None})]
    config = ExperimentConfig(capacity_mode="controlled", active_rule_token_limit=1)
    defender = BaselineDefender(client(tmp_path, steps), config)
    request = RuntimeRequest("a", "Synthetic request")
    update = defender.learn(request, defender.respond(request, snapshot), Feedback(9, False, "fixture"), store, token_counter=lambda text: len(text))
    assert update["status"] == "deferred" and store.snapshot() == snapshot
    store.close()


def test_truncated_host_answer_preserved_but_never_learns(tmp_path):
    from shrek.model_client import Completion
    class TruncatedBackend:
        model_id = "truncated-host-fixture"
        is_mock = True
        def complete(self, role, system, payload, max_output_tokens):
            if role == "classify":
                return Completion('{"label":"other"}', finish_reason="stop")
            if role == "host":
                return Completion("Incomplete synthetic response", finish_reason="length")
            raise AssertionError("An incomplete host answer must not trigger feedback/learning")
    output = tmp_path / "truncated"
    report = run_baseline(baseline_examples()[:1], TruncatedBackend(), ExperimentConfig(), output)
    assert report["answered"] == 1 and report["truncated_answers"] == 1
    assert report["events_with_errors"] == 1 and report["accepted_updates"] == 0
    assert report["usage"]["calls"] == 2
    assert json.loads((output / "answers.jsonl").read_text())["response_truncated"] is True
