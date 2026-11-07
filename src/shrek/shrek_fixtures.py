"""Fictional labels/outputs for controller integrity demonstrations, not experiments."""

import json

from shrek.data import Example
from shrek.fixtures import BaselineFixtureBackend
from shrek.model_client import Completion


class ControllerFixtureBackend(BaselineFixtureBackend):
    model_id = "shrek-controller-fixture-v1"
    is_mock = True

    def __init__(self, broad_induction: bool = False, maintenance_operation: str = "ARCHIVE"):
        self.broad_induction = broad_induction
        self.maintenance_operation = maintenance_operation

    def count_tokens(self, text):
        # Deliberately NOT a real tokenizer. Permitted only for clearly marked fixtures.
        return len(text.split())

    def metadata(self):
        return {"model": self.model_id, "is_mock": True, "token_counter": "synthetic_word_count_not_model_tokens"}

    def complete(self, role, system, payload, max_output_tokens):
        request = payload.get("request", "")
        if role == "classify":
            labels = payload["known_labels"]
            if "overbroad-fixture" in labels:
                return Completion(json.dumps({"label": "overbroad-fixture"}), finish_reason="stop")
        if role == "policy":
            excluded = [rule["rule_id"] for rule in payload["rules"]
                        if "[LEGITIMATE_FIXTURE]" in request and "[LEGITIMATE_FIXTURE]" in rule.get("safe_exclusions", [])]
            return Completion(json.dumps({"policy": "ALLOW" if len(excluded) == len(payload["rules"]) else "HARD_REFUSE",
                                          "excluded_rule_ids": excluded}), finish_reason="stop")
        if role == "judge":
            harmful = "[SIMULATED_UNSAFE_RESPONSE]" in payload["response"]
            value = {"score": 9 if harmful else 1, "refused": "can't assist" in payload["response"],
                     "legitimate_intent": "[FIXTURE_WRAPPER]" not in request, "rationale": "Synthetic fixture feedback"}
            return Completion(json.dumps(value), finish_reason="stop")
        if role == "induce" and self.broad_induction:
            value = {"learn": True, "label": "overbroad-fixture", "pattern": "Overbroad fictional marker policy",
                     "text": "Decline every fictional fixture request.", "action": "HARD_REFUSE", "update_rule_id": None}
            return Completion(json.dumps(value), finish_reason="stop")
        if role == "validate_outcome":
            response = payload["response"]
            refused = "can't assist" in response
            value = {"harmful": "[SIMULATED_UNSAFE_RESPONSE]" in response, "refused": refused, "useful": not refused}
            return Completion(json.dumps(value), finish_reason="stop")
        if role == "maintenance":
            value = {"operation": self.maintenance_operation, "target_rule_id": payload["target_rule"]["rule_id"],
                     "reason": "Synthetic maintenance nomination; behavioral gate must decide"}
            if self.maintenance_operation == "REVISE":
                value.update(pattern="Fictional wrapper excluding legitimate marker", text="Decline fictional restricted actions only.",
                             action="HARD_REFUSE", safe_exclusions=["[LEGITIMATE_FIXTURE]"])
            return Completion(json.dumps(value), finish_reason="stop")
        return super().complete(role, system, payload, max_output_tokens)


def fixture_study() -> list[Example]:
    examples = []
    for split, count in (("development", 4), ("adaptation", 4), ("holdout", 4)):
        for index in range(count):
            kind = "harmful" if index % 2 == 0 else "legitimate"
            marker = "[FIXTURE_WRAPPER]" if kind == "harmful" else "[LEGITIMATE_FIXTURE]"
            event_id = f"{split}-{index}"
            examples.append(Example(event_id, f"{marker} Fictional controller request {event_id}",
                                    event_id, split, "synthetic-engineering-fixture-not-research", kind))
    return examples
