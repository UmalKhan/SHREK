"""SHREK admission, bounded historical feedback, and tested maintenance."""

from collections import Counter
from dataclasses import replace

from shrek.baseline import BaselineDefender, DefenseResult, Feedback, prompt
from shrek.config import ExperimentConfig
from shrek.memory import MemoryStore, StaleProposalError
from shrek.model_client import JsonlLog, ModelOutputError
from shrek.probes import Probe, ProbePool
from shrek.schemas import Proposal, Rule, RuntimeRequest, nonempty
from shrek.validation import ChangeValidator


class ShrekController:
    def __init__(self, store: MemoryStore, learning_defender: BaselineDefender, validator: ChangeValidator,
                 pool: ProbePool, config: ExperimentConfig, audit: JsonlLog, token_counter=None):
        self.store, self.learning, self.validator, self.pool, self.config, self.audit = store, learning_defender, validator, pool, config, audit
        self.token_counter = token_counter
        self.admission_attempts = self.maintenance_attempts = 0
        self.decisions = Counter()
        self.retrieved = Counter()
        self.false_refusals: dict[str, Probe] = {}
        self.last_maintenance_sequence = 0

    def _activate(self, change: Proposal, report: dict, metadata: dict) -> dict:
        if report["decision"] == "accepted":
            try:
                snapshot = self.store.commit(change, {**report, **metadata})
                result = {"status": "accepted", "new_memory_version": snapshot.version, **metadata, "change_id": change.change_id}
            except StaleProposalError:
                report.update(decision="deferred", reason="Memory changed during validation")
                self.store.resolve(change, "deferred", report)
                result = {"status": "deferred", "reason": report["reason"], "change_id": change.change_id}
        else:
            self.store.resolve(change, report["decision"], report)
            result = {"status": report["decision"], "reason": report["reason"], "change_id": change.change_id}
        self.decisions[result["status"]] += 1
        self.audit.append({**report, "activation": result, "is_mock": self.learning.client.backend.is_mock})
        return result

    def learn(self, request: RuntimeRequest, result: DefenseResult, feedback: Feedback, sequence: int, group: str) -> dict:
        if not feedback.violation or result.attack_label == "benign" or result.response_truncated:
            return {"status": "skipped", "reason": "No eligible, complete baseline failure"}
        if self.admission_attempts >= self.config.max_admission_candidates:
            return {"status": "deferred", "reason": "Admission attempt limit reached"}
        self.admission_attempts += 1
        trigger = Probe(request, "harmful", group, sequence, "noisy_online_judge")
        def admission(change, metadata):
            if not self.config.admission_validation:
                report = {"decision": "accepted", "reason": "Baseline immediate admission", "behavioral_validation": False}
            else:
                probes = self.pool.select(request.text, sequence, self.config.validation_probe_limit, exclude_group=group)
                report = self.validator.compare(self.store.snapshot(change.base_version), change, probes, purpose="admission", trigger=trigger)
                report["behavioral_validation"] = True
            return self._activate(change, report, metadata)
        return self.learning.learn(request, result, feedback, self.store, self.token_counter, admission=admission)

    def observe(self, request: RuntimeRequest, result: DefenseResult, feedback: Feedback | None, sequence: int, group: str) -> None:
        for rule_id in result.retrieved_rule_ids:
            self.retrieved[rule_id] += 1
        if feedback is None or feedback.score is None or result.response_truncated:
            return
        kind = "harmful" if feedback.violation else ("legitimate" if feedback.legitimate_intent is True else None)
        if kind is None:
            return
        probe = Probe(request, kind, group, sequence, "noisy_online_judge")
        self.pool.remember(probe)
        if kind == "legitimate" and feedback.refused is True:
            for rule_id in result.retrieved_rule_ids:
                self.false_refusals[rule_id] = probe

    def maintain(self, sequence: int) -> dict:
        if not self.config.maintenance_enabled or sequence % self.config.maintenance_every:
            return {"status": "not_due"}
        if sequence == self.last_maintenance_sequence:
            return {"status": "already_considered"}
        self.last_maintenance_sequence = sequence
        if self.maintenance_attempts >= self.config.max_maintenance_candidates:
            return {"status": "deferred", "reason": "Maintenance attempt limit reached"}
        snapshot = self.store.snapshot()
        if not snapshot.rules:
            return {"status": "skipped", "reason": "No active rules"}
        # At most one candidate per interval, even if a future configuration allows a larger ceiling.
        self.maintenance_attempts += 1
        signatures = Counter((rule.label, rule.pattern.casefold(), rule.text.casefold(), rule.action, rule.safe_exclusions) for rule in snapshot.rules)
        target = min(snapshot.rules, key=lambda rule: (rule.rule_id not in self.false_refusals,
                     signatures[(rule.label, rule.pattern.casefold(), rule.text.casefold(), rule.action, rule.safe_exclusions)] < 2,
                     self.retrieved[rule.rule_id], next(i for i, item in enumerate(snapshot.rules) if item.rule_id == rule.rule_id)))
        trigger = self.false_refusals.get(target.rule_id)
        payload = {"target_rule": target.to_dict(), "retrieval_count": self.retrieved[target.rule_id],
                   "other_rules": [rule.to_dict() for rule in snapshot.rules if rule.rule_id != target.rule_id],
                   "supported_false_refusal": trigger.request.text if trigger else None}
        data = self.learning.client.ask("maintenance", prompt("maintenance"), payload, event_id=f"maintenance:{sequence}")
        operation = data.get("operation")
        if operation == "SKIP":
            return {"status": "skipped", "reason": "Maintenance proposer found no justified change"}
        if operation not in {"REVISE", "ARCHIVE"} or data.get("target_rule_id") != target.rule_id:
            raise ModelOutputError("Maintenance must revise/archive the nominated existing rule")
        try:
            reason = nonempty(data.get("reason"), "maintenance reason", 2000)
            if operation == "ARCHIVE":
                candidate = tuple(rule for rule in snapshot.rules if rule.rule_id != target.rule_id)
            else:
                exclusions = data.get("safe_exclusions", list(target.safe_exclusions))
                if not isinstance(exclusions, list) or len(exclusions) > 3:
                    raise ValueError("Expected at most three safe exclusions")
                revised = replace(target, pattern=nonempty(data.get("pattern"), "pattern", 2000),
                                  text=nonempty(data.get("text"), "text", 4000), action=data.get("action", target.action),
                                  safe_exclusions=tuple(exclusions))
                candidate = tuple(revised if rule.rule_id == target.rule_id else rule for rule in snapshot.rules)
            change = self.store.propose(snapshot, operation, candidate, reason)
        except (TypeError, ValueError) as error:
            raise ModelOutputError("Invalid maintenance candidate") from error
        exclude_group = trigger.group if trigger else None
        probes = self.pool.select(target.pattern + " " + target.text, sequence, self.config.validation_probe_limit, exclude_group)
        report = self.validator.compare(snapshot, change, probes, purpose="maintenance", trigger=trigger if operation == "REVISE" else None)
        report["behavioral_validation"] = True
        return self._activate(change, report, {"target_rule_id": target.rule_id, "sequence": sequence})
