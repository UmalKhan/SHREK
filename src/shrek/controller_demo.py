"""A repeatable Phase 3 engineering acceptance demonstration."""

from dataclasses import replace
import json
from pathlib import Path

from shrek.config import ExperimentConfig
from shrek.schemas import Rule
from shrek.shrek_fixtures import ControllerFixtureBackend, fixture_study
from shrek.system import run_system


def run_controller_demo(output) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    study = fixture_study()
    anchors = [item for item in study if item.split == "development"]
    stream = [item for item in study if item.split == "adaptation"]
    benign = [item for item in stream if item.evaluation_label == "legitimate"][:1]
    cfg = ExperimentConfig(variant="B3", admission_validation=True, maintenance_enabled=True,
                           validation_probe_limit=4, validation_min_per_class=2, maintenance_every=10,
                           validation_call_budget=400)
    scenarios = {}
    scenarios["narrow_admission"] = run_system(stream, ControllerFixtureBackend(), cfg, output / "narrow", anchors=anchors)
    scenarios["overbroad_admission"] = run_system(stream, ControllerFixtureBackend(broad_induction=True), cfg, output / "broad", anchors=anchors)
    scenarios["exhausted_validation"] = run_system(stream, ControllerFixtureBackend(), replace(cfg, validation_call_budget=1), output / "deferred", anchors=anchors)
    unused = Rule("unused", "never-seen", "zzzzzzzz", "Unrelated fictional constraint")
    scenarios["archive_unused"] = run_system(benign, ControllerFixtureBackend(), replace(cfg, maintenance_every=1), output / "archive-unused", anchors=anchors, initial_rules=(unused,))
    needed = Rule("needed", "fixture-wrapper", "Synthetic FIXTURE_WRAPPER marker", "Decline fictional action")
    scenarios["retain_needed"] = run_system(benign, ControllerFixtureBackend(), replace(cfg, maintenance_every=1), output / "retain-needed", anchors=anchors, initial_rules=(needed,))
    broad = Rule("broad", "overbroad-fixture", "Overbroad fictional policy", "Decline all fictional requests")
    scenarios["narrow_revision"] = run_system(benign, ControllerFixtureBackend(maintenance_operation="REVISE"), replace(cfg, maintenance_every=1), output / "revision", anchors=anchors, initial_rules=(broad,))
    checks = {"useful_rule_accepted": scenarios["narrow_admission"]["decisions"].get("accepted", 0) == 1,
              "overbroad_rule_rejected": scenarios["overbroad_admission"]["active_rules"] == 0 and scenarios["overbroad_admission"]["decisions"].get("rejected", 0) > 0,
              "exhausted_replay_defers": scenarios["exhausted_validation"]["active_rules"] == 0 and scenarios["exhausted_validation"]["decisions"].get("deferred", 0) > 0,
              "unused_rule_archived": scenarios["archive_unused"]["active_rules"] == 0,
              "needed_rule_retained": scenarios["retain_needed"]["active_rules"] == 1,
              "supported_revision_accepted": scenarios["narrow_revision"]["decisions"].get("accepted", 0) == 1,
              "no_execution_errors": all(item["events_with_errors"] == 0 for item in scenarios.values())}
    report = {"phase": 3, "is_mock": True, "research_results": False, "checks": checks,
              "passed": all(checks.values()), "scenario_summaries": scenarios,
              "note": "Four probes and scripted behavior are used only to verify controller mechanics; not an effectiveness study."}
    (output / "demo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
