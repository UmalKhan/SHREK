"""Strict TOML configuration, shared by offline and real-model execution."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    variant: str = "B0"
    admission_validation: bool = False
    maintenance_enabled: bool = False
    capacity_mode: str = "paper"
    retrieval_k: int = 2
    per_label_capacity: int = 4
    active_rule_limit: int = 20
    active_rule_token_limit: int = 2000
    max_proposals_per_interaction: int = 1
    validation_probe_limit: int = 12
    max_new_harmful_regressions: int = 0
    max_new_benign_regressions: int = 0
    maintenance_every: int = 10
    maintenance_candidates: int = 1
    validation_min_per_class: int = 4
    validation_call_budget: int = 240
    update_call_budget: int = 120
    replay_limit: int = 100
    validation_cache_limit: int = 256
    max_admission_candidates: int = 4
    max_maintenance_candidates: int = 4

    def __post_init__(self) -> None:
        if self.variant not in {"B0", "B1", "B2", "B3"}:
            raise ValueError("Unknown experimental variant")
        expected = {"B0": (False, False), "B1": (True, False), "B2": (False, True), "B3": (True, True)}
        if (self.admission_validation, self.maintenance_enabled) != expected[self.variant]:
            raise ValueError("Variant flags disagree with variant name")
        if type(self.admission_validation) is not bool or type(self.maintenance_enabled) is not bool:
            raise ValueError("Feature flags must be booleans")
        if self.capacity_mode not in {"paper", "controlled"}:
            raise ValueError("capacity_mode must be paper or controlled")
        for name, value in asdict(self).items():
            if name.startswith("max_new_"):
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a nonnegative integer")
            elif name not in {"variant", "admission_validation", "maintenance_enabled", "capacity_mode"}:
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
        if self.validation_probe_limit % 2 or self.validation_probe_limit < 2 * self.validation_min_per_class:
            raise ValueError("Validation probe count must be even and support both minimum class counts")

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    validation = raw.pop("validation", {})
    maintenance = raw.pop("maintenance", {})
    mapping = {"probe_limit": "validation_probe_limit", "min_per_class": "validation_min_per_class", "call_budget": "validation_call_budget", "cache_limit": "validation_cache_limit", "max_new_harmful_regressions": "max_new_harmful_regressions", "max_new_benign_regressions": "max_new_benign_regressions"}
    other = {"every_interactions": "maintenance_every", "candidates_per_interval": "maintenance_candidates", "max_candidates": "max_maintenance_candidates"}
    for section, names in ((validation, mapping), (maintenance, other)):
        if not isinstance(section, dict) or set(section) - set(names):
            raise ValueError("Unknown configuration section fields")
        raw.update({names[key]: value for key, value in section.items()})
    if set(raw) - set(ExperimentConfig.__dataclass_fields__):
        raise ValueError("Unknown configuration fields")
    return ExperimentConfig(**raw)
