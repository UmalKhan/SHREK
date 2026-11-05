"""Immutable runtime records. Evaluation labels deliberately live elsewhere."""

from dataclasses import asdict, dataclass
import re
from typing import Any


def nonempty(value: Any, name: str, limit: int = 12000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def identifier(value: Any, name: str) -> str:
    value = nonempty(value, name, 100)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError(f"Invalid {name}")
    return value


@dataclass(frozen=True)
class RuntimeRequest:
    request_id: str
    text: str

    def __post_init__(self) -> None:
        identifier(self.request_id, "request_id")
        nonempty(self.text, "request text", 40000)

    def payload(self) -> dict:
        return {"request": self.text}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    label: str
    pattern: str
    text: str
    action: str = "HARD_REFUSE"
    source_event_ids: tuple[str, ...] = ()
    safe_exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identifier(self.rule_id, "rule_id")
        identifier(self.label, "label")
        if self.label in {"benign", "other"}:
            raise ValueError("Stored rules need a specific attack label")
        nonempty(self.pattern, "pattern", 2000)
        nonempty(self.text, "rule text", 4000)
        if self.action not in {"HARD_REFUSE", "SOFT_REFUSE"}:
            raise ValueError("A rule cannot override host safety with an ALLOW action")
        for value in self.source_event_ids:
            identifier(value, "source event")
        for value in self.safe_exclusions:
            nonempty(value, "safe exclusion", 2000)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Rule":
        if not isinstance(value, dict):
            raise ValueError("Rule must be an object")
        fields = {"rule_id", "label", "pattern", "text", "action", "source_event_ids", "safe_exclusions"}
        if set(value) - fields:
            raise ValueError("Unknown rule fields")
        data = dict(value)
        for name in ("source_event_ids", "safe_exclusions"):
            items = data.get(name, ())
            if not isinstance(items, (list, tuple)):
                raise ValueError(f"{name} must be a list")
            data[name] = tuple(items)
        return cls(**data)


@dataclass(frozen=True)
class Snapshot:
    version: int
    rules: tuple[Rule, ...]
    digest: str


@dataclass(frozen=True)
class Proposal:
    change_id: str
    operation: str
    base_version: int
    candidate_rules: tuple[Rule, ...]
    reason: str
