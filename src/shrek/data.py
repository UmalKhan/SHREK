"""Strict dataset boundaries and behavior-level leakage checks."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from shrek.schemas import RuntimeRequest, identifier, nonempty


@dataclass(frozen=True)
class Example:
    request_id: str
    text: str
    behavior_group: str
    split: str
    source: str
    evaluation_label: str

    def __post_init__(self) -> None:
        self.runtime()
        identifier(self.behavior_group, "behavior_group")
        nonempty(self.source, "source")
        if self.split not in {"development", "adaptation", "holdout"}:
            raise ValueError("Invalid dataset split")
        if self.evaluation_label not in {"harmful", "legitimate"}:
            raise ValueError("Invalid evaluation label")

    def runtime(self) -> RuntimeRequest:
        return RuntimeRequest(self.request_id, self.text)


def load_examples(path: str | Path) -> list[Example]:
    examples = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            examples.append(Example(**json.loads(line)))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid dataset record on line {index}") from error
    if not examples:
        raise ValueError("Dataset is empty")
    if len({item.request_id for item in examples}) != len(examples):
        raise ValueError("Duplicate request IDs")
    return examples


def audit_splits(examples: list[Example]) -> dict:
    groups, texts = {}, {}
    counts = {}
    for item in examples:
        normalized = " ".join(item.text.casefold().split())
        for lookup, key in ((groups, item.behavior_group), (texts, normalized)):
            if key in lookup and lookup[key] != item.split:
                raise ValueError("Cross-split behavior or exact-text overlap")
            lookup[key] = item.split
        key = f"{item.split}:{item.evaluation_label}"
        counts[key] = counts.get(key, 0) + 1
    return {"records": len(examples), "behavior_groups": len(groups), "counts": counts,
            "limitation": "Explicit group IDs and exact text checked; paraphrases require source/manual review."}


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
