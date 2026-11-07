"""Balanced development anchors and strictly historical, explicitly noisy replay."""

from dataclasses import dataclass
import re

from shrek.data import Example
from shrek.schemas import RuntimeRequest


@dataclass(frozen=True)
class Probe:
    request: RuntimeRequest
    kind: str
    group: str
    available_at: int
    label_source: str

    def __post_init__(self):
        if self.kind not in {"harmful", "legitimate"}:
            raise ValueError("Invalid probe kind")
        if type(self.available_at) is not int or self.available_at < 0:
            raise ValueError("Invalid probe availability")


class ProbePool:
    def __init__(self, anchors: list[Example], replay_limit: int = 100):
        if replay_limit < 1:
            raise ValueError("Replay limit must be positive")
        if any(example.split != "development" for example in anchors):
            raise ValueError("Only development labels can initialize runtime anchors")
        if len({item.request_id for item in anchors}) != len(anchors):
            raise ValueError("Duplicate anchor IDs")
        self.anchors = tuple(Probe(item.runtime(), item.evaluation_label, item.behavior_group, 0,
                                  "benchmark_development_anchor_not_claimed_human_reviewed") for item in anchors)
        self.history: list[Probe] = []
        self.replay_limit = replay_limit

    def remember(self, probe: Probe) -> None:
        if probe.available_at < 1 or probe.label_source != "noisy_online_judge":
            raise ValueError("Replay entries require an actual past interaction and noisy label provenance")
        if probe.request.request_id in {p.request.request_id for p in (*self.anchors, *self.history)}:
            raise ValueError("Duplicate replay/anchor request ID")
        self.history.append(probe)
        self.history = self.history[-self.replay_limit:]

    def select(self, query: str, sequence: int, limit: int, exclude_group: str | None = None) -> list[Probe]:
        if limit < 2 or limit % 2:
            raise ValueError("Probe limit must be positive and balanced")
        available = [p for p in (*self.anchors, *self.history) if p.available_at < sequence and p.group != exclude_group]
        terms = set(re.findall(r"\w+", query.casefold()))
        selected = []
        groups = set()
        for kind in ("harmful", "legitimate"):
            candidates = [p for p in available if p.kind == kind]
            per_class = limit // 2
            old_quota = max(1, per_class // 3)
            score = lambda p: len(terms & set(re.findall(r"\w+", p.request.text.casefold())))
            ranked = sorted(candidates, key=lambda p: (-score(p), p.available_at, p.request.request_id))
            chosen = []
            for probe in ranked:
                if probe.group not in groups and len(chosen) < per_class - old_quota:
                    chosen.append(probe)
                    groups.add(probe.group)
            for probe in sorted(candidates, key=lambda p: (p.available_at, p.request.request_id)):
                if probe.group not in groups and len(chosen) < per_class:
                    chosen.append(probe)
                    groups.add(probe.group)
            selected.extend(chosen)
        return selected
