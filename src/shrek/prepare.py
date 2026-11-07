"""Reproducible small pilot data from published attacks and legitimate examples."""

from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
from urllib.request import Request, urlopen

from shrek.data import Example, audit_splits


ARTIFACT_REVISION = "909e68c01d94222b8ad2e397a017e2e12e2adb73"
XSTEST_REVISION = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
ARTIFACT_TARGET = "vicuna-13b-v1.5"
SOURCES = {
    "pair.json": f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{ARTIFACT_REVISION}/attack-artifacts/PAIR/black_box/{ARTIFACT_TARGET}.json",
    "xstest.csv": f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_REVISION}/xstest_prompts.csv",
    "artifacts-LICENSE": f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{ARTIFACT_REVISION}/LICENSE",
    "xstest-LICENSE": f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_REVISION}/LICENSE",
}


def group_hash(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()[:20]


def build_examples(artifact: dict, xstest_csv: str, seed=17, development_per_class=10,
                   adaptation_per_class=20, holdout_per_class=10) -> list[Example]:
    sizes = (development_per_class, adaptation_per_class, holdout_per_class)
    if any(type(value) is not int or value < 1 for value in sizes):
        raise ValueError("Each split needs a positive per-class size")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("jailbreaks"), list):
        raise ValueError("Unrecognized published artifact schema")
    harmful = {}
    for item in artifact["jailbreaks"]:
        goal, text = item.get("goal"), item.get("prompt")
        if isinstance(goal, str) and goal.strip() and isinstance(text, str) and text.strip():
            group = "pair-" + group_hash(goal)
            # Never select by the old model's jailbroken/success field.
            harmful.setdefault(group, (text, f"JailbreakBench/artifacts@{ARTIFACT_REVISION}:PAIR:{ARTIFACT_TARGET}:index={item.get('index')}"))
    legitimate = {}
    for item in csv.DictReader(io.StringIO(xstest_csv)):
        if item.get("label") != "safe" or not item.get("prompt", "").strip():
            continue
        # Conservative lexical focus grouping keeps repeated risky words together.
        focus = item.get("focus", "").strip() or item["prompt"]
        group = "xstest-" + group_hash(focus)
        legitimate.setdefault(group, (item["prompt"], f"paul-rottger/xstest@{XSTEST_REVISION}:id={item.get('id')}"))
    total = sum(sizes)
    examples = []
    for kind, groups in (("harmful", harmful), ("legitimate", legitimate)):
        candidates = sorted(groups.items())
        random.Random(f"{seed}:{kind}").shuffle(candidates)
        if len(candidates) < total:
            raise ValueError(f"Need {total} distinct {kind} groups, found only {len(candidates)}; do not fill with duplicates")
        offset = 0
        for split, count in zip(("development", "adaptation", "holdout"), sizes):
            for group, (text, source) in candidates[offset:offset+count]:
                examples.append(Example(group, text, group, split, source, kind))
            offset += count
    # Shuffle within each split without changing the split assignment.
    result = []
    for split in ("development", "adaptation", "holdout"):
        selected = [item for item in examples if item.split == split]
        random.Random(f"{seed}:{split}").shuffle(selected)
        result.extend(selected)
    audit_splits(result)
    return result


def prepare_study(output, *, seed=17, development_per_class=10, adaptation_per_class=20, holdout_per_class=10,
                  source_dir=None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "sources"
    cache.mkdir()
    provenance = {}
    raw = {}
    for filename, url in SOURCES.items():
        if source_dir is not None:
            value = (Path(source_dir) / filename).read_bytes()
        else:
            request = Request(url, headers={"User-Agent": "SHREK-course-research"})
            with urlopen(request, timeout=45) as response:
                value = response.read(20_000_001)
        if len(value) > 20_000_000:
            raise ValueError("Unexpectedly large source file")
        raw[filename] = value
        (cache / filename).write_bytes(value)
        provenance[filename] = {"url": url, "sha256": hashlib.sha256(value).hexdigest(),
                                "origin": "operator_local_snapshot" if source_dir is not None else "pinned_public_revision"}
    examples = build_examples(json.loads(raw["pair.json"]), raw["xstest.csv"].decode("utf-8-sig"), seed,
                              development_per_class, adaptation_per_class, holdout_per_class)
    path = output / "study.jsonl"
    path.write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in examples), encoding="utf-8")
    report = {"seed": seed, "audit": audit_splits(examples), "study_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_files": provenance, "human_reviewed": False, "review_required": True,
              "harmful_grouping": "One published PAIR prompt per normalized underlying goal; old-target success flags ignored",
              "legitimate_grouping": "One safe XSTest prompt per lexical focus; unsafe XSTest counterparts excluded",
              "limitations": ["One published attack family, originally optimized on a different model",
                              "Only non-empty published prompts are available (82/100 in the inspected Vicuna artifact), introducing availability/survivorship bias even though success flags are not used for selection",
                              "Lexical grouping cannot establish semantic independence; review near-duplicates before freezing",
                              "Benchmark labels are not independently re-annotated; review development anchors",
                              "Source selection differs between classes; results are a small pilot, not a deployment prevalence estimate"],
              "attribution": {"attacks": "JailbreakBench (Chao et al., 2024), published PAIR artifacts; preserve supplied license",
                              "legitimate": "XSTest (Roettger et al., 2024), preserve supplied CC-BY license and attribution"}}
    (output / "preparation_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
