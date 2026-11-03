"""Reproduce the student's selection: keep assigned labels 0, 3, 8 and 9."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
KEEP = {"0", "3", "8", "9"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", default="results/photo_labels_f369cab88482/all_assignments.csv")
    args = parser.parse_args()
    source = ROOT / args.assignments
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    paths = [r["path"] for r in rows]
    if len(paths) != 44257 or len(set(paths)) != len(paths):
        raise ValueError("Expected the complete, unique 44,257-photo assignment list")
    dropped_file = ROOT / "track_a/dropped.csv"
    dropped = set()
    if dropped_file.exists():
        with dropped_file.open(encoding="utf-8-sig", newline="") as f:
            dropped = {r["path"] for r in csv.DictReader(f)}
    selected = []
    for row in rows:
        path = PurePosixPath(row["path"])
        if len(path.parts) != 5 or path.parts[:3] != ("data", "starter", "train"):
            raise ValueError(f"Not a training path: {path}")
        relative = PurePosixPath(*path.parts[3:]).as_posix()
        if row["assigned_label"] not in {str(i) for i in range(10)}:
            raise ValueError("Invalid assigned label")
        if row["assigned_label"] in KEEP and relative not in dropped:
            if not (ROOT / path.as_posix()).is_file():
                raise ValueError(f"Missing training photo: {path}")
            selected.append(relative)
    selected.sort()
    counts = Counter(PurePosixPath(p).parts[0] for p in selected)
    if set(counts) != {"busan", "osaka", "seoul", "taipei", "tokyo"}:
        raise ValueError("Selection must cover all five cities")
    output = ROOT / "track_a/selection.csv"
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path"])
        writer.writerows([p] for p in selected)
    manifest = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "kept_assigned_labels": sorted(KEEP), "excluded_assigned_labels": ["1", "2", "4", "5", "6", "7"],
        "human_labels_preserved_in_source": True,
        "previous_dropped_paths_respected": len(dropped),
        "selected": len(selected), "not_selected": len(rows) - len(selected),
        "counts": dict(sorted(counts.items())),
        "selection_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (ROOT / "track_a/selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()