"""Metadata-only checks: no photo is opened and no validation data is read."""

import ast
import csv
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_csv(ROOT / "results/photo_labels_f369cab88482/all_assignments.csv")
        cls.selected = [r["path"] for r in read_csv(ROOT / "track_a/selection.csv")]

    def test_exact_requested_partition(self):
        dropped_file = ROOT / "track_a/dropped.csv"
        dropped = {r["path"] for r in read_csv(dropped_file)} if dropped_file.exists() else set()
        expected = sorted(r["path"].removeprefix("data/starter/train/") for r in self.rows
                          if r["assigned_label"] in {"0", "3", "8", "9"}
                          and r["path"].removeprefix("data/starter/train/") not in dropped)
        self.assertEqual(self.selected, expected)
        self.assertEqual(len(self.selected), len(set(self.selected)))
        self.assertEqual(len(self.selected), 26501)

    def test_human_labels_take_priority(self):
        humans = [r for r in self.rows if r["human_label"] != ""]
        self.assertEqual(len(humans), 1500)
        for row in humans:
            self.assertEqual(row["assigned_label"], row["human_label"])

    def test_manifest_hashes(self):
        manifest = json.loads((ROOT / "track_a/selection_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["selection_sha256"],
                         hashlib.sha256((ROOT / "track_a/selection.csv").read_bytes()).hexdigest())
        self.assertEqual(manifest["source_sha256"],
                         hashlib.sha256((ROOT / manifest["source"]).read_bytes()).hexdigest())

    def test_only_flip_removed_from_transform(self):
        tree = ast.parse((ROOT / "track_a/recipe.py").read_text(encoding="utf-8"))
        transform = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_transform")
        calls = [n.func.attr for n in ast.walk(transform)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(set(calls), {"Compose", "RandomResizedCrop", "ColorJitter"})


if __name__ == "__main__":
    unittest.main(verbosity=2)