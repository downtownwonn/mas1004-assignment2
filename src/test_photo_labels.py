"""Synthetic tests; no real photos, official val, or pretrained downloads."""

import csv
import ast
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import classify_photo_labels as classifier
import colab_photo_labels as runner


class PhotoLabelTests(unittest.TestCase):
    def test_notebook_and_script_syntax(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("classify_photo_labels.py", "colab_photo_labels.py", "test_photo_labels.py"):
            ast.parse((root / "src" / name).read_text(encoding="utf-8"))
        notebook = json.loads((root / "photo_labels_colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_video_groups_and_reproducibility(self):
        paths = [f"data/starter/train/seoul/commons_{i}.jpg" for i in range(200)]
        paths[:2] = [f"data/starter/train/seoul/youtube___AWAZEef6M_{i:05d}.jpg" for i in range(2)]
        labels = np.arange(200) % 10
        train, test = classifier.evaluation_split(paths, labels, 0)
        again = classifier.evaluation_split(paths, labels, 0)
        np.testing.assert_array_equal(test, again[1])
        self.assertEqual(set(train) | set(test), set(range(200)))
        self.assertTrue({classifier.group_key(paths[i]) for i in train}.isdisjoint(
            {classifier.group_key(paths[i]) for i in test}))
        self.assertEqual(set(labels[test]), set(range(10)))
        self.assertEqual(classifier.group_key(paths[0]), "youtube:__AWAZEef6M")

    def test_cached_features_need_no_model_or_images(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            paths = ["data/starter/train/seoul/commons_1.jpg"]
            block = np.ones((1, 512), dtype=np.float32)
            with patch.object(classifier, "resnet18", side_effect=AssertionError("No model download")):
                # First call creates the manifest but cannot read a nonexistent image.
                np.save(cache / "features_000000_000001.npy", block)
                result = classifier.extract_features(paths, [10], cache, torch.device("cpu"), 128)
                np.testing.assert_array_equal(result, block)
                with self.assertRaises(ValueError):
                    classifier.extract_features(paths, [11], cache, torch.device("cpu"), 128)

    def test_train_only_archive_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "starter.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("starter/train/seoul/commons_1.jpg", b"synthetic train bytes")
                z.writestr("starter/val/seoul/forbidden.jpg", b"do not read")
                z.writestr("starter/train/seoul/../../escape.jpg", b"unsafe")
            original = zipfile.ZipFile.open
            def guarded_open(z, name, *args, **kwargs):
                filename = name.filename if isinstance(name, zipfile.ZipInfo) else name
                self.assertNotIn("/val/", filename)
                return original(z, name, *args, **kwargs)
            with patch.object(runner, "EXPECTED", {"seoul": 1}), patch.object(runner, "CITIES", {"seoul"}), patch.object(zipfile.ZipFile, "open", guarded_open):
                runner.prepare(archive, root / "data")
            self.assertEqual((root / "data/starter/train/seoul/commons_1.jpg").read_bytes(), b"synthetic train bytes")
            self.assertFalse((root / "data/starter/val").exists())

    def test_end_to_end_with_synthetic_features_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [f"data/starter/train/seoul/commons_{i}.jpg" for i in range(210)]
            labels = {path: i % 10 for i, path in enumerate(paths[:200])}
            label_file = root / "labels/labels_train.csv"
            label_file.parent.mkdir()
            classifier.write_csv(label_file, ["path", "city", "label", "saved_at"], [
                dict(path=p, city="seoul", label=label, saved_at="test") for p, label in labels.items()
            ])
            before = label_file.read_bytes()
            raw = np.random.default_rng(0).normal(size=(210, 512)).astype(np.float32)
            actual_device = torch.device
            cpu = actual_device("cpu")
            args = ["classify_photo_labels.py", "--out", "results/test", "--epochs", "3"]
            with patch.object(classifier, "ROOT", root), patch.object(classifier, "inventory", return_value=(paths, [10] * 210)), patch.object(classifier, "extract_features", return_value=raw), patch.object(classifier, "run_device", return_value=cpu), patch.object(sys, "argv", args):
                classifier.main()
                classifier.main()
            self.assertEqual(label_file.read_bytes(), before)
            with (root / "results/test/all_assignments.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 210)
            self.assertEqual(sum(r["label_origin"] == "human" for r in rows), 200)
            self.assertTrue(all(int(r["assigned_label"]) == labels[r["path"]] for r in rows if r["path"] in labels))
            grouped = []
            for label in range(10):
                with (root / f"results/test/label_{label}.csv").open(encoding="utf-8-sig", newline="") as f:
                    subset = list(csv.DictReader(f))
                self.assertTrue(all(int(r["assigned_label"]) == label for r in subset))
                grouped.extend(r["path"] for r in subset)
            self.assertEqual(sorted(grouped), sorted(paths))
            metadata = json.loads((root / "results/test/run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["model_assigned"], 10)

    def test_perfect_metrics(self):
        result = classifier.evaluate(np.arange(10), np.eye(10))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()