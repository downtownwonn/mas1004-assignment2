"""Train-only photo-label classification using reusable frozen ResNet18 features."""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
CITIES = ("busan", "osaka", "seoul", "taipei", "tokyo")
LABEL_NAMES = (
    "거리 풍경", "실내", "물건·음식 클로즈업", "지도·글자·포스터·스크린샷",
    "사람 중심", "자연·풍경", "항공샷", "로우앵글로 거리가 안 보이는 사진",
    "이동수단 안에서 찍은 사진", "애매함",
)
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inventory():
    paths, sizes = [], []
    train = (ROOT / "data/starter/train").resolve()
    for city in CITIES:
        folder = train / city
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f"Missing or symlinked training folder: {folder}")
        for path in sorted(folder.glob("*.jpg")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected training file: {path}")
            paths.append(f"data/starter/train/{city}/{path.name}")
            sizes.append(path.stat().st_size)
    if not paths:
        raise ValueError("No training photos found")
    return paths, sizes


def load_labels(path, paths):
    allowed, labels = set(paths), {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name, label = row["path"], int(row["label"])
            if (name not in allowed or name in labels or label not in range(10)
                    or row["city"] != Path(name).parent.name):
                raise ValueError(f"Invalid or duplicate label row: {row}")
            labels[name] = label
    if set(labels.values()) != set(range(10)):
        raise ValueError("All ten labels must have examples")
    return labels


def extract_features(paths, sizes, cache, device, batch_size):
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": "resnet18", "weights": "IMAGENET1K_V1",
        "preprocessing": repr(WEIGHTS.transforms()), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "dimension": 512,
        "dtype": "float32", "batch_size": batch_size, "paths": paths, "bytes": sizes,
        "input_contract": "Immutable starter train photos; path/size checked, not rehashed",
    }
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Cache metadata differs; use a new --cache directory")
    else:
        write_json(manifest_path, manifest)
    model, blocks = None, []
    transform = WEIGHTS.transforms()
    for start in range(0, len(paths), batch_size):
        stop = min(start + batch_size, len(paths))
        shard = cache / f"features_{start:06d}_{stop:06d}.npy"
        if shard.exists():
            block = np.load(shard, allow_pickle=False)
        else:
            if model is None:
                model = resnet18(weights=WEIGHTS)
                model.fc = nn.Identity()
                model = model.to(device).eval()
            images = []
            for name in paths[start:stop]:
                with Image.open(ROOT / name) as image:
                    images.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
            with torch.inference_mode():
                block = model(torch.stack(images).to(device)).cpu().numpy().astype(np.float32)
            temporary = shard.with_suffix(".tmp")
            with temporary.open("wb") as f:
                np.save(f, block, allow_pickle=False)
            temporary.replace(shard)
        if (block.shape != (stop - start, 512) or block.dtype != np.float32
                or not np.isfinite(block).all()):
            raise ValueError(f"Invalid feature shard: {shard}")
        blocks.append(block)
        if start == 0 or stop % (batch_size * 25) == 0 or stop == len(paths):
            print(f"Features ready: {stop}/{len(paths)}", flush=True)
    return np.concatenate(blocks)


def group_key(path):
    match = re.fullmatch(r"youtube_([A-Za-z0-9_-]{11})_([0-9]+)\.jpg", Path(path).name)
    if match:
        return f"youtube:{match[1]}"
    if Path(path).name.startswith("youtube_"):
        raise ValueError(f"Unrecognized YouTube filename: {path}")
    return path


def evaluation_split(paths, labels, seed):
    groups = {}
    for index, path in enumerate(paths):
        groups.setdefault(group_key(path), []).append(index)
    keys, rng = list(groups), random.Random(seed)
    total, best = np.bincount(labels, minlength=10), None
    for _ in range(1000):
        rng.shuffle(keys)
        selected = []
        for key in keys:
            if len(selected) >= round(len(paths) * 0.2):
                break
            selected.extend(groups[key])
        test = np.array(sorted(selected), dtype=np.int64)
        mask = np.ones(len(paths), dtype=bool)
        mask[test] = False
        train = np.flatnonzero(mask)
        counts = np.bincount(labels[test], minlength=10)
        if np.any(counts == 0) or np.any(total - counts == 0):
            continue
        score = abs(len(test) / len(paths) - 0.2) + np.mean(np.abs(counts / total - 0.2))
        if best is None or score < best[0]:
            best = (score, train, test)
    if best is None:
        raise ValueError("Cannot construct a group holdout covering all labels")
    return best[1], best[2]


def fit_head(features, labels, indices, device, seed, epochs):
    torch.manual_seed(seed)
    x, y = features[indices].to(device), labels[indices].to(device)
    counts = torch.bincount(y, minlength=10).float()
    if torch.any(counts == 0):
        raise ValueError("Training partition is missing a label")
    weights = counts.reciprocal()
    weights /= weights.mean()
    model = nn.Linear(512, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y, weight=weights)
        loss.backward()
        optimizer.step()
    return model.eval()


def predict(model, features, device):
    with torch.inference_mode():
        return np.concatenate([
            model(batch.to(device)).softmax(dim=1).cpu().numpy()
            for batch in features.split(2048)
        ])


def evaluate(labels, probabilities):
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    per_label = []
    for label in range(10):
        tp = int(confusion[label, label])
        support, assigned = int(confusion[label].sum()), int(confusion[:, label].sum())
        precision = tp / assigned if assigned else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label.append(dict(label=label, name=LABEL_NAMES[label], support=support,
                              precision=precision, recall=recall, f1=f1))
    return {
        "accuracy": float((predicted == labels).mean()),
        "macro_f1": float(np.mean([r["f1"] for r in per_label])),
        "balanced_accuracy": float(np.mean([r["recall"] for r in per_label])),
        "per_label": per_label, "confusion_rows_true_columns_predicted": confusion.tolist(),
    }


def assignment_rows(paths, human, probabilities):
    rows = []
    for path, prob in zip(paths, probabilities):
        predicted, ordered = int(prob.argmax()), np.sort(prob)
        row = dict(path=path, city=Path(path).parent.name, human_label=human.get(path, ""),
                   predicted_label=predicted, assigned_label=human.get(path, predicted),
                   label_origin="human" if path in human else "model",
                   max_score=float(prob.max()), score_margin=float(ordered[-1] - ordered[-2]))
        row.update({f"score_{i}": float(prob[i]) for i in range(10)})
        rows.append(row)
    return rows


def run_device():
    if not torch.cuda.is_available():
        raise RuntimeError("This run expects a Colab GPU")
    return torch.device("cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="labels/labels_train.csv")
    parser.add_argument("--cache", default="results/resnet18_photo_features_v1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    device = run_device()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out, cache = ROOT / args.out, ROOT / args.cache
    for path in (out, cache):
        if not path.resolve().is_relative_to((ROOT / "results").resolve()) or path.resolve() == (ROOT / "results").resolve():
            raise ValueError("Output and cache must be subdirectories of results/")
    if out.resolve() == cache.resolve() or out.resolve() in cache.resolve().parents or cache.resolve() in out.resolve().parents:
        raise ValueError("Output and cache must be separate directories")
    paths, sizes = inventory()
    label_path = ROOT / args.labels
    label_bytes = label_path.read_bytes()
    human = load_labels(label_path, paths)
    labelled_indices = np.array([i for i, p in enumerate(paths) if p in human], dtype=np.int64)
    labelled_paths = [paths[i] for i in labelled_indices]
    y_array = np.array([human[p] for p in labelled_paths], dtype=np.int64)
    train, test = evaluation_split(labelled_paths, y_array, args.seed)
    assert {group_key(labelled_paths[i]) for i in train}.isdisjoint(
        {group_key(labelled_paths[i]) for i in test})
    config = dict(labels_sha256=hashlib.sha256(label_bytes).hexdigest(), seed=args.seed,
                  epochs=args.epochs, cache=str(cache.relative_to(ROOT)), batch_size=args.batch_size)
    if out.exists():
        run_path = out / "run_config.json"
        if not run_path.exists() or json.loads(run_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Output config differs; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", config)
    test_set = set(test.tolist())
    write_csv(out / "internal_split.csv", ["path", "label", "group", "split"], [
        dict(path=p, label=human[p], group=group_key(p),
             split="internal_eval" if i in test_set else "train")
        for i, p in enumerate(labelled_paths)
    ])
    print(f"Total={len(paths)}, human={len(human)}, fit={len(train)}, internal_eval={len(test)}", flush=True)
    raw = extract_features(paths, sizes, cache, device, args.batch_size)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-length feature detected")
    features, labels = torch.from_numpy(raw / norms), torch.from_numpy(y_array)
    labelled_features = features[labelled_indices]
    evaluation_model = fit_head(labelled_features, labels, train, device, args.seed, args.epochs)
    evaluation_probabilities = predict(evaluation_model, labelled_features[test], device)
    metrics = evaluate(y_array[test], evaluation_probabilities)
    write_json(out / "internal_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    write_csv(out / "internal_eval_predictions.csv", ["path", "human_label", "predicted_label", "max_score"], [
        dict(path=labelled_paths[i], human_label=int(y_array[i]),
             predicted_label=int(prob.argmax()), max_score=float(prob.max()))
        for i, prob in zip(test, evaluation_probabilities)
    ])
    model = fit_head(labelled_features, labels, np.arange(len(labels)), device, args.seed, args.epochs)
    rows = assignment_rows(paths, human, predict(model, features, device))
    assert len(rows) == len(paths) == len({r["path"] for r in rows})
    assert all(r["assigned_label"] == human[r["path"]] for r in rows if r["path"] in human)
    fields = list(rows[0])
    write_csv(out / "all_assignments.csv", fields, rows)
    for label in range(10):
        write_csv(out / f"label_{label}.csv", fields, [r for r in rows if r["assigned_label"] == label])
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "linear_head.pt")
    write_json(out / "run_metadata.json", {
        **config, "classifier": "L2-normalized ResNet18 -> Linear(512, 10)",
        "optimizer": "AdamW(lr=0.01, weight_decay=0.01)", "class_weighting": "inverse frequency",
        "labels": dict(enumerate(LABEL_NAMES)), "total": len(paths), "human_labelled": len(human),
        "model_assigned": len(paths) - len(human), "split_counts": dict(train=len(train), internal_eval=len(test)),
        "assigned_counts": dict(Counter(r["assigned_label"] for r in rows)),
        "score_warning": "Softmax scores are not calibrated probabilities.",
        "evaluation_warning": "YouTube videos grouped; other sources split per image. Photographer/visual overlap remains possible.",
    })
    if label_path.read_bytes() != label_bytes:
        raise RuntimeError("Labels changed during the run; results need a new run")
    print(f"Finished: {out}", flush=True)


if __name__ == "__main__":
    main()