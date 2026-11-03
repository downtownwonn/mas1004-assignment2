"""Prepare only train photos, classify, and pack only this task's results."""

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from get_data import download, drive_file_id

ROOT = Path(__file__).resolve().parents[1]
CITIES = {"busan", "osaka", "seoul", "taipei", "tokyo"}
EXPECTED = {"busan": 5191, "osaka": 7200, "seoul": 11274, "taipei": 7245, "tokyo": 13347}


def train_member(name):
    parts = PurePosixPath(name).parts
    return (len(parts) == 4 and parts[:2] == ("starter", "train")
            and parts[2] in CITIES and parts[3].endswith(".jpg")
            and "\\" not in name and name == "/".join(parts))


def prepare(archive, destination):
    """Read archive directory; decompress only training JPG members (CRC checked)."""
    with zipfile.ZipFile(archive) as z:
        members = [item for item in z.infolist() if train_member(item.filename)]
        counts = {city: sum(PurePosixPath(m.filename).parts[2] == city for m in members)
                  for city in CITIES}
        if counts != EXPECTED or len({m.filename for m in members}) != len(members):
            raise ValueError(f"Unexpected training archive inventory: {counts}")
        if shutil.disk_usage(destination.parent).free < sum(m.file_size for m in members) + 1_000_000_000:
            raise RuntimeError("Not enough local disk space for training photos")
        for index, member in enumerate(members):
            target = destination / member.filename
            if target.exists():
                if target.is_symlink() or target.stat().st_size != member.file_size:
                    raise ValueError(f"Unexpected existing training photo: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".partial")
            with z.open(member) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            temporary.replace(target)
            if (index + 1) % 5000 == 0:
                print(f"Train extracted: {index + 1}/{len(members)}", flush=True)
    print(f"Train only: {counts}; val was not extracted or opened", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    archive = Path(args.archive)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        link = os.environ.get("MAS1004_STARTER_LINK", "").strip()
        if not link:
            raise ValueError("Starter archive missing; supply the link through the notebook prompt")
        partial = archive.with_suffix(".partial")
        download(drive_file_id(link), partial)
        # Validate directory before accepting a completed download; train CRCs
        # are verified on extraction. Do not read validation member contents.
        with zipfile.ZipFile(partial) as z:
            if sum(train_member(n) for n in z.namelist()) != sum(EXPECTED.values()):
                raise ValueError("Downloaded archive has unexpected training contents")
        partial.replace(archive)
    destination = Path("/content/photo_label_data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepare(archive, destination)
    link = ROOT / "data/starter"
    link.parent.mkdir(exist_ok=True)
    expected = destination / "starter"
    if link.is_symlink():
        if link.resolve() != expected.resolve():
            raise ValueError("Existing starter link points elsewhere")
    elif link.exists():
        raise ValueError("Refusing to replace existing starter directory")
    else:
        link.symlink_to(expected, target_is_directory=True)
    labels = ROOT / "labels/labels_train.csv"
    label_digest = hashlib.sha256(labels.read_bytes()).hexdigest()
    script_digest = hashlib.sha256((ROOT / "src/classify_photo_labels.py").read_bytes()).hexdigest()
    run_id = hashlib.sha256((label_digest + script_digest).encode()).hexdigest()[:12]
    out = f"results/photo_labels_{run_id}"
    cache = "results/resnet18_photo_features_v1"
    command = [sys.executable, "-u", "src/classify_photo_labels.py", "--out", out, "--cache", cache]
    log = ROOT / "results/photo_labels_run.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for line in process.stdout:
            print(line, end="", flush=True)
            f.write(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"Classification failed ({status}); see {log}")
    with (ROOT / out / "all_assignments.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != sum(EXPECTED.values()) or len({r["path"] for r in rows}) != len(rows):
        raise RuntimeError("Output inventory verification failed")
    with labels.open(encoding="utf-8-sig", newline="") as f:
        human = {r["path"]: r["label"] for r in csv.DictReader(f)}
    assigned = {r["path"]: r for r in rows}
    if not all(assigned[p]["assigned_label"] == label and assigned[p]["label_origin"] == "human"
               for p, label in human.items()):
        raise RuntimeError("Human label preservation verification failed")
    if hashlib.sha256(labels.read_bytes()).hexdigest() != label_digest:
        raise RuntimeError("Human label file changed")
    output = ROOT / "from_colab_photo_labels.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in (ROOT / out, ROOT / cache):
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(ROOT).as_posix())
        z.write(log, log.relative_to(ROOT).as_posix())
    print(f"Verified {len(rows)} assignments; {len(human)} human labels preserved. Download {output.name}")


if __name__ == "__main__":
    main()