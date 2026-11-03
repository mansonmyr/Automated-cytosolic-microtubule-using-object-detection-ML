"""Prepare a training-label omission experiment without changing source data.

Low local display contrast is a proxy used to choose annotations to remove.
It is not an estimate of photon noise, a biological label, or an MLE score.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import os

import numpy as np
from PIL import Image
import yaml


NAMES = {0: "cytosolic", 1: "centrosomal", 2: "cell"}
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_labels(path):
    raw = path.read_bytes()
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    labels = []
    for index, line in enumerate(lines):
        row = [float(value) for value in line.split()]
        if (len(row) != 5 or not all(math.isfinite(value) for value in row)
                or row[0] not in NAMES or any(not 0 <= value <= 1 for value in row[1:])
                or row[3] <= 0 or row[4] <= 0):
            raise ValueError(f"Invalid label at {path}, row {index + 1}")
        _, cx, cy, width, height = row
        if (cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6
                or cx + width / 2 > 1 + 1e-6 or cy + height / 2 > 1 + 1e-6):
            raise ValueError(f"Box outside image at {path}, row {index + 1}")
        labels.append(row)
    return raw, lines, labels


def local_display_contrast(luminance, box, margin=8):
    """Return p90 inside a box minus the median in its surrounding pixel ring.

    Nearby real structures can enter the ring. This is a reproducible ranking
    proxy, not a clean-background measurement or a signal-to-noise estimate.
    """
    height, width = luminance.shape
    _, cx, cy, bw, bh = box
    x1 = max(0, min(width - 1, math.floor((cx - bw / 2) * width)))
    y1 = max(0, min(height - 1, math.floor((cy - bh / 2) * height)))
    x2 = min(width, max(x1 + 1, math.ceil((cx + bw / 2) * width)))
    y2 = min(height, max(y1 + 1, math.ceil((cy + bh / 2) * height)))
    ox1, oy1 = max(0, x1 - margin), max(0, y1 - margin)
    ox2, oy2 = min(width, x2 + margin), min(height, y2 + margin)
    outer = luminance[oy1:oy2, ox1:ox2]
    mask = np.ones(outer.shape, dtype=bool)
    mask[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False
    ring = outer[mask]
    if not ring.size:
        return None
    return float(np.percentile(luminance[y1:y2, x1:x2], 90) - np.median(ring))


def prepare(data, output, drop_fraction=0.25, seed=42):
    if not 0 < drop_fraction < 1:
        raise ValueError("drop_fraction must be between 0 and 1, exclusive")
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    config = yaml.safe_load(data.read_text())
    if config.get("names") != NAMES:
        raise ValueError("Expected the audited three-class mapping")
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = data.parent / root
    root = root.resolve()
    source_yaml_hash = sha256(data)
    records, candidates = [], []
    before = {split: Counter() for split in ("train", "val")}
    for split in ("train", "val"):
        # Restrict this preparation to the native dataset's audited layout.
        if config.get(split) != f"images/{split}":
            raise ValueError(f"Expected {split}: images/{split}")
        images = sorted(path for path in (root / "images" / split).iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not images or len({path.stem for path in images}) != len(images):
            raise ValueError(f"Missing images or duplicate image stems in {split}")
        labels_present = {path.stem for path in (root / "labels" / split).glob("*.txt")}
        if labels_present != {path.stem for path in images}:
            raise ValueError(f"Images and label files do not match in {split}")
        for path in images:
            label_path = root / "labels" / split / path.with_suffix(".txt").name
            raw, lines, labels = read_labels(label_path)
            before[split].update(int(row[0]) for row in labels)
            with Image.open(path) as image:
                size = list(image.size)
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Expected a single-page image: {path}")
                # Display luminance intentionally avoids a claim of photon calibration.
                if split == "train":
                    luminance = np.asarray(image.convert("L"), dtype=np.float32)
                    for index, row in enumerate(labels):
                        if row[0] == 0:
                            contrast = local_display_contrast(luminance, row)
                            candidates.append(dict(image=path.name, row=index + 1,
                                                   display_contrast=contrast))
            records.append(dict(split=split, image=path, label_path=label_path,
                                raw=raw, lines=lines, labels=labels, size=size,
                                image_sha256=sha256(path), label_sha256=sha256(label_path)))
    eligible = [item for item in candidates if item["display_contrast"] is not None]
    count = math.floor(len(candidates) * drop_fraction)
    if count < 1 or count > len(eligible):
        raise ValueError("Not enough eligible cytosolic annotations for the requested omission fraction")
    # Random order breaks equal-score ties reproducibly; sorting then ranks contrast.
    ranked = random.Random(seed).sample(eligible, len(eligible))
    ranked.sort(key=lambda item: item["display_contrast"])
    omitted = {(item["image"], item["row"]) for item in ranked[:count]}
    dataset = output / "dataset"
    dataset.mkdir(parents=True)
    after = {split: Counter() for split in ("train", "val")}
    manifest = []
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    for record in records:
        split, path = record["split"], record["image"]
        target_image = dataset / "images" / split / path.name
        # Link individual files, not their directory, so YOLO resolves new label paths.
        try:
            target_image.symlink_to(os.path.relpath(path, target_image.parent))
            image_storage = "relative_symlink"
        except OSError:
            shutil.copyfile(path, target_image)
            image_storage = "byte_copy"
        target_label = dataset / "labels" / split / record["label_path"].name
        kept = [(index, row) for index, row in enumerate(record["labels"], start=1)
                if split != "train" or (path.name, index) not in omitted]
        if split == "val":
            target_label.write_bytes(record["raw"])
        else:
            text = "\n".join(record["lines"][index - 1] for index, _ in kept)
            target_label.write_text(text + ("\n" if text else ""))
        after[split].update(int(row[0]) for _, row in kept)
        if sha256(target_image) != record["image_sha256"]:
            raise RuntimeError(f"Image bytes changed: {path}")
        if sha256(record["label_path"]) != record["label_sha256"]:
            raise RuntimeError(f"Source labels changed: {record['label_path']}")
        if split == "val" and target_label.read_bytes() != record["raw"]:
            raise RuntimeError(f"Validation labels changed: {path}")
        manifest.append(dict(split=split, image=path.name, size=record["size"],
                             source_image=str(path), source_image_sha256=record["image_sha256"],
                             source_label_sha256=record["label_sha256"],
                             benchmark_label_sha256=sha256(target_label), image_storage=image_storage,
                             source_label_count=len(record["labels"]), benchmark_label_count=len(kept)))
    if sha256(data) != source_yaml_hash:
        raise RuntimeError("Source data YAML changed")
    dataset_config = dict(path=str(dataset), train="images/train", val="images/val", names=NAMES)
    (dataset / "data.yaml").write_text(yaml.safe_dump(dataset_config, sort_keys=False))
    fields = ["image", "row", "display_contrast", "omitted"]
    with (output / "training_annotation_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow(dict(item, omitted=int((item["image"], item["row"]) in omitted)))
    notes = [
        "Prepared only. No detector was trained and no accuracy improvement was measured.",
        "Original annotations are reference labels, not independently verified ground truth.",
        "Only training cytosolic annotations were deliberately omitted; image bytes are unchanged.",
        "Validation labels and all non-cytosolic training labels are retained.",
        "Low display contrast is a ranking proxy, not proof of faintness or an MLE score.",
        "The surrounding ring can include real structures; intensity calibration is unknown.",
        "Do not expose omitted-label identities or source reference labels to the candidate training method.",
        "The inherited validation split was used in development; this is an internal stress test.",
        "Image symlinks depend on the source dataset. Regenerate after moving datasets.",
    ]
    summary = dict(status="prepared", source_data=str(data), source_yaml_sha256=source_yaml_hash,
                   dataset_yaml=str(dataset / "data.yaml"), seed=seed,
                   requested_omission_fraction=drop_fraction, omitted_training_cytosolic=count,
                   eligible_training_cytosolic=len(eligible),
                   realized_omission_fraction=count / len(candidates),
                   before={s: dict(c) for s, c in before.items()},
                   after={s: dict(c) for s, c in after.items()},
                   images_per_split=dict(Counter(record["split"] for record in records)),
                   notes=notes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare(args.data, args.output, args.drop_fraction, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
