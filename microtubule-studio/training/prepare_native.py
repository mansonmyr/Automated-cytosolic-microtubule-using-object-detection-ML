"""Audit LabelMe/YOLO pairs and prepare whole images without resizing or cropping."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

NAMES = {0: "cytosolic", 1: "centrosomal", 2: "cell"}
CLASS_IDS = {name: idx for idx, name in NAMES.items()}


def prepare(dataset, annotations, output):
    records, fingerprints, clipped = [], {}, []
    # Audit everything before creating the output directory.
    for split, folder in [("train", "train txt file"), ("val", "val txt file")]:
        paths = sorted((dataset / folder).glob("*.tif"))
        if not paths:
            raise ValueError(f"No TIFF images in {dataset / folder}")
        for path in paths:
            with Image.open(path) as image:
                if getattr(image, "n_frames", 1) != 1 or image.size != (2048, 2048):
                    raise ValueError(f"Expected single-page 2048 x 2048 TIFF: {path}")
                if image.mode != "P":
                    raise ValueError(f"Expected the audited 8-bit palette format: {path}")
                rgb = np.asarray(image.convert("RGB"))
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in fingerprints:
                raise ValueError(f"Duplicate image pixels: {path} and {fingerprints[digest]}")
            fingerprints[digest] = str(path)
            annotation = json.loads((annotations / f"{path.stem}.json").read_text())
            if (annotation["imageWidth"], annotation["imageHeight"]) != (2048, 2048):
                raise ValueError(f"Annotation dimensions disagree: {path}")
            rows = [list(map(float, line.split()))
                    for line in path.with_suffix(".txt").read_text().splitlines() if line.strip()]
            if len(rows) != len(annotation["shapes"]):
                raise ValueError(f"Annotation/YOLO count mismatch: {path}")
            labels = []
            for index, (shape, row) in enumerate(zip(annotation["shapes"], rows), start=1):
                if shape["shape_type"] not in ("rectangle", "polygon"):
                    raise ValueError(f"Unsupported annotation shape: {path}, {index}")
                cls = CLASS_IDS[shape["label"]]
                points = np.asarray(shape["points"], dtype=float)
                if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                    raise ValueError(f"Invalid annotation points: {path}, {index}")
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                expected = [(x1+x2)/4096, (y1+y2)/4096, (x2-x1)/2048, (y2-y1)/2048]
                if (len(row) != 5 or row[0] != cls or not np.isfinite(row).all()
                        or not np.allclose(row[1:], expected, atol=1e-6, rtol=0)):
                    raise ValueError(f"Annotation/YOLO coordinates or class disagree: {path}, {index}")
                box = np.clip([x1, y1, x2, y2], 0, 2048)
                if not np.array_equal(box, [x1, y1, x2, y2]):
                    clipped.append({"image": path.name, "shape": index,
                                    "original": [x1, y1, x2, y2], "clipped": box.tolist()})
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Empty bounding box: {path}, {index}")
                labels.append([cls, (x1+x2)/4096, (y1+y2)/4096, (x2-x1)/2048, (y2-y1)/2048])
            records.append({"split": split, "source": str(path.resolve()),
                            "image": path.stem + ".png", "rgb_sha256": digest, "labels": labels})
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    counts = {}
    for record in records:
        split = record["split"]
        target = output / "images" / split / record["image"]
        with Image.open(record["source"]) as image:
            image.convert("RGB").save(target)
        with Image.open(target) as image:
            if hashlib.sha256(np.asarray(image).tobytes()).hexdigest() != record["rgb_sha256"]:
                raise ValueError(f"Pixel verification failed: {target}")
        lines = [str(row[0]) + " " + " ".join(f"{v:.9f}" for v in row[1:]) for row in record.pop("labels")]
        (output / "labels" / split / target.with_suffix(".txt").name).write_text("\n".join(lines) + "\n")
        counts.setdefault(split, Counter()).update(int(line.split()[0]) for line in lines)
    config = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": NAMES}
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    audit = {"image_size": [2048, 2048], "resized": False, "tiled": False,
             "class_names": NAMES, "class_counts": counts,
             "images_per_split": dict(Counter(r["split"] for r in records)),
             "clipped_boundary_boxes": clipped, "pixel_duplicates": 0, "images": records,
             "notes": ["RGB PNG pixels exactly match the decoded TIFF palette colors.",
                       "Cell polygons are converted to bounding boxes, not segmentation masks.",
                       "Existing splits retained; experiment-level independence is unknown.",
                       "Validation images were previously used for model development."]}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("images_per_split", "class_counts", "pixel_duplicates")}, indent=2))
    print(f"Clipped {len(clipped)} boundary boxes. Source files unchanged. Dataset: {output / 'data.yaml'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    prepare(args.dataset, args.annotations, args.output)
