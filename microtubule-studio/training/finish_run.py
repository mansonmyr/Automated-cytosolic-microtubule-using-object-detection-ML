"""Wait for a completed training run, compare checkpoints and verify a native-size export."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

# Set offline/cache options before importing Ultralytics.
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "training_runs" / "mpl-cache"))

import numpy as np
import torch
from ultralytics import YOLO
import yaml

NAMES = {0: "cytosolic", 1: "centrosomal", 2: "cell"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wait_for_training(run, output, wait):
    status_path = run / "training_status.json"
    while True:
        # Training updates are short writes; retry on a partially read document.
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if status["status"] == "completed":
            return status
        if status["status"] in ("failed", "interrupted"):
            raise RuntimeError(f"Training {status['status']}: {status.get('error', '')}")
        if not wait:
            raise RuntimeError("Training is still running. Use --wait to evaluate when it completes.")
        if time.time() - status_path.stat().st_mtime > 1800:
            raise RuntimeError("No training status update for 30 minutes. Check training before restarting this step.")
        write_json(output / "finish_status.json", {"status": "waiting_for_training",
                   "completed_epochs": status.get("completed_epochs", 0), "checked_at_unix": time.time()})
        time.sleep(15)


def load_model(weights):
    model = YOLO(str(weights))
    if model.model.model[-1].nc != 3:
        raise ValueError(f"Expected the three-class detector: {weights}")
    # Original checkpoints have incorrect display names; their class IDs match the audited labels.
    model.model.names = dict(NAMES)
    return model


def evaluate(weights, label, size, data, device, output):
    print(f"Evaluating {label} at {size} x {size}", flush=True)
    metrics = load_model(weights).val(data=str(data), imgsz=size, batch=1, device=device,
                   workers=0, conf=0.001, iou=0.7, max_det=300, half=False,
                   augment=False, plots=True, project=str(output), name=label, exist_ok=True)
    per_class = {}
    for i, cls in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(i)
        per_class[NAMES[int(cls)]] = dict(precision=float(precision), recall=float(recall),
                                         mAP50=float(ap50), mAP50_95=float(ap))
    result = {"name": label, "weights": str(weights), "input_size": size,
              "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
              "per_class": per_class}
    write_json(output / f"{label}.json", result)
    return result


def export_and_verify(weights, data, device):
    model = load_model(weights)
    exported = Path(model.export(format="torchscript", imgsz=2048, batch=1,
                                 device="cpu", optimize=False, half=False))
    config = yaml.safe_load(data.read_text())
    images = sorted((Path(config["path"]) / config["val"]).glob("*.png"))
    # Verify actual inference with both a sparse and a dense annotated validation image.
    def label_count(image):
        path = Path(config["path"]) / "labels" / "val" / image.with_suffix(".txt").name
        return len(path.read_text().splitlines())
    chosen = [min(images, key=label_count), max(images, key=label_count)]
    original, traced = load_model(weights), YOLO(str(exported))
    checks = []
    for image in chosen:
        settings = dict(source=str(image), imgsz=2048, conf=0.25, iou=0.7, max_det=300,
                        device=device, half=False, verbose=False)
        a = original.predict(**settings)[0].boxes
        b = traced.predict(**settings)[0].boxes
        # Compare stable class/score order; the model's end-to-end head needs no NMS.
        a, b = a.data.cpu().numpy(), b.data.cpu().numpy()
        a = a[np.lexsort((-a[:, 4], a[:, 5]))]
        b = b[np.lexsort((-b[:, 4], b[:, 5]))]
        if (a.shape != b.shape or not np.array_equal(a[:, 5], b[:, 5])
                or not np.allclose(a[:, :4], b[:, :4], atol=0.5, rtol=1e-4)
                or not np.allclose(a[:, 4], b[:, 4], atol=0.002, rtol=1e-4)):
            raise RuntimeError(f"TorchScript output differs from PyTorch on {image.name}")
        checks.append({"image": image.name, "detections": len(a), "passed": True})
    return {"path": str(exported), "input_shape": [1, 3, 2048, 2048], "checks": checks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--refined", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Check configuration without evaluating or waiting")
    args = p.parse_args()
    for key in ("run", "data", "baseline", "refined", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if not (args.run / "training_status.json").is_file():
        p.error("Training status file is missing")
    for path in (args.data, args.baseline, args.refined):
        if not path.is_file():
            p.error(f"Missing file: {path}")
    if args.check_only:
        print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    status_file = args.output / "finish_status.json"
    if status_file.is_file() and json.loads(status_file.read_text()).get("status") == "completed":
        print("Comparison and export already completed.")
        return
    torch.set_num_threads(args.threads)
    try:
        training = wait_for_training(args.run, args.output, args.wait)
        write_json(status_file, {"status": "evaluating", "completed_epochs": training["completed_epochs"]})
        best = args.run / "weights" / "best.pt"
        # Same-resolution comparisons plus each earlier model's original training size.
        comparisons = [(args.baseline, "baseline_2048", 2048), (args.refined, "refined_2048", 2048),
                       (best, "native_2048", 2048), (args.baseline, "baseline_640", 640),
                       (args.refined, "refined_1024", 1024)]
        results = [evaluate(weights, label, size, args.data, args.device, args.output)
                   for weights, label, size in comparisons]
        write_json(args.output / "comparison.json", {"results": results})
        with (args.output / "comparison.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "input_size", "class", "precision", "recall", "mAP50", "mAP50_95"])
            for result in results:
                for cls, values in result["per_class"].items():
                    writer.writerow([result["name"], result["input_size"], cls, values["precision"],
                                     values["recall"], values["mAP50"], values["mAP50_95"]])
        export = None
        if not args.skip_export:
            write_json(status_file, {"status": "exporting", "completed_epochs": training["completed_epochs"]})
            export = export_and_verify(best, args.data, args.device)
            write_json(args.output / "export_verification.json", export)
        lines = ["# Native-resolution model comparison", "",
                 "These are development-validation results on 25 previously used images, not an independent test.", "",
                 "All runs use conf=0.001, IoU=0.7 and max_det=300 for validation.",
                 "Precision and recall are library summaries at its selected operating point.", "",
                 "| Checkpoint | Input size | mAP50 | mAP50–95 |", "|---|---:|---:|---:|"]
        for result in results:
            m = result["metrics"]
            lines.append(f"| {result['name']} | {result['input_size']} | {m['metrics/mAP50(B)']:.4f} | {m['metrics/mAP50-95(B)']:.4f} |")
        lines += ["", "Per-class results are in comparison.csv and comparison.json.",
                  "The 2048 comparisons use the same input size. The last two rows show the earlier input sizes.",
                  "Training settings also changed, so this is not a controlled experiment isolating resolution.",
                  "", "The export check compares outputs on two validation images; it is not independent accuracy validation.",
                  "The browser integration requires a separate compatible runtime and input-size check."]
        (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
        write_json(status_file, {"status": "completed", "completed_epochs": training["completed_epochs"],
                   "report": str(args.output / "comparison.md"), "export": export})
        print(f"Comparison completed: {args.output / 'comparison.md'}", flush=True)
    except BaseException as error:
        write_json(status_file, {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
