"""Run the separate 25% omission arm and score the paired checkpoints on original references."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
BENCH = PROJECT / "evaluation/annotation_quality"
RUN = PROJECT / "training_runs/native2048_omit25"
ORIGINAL = PROJECT / "training_runs/native2048"
REFERENCE = PROJECT / "analysis_runs/per_cell_v1"
ENV = dict(os.environ, YOLO_OFFLINE="true", OPENCV_LOG_LEVEL="ERROR", MPLCONFIGDIR=str(PROJECT/"training_runs/mpl-cache"),
           VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
STATUS = BENCH / "comparison_status.json"


def record(stage, **details):
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(stage=stage, updated_unix=time.time(), **details), indent=2)+"\n")
    temp.replace(STATUS)


def run(script, args, log):
    with log.open("a") as stream:
        subprocess.run([sys.executable, "-u", str(PROJECT/script), *map(str, args)],
                       cwd=PROJECT, env=ENV, stdout=stream, stderr=subprocess.STDOUT, check=True)


def audit(label, weights):
    out = BENCH/"results"/label
    summary = out/"evaluation_summary.json"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if summary.exists():
        if json.loads(summary.read_text())["weights_sha256"] != digest:
            raise ValueError(f"Checkpoint changed for {label}. Use a new comparison directory.")
        return out
    out.mkdir(parents=True, exist_ok=True)
    for name in ("cytosolic_ownership.csv", "cell_annotation_counts.csv", "image_manifest.csv", "audit_summary.json"):
        shutil.copyfile(REFERENCE/name,out/name)
    run("analysis/evaluate_faint.py", ["--root",PROJECT.parent,"--audit",out,"--weights",weights,"--imgsz",2048],out/"audit.log")
    return out


def score(label, original, omission):
    output = BENCH/"results"/label
    output.mkdir(parents=True, exist_ok=True)
    run("analysis/compare_checkpoints.py",
        ["--audit","original",original,"--audit","omission",omission,"--output",output], output/"comparison.log")


def resume_interrupted(status):
    pid = status.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("Missing trainer PID; check the run before resuming.")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError(f"Trainer PID {pid} still exists; no second trainer started.")

    import torch
    weights = RUN/"weights/last.pt"
    data = BENCH/"dataset/data.yaml"
    if not weights.with_name("resume_state.pth").is_file():
        raise RuntimeError("Full-precision recovery state is unavailable. Inspect this run before resuming.")
    ckpt = torch.load(weights, weights_only=False, map_location="cpu")
    epoch = ckpt["epoch"] + 1
    settings = ckpt["train_args"]
    if (epoch != status["completed_epochs"] or not (0 < epoch < settings["epochs"])
            or Path(settings["data"]).resolve() != data.resolve()
            or settings["name"] != RUN.name or settings["imgsz"] != 2048
            or not (ckpt.get("optimizer") or {}).get("state")):
        raise ValueError("Checkpoint does not match the recorded omission run.")
    del ckpt
    backup = RUN/f"before_resume_epoch{epoch}_{time.time_ns()}"
    backup.mkdir()
    for src in [weights, weights.with_name("resume_state.pth"), RUN/"weights/best.pt", RUN/"results.csv", RUN/"training_status.json"]:
        shutil.copy2(src, backup/src.name)
    with (BENCH/"resume_history.jsonl").open("a") as handle:
        print(json.dumps({"unix": time.time(), "after_completed_epoch": epoch, "backup": str(backup),
                          "optimizer_reset": False, "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}), file=handle)
    record("training", resumed_after_epoch=epoch, log=str(BENCH/"training.log"))
    run("training/train.py", ["--data",data,"--weights",weights,"--resume","--full-precision-state","--threads",4,"--device","cpu"], BENCH/"training.log")


def main(resume=False):
    design = json.loads((BENCH/"comparison_design.json").read_text())
    initial = PROJECT.parent/"results/yolov10n.pt"
    if hashlib.sha256(initial.read_bytes()).hexdigest() != design["initial_weights_sha256"]:
        raise ValueError("Initialization differs from the recorded comparison design.")
    train_status = RUN/"training_status.json"
    if resume and not train_status.is_file():
        raise ValueError("Cannot resume without the existing training status.")
    if not RUN.exists():
        record("training", primary_budget_epochs=26, total_planned_epochs=80, log=str(BENCH/"training.log"))
        run("training/train.py", ["--data",BENCH/"dataset/data.yaml","--weights",initial,"--epochs",80,"--batch",1,
                                "--threads",4,"--device","cpu","--name",RUN.name,"--full-precision-state"], BENCH/"training.log")
    elif resume:
        status = json.loads(train_status.read_text())
        if status.get("status") != "completed":
            resume_interrupted(status)
    if not train_status.exists() or json.loads(train_status.read_text()).get("status") != "completed":
        raise RuntimeError(f"Inspect {RUN} before resuming. A second trainer would invalidate the comparison.")
    record("scoring")
    marker = "epoch25.pt"
    if (RUN/"weights"/marker).exists():
        score("paired_epoch26", audit("original_epoch26",ORIGINAL/"weights"/marker),
              audit("omission_epoch26",RUN/"weights"/marker))
        primary = "paired_epoch26/paired_comparison.json"
    else:
        primary = "Unavailable because the omission arm stopped before epoch 26."
    score("exploratory_best",PROJECT/"analysis_runs/native2048",audit("omission_best",RUN/"weights/best.pt"))
    record("completed",primary=primary,secondary="exploratory_best/paired_comparison.json")


def primary_when_ready():
    """Publish the fixed-budget result while later training epochs continue."""
    milestone = RUN/"weights/epoch25.pt"
    status_file = RUN/"training_status.json"
    while True:
        try:
            status = json.loads(status_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(20)
            continue
        # The trainer writes checkpoint files before it records a completed epoch.
        if status.get("status") != "running":
            return  # The main supervisor scores available checkpoints after training.
        if (status.get("completed_epochs") or 0) >= 26 and milestone.is_file():
            break
        time.sleep(30)
    original = audit("original_epoch26", ORIGINAL/"weights/epoch25.pt")
    omission = audit("omission_epoch26", milestone)
    score("paired_epoch26", original, omission)


if __name__ == "__main__":
    import argparse
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-when-ready", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted omission run from its last complete epoch.")
    parser.add_argument("--standby", action="store_true", help="Wait for an existing supervisor, then resume only if needed.")
    args = parser.parse_args()
    if args.standby and not (args.resume or args.primary_when_ready):
        parser.error("--standby requires --resume or --primary-when-ready")
    name = "primary.lock" if args.primary_when_ready else "comparison.lock"
    with (BENCH/name).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not args.standby:
                sys.exit("This supervisor is already active.")
            print("Following the current run. Terminal will take over if its supervisor exits.", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        if args.standby and STATUS.is_file() and json.loads(STATUS.read_text()).get("stage") == "completed":
            sys.exit(0)
        if args.primary_when_ready:
            if not (BENCH/"results/paired_epoch26/paired_comparison.json").is_file():
                primary_when_ready()
        else:
            try:
                main(resume=args.resume)
            except BaseException as error:
                record("stopped",error=str(error))
                raise
