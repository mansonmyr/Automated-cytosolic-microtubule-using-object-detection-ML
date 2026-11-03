"""Fine-tune the refined checkpoint on all original native-resolution annotations."""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT/"evaluation/refined_to2048"
RUN = PROJECT/"training_runs/refined_to2048"
DATA = PROJECT/"prepared_native2048/data.yaml"
INITIAL = PROJECT.parent/"results/runs/detect/runs/yolov10n_refined/weights/best.pt"
ENV = dict(os.environ, YOLO_OFFLINE="true", MPLCONFIGDIR=str(PROJECT/"training_runs/mpl-cache"),
           OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENCV_LOG_LEVEL="ERROR")
STATE = OUTPUT/"workflow_status.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(stage, **extra):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(stage=stage,updated_unix=time.time(),**extra),indent=2)+"\n")
    tmp.replace(STATE)


def execute(script, args, log):
    with log.open("a") as handle:
        subprocess.run([sys.executable,"-u",str(PROJECT/script),*map(str,args)],
                       cwd=PROJECT,env=ENV,stdout=handle,stderr=subprocess.STDOUT,check=True)


def prepare(epoch_limit=40, check_only=False):
    from train import native_size
    if native_size(DATA) != 2048:
        raise ValueError("Expected original 2048-pixel images")
    # Check that the source labels still match the benchmark preparation manifest.
    manifest = json.loads((PROJECT/"evaluation/annotation_quality/manifest.json").read_text())
    for item in manifest:
        image = Path(item["source_image"])
        label = DATA.parent/"labels"/item["split"]/(image.stem+".txt")
        if digest(image)!=item["source_image_sha256"] or digest(label)!=item["source_label_sha256"]:
            raise ValueError(f"Source changed since the annotation audit: {image.name}")
    plan = dict(initial_weights=str(INITIAL),initial_sha256=digest(INITIAL),data=str(DATA),
                data_sha256=digest(DATA),input_size=2048,training_sources=80,development_sources=25,
                all_original_cytosolic_training_positives=1484,maximum_additional_epochs=epoch_limit,
                optimizer="AdamW",lr0=.0001,patience=10,warmup_epochs=1,seed=42,batch=1,
                original_dimensions_preserved=True,
                interpretation="Development comparison on reused source images. All cells require reviewed ownership. The omission model is an experimental arm.")
    path = OUTPUT/"design.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        previous=json.loads(path.read_text())
        comparable=dict(previous,maximum_additional_epochs=epoch_limit)
        if comparable!=plan or epoch_limit<previous["maximum_additional_epochs"]:
            raise ValueError("Fine-tuning inputs differ from the recorded design")
        if not check_only:
            shutil.copy2(path,OUTPUT/f"design_before_extension_{time.time_ns()}.json")
    if not check_only:
        path.write_text(json.dumps(plan,indent=2)+"\n")
    return plan


def extend_checkpoint(weights, state, epoch_limit):
    """Change the epoch budget in a copy while preserving the saved FP32 state."""
    import torch
    checkpoint=torch.load(weights,map_location="cpu",weights_only=False)
    previous=checkpoint["train_args"]["epochs"]
    if previous==epoch_limit:
        return weights
    if previous>epoch_limit:
        raise ValueError("Refusing to reduce the checkpoint epoch budget")
    folder=RUN/f"extension_to_{epoch_limit}_{time.time_ns()}"
    folder.mkdir()
    checkpoint["train_args"]["epochs"]=epoch_limit
    extended=folder/"last.pt"
    torch.save(checkpoint,extended)
    restored=dict(state,checkpoint_sha256=digest(extended))
    torch.save(restored,extended.with_name("resume_state.pth"))
    reloaded=torch.load(extended.with_name("resume_state.pth"),map_location="cpu",weights_only=True)
    if reloaded["checkpoint_sha256"]!=digest(extended):
        raise ValueError("Extended checkpoint and recovery state differ")
    (folder/"change.json").write_text(json.dumps(dict(previous_budget=previous,new_budget=epoch_limit,
        completed_epochs=state["completed_epochs"],optimizer_reset=False,
        note="The learning-rate schedule uses the extended budget after this restart."),indent=2)+"\n")
    return extended


def audit(weights):
    folder=OUTPUT/"audit"
    summary=folder/"evaluation_summary.json"
    if summary.exists():
        if json.loads(summary.read_text())["weights_sha256"] != digest(weights):
            raise ValueError("Existing audit belongs to a different checkpoint")
        return folder
    folder.mkdir(exist_ok=True)
    for name in ["image_manifest.csv","cytosolic_ownership.csv","cell_annotation_counts.csv","audit_summary.json"]:
        shutil.copy2(PROJECT/"analysis_runs/per_cell_v1"/name,folder/name)
    execute("analysis/evaluate_faint.py",["--root",PROJECT.parent,"--audit",folder,"--weights",weights,"--imgsz",2048],folder/"evaluation.log")
    return folder


def train_and_score(epoch_limit=40):
    if STATE.exists() and json.loads(STATE.read_text()).get("stage") in ["completed","completed_export_pending"]:
        print("Training and comparison have already finished.",flush=True)
        return
    args=["--data",DATA,"--device","cpu","--threads",4,"--full-precision-state"]
    if RUN.exists():
        status_path=RUN/"training_status.json"
        if not status_path.exists():
            raise RuntimeError("The run directory exists without a training status. Inspect it before continuing.")
        status=json.loads(status_path.read_text())
        if status["status"]!="completed":
            try:
                os.kill(status["pid"],0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("The recorded trainer still exists")
            prepare(epoch_limit)
            import torch
            weights=RUN/"weights/last.pt"
            state=torch.load(weights.with_name("resume_state.pth"),map_location="cpu",weights_only=True)
            if state["checkpoint_sha256"]!=digest(weights) or state["completed_epochs"]!=status["completed_epochs"]:
                raise ValueError("Recovery state does not match the last complete checkpoint")
            weights=extend_checkpoint(weights,state,epoch_limit)
            record("training",resumed_after_epoch=state["completed_epochs"])
            execute("training/train.py",[*args,"--weights",weights,"--resume"],OUTPUT/"training.log")
    else:
        prepare(epoch_limit)
        record("training")
        execute("training/train.py",[*args,"--weights",INITIAL,"--epochs",epoch_limit,"--batch",1,
            "--name",RUN.name,"--optimizer","AdamW","--lr0",.0001,"--patience",10,"--warmup-epochs",1],OUTPUT/"training.log")
    status=json.loads((RUN/"training_status.json").read_text())
    if status["status"]!="completed":
        raise RuntimeError("Training has not completed")
    weights=RUN/"weights/best.pt"
    record("evaluating",completed_epochs=status["completed_epochs"])
    folder=audit(weights)
    execute("analysis/compare_checkpoints.py",[
        "--audit","refined1024",PROJECT/"analysis_runs/refined1024",
        "--audit","native2048",PROJECT/"analysis_runs/native2048",
        "--audit","fine_tuned2048",folder,"--output",OUTPUT/"comparison"],OUTPUT/"comparison.log")
    # This compares operational direct whole-image predictions, including original-coordinate boxes.
    export = None
    if all(importlib.util.find_spec(x) is not None for x in ["onnx","onnxruntime"]):
        from export_browser import export_and_verify
        record("exporting")
        export=export_and_verify(weights,2048,OUTPUT/"browser")
    stage="completed" if export else "completed_export_pending"
    record(stage,completed_epochs=status["completed_epochs"],comparison=str(OUTPUT/"comparison/paired_comparison.md"),
           browser_export=export,export_requires=None if export else ["onnx","onnxruntime"])
    print("Fine-tuning and comparison completed.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--epoch-limit",type=int,default=40)
    args=parser.parse_args()
    if args.epoch_limit<1:
        parser.error("--epoch-limit must be positive")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.check_only:
        print(json.dumps(prepare(args.epoch_limit,check_only=True),indent=2))
        sys.exit(0)
    with (OUTPUT/"workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("This workflow is already running.")
        try:
            train_and_score(args.epoch_limit)
        except BaseException as error:
            record("stopped",error=str(error))
            raise
