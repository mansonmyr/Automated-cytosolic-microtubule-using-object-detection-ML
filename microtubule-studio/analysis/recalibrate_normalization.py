"""Recompute batch-normalization buffers on training sources for a separate candidate copy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from ultralytics import YOLO


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def recalibrate(weights, prepared, manifest, output):
    sources = pd.read_csv(manifest)
    sources = sources.loc[sources.subset=="fit", ["source_id","split"]].sort_values("source_id")
    if set(sources.split) != {"train"} or len(sources) != 60 or sources.source_id.duplicated().any():
        raise ValueError("Expected the 60 distinct fit sources from the original training split")
    if output.exists() or output.resolve()==weights.resolve():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(weights))
    model = detector.model.float().eval()
    original_parameters = parameter_hash(model)
    layers = [m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    for source in sources.source_id:
        path = prepared/"images/train"/(source+".png")
        with Image.open(path) as image:
            if image.size != (2048,2048):
                raise ValueError(path)
            rgb = np.asarray(image.convert("RGB")).copy()
        tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().div_(255)
        model(tensor)
    if not layers or any(int(layer.num_batches_tracked) != len(sources) for layer in layers):
        raise RuntimeError("Not every normalization layer saw every fit source")
    model.eval()
    if parameter_hash(model) != original_parameters:
        raise RuntimeError("A trainable parameter changed")
    # Save FP32 buffers. Casting small estimated variances to FP16 can change evaluation.
    checkpoint = {"model":model,"ema":None,"epoch":-1,"optimizer":None,
                  "train_args":dict(detector.ckpt.get("train_args",{}))}
    temp = output.with_suffix(".tmp.pt")
    torch.save(checkpoint,temp)
    restored = YOLO(str(temp)).model.float().eval()
    if parameter_hash(restored) != original_parameters:
        raise RuntimeError("Reload changed the serialized trainable parameters")
    for (name,a),(other,b) in zip(model.named_buffers(),restored.named_buffers()):
        if name != other or not torch.equal(a,b):
            raise RuntimeError(f"Buffer reload mismatch at {name}")
    temp.replace(output)
    provenance = {"parent_checkpoint_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
                  "candidate_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                  "trainable_parameter_hash_before_and_after":original_parameters,
                  "batch_norm_layers":len(layers),"fit_source_ids":sources.source_id.tolist(),
                  "input":"One pass in source order at 2048 pixels, RGB/255, equal-weight average of per-image batch statistics.",
                  "primary_comparison":"Fixed score 0.25 on 25 development images. Report count MAE on common resolved cells, faint recall and reference-unmatched proposals. The normalization hypothesis arose from the two-source fit diagnostic."}
    output.with_suffix(".json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("weights","prepared","manifest","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    recalibrate(a.weights,a.prepared,a.manifest,a.output)
