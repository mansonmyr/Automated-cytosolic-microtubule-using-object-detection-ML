"""Export a square whole-image detector and compare ONNX Runtime with PyTorch on two sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("YOLO_OFFLINE","true")

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def export_and_verify(weights, size, output, all_validation=False):
    import cv2
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    project=Path(__file__).resolve().parents[1]
    output.mkdir(parents=True,exist_ok=True)
    local=output/"checkpoint.pt"
    if not local.exists() or hashlib.sha256(local.read_bytes()).digest()!=hashlib.sha256(weights.read_bytes()).digest():
        shutil.copy2(weights,local)
    model=YOLO(str(local))
    model.model.names={0:"cytosolic",1:"centrosomal",2:"cell"}
    torch.set_num_threads(1)
    exported=Path(model.export(format="onnx",imgsz=size,batch=1,dynamic=False,simplify=False,opset=17,device="cpu"))
    graph=onnx.load(exported);onnx.checker.check_model(graph)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(exported),sess_options=options,providers=["CPUExecutionProvider"])
    inp=session.get_inputs()[0]
    if inp.shape != [1,3,size,size] or session.get_outputs()[0].shape[-1]!=6:
        raise ValueError("ONNX tensor shapes differ from the supported detector")
    images=sorted((project/"prepared_native2048/images/val").glob("*.png"))
    def density(path):
        return len((project/"prepared_native2048/labels/val"/(path.stem+".txt")).read_text().splitlines())
    selected=images if all_validation else [min(images,key=density),max(images,key=density)]
    checks=[]
    for path in selected:
        rgb=np.array(Image.open(path).convert("RGB"))
        width,height=rgb.shape[1],rgb.shape[0]
        if width!=height:
            raise ValueError("Verification requires square source images")
        # The app's whole-image mode must use this RGB resize and scale boxes back.
        resized=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_LINEAR) if width!=size else rgb
        tensor=np.ascontiguousarray(resized.transpose(2,0,1)[None],dtype=np.float32)/255
        raw=session.run(None,{inp.name:tensor})[0]
        if raw.shape!=(1,300,6) or not np.isfinite(raw).all():
            raise ValueError("Unexpected ONNX predictions")
        b=raw[0];b=b[b[:,4]>=.05].copy()
        b[:,:4]=np.clip(b[:,:4],0,size)*(width/size)
        a=model.predict(str(path),imgsz=size,device="cpu",conf=.05,max_det=300,verbose=False)[0].boxes.data.cpu().numpy()
        if a.shape!=b.shape:
            raise ValueError(f"Prediction counts differ on {path.name}")
        if len(a):
            cost=np.abs(a[:,None,:4]-b[None,:,:4]).sum(axis=2)+(a[:,None,5]!=b[None,:,5])*1e6
            ai,bi=linear_sum_assignment(cost);a,b=a[ai],b[bi]
            if (not np.array_equal(a[:,5],b[:,5]) or not np.allclose(a[:,:4],b[:,:4],atol=.5,rtol=1e-4)
                    or not np.allclose(a[:,4],b[:,4],atol=.002,rtol=1e-4)):
                raise ValueError(f"ONNX parity failed on {path.name}")
        checks.append(dict(image=path.name,detections=len(a),passed=True))
    result=dict(path=str(exported),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                onnx_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),input_shape=inp.shape,
                mode="whole_image",classes=["cytosolic","centrosomal","cell"],checks=checks,
                verification="ONNX Runtime CPU parity. Browser WASM and original-coordinate mapping still require an integration check.")
    (output/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",type=Path,required=True);p.add_argument("--size",type=int,default=2048)
    p.add_argument("--all-validation",action="store_true",help="Check every development image")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    print(json.dumps(export_and_verify(a.weights,a.size,a.output,a.all_validation),indent=2))
