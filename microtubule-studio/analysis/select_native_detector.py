"""Run native detectors on the source images and score the candidates considered for deployment."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from ultralytics import YOLO
from evaluate_ensembles import ensemble, BOXES, KEY
from evaluate_faint import detection_summary, match_predictions, ious
from ownership_audit import cell_masks, owner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"evaluation/native_selection"
REFERENCE = ROOT/"analysis_runs/per_cell_v1"
MODELS = {
    "baseline640": (ROOT.parent/"results/runs/detect/train/weights/best.pt", 640),
    "refined1024": (ROOT.parent/"results/runs/detect/runs/yolov10n_refined/weights/best.pt", 1024),
    "native2048": (ROOT/"training_runs/native2048/weights/best.pt", 2048),
    "fine_tuned2048": (ROOT/"training_runs/refined_to2048/weights/best.pt", 2048),
}
GT_BOX = ["annotation_x1","annotation_y1","annotation_x2","annotation_y2"]


def predict(name, path, size, manifest):
    model = YOLO(str(path))
    rows = []
    for source, subset in manifest[["source_id","subset"]].itertuples(index=False):
        folder = ROOT.parent/"channel2_input2"
        polygons = cell_masks(json.loads((folder/"json"/(source+".json")).read_text()))
        result = model.predict(str(folder/(source+".tif")), imgsz=size, conf=.001, iou=.7,
                               max_det=300, device="cpu", verbose=False)[0]
        assert result.orig_shape == (2048,2048)
        for x1,y1,x2,y2,score,cls in result.boxes.data.cpu().numpy():
            if cls!=0:
                continue
            extent=[max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),
                    min(2048,int(np.ceil(x2))+1),min(2048,int(np.ceil(y2))+1)]
            rows.append(dict(source_id=source,subset=subset,member=name,x1=x1,y1=y1,x2=x2,y2=y2,
                             yolo_score=float(score),**owner(extent,polygons)))
    pred=pd.DataFrame(rows)
    pred.to_csv(OUT/(name+"_native.csv"),index=False)
    print("Native inference finished:",name,len(manifest),"sources",flush=True)
    return pred


def object_scores(pred, references, box_names):
    scores = []
    for source, group in references.groupby("source_id"):
        pp=pred[pred.source_id==source]
        overlaps=ious(group[box_names], pp[BOXES])
        scores.extend(np.where(overlaps>=.5,pp.yolo_score.to_numpy(),0).max(axis=1,initial=0))
    return np.array(scores)


def evaluate(name, pred, gt, cells, controls):
    pred=pred.reset_index(drop=True)
    cal=pred[pred.subset=="calibration"]
    test=pred[pred.subset=="heldout_development"]
    calgt=gt[gt.subset=="calibration"]
    truth=gt[gt.subset=="heldout_development"]
    tp,_=match_predictions(cal,calgt,"yolo_score")
    false=cal.loc[(~tp)&(cal.owner_status=="reliable"),"yolo_score"].sort_values(ascending=False).to_numpy()
    allowed=int(.1*(cells.subset=="calibration").sum())
    # Smallest three-decimal cutoff strictly above the first disallowed score.
    threshold=max(.05,(np.floor(false[allowed]*1000)+1)/1000) if len(false)>allowed else .05
    metrics,_=detection_summary(test,truth,"yolo_score",threshold)
    chosen=test[test.yolo_score>=threshold]
    tp,_=match_predictions(chosen,truth,"yolo_score")
    positive_hits=int(tp.sum())
    fp=len(chosen)-positive_hits
    fn=len(truth)-positive_hits
    cohort=cells[(cells.subset=="heldout_development")&cells.ownership_resolved].set_index(KEY)
    estimated=chosen[chosen.owner_status=="reliable"].groupby(KEY).size().reindex(cohort.index,fill_value=0)
    error=estimated-cohort.certain_annotation_count
    # ROC on a specified positive/negative population, with zero for locations without a proposal.
    pos=object_scores(test,truth,GT_BOX)
    neg=object_scores(test,controls[controls.subset=="heldout_development"],BOXES)
    n,m=len(pos),len(neg)
    auc=(rankdata(np.r_[pos,neg])[:n].sum()-n*(n+1)/2)/(n*m)
    result=dict(method=name,threshold=float(threshold),n_cells=len(cohort),count_MAE=float(error.abs().mean()),
                count_RMSE=float(np.sqrt(np.mean(error**2))),count_bias=float(error.mean()),
                precision=metrics["precision_at_threshold"],recall=metrics["recall_at_threshold"],
                f1=2*positive_hits/(2*positive_hits+fp+fn),faint_recall=metrics["contrast_groups"]["low"]["recall"],
                AP50=metrics["AP50_proposal_ranking"],restricted_object_ROC_AUC=float(auc),
                ROC_cytosolic=n,ROC_centrosomal=m,matched=positive_hits,unmatched=fp,missed=fn,
                unmatched_owned_per_cell=float(((~tp)&(chosen.owner_status=="reliable")).sum()/(cells.subset=="heldout_development").sum()))
    count_table=pd.DataFrame(dict(method=name,reference=cohort.certain_annotation_count,predicted=estimated,error=error)).reset_index()
    return result,count_table


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    manifest=pd.read_csv(REFERENCE/"image_manifest.csv")
    manifest=manifest[manifest.subset.isin(["calibration","heldout_development"])]
    assert manifest.groupby("subset").size().to_dict()=={"calibration":20,"heldout_development":25}
    gt=pd.read_csv(REFERENCE/"cytosolic_ownership.csv")
    cells=pd.read_csv(REFERENCE/"cell_annotation_counts.csv")
    controls=pd.read_csv(REFERENCE/"annotated_centrosomal_controls.csv")
    design=dict(selection="Lowest development per-cell MAE; RMSE as tie-breaker. This is model selection on reused development images, not an independent test.",
        calibration="20 training-side sources. For each policy cap unmatched owned predictions at 0.1 per annotated calibration cell, using a three-decimal threshold.",
        candidates="Four single full-label checkpoints with and without cyto NMS at IoU >0.5, and either 2048 checkpoint combined with refined1024 using the same NMS.",
        counting="Count reliably owned predictions using supplied polygons in all 135 reference-resolved development cells.",
        ROC="Annotated cytosolic regions versus annotated nonoverlapping-patch centrosomal controls. Region score = largest retained cyto proposal score at box IoU >=0.5, else 0. Ties receive half credit. Not a field-wide detector ROC.",
        weights={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size=s) for k,(p,s) in MODELS.items()})
    (OUT/"design.json").write_text(json.dumps(design,indent=2)+"\n")
    models={k:predict(k,p,s,manifest) for k,(p,s) in MODELS.items()}
    methods=dict(models)
    methods.update({k+"_nms":ensemble(v) for k,v in models.items()})
    for k in ("native2048","fine_tuned2048"):
        methods["refined1024_plus_"+k]=ensemble(models["refined1024"],models[k])
    rows=[];counts=[]
    for name,pred in methods.items():
        row,ct=evaluate(name,pred,gt,cells,controls)
        rows.append(row);counts.append(ct)
    ranking=pd.DataFrame(rows).sort_values(["count_MAE","count_RMSE"])
    ranking.to_csv(OUT/"comparison.csv",index=False)
    pd.concat(counts).to_csv(OUT/"per_cell_counts.csv",index=False)
    (OUT/"selected.json").write_text(json.dumps(ranking.iloc[0].to_dict(),indent=2)+"\n")
    print(ranking[["method","threshold","count_MAE","count_RMSE","restricted_object_ROC_AUC"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
