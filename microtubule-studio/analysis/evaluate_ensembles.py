"""Exploratory ensemble check using cached whole-image predictions in original coordinates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_checkpoints import owned_curve, paired_interval
from evaluate_faint import detection_summary, ious, match_predictions

PROJECT=Path(__file__).resolve().parents[1]
OUTPUT=PROJECT/"evaluation/ensemble_pilot"
BOXES=["x1","y1","x2","y2"]
KEY=["source_id","cell_id"]


def suppress(group, threshold=.5):
    """Retain the highest-scoring box from an overlapping group, regardless of model."""
    group=group.sort_values("yolo_score",ascending=False,kind="stable")
    boxes=group[BOXES].to_numpy()
    keep=[]
    pending=np.arange(len(group))
    while len(pending):
        first=pending[0]
        keep.append(first)
        remaining=pending[1:]
        overlap=ious(boxes[first:first+1],boxes[remaining])[0]
        pending=remaining[overlap<=threshold]
    return group.iloc[keep]


def ensemble(a,b=None):
    rows=pd.concat([a,b],ignore_index=True) if b is not None else a.copy()
    return pd.concat([suppress(group) for _,group in rows.groupby("source_id")],ignore_index=True)


def main():
    OUTPUT.mkdir(exist_ok=True,parents=True)
    folders={"refined1024":PROJECT/"analysis_runs/refined1024",
             "native2048":PROJECT/"analysis_runs/native2048",
             "fine_tuned2048":PROJECT/"evaluation/refined_to2048/audit"}
    predictions={}
    provenance={}
    for name,folder in folders.items():
        path=folder/"all_cytosolic_candidates.csv"
        pred=pd.read_csv(path).fillna({"cell_id":"","candidate_cells":""})
        pred["member"]=name
        pred=pred[pred.subset.isin(["calibration","heldout_development"])].copy()
        assert np.isfinite(pred[BOXES+["yolo_score"]].to_numpy()).all()
        predictions[name]=pred
        provenance[name]={"predictions_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                          "weights_sha256":json.loads((folder/"evaluation_summary.json").read_text())["weights_sha256"]}
    gt=pd.read_csv(PROJECT/"analysis_runs/per_cell_v1/cytosolic_ownership.csv")
    cells=pd.read_csv(PROJECT/"analysis_runs/per_cell_v1/cell_annotation_counts.csv")
    original=predictions["refined1024"]
    methods={"refined1024":original,"refined1024_nms_control":ensemble(original),
             "refined1024_plus_native2048":ensemble(original,predictions["native2048"]),
             "refined1024_plus_fine_tuned2048":ensemble(original,predictions["fine_tuned2048"])}
    # These rules are fixed before reading the ensemble's development outcomes.
    design={"members":provenance,"fusion":"Union cytosolic candidates; confidence-ordered NMS at IoU >0.5, same original source coordinates. Coordinates and winning member scores are retained. Stable equal-score ties prefer refined1024.",
            "policies":["fixed confidence >=0.25","Training-side calibration cap <=0.1 unmatched reliably owned proposal per annotated cell"],
            "calibration_sources":20,"development_sources":25,"actual_independent_calibration":False,
            "reference_budget_curve":"Retrospective development curve at <=16 unmatched owned proposals over 166 annotated cells; scores matched to GT within the same cell.",
            "count_cohort":"All reference-resolved development cells, fixed before comparing methods. Count reliably owned predictions; defer ambiguously owned ones.",
            "limitations":["Calibration sources were seen during detector training.","The 25 development sources were previously used for model selection.",
                           "Unmatched reference proposals can be real unlabelled objects.","Counts use supplied cell polygons, not automatically detected cell boundaries.",
                           "Native and refined models have different training and restart histories. This pilot compares available checkpoints.",
                           "Scores from different members are uncalibrated ranking values."]}
    (OUTPUT/"design.json").write_text(json.dumps(design,indent=2)+"\n")
    calgt=gt[gt.subset=="calibration"]
    testgt=gt[gt.subset=="heldout_development"]
    cal_cells=int((cells.subset=="calibration").sum())
    test_cells=int((cells.subset=="heldout_development").sum())
    sources=sorted(testgt.source_id.unique())
    draws=np.random.default_rng(2026).integers(len(sources),size=(4000,len(sources)))
    summary={}; found_sets={}; errors={}; count_rows=[]
    for name,pred in methods.items():
        pred=pred.reset_index(drop=True)
        cal=pred[pred.subset=="calibration"]
        test=pred[pred.subset=="heldout_development"]
        tp,_=match_predictions(cal,calgt,"yolo_score")
        unmatched=cal.loc[(~tp)&(cal.owner_status=="reliable"),"yolo_score"].sort_values(ascending=False).to_numpy()
        allowed=int(np.floor(.1*cal_cells))
        cutoff=float(np.nextafter(unmatched[allowed],np.inf)) if len(unmatched)>allowed else .001
        summary[name]={"policies":{},"matched_reference_budget":owned_curve(test,testgt,"yolo_score",test_cells)}
        for policy,threshold in [("fixed_025",.25),("calibration_policy",cutoff)]:
            selected=test[test.yolo_score>=threshold]
            metrics,found=detection_summary(test,testgt,"yolo_score",threshold)
            matched_tp,_=match_predictions(selected,testgt,"yolo_score")
            cal_selected=cal[cal.yolo_score>=threshold]
            cal_tp,_=match_predictions(cal_selected,calgt,"yolo_score")
            cohort=cells[(cells.subset=="heldout_development")&cells.ownership_resolved].set_index(KEY)
            estimates=selected[selected.owner_status=="reliable"].groupby(KEY).size().reindex(cohort.index,fill_value=0)
            err=estimates-cohort.certain_annotation_count
            errors[name,policy]=err.abs()
            found_sets[name,policy]=found
            metrics.update(count_cells=len(cohort),cell_MAE=float(err.abs().mean()),cell_bias=float(err.mean()),
                           ambiguous_predictions_deferred=int((selected.owner_status=="ambiguous").sum()),
                           unmatched_owned_per_cell=float(((~matched_tp)&(selected.owner_status=="reliable")).sum()/test_cells),
                           calibration_unmatched_owned_per_cell=float(((~cal_tp)&(cal_selected.owner_status=="reliable")).sum()/cal_cells))
            summary[name]["policies"][policy]=metrics
            for (source,cell),error in err.items():
                count_rows.append(dict(method=name,policy=policy,source_id=source,cell_id=cell,
                                       error=int(error),absolute_error=abs(int(error))))
        pred.to_csv(OUTPUT/(name+"_candidates.csv"),index=False)
        print("Scored",name,flush=True)
    contrasts=[]
    faint=testgt[testgt.contrast_group=="low"]
    for name in methods:
        if name=="refined1024":
            continue
        for policy in ["fixed_025","calibration_policy"]:
            base=found_sets["refined1024",policy]; other=found_sets[name,policy]
            change=pd.DataFrame({"source_id":faint.source_id,
                     "delta":[int(i in other)-int(i in base) for i in faint.index]})
            agg=change.groupby("source_id").delta.agg(["sum","count"]).reindex(sources,fill_value=0)
            diff=errors[name,policy]-errors["refined1024",policy]
            counts=diff.groupby(level="source_id").agg(["sum","count"]).reindex(sources,fill_value=0)
            contrasts.append(dict(method=name,policy=policy,faint_recall_change=float(change.delta.mean()),
                    faint_recall_change_95CI=paired_interval(agg["sum"],agg["count"],draws),
                    count_MAE_change=float(diff.mean()),
                    count_MAE_change_95CI=paired_interval(counts["sum"],counts["count"],draws)))
    # Individual-model hit sets describe complementary detections before fusion.
    complement={}
    _,base_hits=detection_summary(original[original.subset=="heldout_development"],testgt,"yolo_score",.25)
    for name,pred in predictions.items():
        if name=="refined1024":continue
        _,hits=detection_summary(pred[pred.subset=="heldout_development"],testgt,"yolo_score",.25)
        complement[name]=dict(individual_hits=len(hits),hits_missing_from_1024=len(hits-base_hits),
                               faint_hits_missing_from_1024=sum(i in hits-base_hits for i in faint.index),
                               note="Complementarity of separately matched boxes, not an achievable detection guarantee.")
    doc=dict(methods=summary,paired_source_bootstrap=contrasts,complementary_annotations=complement)
    (OUTPUT/"results.json").write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(count_rows).to_csv(OUTPUT/"per_cell_errors.csv",index=False)
    lines=["# Ensemble pilot on the existing source images","",
           "The 1024 and 2048 predictions were mapped to original image coordinates before merging. A score-ordered union removes boxes with IoU above 0.5. Cytosolic boxes only were tested. The NMS-only control measures the effect of overlap removal without an additional model.","",
           "| Method | Policy | Threshold | Recall | Faint recall | Count MAE | Unmatched owned / cell |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for name,item in summary.items():
        for policy,r in item["policies"].items():
            lines.append(f"| {name} | {policy} | {r['threshold']:.3f} | {r['recall_at_threshold']:.1%} | {r['contrast_groups']['low']['recall']:.1%} | {r['cell_MAE']:.2f} | {r['unmatched_owned_per_cell']:.2f} |")
    lines+=["","All count rows use the same reference-resolved cells. Ambiguous assignments are deferred. This cohort differs from previous comparisons that excluded any cell with an ambiguous accepted prediction.","",
            "| Method | Faint hits at common reference budget | Faint owned positives |",
            "|---|---:|---:|"]
    for name,item in summary.items():
        r=item["matched_reference_budget"]
        lines.append(f"| {name} | {r['owned_faint_hits']} | {r['owned_faint_positives']} |")
    lines+=["","The common curve budget is 16 unmatched, reliably owned proposals over 166 annotated cells. It describes development-set ranking, and does not establish a biological false-positive rate or a deployable cutoff.",""]
    for name,r in complement.items():
        lines.append(f"{name} individually recovered {r['hits_missing_from_1024']} annotations missed by 1024 at score 0.25, including {r['faint_hits_missing_from_1024']} low-contrast annotations.")
        lines.append("")
    for r in contrasts:
        if r["policy"]!="fixed_025":continue
        a,b=r["faint_recall_change_95CI"];c,d=r["count_MAE_change_95CI"]
        lines.append(f"At score 0.25, {r['method']} minus refined1024 changed faint recall by {100*r['faint_recall_change']:+.1f} percentage points (source-bootstrap 95% interval {100*a:+.1f} to {100*b:+.1f}). Count MAE changed by {r['count_MAE_change']:+.2f} (interval {c:+.2f} to {d:+.2f}).")
        lines.append("")
    lines+=["Both detectors had seen the 20 calibration sources during training. The 25 development sources had previous model-selection exposure. Scores across models need separate calibration for deployment. The pilot uses manually supplied cell polygons. Automated cell segmentation remains a separate part of the intended per-cell application.","",
            "A two-member ensemble requires two inference passes. Browser execution, memory and coordinate mapping have not been tested by this offline pilot."]
    (OUTPUT/"Ensemble_Pilot.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
