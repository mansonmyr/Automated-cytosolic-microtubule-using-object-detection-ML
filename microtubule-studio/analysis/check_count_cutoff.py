"""Test a single native-checkpoint cutoff adjustment using the existing 20 calibration sources."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_faint import detection_summary, match_predictions

def evaluate(audit, out):
    pred = pd.read_csv(audit/"all_cytosolic_candidates.csv")
    gt = pd.read_csv(audit/"cytosolic_ownership.csv")
    cells = pd.read_csv(audit/"cell_annotation_counts.csv")
    resolved = cells[cells.ownership_resolved].set_index(["source_id","cell_id"])
    def cell_errors(split, cutoff):
        cohort = resolved[resolved.subset == split]
        chosen = pred[(pred.subset == split) & (pred.owner_status == "reliable") & (pred.yolo_score >= cutoff)]
        estimates = chosen.groupby(["source_id","cell_id"]).size().reindex(cohort.index, fill_value=0)
        return estimates - cohort.certain_annotation_count
    # Fix grid and tie rule before computing the adjustment's development-image outcomes.
    grid = np.arange(1,100)/100
    losses = [cell_errors("calibration", t).abs().mean() for t in grid]
    threshold = float(max(t for t,l in zip(grid, losses) if l == min(losses)))
    doc = {"criterion":"Minimize mean absolute reliable-assignment count error over reference-resolved calibration cells; larger cutoff breaks ties. Ambiguous predictions are deferred.",
           "calibration_sources":int(cells[cells.subset=="calibration"].source_id.nunique()),
           "calibration_cells":len(cell_errors("calibration",.25)),"grid":"0.01 to 0.99 in steps of 0.01",
           "selection_exposure":"Calibration sources were in native detector training. Evaluate the frozen cutoff on the separate development sources.",
           "results":{}}
    hold = pred[pred.subset == "heldout_development"]
    ref = gt[gt.subset == "heldout_development"]
    n_cells = int((cells.subset == "heldout_development").sum())
    for label,t in [("original",.25),("count_selected",threshold)]:
        err=cell_errors("heldout_development",t)
        metrics,_=detection_summary(hold,ref,"yolo_score",t)
        accepted = hold[hold.yolo_score >= t]
        tp,_ = match_predictions(accepted,ref,"yolo_score")
        doc["results"][label]={"threshold":t,"calibration_MAE":float(cell_errors("calibration",t).abs().mean()),
                             "development_cells":len(err),"development_MAE":float(err.abs().mean()),
                             "development_bias":float(err.mean()),"faint_recall":metrics["contrast_groups"]["low"]["recall"],
                             "precision":metrics["precision_at_threshold"],"recall":metrics["recall_at_threshold"],
                             "deferred_ambiguous_predictions":int((accepted.owner_status=="ambiguous").sum()),
                             "unmatched_owned_per_cell":float(((~tp)&(accepted.owner_status=="reliable")).sum()/n_cells)}
    err=cell_errors("heldout_development",threshold).abs()-cell_errors("heldout_development",.25).abs()
    agg=err.groupby(level=0).agg(["sum","count"]).reindex(sorted(ref.source_id.unique()),fill_value=0)
    rng=np.random.default_rng(42);draws=rng.integers(len(agg),size=(4000,len(agg)))
    sums=agg.to_numpy()[draws].sum(axis=1);valid=sums[:,1]>0
    doc["development_MAE_change_source_bootstrap_95CI"]=np.quantile(sums[valid,0]/sums[valid,1],[.025,.975]).tolist()
    out.write_text(json.dumps(doc, indent=2,allow_nan=False)+"\n")

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--audit",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path); a=p.parse_args(); evaluate(a.audit,a.output)
