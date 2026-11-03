"""Compare saved per-cell audits on the same source images and the same resolved cells."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_faint import ious, match_predictions

SCORES = {"yolo_default": "yolo_score", "yolo_rate_matched": "yolo_score",
          "filament_rate_matched": "log_likelihood_ratio"}
CELL_KEY = ["source_id", "cell_id"]


def paired_interval(numerators, denominators, draws):
    num = np.asarray(numerators, dtype=float)[draws].sum(axis=1)
    den = np.asarray(denominators, dtype=float)[draws].sum(axis=1)
    return np.quantile(num[den > 0] / den[den > 0], [.025, .975]).tolist()


def owned_curve(predictions, gt, score, cell_count):
    # A hit must match a reference in the same cell. Tied scores enter as a group.
    pred = predictions[predictions.owner_status == "reliable"].copy()
    ref = gt[gt.owner_status == "reliable"].copy()
    for frame in (pred, ref):
        frame["source_id"] = frame.source_id + "/" + frame.cell_id
    tp, matched = match_predictions(pred, ref, score)
    low = set(ref.index[ref.contrast_group == "low"])
    pred["unmatched"] = (~tp).astype(int)
    pred["faint_hit"] = [int(matched.get(i) in low) for i in pred.index]
    events = pred.groupby(score)[["unmatched", "faint_hit"]].sum().sort_index(ascending=False).cumsum()
    allowed = int(.1 * cell_count)
    eligible = events[events.unmatched <= allowed]
    fp, hits = map(int, eligible.iloc[-1]) if len(eligible) else (0, 0)
    return {"unmatched_budget": allowed, "annotated_cells": cell_count, "unmatched": fp,
            "owned_faint_hits": hits, "owned_faint_positives": len(low), "recall": hits/len(low)}


def compare(inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    detail = {}
    image_frames, count_frames, refs = [], [], {}
    fixed_sources = None
    for name, path in inputs.items():
        gt = pd.read_csv(path/"heldout_detection_failures.csv")
        cells = pd.read_csv(path/"heldout_per_cell_counts.csv")
        pred = pd.read_csv(path/"all_cytosolic_candidates.csv")
        pred = pred[pred.subset == "heldout_development"].copy()
        summary = json.loads((path/"evaluation_summary.json").read_text())
        keys = set(gt.annotation_id)
        if fixed_sources is None:
            fixed_sources = keys
        if keys != fixed_sources:
            raise ValueError(f"Reference annotations differ for {name}")
        sources = sorted(gt.source_id.unique())
        cell_count = len(cells[CELL_KEY].drop_duplicates())
        refs[name] = gt.set_index("annotation_id")
        detail[name] = {"weights_sha256": summary["weights_sha256"], "methods": {},
                        "owned_faint_at_matched_reference_budget": {}}
        for score in set(SCORES.values()):
            detail[name]["owned_faint_at_matched_reference_budget"][score] = owned_curve(pred, gt, score, cell_count)
        for method, score in SCORES.items():
            result = summary["methods"][method]
            selected = pred[pred[score] >= result["threshold"]]
            tp, _ = match_predictions(selected, gt, score)
            unmatched = int(((~tp) & (selected.owner_status == "reliable")).sum())
            totals = gt.groupby("source_id").size().reindex(sources)
            found = gt.groupby("source_id")["detected_" + method].sum().reindex(sources)
            estimates = selected.groupby("source_id").size().reindex(sources, fill_value=0)
            errors = estimates - totals
            image_frames.append(pd.DataFrame(dict(model=name, method=method, source_id=sources,
                                      reference=totals.values, predicted=estimates.values,
                                      detected=found.values, count_error=errors.values)))
            count_frames.append(cells[cells.method == method].assign(model=name))
            detail[name]["methods"][method] = {
                "threshold": result["threshold"], "precision": result["precision_at_threshold"],
                "recall": result["recall_at_threshold"], "contrast": result["contrast_groups"],
                "image_count_MAE": float(errors.abs().mean()), "image_count_bias": float(errors.mean()),
                "unmatched_owned_per_annotated_cell": unmatched/cell_count}
        # Diagnose fixed-score misses, relative to all retained cytosolic proposals at this input size.
        missed = gt[~gt.detected_yolo_default]
        causes = {"no_proposal_at_IoU_0.1": 0, "best_IoU_0.1_to_below_0.5": 0,
                  "overlap_0.5_only_below_score_0.25": 0, "overlap_0.5_competing_match": 0}
        for source, group in missed.groupby("source_id"):
            pp = pred[pred.source_id == source]
            matrix = ious(group[["annotation_x1","annotation_y1","annotation_x2","annotation_y2"]], pp[["x1","y1","x2","y2"]])
            for row in matrix:
                best = row.max(initial=0)
                if best < .1:
                    reason = "no_proposal_at_IoU_0.1"
                elif best < .5:
                    reason = "best_IoU_0.1_to_below_0.5"
                elif not np.any((row >= .5) & (pp.yolo_score.to_numpy() >= .25)):
                    reason = "overlap_0.5_only_below_score_0.25"
                else:
                    reason = "overlap_0.5_competing_match"
                causes[reason] += 1
        detail[name]["misses_at_0.25"] = causes

    all_cells = pd.concat(count_frames, ignore_index=True)
    all_images = pd.concat(image_frames, ignore_index=True)
    expected = len(inputs)*len(SCORES)
    keyed = all_cells.groupby(CELL_KEY)
    if not (keyed.size() == expected).all() or (keyed.reference_lower.nunique() != 1).any():
        raise ValueError("Cell references differ across models or methods")
    keep = (keyed.ownership_resolved.all() & (keyed.ambiguous_predictions.max() == 0)).rename("included")
    common = all_cells.merge(keep.reset_index(), on=CELL_KEY)
    common["error"] = common.predicted_lower - common.reference_lower
    common["absolute_error"] = common.error.abs()
    kept = common[common.included]
    for (name, method), group in kept.groupby(["model", "method"]):
        detail[name]["methods"][method].update(common_cells=len(group), cell_MAE=float(group.absolute_error.mean()),
                                               cell_bias=float(group.error.mean()))
    draws = np.random.default_rng(42).integers(0, len(sources), (4000, len(sources)))
    pairs = []
    for name in inputs:
        current = refs[name]
        lo = current[current.contrast_group == "low"].groupby("source_id").detected_yolo_default.agg(["sum","count"]).reindex(sources,fill_value=0).to_numpy()
        hi = current[current.contrast_group == "high"].groupby("source_id").detected_yolo_default.agg(["sum","count"]).reindex(sources,fill_value=0).to_numpy()
        a,b = lo[draws].sum(axis=1), hi[draws].sum(axis=1)
        ok = (a[:,1]>0) & (b[:,1]>0)
        detail[name]["low_minus_high_recall_95CI"] = np.quantile(a[ok,0]/a[ok,1] - b[ok,0]/b[ok,1],[.025,.975]).tolist()
        base = next(iter(inputs))
        if name == base:
            continue
        for method in SCORES:
            detections = "detected_" + method
            diff = current[current.contrast_group == "low"][["source_id", detections]].copy()
            diff["delta"] = diff[detections].astype(int) - refs[base].loc[diff.index,detections].astype(int)
            agg = diff.groupby("source_id").delta.agg(["sum","count"]).reindex(sources,fill_value=0)
            wide = kept[kept.method == method].pivot(index=CELL_KEY, columns="model",values="absolute_error")
            ce = (wide[name]-wide[base]).groupby(level="source_id").agg(["sum","count"]).reindex(sources,fill_value=0)
            pairs.append({"model": name, "minus": base, "method":method,
                          "faint_recall_difference":float(diff.delta.mean()),
                          "faint_95CI":paired_interval(agg["sum"],agg["count"],draws),
                          "cell_MAE_difference":float(ce["sum"].sum()/ce["count"].sum()),
                          "cell_MAE_95CI":paired_interval(ce["sum"],ce["count"],draws)})
    common.to_csv(output/"paired_cell_errors.csv",index=False)
    all_images.to_csv(output/"image_count_errors.csv",index=False)
    (output/"paired_comparison.json").write_text(json.dumps({"models":detail,"paired_source_bootstrap":pairs}, indent=2, allow_nan=False)+"\n")
    lines = ["# Paired cytosolic detection comparison", "",
             "All rows use the same 25 development sources and the supplied cell polygons. The per-cell comparison includes only cells with resolved reference counts and no ambiguously assigned detections under any compared policy.", "",
             "| Checkpoint | Score cutoff | Cytosolic recall | Low-contrast recall | Count MAE | Common cells |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, entry in detail.items():
        r = entry["methods"]["yolo_default"]
        lines.append(f"| {name} | {r['threshold']:.2f} | {r['recall']:.1%} | {r['contrast']['low']['recall']:.1%} | {r['cell_MAE']:.2f} | {r['common_cells']} |")
    lines.extend(["", "At a fixed budget of 16 unmatched owned proposals across the 166 reference cells, detector-score ranking yielded the following low-contrast recall among annotations with a reliable owner.", ""])
    for name, entry in detail.items():
        result = entry["owned_faint_at_matched_reference_budget"]["yolo_score"]
        lines.append(f"- {name} recovered {result['owned_faint_hits']}/{result['owned_faint_positives']} ({result['recall']:.1%}) with {result['unmatched']} reference-unmatched proposals.")
    lines.extend(["", "The comparison uses existing annotations. An unmatched proposal can contain a real, unlabelled structure. The fixed-budget curve describes these development images. Thresholds read off that curve still need separate validation.", ""])
    for pair in pairs:
        if pair["method"] != "yolo_default":
            continue
        lo, hi = pair["faint_95CI"]
        low_mae, high_mae = pair["cell_MAE_95CI"]
        lines.append(f"At score 0.25, {pair['model']} minus {pair['minus']} changed faint recall by {100*pair['faint_recall_difference']:+.1f} percentage points (paired source-bootstrap 95% interval {100*lo:+.1f} to {100*hi:+.1f}). Count MAE changed by {pair['cell_MAE_difference']:+.2f} objects per cell (95% interval {low_mae:+.2f} to {high_mae:+.2f}).")
        lines.append("")
    (output/"paired_comparison.md").write_text("\n".join(lines)+"\n")



if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", nargs=2, action="append", metavar=("NAME","DIRECTORY"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    compare({k:Path(v) for k,v in args.audit}, args.output)
