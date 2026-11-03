"""Rebuild release figures from recorded research outputs, without rerunning inference.

Run from the organized workspace: python3 01_GITHUB_UPLOAD/microtubule-studio/analysis/build_release_figures.py --research 04_RESEARCH_WORKSPACE
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mt-release-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_ensembles import ensemble, BOXES
from evaluate_faint import ious, match_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, required=True)
    args = parser.parse_args()
    release = HERE.parents[1]
    metrics = release / 'microtubule-studio/metrics'
    figures = release / 'docs/figures'
    outputs = release / 'docs/outputs'
    figures.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    research = args.research.resolve()
    cached = research / 'microtubule-studio/evaluation/native_selection/refined1024_native.csv'
    refroot = research / 'microtubule-studio/analysis_runs/per_cell_v1'
    pred = ensemble(pd.read_csv(cached))
    pred = pred[pred.subset == 'heldout_development'].reset_index(drop=True)
    gt = pd.read_csv(refroot / 'cytosolic_ownership.csv')
    gt = gt[gt.subset == 'heldout_development']
    controls = pd.read_csv(refroot / 'annotated_centrosomal_controls.csv')
    controls = controls[controls.subset == 'heldout_development']
    selected = json.loads((metrics / 'selected.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 140, 'savefig.dpi': 180})
    blue, pink = '#244e89', '#d14e8a'
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / name, bbox_inches='tight')
        plt.close(fig)

    # Scores follow select_native_detector.object_scores, including zero-scored misses.
    def scores(ref, columns):
        rows = []
        for source, group in ref.groupby('source_id'):
            p = pred[pred.source_id == source]
            values = np.where(ious(group[columns], p[BOXES]) >= .5,
                              p.yolo_score.to_numpy(), 0).max(axis=1, initial=0)
            rows.extend({'source_id': source, 'score': float(s)} for s in values)
        return pd.DataFrame(rows)
    pos = scores(gt, ['annotation_x1', 'annotation_y1', 'annotation_x2', 'annotation_y2'])
    neg = scores(controls, BOXES)
    pos['positive'] = 1
    neg['positive'] = 0
    regions = pd.concat([pos, neg], ignore_index=True)
    regions.to_csv(outputs / 'restricted_roc_region_scores.csv', index=False)
    p, n = pos.score.to_numpy(), neg.score.to_numpy()
    auc = (rankdata(np.r_[p, n])[:len(p)].sum() - len(p)*(len(p)+1)/2)/(len(p)*len(n))
    assert np.isclose(auc, selected['restricted_object_ROC_AUC'], atol=1e-12)
    ordered = regions.sort_values('score', ascending=False, kind='stable')
    end = ordered.score.ne(ordered.score.shift(-1)).to_numpy()
    tpr = np.r_[0, ordered.positive.cumsum().to_numpy()[end]/len(p)]
    fpr = np.r_[0, (1-ordered.positive).cumsum().to_numpy()[end]/len(n)]
    pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_csv(outputs/'restricted_roc_curve.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.plot(fpr, tpr, color=blue, label=f'Restricted AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set(xlabel='False acceptance fraction among centrosomal controls',
           ylabel='Accepted fraction of annotated cytosolic regions',
           title='Restricted annotated-object ROC', xlim=(0,1), ylim=(0,1.02))
    ax.legend(loc='lower right')
    fig.text(.5, -.02, f'{len(p):,} cytosolic regions; {len(n)} centrosomal controls. Development sources.\nNot a field-wide specificity estimate.', ha='center', fontsize=9)
    save(fig, 'restricted_roc.png')

    tp, _ = match_predictions(pred, gt, 'yolo_score')
    order = pred.sort_values('yolo_score', ascending=False).index
    ctp = np.cumsum(tp.loc[order].to_numpy())
    precision = ctp/np.arange(1, len(order)+1)
    recall = ctp/len(gt)
    ap = np.mean([precision[recall >= q].max() if np.any(recall >= q) else 0 for q in np.linspace(0,1,101)])
    assert np.isclose(ap, selected['AP50'], atol=1e-12)
    pd.DataFrame({'score': pred.loc[order, 'yolo_score'], 'precision': precision, 'recall': recall}).to_csv(outputs/'selected_pr_curve.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(recall, precision, color=blue, label=f'101-point AP50 = {ap:.4f}')
    ax.scatter([selected['recall']], [selected['precision']], color=pink, zorder=3, label='Confidence 0.174')
    ax.set(xlabel='Recall against cytosolic annotations', ylabel='Precision against cytosolic annotations', title='Selected pipeline precision–recall', xlim=(0,1), ylim=(0,1.02))
    ax.legend()
    save(fig, 'selected_precision_recall.png')

    comparison = pd.read_csv(metrics/'comparison.csv').sort_values('count_MAE')
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(comparison.method, comparison.count_MAE, color=[pink]+[blue]*9)
    for k, val in enumerate(comparison.count_MAE):
        ax.text(val+.07, k, f'{val:.2f}', va='center', fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel='Mean absolute count error (microtubules/cell)', title='Ten deployment policies: 135 reference-resolved development cells', xlim=(0,9))
    save(fig, 'policy_comparison.png')

    counts = pd.read_csv(metrics/'per_cell_counts.csv')
    counts = counts[counts.method == selected['method']]
    counts.to_csv(outputs/'selected_per_cell_counts.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(counts.reference, counts.predicted, color=blue, alpha=.6, s=24)
    lim = max(counts.reference.max(), counts.predicted.max())+2
    axes[0].plot([0,lim], [0,lim], '--', color='gray')
    axes[0].set(xlabel='Reference cytosolic count', ylabel='Predicted cytosolic count', title='Counts in 135 resolved cells', xlim=(0,lim), ylim=(0,lim))
    axes[1].hist(counts.error, bins=np.arange(counts.error.min()-.5, counts.error.max()+1.5), color=blue)
    axes[1].axvline(0, color=pink, linestyle='--')
    axes[1].set(xlabel='Prediction − reference (microtubules/cell)', ylabel='Cells', title=f'MAE {selected["count_MAE"]:.2f}; bias {selected["count_bias"]:.2f}')
    save(fig, 'per_cell_counts.png')

    train = pd.read_csv(metrics/'refined1024-training.csv')
    train.columns = train.columns.str.strip()
    fig, axes = plt.subplots(1, 2, figsize=(10,4.1))
    for col, label in [('train/box_loss', 'Box'), ('train/cls_loss', 'Classification'), ('train/dfl_loss', 'Distribution focal')]:
        axes[0].plot(train.epoch, train[col], label=label)
    for col, label in [('metrics/mAP50(B)', 'All-class mAP50'), ('metrics/mAP50-95(B)', 'All-class mAP50–95')]:
        axes[1].plot(train.epoch, train[col], label=label)
    axes[0].set(xlabel='Epoch', ylabel='Training loss', title='Recorded refined1024 training')
    axes[1].set(xlabel='Epoch', ylabel='Validation AP', title='Historical training-time validation')
    for ax in axes: ax.legend(fontsize=8)
    save(fig, 'training_history.png')

    original = research/'results/runs/detect/runs/yolov10n_refined'
    for name in ['confusion_matrix.png', 'confusion_matrix_normalized.png', 'BoxPR_curve.png', 'BoxF1_curve.png', 'BoxP_curve.png', 'BoxR_curve.png', 'results.png']:
        shutil.copy2(original/name, figures/('training_'+name))

    # One named example, selected for continuity with the recorded browser check.
    source = 'Experiment-1678_C2_sum'
    from PIL import Image
    image = Image.open(research/'channel2_input2'/f'{source}.tif').convert('RGB')
    chosen = pred[(pred.source_id == source)&(pred.yolo_score >= selected['threshold'])]
    chosen[BOXES+['yolo_score']].to_csv(outputs/(source+'_cytosolic.csv'), index=False)
    fig, axes = plt.subplots(1,2,figsize=(11,5.8))
    for ax in axes: ax.imshow(image); ax.axis('off')
    axes[0].set_title('Input fluorescence image')
    axes[1].set_title(f'{len(chosen)} cytosolic candidates at confidence ≥0.174')
    from matplotlib.patches import Rectangle
    for row in chosen.itertuples():
        axes[1].add_patch(Rectangle((row.x1,row.y1),row.x2-row.x1,row.y2-row.y1, fill=False, edgecolor='#ff8cbd', linewidth=.65))
    fig.suptitle(source+' · saved native predictions after overlap suppression', fontsize=11)
    save(fig, 'example_detection.png')
    inputs = [cached, refroot/'cytosolic_ownership.csv', refroot/'annotated_centrosomal_controls.csv']
    (outputs/'figure_provenance.json').write_text(json.dumps({'restricted_ROC_AUC':auc, 'AP50':ap, 'inputs_sha256':{str(p.relative_to(research)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'note':'Recomputed from cached development predictions; no new inference or independent testing.'},indent=2)+'\n')
    print(f'Figures created. ROC AUC={auc:.12f}; AP50={ap:.12f}; example={len(chosen)} candidates.')


if __name__ == '__main__':
    main()
