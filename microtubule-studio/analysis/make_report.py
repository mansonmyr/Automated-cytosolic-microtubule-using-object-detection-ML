"""Create the audit report and standalone scientific figures from recorded results."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/per-cell-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from evaluate_faint import match_predictions


def main():
    output=Path(__file__).resolve().parents[1]/'analysis_runs/per_cell_v1'
    audit=json.loads((output/'audit_summary.json').read_text());evaluation=json.loads((output/'evaluation_summary.json').read_text());supp=json.loads((output/'supplementary_diagnostics.json').read_text());fit=json.loads((output/'likelihood_model.json').read_text())
    objects=pd.read_csv(output/'cytosolic_ownership.csv');pred=pd.read_csv(output/'all_cytosolic_candidates.csv');counts=pd.read_csv(output/'heldout_per_cell_counts.csv');noise=pd.read_csv(output/'intensity_diagnostics.csv');manifest=pd.read_csv(output/'image_manifest.csv')
    # Both the calibration and held-out operational rates remain reference-unmatched proxies.
    rates={}
    for method,score in [('yolo_default','yolo_score'),('yolo_rate_matched','yolo_score'),('filament_rate_matched','log_likelihood_ratio')]:
        threshold=evaluation['methods'][method]['threshold'];rates[method]={}
        for subset in ['calibration','heldout_development']:
            selected=pred[(pred.subset==subset)&(pred[score]>=threshold)];gt=objects[objects.subset==subset];tp,_=match_predictions(selected,gt,score)
            unmatched=int(((~tp)&(selected.owner_status=='reliable')).sum());celln=int(manifest.loc[manifest.subset==subset,'n_cells'].sum())
            rates[method][subset]={'reference_unmatched_owned_proposals':unmatched,'annotated_cells':celln,'per_cell':unmatched/celln}
    supp['reference_unmatched_rates']=rates
    (output/'supplementary_diagnostics.json').write_text(json.dumps(supp,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8));pink='#c73585';blue='#2c58ab';grey='#767f89'
    groups=['low','medium','high'];x=np.arange(3)
    for k,(method,color,label) in enumerate([('yolo_default',grey,'YOLO score ≥0.25'),('yolo_rate_matched',pink,'YOLO: calibration policy'),('filament_rate_matched',blue,'Filament: calibration policy')]):
        axes[0,0].bar(x+(k-1)*.24,[evaluation['methods'][method]['contrast_groups'][g]['recall'] for g in groups],width=.23,color=color,label=label)
    axes[0,0].set_xticks(x,groups);axes[0,0].set_ylim(0,1);axes[0,0].set_ylabel('Cytosolic detection recall (IoU ≥0.5)');axes[0,0].set_title('Held-out development source images');axes[0,0].legend(fontsize=7)
    common=counts.groupby(['source_id','cell_id']).agg(resolved=('ownership_resolved','all'),ambiguous=('ambiguous_predictions','max'));common=common[common.resolved&(common.ambiguous==0)].reset_index()[['source_id','cell_id']]
    for method,color,label in [('yolo_rate_matched',pink,'YOLO calibration policy'),('filament_rate_matched',blue,'Filament calibration policy')]:
        c=counts[counts.method==method].merge(common,on=['source_id','cell_id']);axes[0,1].scatter(c.reference_lower,c.predicted_lower,s=14,alpha=.55,color=color,label=label)
    limit=float(counts.reference_upper.max()+2);axes[0,1].plot([0,limit],[0,limit],':',color=grey);axes[0,1].set_xlabel('Annotated cytosolic count');axes[0,1].set_ylabel('Predicted owned count');axes[0,1].set_title(f'Paired ownership-resolved cells (n={len(common)})');axes[0,1].legend(fontsize=7)
    axes[1,0].scatter(noise['mean'],noise.variance_mle,s=8,alpha=.3,color=blue);axes[1,0].set_xlabel('Local mean (stored byte-value units)');axes[1,0].set_ylabel('Planar residual variance (MLE)');axes[1,0].set_title('Residual variance is signal-dependent')
    axes[1,1].hist(noise.lag1_correlation.dropna(),bins=30,color=pink,alpha=.8);axes[1,1].axvline(noise.lag1_correlation.median(),color=blue,label=f'Median {noise.lag1_correlation.median():.3f}');axes[1,1].set_xlabel('Horizontal lag-1 residual correlation');axes[1,1].set_ylabel('16 × 16 diagnostic patches');axes[1,1].set_title('Independent-pixel assumption is unsupported');axes[1,1].legend(fontsize=8)
    fig.suptitle('Per-cell cytosolic microtubule audit: exploratory results',fontsize=14)
    fig.tight_layout();fig.savefig(output/'scientific_summary.png',dpi=180);plt.close(fig)
    low=evaluation['methods']['yolo_default']['contrast_groups']['low'];high=evaluation['methods']['yolo_default']['contrast_groups']['high']
    lines=['# Per-cell cytosolic microtubule analysis', '',
    '## Decision', '',
    'Keep all supplied positive annotations, including faint objects. Preserve uncertain ownership and candidate detections for review. The current working likelihood-ratio reranker is not supported as a deployment improvement. No annotation-aware retraining correction has been activated because no missing cytosolic label was verified in this limited visual audit.', '',
    'This is a completed exploratory audit of the existing images, with explicit unresolved annotation questions. It is not independent validation of the model currently training.', '',
    '## Images and cell ownership', '',
    f"There are {audit['source_images']} source images, {audit['cells']} cell polygons and {audit['cytosolic_annotations']} cytosolic annotations. A candidate owner requires a unique centre-containing polygon, at least 80% box coverage and at most 10% coverage by another polygon. Self-crossing cell contours are flagged. Nearest-cell assignment is not used.", '',
    '| Geometric ownership | Cytosolic annotations |','|---|---:|']
    for name in ['reliable','ambiguous','unowned']:lines.append(f"| {name} | {audit['ownership'][name]} |")
    lines += ['', '“Reliable” is a conservative geometric assignment relative to the supplied cell polygons, not independent confirmation of biological cell boundaries. The 337 unresolved annotations remain positive examples. Their candidate cells and coverage fractions are in cytosolic_ownership.csv.', '',
    'One cell contour has a self-intersection: Experiment-1495_C2_sum, cell-3, source shape index 2. The cell table retains marginal lower/upper annotated-count bounds. Upper bounds must not be summed across cells because an ambiguous object can occur in more than one candidate-cell bound.', '',
    '## Split and train-defined contrast', '',
    'The original 80 training images were divided by source file into 60 likelihood-fit images and 20 calibration images using seed 42. The original 25 validation source images are used for held-out development evaluation. They were used for earlier model development and checkpoint selection, so they are not a pristine independent test. No tiles from a source cross these assignments.', '',
    'Contrast is (within-box intensity 90th percentile − local ring median) / max(1, 1.4826 × local ring MAD). All annotated object regions plus a two-pixel margin are excluded from the ring. The ring is restricted to the assigned cell when ownership is resolved. This is an operational local contrast index, not photon SNR.', '',
    f"Cutpoints are the fit-positive tertiles: {audit['contrast_cutpoints_fit_only'][0]:.4f} and {audit['contrast_cutpoints_fit_only'][1]:.4f}. The same cutpoints were frozen for calibration and held-out images. Faint positives and annotations without a usable background ring were retained.", '',
    '| Contrast group | Held-out positives | Recall at YOLO score 0.25 | Miss rate |','|---|---:|---:|---:|']
    for group in groups:
        r=evaluation['methods']['yolo_default']['contrast_groups'][group];lines.append(f"| {group} | {r['positives']} | {r['recall']:.1%} | {1-r['recall']:.1%} |")
    ci=supp['low_minus_high_recall_cluster_bootstrap_95CI']
    lines += ['',f"Low-minus-high recall is {low['recall']-high['recall']:+.1%}; its source-cluster bootstrap 95% interval is [{ci[0]:+.1%}, {ci[1]:+.1%}] (2,000 resamples). These results do not show detection failures concentrated in the low-contrast group under this contrast definition. This is descriptive, not a causal contrast-effect estimate. Object size and crowding can also affect failures. It is conditional on currently labelled objects; verified missing faint labels could change the pattern.", '',
    'Calibration is sparse relative to evaluation: the calibration images contain 104 cytosolic annotations, while the held-out development images contain 1,970. The original detector had trained on the calibration images. Likelihood parameters did not use them. This limits calibration-policy transfer and end-to-end inference.', '',
    '## TIFF intensities and likelihood assumptions', '',
    'Every TIFF is single-page, 2048 × 2048, 8-bit palette-colour data (PhotometricInterpretation=3), saved by ImageJ. The green palette is exactly the identity byte-value mapping. TIFF decoding and the prepared RGB PNGs were checked for pixel equality; a representative prediction comparison was also identical. No gain, offset or raw photon-count calibration is available. Palette indices must not be asserted to be photon counts.', '',
    f"Across 800 sampled unlabelled cell-interior diagnostic patches, median lag-1 planar-residual correlation is {audit['intensity']['median_lag1_correlation']:.3f}. Mean/residual-variance Spearman correlation is {supp['intensity_mean_variance_spearman']:.3f}. The median residual variance/mean ratio is {supp['residual_variance_to_mean_quantiles'][1]:.3f} in stored byte units. These single-image structural diagnostics do not identify camera noise parameters. They do not support a simple independent homoscedastic Gaussian or unit-gain Poisson model. Marginal normality-test p-values are diagnostic only because the observations are correlated.", '',
    'The working hypotheses use a planar background plus a nonnegative isotropic spot (H0), versus a planar background plus a nonnegative oriented anisotropic Gaussian filament (H1). Each patch is 31 × 31 original pixels with no resampling. Planar coefficients and amplitude are fitted by least squares under a nonnegative amplitude constraint. Variance is the constrained Gaussian MLE max(RSS/n, 1/12), with an explicit numerical floor in squared byte-value units. A grid MLE on fit-source annotations chooses the shared filament width/length and null-spot radius. Orientation is profiled per patch.', '',
    f"The selected filament scales are width {fit['line_width_pixels']:.0f} px and length {fit['line_length_scale_pixels']:.0f} px; null spot radius is {fit['null_spot_radius_pixels']:.0f} px. These values touch the fitted grid boundaries and are not reliable physical filament measurements. Correlated residuals and shape misspecification invalidate a nominal chi-square interpretation. Scores are working log-likelihood ratios, not probabilities or truth labels.", '',
    '## Fixed-FPR experiment and end-to-end evaluation', '',
    'The only automatically identified negative controls are existing centrosomal annotations without overlap with a cytosolic annotation in the scored patch. These are restricted known-class controls, not a representative background-negative sample. A descriptive ROC on oracle-centred held-out crops gives faint-positive recall of 63.8% for the filament score at an interpolated 5% centrosomal-control FPR, versus 0% for a contrast-only score. Because the locations are supplied by annotations, this is not a blind detection result.', '',
    'A threshold fixed independently on 59 calibration controls allowed 2/59 false acceptances (3.39%, the conservative empirical operating point below 5%). On 198 held-out centrosomal controls, the filament score accepted 1/198 (0.51%; exact binomial 95% interval approximately 0.01–2.78%) and 283/644 low-contrast positive crops (43.94%). The contrast-only threshold did not transfer: it accepted 109/198 controls (55.05%). A stable field-wide biological FPR has therefore not been established.', '',
    'For operational evaluation, the old YOLOv10n checkpoint generated cytosolic proposals at nominal 640-pixel input, score ≥0.001 and at most 300 detections across classes. Both rerankers use the identical cached proposals. A source-cluster bootstrap was used for recall differences. The 2048-pixel model is still training and is not the model evaluated here.', '',
    'The two calibration policies allow at most 0.1 reference-unmatched, reliably owned proposals per annotated calibration cell. “Unmatched” is not equivalent to a true false positive when labels may be missing. The two policies do not achieve identical held-out false-positive rates and must not be presented as a fixed biological-FPR comparison.', '',
    '| Frozen policy | Cytosolic recall | Low-contrast recall | Proposal-ranking AP50 | Count MAE | Count bias |','|---|---:|---:|---:|---:|---:|']
    for method,label in [('yolo_default','YOLO score 0.25'),('yolo_rate_matched','YOLO calibration policy'),('filament_rate_matched','Filament calibration policy')]:
        r=evaluation['methods'][method];c=r['per_cell_annotation_count_error'];lines.append(f"| {label} | {r['recall_at_threshold']:.2%} | {r['contrast_groups']['low']['recall']:.2%} | {r['AP50_proposal_ranking']:.3f} | {c['MAE']:.3f} | {c['bias']:.3f} |")
    lines += ['', 'Count errors use the same 112 paired cells with resolved reference ownership and no ambiguous accepted prediction under any compared policy. They compare predictions with annotated counts, not independently verified complete biological counts. All 166 held-out cells retain their count intervals in heldout_per_cell_counts.csv. The paired subset may exclude difficult cells, so its errors must not be generalized to all cells.', '',
    'AP50 here uses score-ordered one-to-one matching at IoU 0.5. It is not interchangeable with the earlier library-validator AP: the standard rectangular validation loader was observed to form 672 × 672 tensors for nominal imgsz=640, while this direct prediction path uses 640 × 640. Its matcher also differs. Comparisons in the table use one explicit, fixed protocol.', '',
    'The filament policy lowers recall and increases count error in this exploratory operational comparison. This does not establish that every likelihood-based approach fails; it provides no support for deploying this particular model and calibration policy. All 18,367 raw proposals remain available in all_cytosolic_candidates.csv, including decision disagreements and uncertain ownership.', '',
    '## Limited missing-label audit and correction gate', '',
    'Eight candidate contexts from six likelihood-fit images were inspected at the original display scale, alongside existing labels and cell boundaries. Filament-like signals, shared structures and proximity to centrosomal asters prevented confident classification as distinct missing cytosolic objects. These observations are recorded in missing_label_review.csv, with decision=ambiguous. The screening was targeted, not an exhaustive annotation-completeness audit.', '',
    'No missing cytosolic positive was verified. No labels were removed or automatically added. The correction gate remains not_activated. If a named confirming reviewer verifies missing cytosolic boxes, prepare_label_correction.py can append them only in fit images while retaining every original label. It prepares matched baseline/corrected source-image splits; both variants then require training from the same initialization with identical settings. It refuses to modify calibration or held-out source labels.', '',
    '## Reproduce', '', 'Run from the source project directory:', '', '```bash',
    'python3 analysis/ownership_audit.py --root .. --output analysis_runs/per_cell_v1',
    'OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 analysis/evaluate_faint.py --root .. --audit analysis_runs/per_cell_v1 --weights ../results/runs/detect/train/weights/best.pt --imgsz 640',
    'python3 analysis/prepare_label_correction.py --base prepared_native2048 --reviews analysis_runs/per_cell_v1/missing_label_review.csv --manifest analysis_runs/per_cell_v1/image_manifest.csv --output analysis_runs/per_cell_v1/corrected_training --gate analysis_runs/per_cell_v1/correction_gate.json',
    'python3 -m unittest discover -s tests -p test_scientific_audit.py -v', '```', '',
    'Source-image split membership is stored in image_manifest.csv. Manual visual observations must be re-entered after regenerating the initial review queue; the regenerated queue defaults to ambiguous. The score grids and split seed are declared in source. The dataset and checkpoint are not included in this deliverable. The TIFFs supplied in the original workspace are required.', '',
    '## Method references', '',
    '- Foi et al. (2008). Practical Poissonian–Gaussian noise modeling and fitting for single-image raw-data. https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf',
    '- TIFF 6.0 specification, palette-colour interpretation. https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf', '',
    'The first reference concerns raw sensor data and does not validate a physical-noise interpretation of these TIFFs. The mathematical likelihood calculation is a working model whose empirical behaviour is evaluated above.']
    (output/'Per_Cell_MT_Audit_Report.md').write_text('\n'.join(lines)+'\n')
    print(output/'Per_Cell_MT_Audit_Report.md')


if __name__=='__main__':main()
