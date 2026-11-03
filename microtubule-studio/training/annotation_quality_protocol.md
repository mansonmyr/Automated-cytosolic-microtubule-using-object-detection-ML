# Learning from imperfect microtubule annotations

The functional target is cytosolic microtubule detections assigned to individual cells. This protocol tests whether training remains useful when real targets lack annotations. It does not assume that faint image signal means a bad annotation.

## Current evidence

There are 105 images, with 80 assigned to training and 25 to development validation. The original LabelMe files contain 3,454 cytosolic annotations and 424 cell annotations. Most annotations lack explicit cell IDs. Cell ownership needs geometry checks against the original cell boundaries. Cell bounding boxes are insufficient for accurate boundary assignment.

Training images have a median of 10.5 annotated objects; validation images have a median of 92. This difference warrants investigation. It does not establish annotation incompleteness. Existing validation images have already been used in model development.

The available TIFFs are 8-bit palette images. Display contrast can rank candidates, but it is not calibrated photon intensity. MLE requires a defensible signal model and noise assumptions. A high likelihood score alone does not verify biological identity.

## One controlled omission experiment

From the microtubule-studio directory:

```bash
python3 training/prepare_annotation_benchmark.py --data prepared_native2048/data.yaml --output evaluation/annotation_quality --drop-fraction 0.25 --seed 42
python3 -m unittest discover -s tests -p 'test_annotation_benchmark.py'
```

The script makes a separate dataset configuration. It removes a fixed fraction of training cytosolic annotations selected by low local display contrast. The score is the 90th percentile of display luminance inside a box minus the median in an eight-pixel surrounding ring. Other real structures may enter this ring. The ranking is a reproducible proxy for this experiment, not an annotation-quality verdict.

The script preserves every image byte and all validation label bytes. Non-cytosolic training labels are retained. Source files are read only. Equal-score ties are resolved using the recorded seed. The output contains per-annotation scores and a hash manifest. Image files use relative symlinks where supported; regenerate the experiment if its source dataset is moved.

Original labels remain reference annotations, not independently established ground truth. Deliberate removals create known omissions relative to that reference. They do not resolve uncertainty already present in the original labels.

## Comparison to run after the current training job

Use the current standard-training run as the original-label reference. Train the same architecture on the omission dataset with the same initialization and training settings. This measures sensitivity to the controlled omissions.

If that sensitivity is substantial, compare one missing-annotation-aware training method against standard training on the exact same omission dataset. Positive-unlabeled classification or an ignore-region loss is a candidate. These require implementation and checking in the YOLOv10 training path; they are not enabled by this preparation script. Do not start a broad architecture search.

Neither method may access the omitted annotation identities, the complete source labels, or validation labels while updating model parameters. Automated quality scores prioritize review; they do not authorize replacing annotations with model predictions.

Before the comparison, record the initialization hash and the inference settings. Keep training budgets equal. Evaluate at the same native resolution. Select operating thresholds on an inner training/validation split, separate from the final evaluation images. Any new folds must start from generic pretrained weights rather than checkpoints that have seen their held-out images. All crops of a source image stay in that image's fold. Use acquisition groups only when actual grouping metadata is available.

## Evidence required for a useful result

- Establish a fully reviewed subset of existing images for evaluation, with ambiguous structures explicitly recorded. A second blinded microscopy reader provides stronger annotation evidence where available. Do not use evaluation corrections as training supervision for the same comparison.
- Compare cytosolic recall at a matched false-positive rate per cell. Also measure absolute count error per cell. Review overlaps and border assignments before treating cell counts as reference values.
- Report results separately for low-contrast annotated objects. Define contrast categories on training data, without optimizing them on final evaluation images.
- Estimate uncertainty from paired image-level results, or acquisition groups when identifiable. Boxes within one image are not independent replicates.
- Include an unchanged-label baseline and report the loss from known omissions before claiming robustness. A candidate helps only if it improves the corrupted-label comparison without an unacceptable loss on the original-label task.

Use the inherited 25-image validation set for an exploratory stress test only. A new internal cross-validation design does not erase earlier development exposure. A null or inconclusive result is a valid outcome.

## Functional scale

Accuracy and throughput are separate questions. A later batch runner should measure seconds per image and peak memory on the native input size. It should retain data provenance and route uncertain cells for review. Repeatedly processing the same 105 images can measure throughput, but does not create evidence about new experimental conditions. Batch-level failure recovery also needs testing before unattended use.

The present output is benchmark preparation. It does not demonstrate robust training, reliable faint-object detection, or performance on a large biological dataset.

## Primary sources

- [ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data](https://arxiv.org/abs/2309.00832).
- [Object Detection as a Positive-Unlabeled Problem](https://arxiv.org/abs/2002.04672).
- [Cleanlab object-detection label-quality tutorial](https://docs.cleanlab.ai/stable/tutorials/object_detection.html).
- [Grouped cross-validation documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
