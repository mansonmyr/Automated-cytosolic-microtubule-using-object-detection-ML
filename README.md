# Per-cell microtubule analysis with Refined1024

Ooi Kai Hong · [ORCID 0000-0003-3803-7864](https://orcid.org/0000-0003-3803-7864)

The browser app combines Refined YOLOv10n at 1024 pixels with a MobileNetV3-Small cell segmentation model. Review the proposed outlines before assigning cytosolic detections and exporting counts. Images are processed locally in the browser.

## Run the app

From the repository root, start a local server and open [the app](http://127.0.0.1:8765/).

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory microtubule-studio/dist
```

1. Open a TIFF, PNG or JPEG and choose **Detect cells + microtubules**.
2. Remove partial or merged outlines. Complete cells can touch the image edge or lack a detected centrosome. Confirm that the retained cells are complete.
3. Select **Flip agreement** under **Counting input** to compare the original image with its horizontal reflection. Continue to per-cell counting and run detection.
4. Review the boxes and assignments, then export the per-cell counts CSV or save a review JSON. Resume a saved review with the same source image.

The square preview shows outlines, detection boxes and provisional counts. Use the mouse wheel to zoom at the cursor, middle-button or Alt-drag to pan, and double-click to fit. Display brightness changes the preview only. The ONNX runtime and TIFF decoder load from jsDelivr.

## Detection evaluation

Flip agreement averages matching boxes and scores across the two views and halves scores for boxes seen in just one view. The mode uses cutoff 0.075 and takes an extra inference pass. The original-image default uses cutoff 0.174.

| Cytosolic detection mode | Cutoff | Validation precision | Validation recall | Validation F1 | Development F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original default | 0.174 | 57.7% | 14.4% | 23.1% | 54.9% |
| Original, best tested validation F1 | 0.050 | 43.8% | 44.2% | 44.0% | 52.6% |
| Flip agreement, best tested validation F1 | 0.075 | 59.5% | 42.3% | 49.4% | 56.6% |

The sweep selected cutoffs on 20 validation images and applied those cutoffs to 25 development images. It matched cytosolic boxes one-to-one at IoU ≥ 0.5 across the annotated fields. Development images were used in earlier evaluations. The comparison measures detection F1 and image-level count error. Per-cell count accuracy with flip agreement still needs evaluation. [Inference comparison](microtubule-studio/metrics/flip_agreement_evaluation.json)

The selected segmentation model reached 79.2% cell F1 on validation and 60.9% on development at polygon IoU ≥ 0.5. It matched 106 of 166 annotated development cells. Touching cells can merge, and the review step is needed before counting. [Segmentation evaluation](microtubule-studio/metrics/cell_segmentation_evaluation.json)

The optional example filter compares shape and intensity to two user-rejected outlines. It reproduced those exclusions in Experiment 1475 and removed no proposals across 44 other evaluated images. Use **Restore automatically excluded cells** to inspect a rejected outline. Broader completeness detection requires more reviewed examples.

## Verification and earlier work

```bash
node microtubule-studio/scripts/check-deployment.cjs
node --test microtubule-studio/tests/review.test.cjs archive/image-level-reviewer/tests/app.test.cjs
```

The deployment check verifies both ONNX hashes and the static app assets. The earlier technical report describes the image-level reviewer and its evaluation, including the restricted ROC-AUC, precision–recall figures and annotated-polygon count metrics. [Earlier technical report](TECHNICAL_PAPER.md)

[Software citation](citation.cff) · [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
