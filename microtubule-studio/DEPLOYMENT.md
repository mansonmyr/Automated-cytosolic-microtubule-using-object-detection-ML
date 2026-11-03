# Deploy the cytosolic microtubule detector

The selected refined 1024 ONNX model is in `dist/refined1024.onnx`. The app starts at confidence 0.174 and processes images in browser memory. The runtime and TIFF decoder load from jsDelivr.

## Repository layout

The existing GitHub project is https://github.com/mansonmyr/Automated-cytosolic-microtubule-using-object-detection-ML. Its app is under `microtubule-studio/`. The current release installs the selected ONNX model in that directory. The historical snapshot is absent from this local release; see `archive/README.md`.

The root provider configurations publish `microtubule-studio/dist` when the repository root is the hosting root/base directory. App-level configurations also support the following existing app-directory setup.

For the existing GitHub layout, use either of these configurations:

| Provider | App directory setting | Build command | Publish directory |
| --- | --- | --- | --- |
| Vercel | Root Directory: `microtubule-studio` | `node scripts/check-deployment.cjs` | `dist` |
| Netlify | Base directory: `microtubule-studio` | `node scripts/check-deployment.cjs` | `dist` |

The configuration files beside this document use these paths. If this app directory is instead the entire repository, leave the provider's root/base directory at the repository root. The same configuration files work without edits.

With repository-root hosting, the root configurations use `node microtubule-studio/scripts/check-deployment.cjs` and publish `microtubule-studio/dist`. Neither provider publishes `archive/`.

## Local verification

Run these commands from the directory containing this document:

```bash
node scripts/check-deployment.cjs
node --test tests/*.test.cjs
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Open http://localhost:8765. The model checksum and all 18 JavaScript tests passed.

## Safari verification

Safari decoded the 2048 × 2048 TIFF `Experiment-1678_C2_sum.tif` and ran the selected model with ONNX Runtime Web using WebAssembly. At confidence 0.174, the reviewer showed 66 cytosolic candidates, 6 centrosomal candidates and 5 cell boxes. The cytosolic count matched the native reference for that image.

Selecting a box showed its coordinates in the original image. Deleting it reduced the cytosolic count to 65. Undo restored 66. In a second run the reviewer exported the corrected state to a JSON blob. Reopening the same TIFF and importing that blob restored 65 cytosolic candidates and matched the image fingerprint.

The local test fixture supplied the model bytes and configuration directly. It loaded the production TIFF decoder and ONNX runtime from their configured CDN. The deployed model URL and the browser’s file download dialog still need a hosted check.

## Hosted verification

Open the site over HTTPS or localhost. Upload a 2048 × 2048 TIFF and run detection. Confirm the footer names Refined YOLOv10n and inspect the candidate boxes. Select a box, zoom in and verify its coordinates remain in original image pixels. Save a review JSON, reopen the same source image and import the review. Confirm the corrections are restored.

## Model interpretation

The app displays image-level candidate totals. Native development evaluation used supplied cell polygons and 135 cells with resolved reference counts. Count MAE was 3.07 and RMSE was 5.32 at confidence 0.174. These development images had prior model-selection exposure. The browser app does not yet provide validated per-cell counts.
