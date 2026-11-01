# Microtubule Studio prototype

A research prototype for displaying candidate detections in fluorescence microscopy images. It uses the original 640-pixel YOLOv10n checkpoint exported to ONNX. Correct class IDs are 0 cytosolic, 1 centrosomal, and 2 cell. The model metadata uses different names; the app applies the mapping found in the original annotations. Boxes and counts need human review and independent validation.

## Web interface

`dist/index.html` runs inference in the browser using `dist/model.onnx`. A web server is required. The interface loads ONNX Runtime Web and UTIF.js from jsDelivr. These libraries and the model must be accessible for inference; images remain in browser memory. Only single-page TIFF files, PNG, and JPEG are supported. The baseline model uses 640 × 640 input. For larger images, the app runs overlapping 640-pixel tiles at native image resolution and merges same-class boxes using an overlap threshold. This preserves image detail but can still miss or duplicate detections, especially near tile edges. Do not treat predicted box counts as microtubule length or fluorescence intensity.

From this directory, start a static server and open `http://localhost:8765`:

```bash
python3 -m http.server 8765 --directory dist
```

The source repository includes the interface and training scripts. Trained weights, ONNX files and source images are excluded from Git. The local workspace has the baseline `dist/model.onnx`; for another checkout, supply a compatible model separately. For example, export a three-class YOLOv10n checkpoint with `YOLO('/path/to/best.pt').export(format='onnx', imgsz=640, dynamic=False, simplify=False)` and copy the output to `dist/model.onnx`. ONNX export requires the corresponding Python dependencies. This does not establish browser runtime compatibility or model accuracy.

### Review and correct boxes

- Use Pan to drag a zoomed image. The +/− buttons change zoom; Fit shows the whole image and 100% shows one image pixel per display CSS pixel. Ctrl/Command plus scrolling zooms around the pointer.
- Use Select and click inside a box to inspect its original image coordinates. A small box takes priority over a larger cell box. Delete selected removes it; Undo restores the last correction.
- Use Draw box, select its class and drag between opposite corners. Manual boxes use dashed outlines and have no model confidence score. Reset corrections restores the last model predictions and can itself be undone.
- The confidence slider filters cached model predictions without another inference pass. Manual boxes remain visible. Raising the threshold hides predictions without deleting them; model scores below 0.05 are not retained.

The annotated PNG export is rendered at the original image dimensions. CSV includes only visible boxes, with coordinates in original image pixels and an `origin` column. Its confidence field is empty for manual boxes. Review JSON includes the threshold and model settings, along with all retained boxes and the removed model predictions. JSON is an audit export; importing a saved review is not implemented. Corrections stay in page memory until downloaded. Opening another image or rerunning detection asks before replacing corrections.

`dist/model-config.json` declares the model path and input size, along with tile overlap and the fixed class order. The app uses that configuration to prepare input tensors. A new 2048-pixel ONNX export requires `inputSize: 2048` and a compatible overlap such as `512`. A 2048 × 2048 image then uses one whole-image inference pass. The configured size must match the actual ONNX input; changing the configuration does not convert a 640-pixel model. TorchScript needs a Python inference runtime and cannot be loaded directly by this browser interface.

### Verification

```bash
node --test tests/*.test.cjs
```

The checks exercise geometry and correction history, plus an application harness with simulated DOM/canvas and inference objects. The harness verifies a configured 2048-pixel tensor, original-coordinate exports and failure/race handling. These checks do not replace a real-browser inference test. Browser preview and live Site publication were unavailable in the development environment because local preview was blocked and the Site source server could not be reached.

## Retrain at original whole-image resolution

The supplied images are all 2048 × 2048 single-page, 8-bit palette TIFFs. `training/prepare_native.py` audits every YOLO row against its original LabelMe shape, fixes the class mapping to **0 cytosolic, 1 centrosomal, 2 cell**, and clips boxes to the image boundaries. It writes RGB PNGs at exactly 2048 × 2048 without resizing or cropping, then verifies that their decoded RGB pixels exactly match the source TIFF palette colors. These are display-prepared images; conversion does not recover higher-bit-depth microscope acquisition data.

From this directory:

```bash
python3 training/prepare_native.py --dataset ../yolo_dataset --annotations ../channel2_input2/json --output prepared_native2048
python3 training/train.py --data prepared_native2048/data.yaml --weights ../results/yolov10n.pt --dry-run
python3 -u training/train.py --data prepared_native2048/data.yaml --weights ../results/yolov10n.pt --epochs 80 --batch 1 --device auto --project training_runs --name native2048
```

Use a new output folder if the prepared dataset already exists, or reuse its `data.yaml` without repeating preparation. Dependencies used locally: Ultralytics 8.4.9, PyTorch 2.11.0, Pillow, NumPy and PyYAML. `--device auto` selects CUDA if available, then Apple MPS, then CPU. `--device 0` selects a Kaggle/Colab CUDA GPU; the prepared YAML must point to its dataset location on that machine.

The trainer reads the actual image dimensions and uses that size for both training and validation. It disables mosaic and random scaling, along with translation and color augmentation. It also disables the library's optional Albumentations defaults. Horizontal/vertical flips preserve pixel spacing. The initial checkpoint is the generic pretrained YOLOv10n, with a new three-class head; existing experiment checkpoints are retained. Batch size is 1 to limit memory use. The run uses at most 80 epochs and patience 20, with `last.pt` saved after each epoch and the best validation checkpoint saved as `best.pt`.

The audit found 80 training images (2,032 boxes) and 25 validation images (2,334 boxes). All 4,366 YOLO boxes matched their corresponding LabelMe shape's bounding box and class ID. Four source shapes extended beyond an image boundary and were clipped in the prepared copy. There were no exact duplicate decoded RGB images. Cell polygons remain detection bounding boxes; this run does not train segmentation masks. The old conversion notebook assigned class IDs through an unordered Python set, and its training YAML assigned incorrect names. Do not regenerate labels using that notebook without fixing the mapping.

The current run's configuration, metrics and checkpoints are under `training_runs/native2048/`. `training_status.json` records the verified training tensor size, completed epochs and validation metrics. The CPU run's console log is `training-native2048.log`. To resume an interrupted run after its first checkpoint:

```bash
python3 training/train.py --data prepared_native2048/data.yaml --weights training_runs/native2048/weights/last.pt --resume --device auto
```

This split has been used in earlier model development. No experiment/batch grouping metadata or independent test set was provided. Exact duplicate checks cannot establish split independence. For a fair model comparison, evaluate both old and new checkpoints on the same whole validation images at the same input size. Compare per-class precision/recall and mAP, then inspect counts and overlays. A larger input alone does not establish improved accuracy.

### Watch progress and finish evaluation

The training console writes live batch progress to `training-native2048.log`. To watch it from this directory, use `tail -f training-native2048.log`. Press Control+C to stop watching the log; this does not stop the training process. Completed-epoch metrics are also written to `training_runs/native2048/results.csv`.

`training/finish_run.py` waits for successful completion, compares the new checkpoint with the baseline and refined checkpoints at 2048 pixels, and also evaluates each earlier checkpoint at its original training input size. It then exports TorchScript at 2048 pixels and compares its predictions with PyTorch on two validation images. Reports are written to `evaluation/native2048/`; `finish_status.json` records whether the process is waiting, evaluating or complete. Redirect its output to `evaluation-native2048.log` to retain a console log.

To run this follow-up for another completed run, or restart it if it is no longer running:

```bash
python3 -u training/finish_run.py --run training_runs/native2048 --data prepared_native2048/data.yaml --baseline ../results/runs/detect/train/weights/best.pt --refined ../results/runs/detect/runs/yolov10n_refined/weights/best.pt --output evaluation/native2048 --wait
```

Do not start duplicate follow-up processes for the same output directory. The script stops if training reports failure or its status has not updated for 30 minutes. Evaluation does not install the checkpoint in the browser app. The comparison reflects both resolution and changed training settings; it does not isolate the effect of resolution alone.

The browser still uses the previous 640-pixel ONNX model and 640-pixel tiles. A native-2048 checkpoint requires evaluation and a compatible export, followed by browser input-size changes and testing before installation. TorchScript export for a completed run can be performed separately with `YOLO('training_runs/native2048/weights/best.pt').export(format='torchscript', imgsz=2048)`.

## Optional training with native-resolution tiles

Install Pillow, PyYAML, PyTorch and Ultralytics 8.4.9 in a Python environment. Run the following from this directory, replacing paths as needed:

```bash
python training/prepare_tiles.py --dataset ../yolo_dataset --output ./prepared_tiles
python training/train.py --data ./prepared_tiles/data.yaml --weights ../results/yolov10n.pt --device cpu --name tiled1024
```

Images are saved as lossless PNG tiles with overlapping edges. This preserves the pixel spacing inside each crop but loses whole-image context. Labels are clipped to each tile. Train and validation images remain in their existing source splits before tiling. Review the split for independence at experiment, slide or acquisition level. Tiles from one image must not cross splits. Tiled validation metrics are not whole-image metrics: overlapping tiles can contain the same object more than once. Compare full-image predictions with tile merging and class-aware overlap suppression on the existing validation images. Because those images were already used in development, results on them do not establish performance on new experiments.

The trained `best.pt` from a new run is not automatically installed in the app. It needs evaluation, ONNX export, compatibility checks and a new deployment. Use GPU training when available; 1024-pixel CPU training can take a long time. A one-epoch synthetic smoke test only checks execution, not model quality.

## Dataset and benchmark

The available dataset contains 80 train and 25 validation images. Both sets were previously used for development; no independent test set was supplied. In a same-settings comparison at 640 pixels, original and refined checkpoints achieved validation mAP50 of approximately 0.613 and 0.582 respectively; these results do not demonstrate performance on new experiments. TIFF samples inspected were 2048 × 2048, with an 8-bit green image palette. The original training resized images to 640 or 1024.

The website and training code are prototypes. External JavaScript, ONNX output interpretation, different TIFF formats, inference, and deployment require end-to-end browser testing. Dataset and trained weights may carry separate sharing and licensing conditions.
