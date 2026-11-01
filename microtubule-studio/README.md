# Microtubule Studio prototype

A research prototype for displaying candidate detections in fluorescence microscopy images. It uses the original 640-pixel YOLOv10n checkpoint exported to ONNX. Correct class IDs are 0 cytosolic, 1 centrosomal, and 2 cell. The model metadata uses different names; the app applies the mapping found in the original annotations. Boxes and counts need human review and independent validation.

## Web interface

`dist/index.html` runs inference in the browser using `dist/model.onnx`. A web server is required. The interface loads ONNX Runtime Web and UTIF.js from jsDelivr. These libraries and the model must be accessible for inference; images remain in browser memory. Only single-page TIFF files, PNG, and JPEG are supported. The baseline model uses 640 × 640 input, so it does not yet use tiles for inference. Do not treat predicted box counts as microtubule length or fluorescence intensity.

## Prepare full-resolution training tiles

Install Pillow, PyYAML, PyTorch and Ultralytics 8.4.9 in a Python environment. Run the following from this directory, replacing paths as needed:

```bash
python training/prepare_tiles.py --dataset ../yolo_dataset --output ./prepared_tiles
python training/train.py --data ./prepared_tiles/data.yaml --weights ../results/yolov10n.pt --device cpu
```

Images are saved as lossless PNG tiles with overlapping edges. Labels are clipped to each tile. Train and validation images remain in their existing source splits before tiling. Review the split for independence at experiment, slide or acquisition level. Tiles from one image must not cross splits. Tiled validation metrics are not whole-image metrics: overlapping tiles can contain the same object more than once. Evaluate on independent full images, with tile merging and class-aware overlap suppression, before claiming improved performance.

The trained `best.pt` from a new run is not automatically installed in the app. It needs evaluation, ONNX export, compatibility checks and a new deployment. Use GPU training when available; 1024-pixel CPU training can take a long time. A one-epoch synthetic smoke test only checks execution, not model quality.

## Dataset and benchmark

The available dataset contains 80 train and 25 validation images. Both sets were previously used for development; no independent test set was supplied. In a same-settings comparison at 640 pixels, original and refined checkpoints achieved validation mAP50 of approximately 0.613 and 0.582 respectively; these results do not demonstrate performance on new experiments. TIFF samples inspected were 2048 × 2048, with an 8-bit green image palette. The original training resized images to 640 or 1024.

The website and training code are prototypes. External JavaScript, ONNX output interpretation, different TIFF formats, inference, and deployment require end-to-end browser testing. Dataset and trained weights may carry separate sharing and licensing conditions.
