# Full-resolution tiled training

These scripts prepare lossless PNG tiles from the original TIFF images and train a YOLOv10 detector with PyTorch and Ultralytics. They are an experimental training pipeline. A trained tiled model and a deployed web application are not yet included here.

## Prepare and train

Install Python, PyTorch, Pillow, PyYAML and Ultralytics 8.4.9. Download the original dataset and generic YOLOv10n pretrained weights separately. The dataset directory must contain `train txt file` and `val txt file`, each with paired `.tif` images and YOLO `.txt` labels.

```bash
python microtubule-studio/training/prepare_tiles.py --dataset /path/to/yolo_dataset --output /path/to/prepared_tiles
python microtubule-studio/training/train.py --data /path/to/prepared_tiles/data.yaml --weights /path/to/yolov10n.pt --device cpu
```

The output directory must not already exist. The default uses 1024-pixel tiles with 256 pixels of overlap, and keeps each source image in its existing split. CPU training can take many hours. Use an appropriate supported GPU device if available. The scripts do not upload the source images.

## Interpretation

Existing train and validation splits require checking for independent experiments and acquisitions. The original validation set has already been used during development. Overlapping tiles can contain the same object more than once, and boxes at tile edges are clipped. Tile-level scores are not independent full-image performance measures. Assess a new model on held-out full images with tile merging and class-aware duplicate suppression before using it for research conclusions.

The class IDs used by these scripts are 0 cytosolic, 1 centrosomal and 2 cell. Confirm these against your annotations. Bounding-box counts are not measurements of filament length or fluorescence intensity.
