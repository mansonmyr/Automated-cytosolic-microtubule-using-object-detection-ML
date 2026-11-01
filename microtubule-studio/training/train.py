"""Train whole images (or prepared square tiles) at their native pixel size."""
import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image
import torch
from ultralytics import YOLO
import ultralytics
import yaml


def native_size(data):
    config = yaml.safe_load(data.read_text())
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = data.parent / root
    sizes = set()
    for split in ("train", "val"):
        paths = sorted((root / config[split]).glob("*"))
        images = [p for p in paths if p.suffix.lower() in (".png", ".tif", ".jpg", ".jpeg")]
        if not images:
            raise ValueError(f"No images for {split}")
        for path in images:
            with Image.open(path) as image:
                sizes.add(image.size)
    if len(sizes) != 1:
        raise ValueError(f"Native mode needs identical square image dimensions, found {sizes}")
    width, height = sizes.pop()
    if width != height or width % 32:
        raise ValueError("Native mode requires square dimensions divisible by model stride (32)")
    return width


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--device', default='auto', help='auto, cpu, mps, or CUDA index 0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--project', type=Path, default=Path('training_runs'))
    p.add_argument('--name', default='native2048')
    p.add_argument('--resume', action='store_true', help='Resume --weights last.pt with its saved settings')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        p.error('Data YAML and local .pt weights must exist')
    if min(args.epochs, args.batch, args.threads) < 1:
        p.error('Epochs, batch size and thread count must be positive')
    torch.set_num_threads(args.threads)
    device = args.device
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    size = native_size(args.data.resolve())
    config = dict(data=str(args.data.resolve()), imgsz=size, epochs=args.epochs, batch=args.batch,
                  device=device, workers=0, project=str(args.project.resolve()), name=args.name,
                  exist_ok=False, seed=42, deterministic=True, patience=20,
                  cache=False, amp=device not in ('cpu', 'mps'), save=True, save_period=5,
                  # Flips preserve pixel spacing. Disable rescaling and image/color mixing.
                  mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
                  scale=0.0, translate=0.0, degrees=0.0, shear=0.0, perspective=0.0,
                  multi_scale=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
                  augmentations=[], fliplr=0.5, flipud=0.5, plots=True)
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        return
    os.environ['YOLO_OFFLINE'] = 'true'
    model = YOLO(str(args.weights.resolve()))
    started = time.monotonic()

    def record_start(trainer):
        shape = tuple(trainer.train_loader.dataset[0]['img'].shape)
        if shape != (3, size, size):
            raise RuntimeError(f'Unexpected training tensor shape: {shape}')
        plan = {'status': 'running', 'pid': os.getpid(), 'ultralytics': ultralytics.__version__,
                'torch': torch.__version__, 'initial_weights': str(args.weights.resolve()),
                'configuration': vars(trainer.args), 'image_tensor_shape': list(shape)}
        (trainer.save_dir / 'training_status.json').write_text(json.dumps(plan, indent=2, default=str) + '\n')

    def record_epoch(trainer):
        path = trainer.save_dir / 'training_status.json'
        plan = json.loads(path.read_text())
        plan.update(completed_epochs=trainer.epoch + 1, elapsed_seconds=round(time.monotonic()-started),
                    metrics={k: float(v) for k, v in trainer.metrics.items()},
                    last_checkpoint=str(trainer.last), best_checkpoint=str(trainer.best))
        path.write_text(json.dumps(plan, indent=2, default=str) + '\n')

    def record_end(trainer):
        path = trainer.save_dir / 'training_status.json'
        plan = json.loads(path.read_text())
        plan.update(status='completed', elapsed_seconds=round(time.monotonic()-started))
        path.write_text(json.dumps(plan, indent=2, default=str) + '\n')

    model.add_callback('on_train_start', record_start)
    model.add_callback('on_fit_epoch_end', record_epoch)
    model.add_callback('on_train_end', record_end)
    try:
        model.train(**({'resume': True, 'device': device} if args.resume else config))
    except BaseException as error:
        trainer = getattr(model, 'trainer', None)
        if trainer is not None and trainer.save_dir.exists():
            path = trainer.save_dir / 'training_status.json'
            plan = json.loads(path.read_text()) if path.exists() else {}
            plan.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            path.write_text(json.dumps(plan, indent=2, default=str) + '\n')
        raise


if __name__ == '__main__':
    main()
