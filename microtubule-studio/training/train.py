"""Train whole images (or prepared square tiles) at their native pixel size."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image
import torch
from ultralytics import YOLO
import ultralytics
import yaml


def save_training_state(trainer):
    """Keep full-precision online weights and optimizer moments for interrupted CPU runs."""
    dest = trainer.last.with_name('resume_state.pth')
    temp = dest.with_suffix('.tmp')
    optimizer = trainer.optimizer.state_dict()
    for group in optimizer['param_groups']:
        group['lr'] = float(group['lr'])
    torch.save({'completed_epochs': trainer.epoch + 1,
                'checkpoint_sha256': hashlib.sha256(trainer.last.read_bytes()).hexdigest(),
                'model': trainer.model.state_dict(), 'ema': trainer.ema.ema.state_dict(),
                'optimizer': optimizer, 'ema_updates': int(trainer.ema.updates)},
               temp)
    temp.replace(dest)


def restore_training_state(trainer, weights):
    state = torch.load(weights.with_name('resume_state.pth'), map_location='cpu', weights_only=True)
    if (state['completed_epochs'] != trainer.start_epoch
            or state['checkpoint_sha256'] != hashlib.sha256(weights.read_bytes()).hexdigest()):
        raise ValueError('Full-precision training state does not match the last complete checkpoint.')
    trainer.model.load_state_dict(state['model'])
    trainer.ema.ema.load_state_dict(state['ema'])
    trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.ema.updates = state['ema_updates']


def native_size(data):
    config = yaml.safe_load(data.read_text())
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = data.parent / root
    sizes = set()
    for split in ("train", "val"):
        inputs = config[split] if isinstance(config[split], list) else [config[split]]
        paths = []
        for entry in inputs:
            source = root / entry
            if source.is_file() and source.suffix.lower() == '.txt':
                paths.extend((source.parent / line.strip()) if not Path(line.strip()).is_absolute() else Path(line.strip())
                             for line in source.read_text().splitlines() if line.strip())
            else:
                paths.extend(sorted(source.glob('*')) if source.is_dir() else [source])
        images = [p for p in paths if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg")]
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
    p.add_argument('--optimizer', choices=['auto', 'AdamW', 'Adam', 'SGD'])
    p.add_argument('--lr0', type=float)
    p.add_argument('--patience', type=int)
    p.add_argument('--warmup-epochs', type=float)
    p.add_argument('--resume', action='store_true', help='Resume --weights last.pt with its saved settings')
    p.add_argument('--full-precision-state', action='store_true',
                   help='Save a CPU training-state sidecar each epoch and restore it on resume')
    p.add_argument('--reset-optimizer', action='store_true',
                   help='For recovery only: keep resumed weights/epoch but clear optimizer moments')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.reset_optimizer and not args.resume:
        p.error('--reset-optimizer requires --resume')
    if args.full_precision_state and (args.reset_optimizer or args.device != 'cpu'):
        p.error('--full-precision-state requires CPU training without an optimizer reset')
    if not args.data.is_file() or not args.weights.is_file():
        p.error('Data YAML and local .pt weights must exist')
    if min(args.epochs, args.batch, args.threads) < 1:
        p.error('Epochs, batch size and thread count must be positive')
    if args.lr0 is not None and (not 0 < args.lr0 <= 1):
        p.error('--lr0 must be greater than zero and at most one')
    if args.lr0 is not None and args.optimizer in (None, 'auto'):
        p.error('Choose an explicit --optimizer when setting --lr0')
    if args.patience is not None and args.patience < 0:
        p.error('--patience cannot be negative')
    if args.warmup_epochs is not None and not 0 <= args.warmup_epochs <= args.epochs:
        p.error('--warmup-epochs must be between zero and the epoch budget')
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
    for option in ('optimizer', 'lr0', 'patience', 'warmup_epochs'):
        value = getattr(args, option)
        if value is not None:
            config[option] = value
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
        if args.full_precision_state and args.resume:
            restore_training_state(trainer, args.weights)
            print('Restored full-precision online weights, EMA and optimizer state.', flush=True)
        if args.reset_optimizer:
            trainer.optimizer.state.clear()
            print('Recovery: optimizer moments reset; resumed weights, epoch and learning-rate schedule retained.',
                  flush=True)
        plan = {'status': 'running', 'pid': os.getpid(), 'ultralytics': ultralytics.__version__,
                'torch': torch.__version__, 'initial_weights': str(args.weights.resolve()),
                'completed_epochs': trainer.start_epoch,
                'optimizer_state_reset': args.reset_optimizer,
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
    if args.full_precision_state:
        model.add_callback('on_model_save', save_training_state)
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
