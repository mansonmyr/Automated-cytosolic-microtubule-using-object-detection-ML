"""Train a PyTorch YOLO detector using the prepared tiles."""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True,help='Local generic pretrained YOLOv10n .pt weights')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--device',default='cpu',help='cpu, mps, 0, etc.')
    p.add_argument('--project',default='training_runs')
    args=p.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        p.error('Data YAML and pretrained weights must exist locally')
    YOLO(str(args.weights)).train(data=str(args.data.resolve()),imgsz=1024,epochs=args.epochs,batch=args.batch,
        device=args.device,workers=0,project=args.project,name='tiled',exist_ok=False,
        seed=42,patience=20,hsv_h=0,hsv_s=0,hsv_v=0,fliplr=0.5,flipud=0.5)
if __name__=='__main__': main()
