"""Create lossless image tiles from the existing split. No source files are changed."""
import argparse
from pathlib import Path
from PIL import Image
import yaml

NAMES = {0: 'cytosolic', 1: 'centrosomal', 2: 'cell'}

def positions(length, tile, stride):
    if length <= tile:
        return [0]
    out = list(range(0, length - tile + 1, stride))
    if out[-1] != length - tile:
        out.append(length - tile)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True, help='Folder containing train txt file and val txt file')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=256)
    args = parser.parse_args()
    if args.tile_size <= 0 or not 0 <= args.overlap < args.tile_size:
        parser.error('Require 0 <= overlap < tile size')
    args.output.mkdir(parents=True, exist_ok=False)
    stride = args.tile_size - args.overlap
    for split, folder in [('train', 'train txt file'), ('val', 'val txt file')]:
        src = args.dataset / folder
        images = sorted(src.glob('*.tif'))
        if not images:
            raise ValueError(f'No TIFF files in {src}')
        imgout = args.output / 'images' / split
        labout = args.output / 'labels' / split
        imgout.mkdir(parents=True); labout.mkdir(parents=True)
        count = 0
        for path in images:
            label = path.with_suffix('.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
            with Image.open(path) as im:
                image = im.convert('RGB')
            width, height = image.size
            boxes = []
            for line in label.read_text().splitlines():
                cls,cx,cy,bw,bh = map(float,line.split())
                if int(cls) not in NAMES or cls != int(cls) or not all(0 <= x <= 1 for x in (cx,cy,bw,bh)):
                    raise ValueError(f'Invalid label in {label}: {line}')
                boxes.append((int(cls),(cx-bw/2)*width,(cy-bh/2)*height,(cx+bw/2)*width,(cy+bh/2)*height))
            for y in positions(height,args.tile_size,stride):
                for x in positions(width,args.tile_size,stride):
                    tw=min(args.tile_size,width-x); th=min(args.tile_size,height-y)
                    crop=image.crop((x,y,x+tw,y+th)); name=f'{path.stem}__x{x}_y{y}'
                    crop.save(imgout / f'{name}.png')
                    entries=[]
                    for cls,x1,y1,x2,y2 in boxes:
                        a,b,c,d=max(x1,x),max(y1,y),min(x2,x+tw),min(y2,y+th)
                        if c-a < 1 or d-b < 1: continue
                        entries.append(f'{cls} {((a+c)/2-x)/tw:.6f} {((b+d)/2-y)/th:.6f} {(c-a)/tw:.6f} {(d-b)/th:.6f}')
                    (labout / f'{name}.txt').write_text('\n'.join(entries) + ('\n' if entries else ''))
                    count += 1
        print(f'{split}: {len(images)} original images, {count} tiles')
    data={'path':str(args.output.resolve()),'train':'images/train','val':'images/val','names':NAMES}
    (args.output/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    print('Tiles from each source image remain in its original split. Check that experiments are independent across splits.')
if __name__ == '__main__': main()
