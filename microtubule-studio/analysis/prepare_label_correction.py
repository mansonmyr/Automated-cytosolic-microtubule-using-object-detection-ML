"""Prepare a paired training comparison only after missing cytosolic labels are verified."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import yaml
from evaluate_faint import ious


def prepare(base, reviews, manifest, output, gate):
    decisions=pd.read_csv(reviews).fillna('');images=pd.read_csv(manifest)
    verified=decisions[decisions.decision=='verified_cytosolic']
    if not len(verified):
        result={'status':'not_activated','verified_missing_positives':0,
                'reason':'No missing cytosolic annotations have been verified. Original positive labels, including faint examples, remain unchanged.'}
        gate.write_text(json.dumps(result,indent=2)+'\n');return result
    fit=set(images.loc[images.subset=='fit','source_id']);cal=set(images.loc[images.subset=='calibration','source_id'])
    additions={}
    for row in verified.to_dict('records'):
        if row['source_id'] not in fit:raise ValueError('Corrections may be added only to fit source images, never calibration or held-out images.')
        reviewer=str(row.get('reviewer','')).strip()
        if not reviewer or 'pending' in reviewer.lower() or 'assistant visual screening' in reviewer.lower() or not str(row.get('reason','')).strip():
            raise ValueError('A named confirming reviewer and verification reason are required.')
        box=np.asarray([row[k] for k in ['x1','y1','x2','y2']],float)
        if not np.isfinite(box).all() or box.min()<0 or box.max()>2048 or np.any(box[2:]<=box[:2]):raise ValueError('Invalid verified box.')
        label=base/'labels/train'/f"{row['source_id']}.txt"
        rows=np.loadtxt(label,ndmin=2);cyto=rows[rows[:,0]==0]
        existing=np.c_[cyto[:,1:3]-cyto[:,3:5]/2,cyto[:,1:3]+cyto[:,3:5]/2]*2048
        if len(existing) and ious([box],existing).max()>.5:raise ValueError('Verified box duplicates an existing positive annotation.')
        previous=additions.setdefault(row['source_id'],[])
        if previous and ious([box],previous).max()>.5:raise ValueError('Verified corrections overlap as probable duplicates.')
        previous.append(box)
    output.mkdir(parents=True,exist_ok=False)
    (output/'images').symlink_to((base/'images').resolve(),target_is_directory=True)
    shutil.copytree(base/'labels',output/'labels')
    for source,boxes in additions.items():
        target=output/'labels/train'/f'{source}.txt';original=target.read_text();lines=[]
        for x1,y1,x2,y2 in boxes:lines.append(f'0 {(x1+x2)/4096:.9f} {(y1+y2)/4096:.9f} {(x2-x1)/2048:.9f} {(y2-y1)/2048:.9f}')
        target.write_text(original+('' if original.endswith('\n') or not original else '\n')+'\n'.join(lines)+'\n')
        if not target.read_text().startswith(original):raise RuntimeError('Original positive labels were modified.')
    for key,sources in [('train',fit),('val',cal)]:
        (output/f'{key}.txt').write_text('\n'.join(str(output/'images/train'/f'{s}.png') for s in sorted(sources))+'\n')
        (output/f'baseline_{key}.txt').write_text('\n'.join(str(base/'images/train'/f'{s}.png') for s in sorted(sources))+'\n')
    for name,prefix in [('corrected',''),('baseline','baseline_')]:
        config={'path':str(output),'train':str(output/f'{prefix}train.txt'),'val':str(output/f'{prefix}val.txt'),
                'test':str(base/'images/val'),'names':{0:'cytosolic',1:'centrosomal',2:'cell'}}
        (output/f'{name}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    result={'status':'prepared_not_trained','verified_missing_positives':len(verified),'changed_fit_images':len(additions),
            'fit_images':len(fit),'calibration_images':len(cal),'original_labels_retained':True,
            'required_comparison':'Train both YAML variants from the same generic pretrained checkpoint with identical settings and seeds. Use calibration for selection; evaluate once on held-out sources.'}
    gate.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True,type=Path);p.add_argument('--reviews',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--gate',required=True,type=Path);a=p.parse_args()
    print(json.dumps(prepare(a.base.resolve(),a.reviews,a.manifest,a.output.resolve(),a.gate),indent=2))
