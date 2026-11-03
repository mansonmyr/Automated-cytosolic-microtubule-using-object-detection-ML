"""Source-image ownership and train-only contrast audit. Never changes supplied labels."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/per-cell-mpl')
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.stats import normaltest, skew, kurtosis


def bounds(points, width, height):
    points = np.asarray(points, float)
    return [max(0, int(np.floor(points[:,0].min()))), max(0, int(np.floor(points[:,1].min()))),
            min(width, int(np.ceil(points[:,0].max()))+1), min(height, int(np.ceil(points[:,1].max()))+1)]


def polygon_is_simple(points):
    points=np.asarray(points,float)
    if len(points)>1 and np.allclose(points[0],points[-1]):points=points[:-1]
    if len(points)<3 or not np.isfinite(points).all():return False
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    for i in range(len(points)):
        a,b=points[i],points[(i+1)%len(points)]
        if np.linalg.norm(a-b)<1e-6:return False
        for j in range(i+2,len(points)):
            if i==0 and j==len(points)-1:continue
            c,d=points[j],points[(j+1)%len(points)]
            if cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0:return False
    return True


def cell_masks(annotation):
    height, width = annotation['imageHeight'], annotation['imageWidth']
    cells = []
    for index, shape in enumerate(annotation['shapes']):
        if shape['label'] != 'cell':
            continue
        mask = np.zeros((height,width), np.uint8)
        points = np.asarray(shape['points'], float)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        cells.append({'cell_id':f'cell-{len(cells)+1}', 'shape_index':index, 'points':points.tolist(),
                      'geometry_valid':polygon_is_simple(points) and mask.sum()>=100, 'mask':mask})
    return cells


def owner(box, cells):
    x1,y1,x2,y2 = map(int,box)
    if x2 <= x1 or y2 <= y1:
        return {'owner_status':'invalid_box','cell_id':'','candidate_cells':'','owner_fraction':0,'second_fraction':0}
    fractions = np.array([c['mask'][y1:y2,x1:x2].mean() for c in cells])
    if not len(fractions):
        return {'owner_status':'unowned','cell_id':'','candidate_cells':'','owner_fraction':0,'second_fraction':0}
    order = np.argsort(-fractions); first=int(order[0]); top=float(fractions[first]); second=float(fractions[order[1]]) if len(order)>1 else 0.
    cy,cx=(y1+y2-1)//2,(x1+x2-1)//2
    inside=[i for i,c in enumerate(cells) if c['mask'][cy,cx]]
    candidates=[c['cell_id'] for i,c in enumerate(cells) if fractions[i]>.05 or i in inside]
    reliable = len(inside)==1 and inside[0]==first and top>=.8 and second<=.1 and cells[first].get('geometry_valid',True)
    return {'owner_status':'reliable' if reliable else 'ambiguous' if top>0 else 'unowned',
            'cell_id':cells[first]['cell_id'] if reliable else '', 'candidate_cells':'|'.join(candidates),
            'owner_fraction':top,'second_fraction':second}


def scalar_image(path):
    with Image.open(path) as image:
        a=np.array(image)
        palette=np.asarray(image.getpalette()).reshape(-1,3) if image.mode=='P' else None
        tags={'mode':image.mode,'bits':list(image.tag_v2.get(258,())),
              'photometric':int(image.tag_v2.get(262,-1)),'description':str(image.tag_v2.get(270,'')),
              'zero_fraction':float(np.mean(a==0)),'saturation_fraction':float(np.mean(a==255))}
        tags['green_identity_palette']=bool(palette is not None and np.array_equal(palette[:,1],np.arange(256))
                                             and not palette[:,0].any() and not palette[:,2].any())
        if not tags['green_identity_palette'] or a.dtype != np.uint8:
            raise ValueError(f'Unrecognised intensity encoding: {path}')
    return a.astype(np.float32),tags


def exclusion_mask(annotation):
    mask=np.zeros((annotation['imageHeight'],annotation['imageWidth']),bool)
    for s in annotation['shapes']:
        if s['label']=='cell':continue
        x1,y1,x2,y2=bounds(s['points'],mask.shape[1],mask.shape[0]);mask[y1:y2,x1:x2]=True
    return binary_dilation(mask,iterations=2)


def contrast(image,box,excluded,cell=None):
    x1,y1,x2,y2=map(int,box);pad=max(8,min(40,int(max(x2-x1,y2-y1)*.5)))
    a,b,c,d=max(0,x1-pad),max(0,y1-pad),min(image.shape[1],x2+pad),min(image.shape[0],y2+pad)
    eligible=~excluded[b:d,a:c].copy();eligible[y1-b:y2-b,x1-a:x2-a]=False
    if cell is not None:eligible &= cell[b:d,a:c].astype(bool)
    background=image[b:d,a:c][eligible]
    if len(background)<32:return {'contrast':np.nan,'background_n':len(background),'background_median':np.nan,'background_mad':np.nan,'signal_p90':np.nan}
    median=float(np.median(background));mad=float(np.median(np.abs(background-median)));q90=float(np.percentile(image[y1:y2,x1:x2],90))
    return {'contrast':(q90-median)/max(1.,1.4826*mad),'background_n':len(background),
            'background_median':median,'background_mad':mad,'signal_p90':q90}


def write_contact_sheet(rows, root, output, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(len(rows),2,figsize=(10,4*len(rows)),squeeze=False)
    for k,row in enumerate(rows):
        image,_=scalar_image(root/'channel2_input2'/f"{row['source_id']}.tif")
        annotation=json.loads((root/'channel2_input2/json'/f"{row['source_id']}.json").read_text())
        box=np.array([row[x] for x in ['x1','y1','x2','y2']],int)
        pad=70;a,b,c,d=max(0,box[0]-pad),max(0,box[1]-pad),min(2048,box[2]+pad),min(2048,box[3]+pad)
        crop=image[b:d,a:c]
        for col in range(2):
            ax=axes[k,col];ax.imshow(crop,cmap='gray',vmin=0,vmax=255);ax.axis('off')
            if col:
                for sh in annotation['shapes']:
                    if sh['label']=='cell':
                        p=np.asarray(sh['points']);ax.plot(np.r_[p[:,0],p[0,0]]-a,np.r_[p[:,1],p[0,1]]-b,lw=.8,color='cyan')
                    else:
                        bb=bounds(sh['points'],2048,2048)
                        if bb[2]<a or bb[0]>c or bb[3]<b or bb[1]>d:continue
                        ax.add_patch(Rectangle((bb[0]-a,bb[1]-b),bb[2]-bb[0],bb[3]-bb[1],fill=False,lw=.7,edgecolor='lime' if sh['label']=='cytosolic' else 'orange'))
                ax.add_patch(Rectangle((box[0]-a,box[1]-b),box[2]-box[0],box[3]-box[1],fill=False,lw=1.4,edgecolor='magenta'))
                ax.set_xlim(-.5,c-a-.5);ax.set_ylim(d-b-.5,-.5)
            ax.set_title(f"{row['annotation_id']} | {row['owner_status']} | C={row['contrast']:.2f}"+ (' | labels' if col else ' | original display scale'),fontsize=8)
    fig.tight_layout();fig.savefig(output/name,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    records=[];cells_out=[];image_records=[];patches=[]
    train_paths=sorted((root/'yolo_dataset/train txt file').glob('*.tif'))
    # Calibration files are withheld from likelihood-parameter fitting, including their positive labels.
    rng=np.random.default_rng(42);order=rng.permutation(len(train_paths));calibration={train_paths[i].stem for i in order[:20]}
    for split,folder in [('train','train txt file'),('heldout_development','val txt file')]:
        for path in sorted((root/'yolo_dataset'/folder).glob('*.tif')):
            j=json.loads((root/'channel2_input2/json'/f'{path.stem}.json').read_text());image,tags=scalar_image(path);cells=cell_masks(j);excluded=exclusion_mask(j)
            subset='calibration' if path.stem in calibration else 'fit' if split=='train' else 'heldout_development'
            image_records.append(dict(source_id=path.stem,split=split,subset=subset,n_cells=len(cells),**tags))
            current=[]
            for index,shape in enumerate(j['shapes']):
                if shape['label']!='cytosolic':continue
                box=bounds(shape['points'],2048,2048);ownership=owner(box,cells)
                cell=next((c['mask'] for c in cells if c['cell_id']==ownership['cell_id']),None)
                row=dict(source_id=path.stem,split=split,subset=subset,shape_index=index,annotation_id=f'{path.stem}:shape-{index}',
                         x1=box[0],y1=box[1],x2=box[2],y2=box[3],**ownership,**contrast(image,box,excluded,cell))
                pts=np.asarray(shape['points'],float)
                row.update(annotation_x1=float(np.clip(pts[:,0].min(),0,2048)),annotation_y1=float(np.clip(pts[:,1].min(),0,2048)),
                           annotation_x2=float(np.clip(pts[:,0].max(),0,2048)),annotation_y2=float(np.clip(pts[:,1].max(),0,2048)))
                records.append(row);current.append(row)
            for c in cells:
                certain=sum(r['cell_id']==c['cell_id'] for r in current)
                uncertain=sum(r['owner_status']=='ambiguous' and c['cell_id'] in r['candidate_cells'].split('|') for r in current)
                overlap=np.any(np.stack([d['mask'] for d in cells if d is not c]),axis=0) if len(cells)>1 else np.zeros_like(c['mask'])
                cells_out.append(dict(source_id=path.stem,split=split,subset=subset,cell_id=c['cell_id'],shape_index=c['shape_index'],
                   area_pixels=int(c['mask'].sum()),overlap_fraction=float(np.sum(c['mask']*overlap)/max(1,c['mask'].sum())),
                   geometry_valid=c['geometry_valid'],
                   certain_annotation_count=certain,ambiguous_annotation_count=uncertain,upper_annotation_count=certain+uncertain,
                   ownership_resolved=uncertain==0 and c['geometry_valid']))
            # Single-frame structure diagnostics, not a camera calibration or a sample of verified negatives.
            if split=='train':
                union=np.any(np.stack([c['mask'] for c in cells]),axis=0) if cells else np.zeros_like(excluded)
                candidates=[]
                for y in range(16,2032,32):
                    for x in range(16,2032,32):
                        if union[y-8:y+8,x-8:x+8].all() and not excluded[y-8:y+8,x-8:x+8].any():candidates.append((x,y))
                for i in rng.choice(len(candidates),size=min(10,len(candidates)),replace=False) if candidates else []:
                    x,y=candidates[i];v=image[y-8:y+8,x-8:x+8];yy,xx=np.mgrid[-1:1:16j,-1:1:16j];X=np.c_[np.ones(256),xx.ravel(),yy.ravel()];r=v.ravel()-X@np.linalg.lstsq(X,v.ravel(),rcond=None)[0];rr=r.reshape(16,16)
                    var=float(np.mean(r*r));corr=float(np.corrcoef(rr[:,:-1].ravel(),rr[:,1:].ravel())[0,1]) if var>0 else np.nan
                    patches.append(dict(source_id=path.stem,subset=subset,x=x,y=y,mean=float(v.mean()),variance_mle=var,
                                        lag1_correlation=corr,skew=float(skew(r)),excess_kurtosis=float(kurtosis(r)),normality_p=float(normaltest(r).pvalue)))
    objects=pd.DataFrame(records);images=pd.DataFrame(image_records);cell_table=pd.DataFrame(cells_out);noise=pd.DataFrame(patches)
    # Cutpoints use only the 60 fit source images, with each available positive retained.
    thresholds=np.quantile(objects.loc[(objects.subset=='fit')&objects.contrast.notna(),'contrast'],[1/3,2/3])
    objects['contrast_group']=np.where(objects.contrast.isna(),'unavailable',np.where(objects.contrast<=thresholds[0],'low',np.where(objects.contrast<=thresholds[1],'medium','high')))
    objects.to_csv(out/'cytosolic_ownership.csv',index=False);cell_table.to_csv(out/'cell_annotation_counts.csv',index=False);images.to_csv(out/'image_manifest.csv',index=False);noise.to_csv(out/'intensity_diagnostics.csv',index=False)
    summary={'source_images':len(images),'cells':len(cell_table),'cytosolic_annotations':len(objects),'ownership':objects.owner_status.value_counts().to_dict(),
             'ownership_by_split':pd.crosstab(objects.split,objects.owner_status).to_dict(),'contrast_cutpoints_fit_only':thresholds.tolist(),
             'contrast_definition':'(90th percentile within positive box - median local ring) / max(1, 1.4826*ring MAD); exclude all annotated objects plus a 2-pixel margin; restrict ring to reliable owner when available',
             'fit_images':int((images.subset=='fit').sum()),'calibration_images':int((images.subset=='calibration').sum()),'heldout_development_images':int((images.subset=='heldout_development').sum()),
             'ownership_rule':'unique centre-containing polygon; >=80% box coverage; second polygon <=10%; no nearest-cell assignment',
             'ownership_resolved_cells':int(cell_table.ownership_resolved.sum()),
             'invalid_cell_polygons':int((~cell_table.geometry_valid).sum()),
             'intensity':{'all_8bit_green_palette':bool(images.green_identity_palette.all()),'mean_saturation_fraction':float(images.saturation_fraction.mean()),
               'diagnostic_patches':len(noise),'median_lag1_correlation':float(noise.lag1_correlation.median()),'fraction_normality_p_below_05':float((noise.normality_p<.05).mean())},
             'limitations':['Original validation files were previously used for development; they are not an untouched independent test.',
                            'Reliable ownership is a conservative geometric assignment, not an independent cell-segmentation validation.',
                            'Unlabelled locations are not assumed to be biological negatives.',
                            'All supplied positive annotations are retained regardless of contrast or ownership.']}
    (out/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    audit=objects[(objects.subset=='fit')&objects.contrast.notna()].sort_values('contrast')
    choices=pd.concat([audit.head(3),audit[audit.owner_status!='reliable'].head(3),audit[audit.owner_status=='reliable'].tail(2)]).drop_duplicates('annotation_id')
    choices.to_csv(out/'visual_audit_selection.csv',index=False)
    write_contact_sheet(choices.to_dict('records'),root,out,'ownership_contrast_review.png')


if __name__=='__main__':main()
