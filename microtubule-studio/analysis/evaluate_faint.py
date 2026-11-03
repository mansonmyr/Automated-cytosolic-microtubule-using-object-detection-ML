"""Development-source evaluation with a restricted-negative LRT experiment and explicit ownership uncertainty."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/per-cell-mpl')
os.environ.setdefault('YOLO_OFFLINE','true')
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from ownership_audit import scalar_image,cell_masks,owner,bounds,exclusion_mask,contrast,write_contact_sheet
from filament_likelihood import FilamentLikelihood,patch_at,threshold_at_fpr


def ious(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    intersection=wh.prod(axis=2);aa=(a[:,2:]-a[:,:2]).prod(axis=1);bb=(b[:,2:]-b[:,:2]).prod(axis=1)
    return intersection/np.maximum(1e-12,aa[:,None]+bb[None,:]-intersection)


def match_predictions(predictions,positives,score):
    matched={};tp=pd.Series(False,index=predictions.index)
    for source,group in predictions.groupby('source_id'):
        gt=positives[positives.source_id==source]
        matrix=ious(group[['x1','y1','x2','y2']],gt[['annotation_x1','annotation_y1','annotation_x2','annotation_y2']]);used=set()
        locations={idx:k for k,idx in enumerate(group.index)}
        for idx in group.sort_values(score,ascending=False).index:
            overlaps=matrix[locations[idx]]
            for k in np.argsort(-overlaps):
                if overlaps[k]<.5:break
                if k not in used:
                    used.add(int(k));tp.loc[idx]=True;matched[idx]=int(gt.index[k]);break
    return tp,matched


def detection_summary(predictions,positives,score,threshold):
    selected=predictions[predictions[score]>=threshold]
    tp,matched=match_predictions(selected,positives,score)
    found=set(matched.values());groups={}
    for group,gt in positives.groupby('contrast_group'):
        hits=sum(k in found for k in gt.index);groups[group]={'positives':len(gt),'detected':hits,'recall':hits/len(gt)}
    precision=float(tp.sum()/len(selected)) if len(selected) else 0.
    recall=len(found)/len(positives)
    # All-points AP at IoU .5, using the retained proposal scores (no test-set tuning).
    all_tp,_=match_predictions(predictions,positives,score);order=predictions.sort_values(score,ascending=False).index
    ctp=np.cumsum(all_tp.loc[order].to_numpy());p=ctp/np.maximum(1,np.arange(len(ctp))+1);r=ctp/max(1,len(positives))
    ap=float(np.mean([p[r>=q].max() if np.any(r>=q) else 0 for q in np.linspace(0,1,101)]))
    return {'threshold':threshold,'precision_at_threshold':precision,'recall_at_threshold':recall,'AP50_proposal_ranking':ap,
            'selected_predictions':len(selected),'unmatched_predictions':int((~tp).sum()),'contrast_groups':groups},found


def score_batches(model,patches):
    return np.concatenate([model.score(patches[i:i+128]) for i in range(0,len(patches),128)]) if len(patches) else np.array([])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--imgsz',type=int,default=640);args=p.parse_args()
    root=args.root.resolve();out=args.audit.resolve();gt=pd.read_csv(out/'cytosolic_ownership.csv');manifest=pd.read_csv(out/'image_manifest.csv');cells_table=pd.read_csv(out/'cell_annotation_counts.csv')
    assert 'annotation_x1' in gt,'Run the current ownership audit first.'
    positives=[];negative_rows=[];negative_patches=[]
    for row in manifest.to_dict('records'):
        source=row['source_id'];im,_=scalar_image(root/'channel2_input2'/f'{source}.tif');j=json.loads((root/'channel2_input2/json'/f'{source}.json').read_text());excluded=exclusion_mask(j)
        current=gt[gt.source_id==source]
        for idx,r in current.iterrows():positives.append((idx,patch_at(im,(r.annotation_x1+r.annotation_x2)/2,(r.annotation_y1+r.annotation_y2)/2)))
        for index,s in enumerate(j['shapes']):
            if s['label']!='centrosomal':continue
            bb=bounds(s['points'],2048,2048);cx=(bb[0]+bb[2]-1)/2;cy=(bb[1]+bb[3]-1)/2
            # An explicitly annotated different class is a restricted negative, not generic unlabelled background.
            patch_box=[cx-15,cy-15,cx+16,cy+16]
            if len(current) and np.any(ious([patch_box],current[['annotation_x1','annotation_y1','annotation_x2','annotation_y2']])>0):continue
            negative_rows.append(dict(source_id=source,subset=row['subset'],shape_index=index,x1=bb[0],y1=bb[1],x2=bb[2],y2=bb[3],**contrast(im,bb,excluded)))
            negative_patches.append(patch_at(im,cx,cy))
    positives.sort(key=lambda x:x[0]);patches=np.stack([x[1] for x in positives]);negative_patches=np.asarray(negative_patches);neg=pd.DataFrame(negative_rows)
    model=FilamentLikelihood();fit=model.fit(patches[(gt.subset=='fit').to_numpy()],negative_patches[(neg.subset=='fit').to_numpy()])
    fit.update(fit_source_ids=manifest.loc[manifest.subset=='fit','source_id'].tolist(),calibration_source_ids=manifest.loc[manifest.subset=='calibration','source_id'].tolist())
    (out/'likelihood_model.json').write_text(json.dumps(fit,indent=2)+'\n')
    gt['log_likelihood_ratio']=score_batches(model,patches);neg['log_likelihood_ratio']=score_batches(model,negative_patches)
    gt.to_csv(out/'cytosolic_likelihoods.csv',index=False);neg.to_csv(out/'annotated_centrosomal_controls.csv',index=False)
    restricted={}
    for score in ['contrast','log_likelihood_ratio']:
        cal=neg[(neg.subset=='calibration')&neg[score].notna()][score].to_numpy();threshold=threshold_at_fpr(cal,.05)
        test_neg=neg[(neg.subset=='heldout_development')&neg[score].notna()];k=int((test_neg[score]>=threshold).sum());n=len(test_neg);ci=binomtest(k,n).proportion_ci() if n else None
        groups={}
        for category,group in gt[gt.subset=='heldout_development'].groupby('contrast_group'):
            eligible=group[score].notna();groups[category]={'positives':int(eligible.sum()),'accepted':int((group.loc[eligible,score]>=threshold).sum()),'recall':float((group.loc[eligible,score]>=threshold).mean()) if eligible.any() else None}
        restricted[score]={'calibration_threshold':threshold,'target_FPR':.05,'calibration_negatives':len(cal),'calibration_FPR':float((cal>=threshold).mean()),
                           'heldout_negative_regions':n,'heldout_false_acceptances':k,'heldout_FPR':k/n if n else None,
                           'heldout_FPR_95CI':[float(ci.low),float(ci.high)] if ci else None,'contrast_groups':groups}
    (out/'restricted_negative_experiment.json').write_text(json.dumps(restricted,indent=2)+'\n')
    print('Restricted-negative likelihood experiment completed.',flush=True)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1);detector=YOLO(str(args.weights.resolve()));detector.model.names={0:'cytosolic',1:'centrosomal',2:'cell'}
    predictions=[]
    for index,row in enumerate(manifest.to_dict('records')):
        source=row['source_id'];im,_=scalar_image(root/'channel2_input2'/f'{source}.tif');j=json.loads((root/'channel2_input2/json'/f'{source}.json').read_text());cell_info=cell_masks(j)
        result=detector.predict(str(root/'channel2_input2'/f'{source}.tif'),imgsz=args.imgsz,conf=.001,iou=.7,max_det=300,device='cpu',verbose=False)[0]
        boxes=result.boxes.data.cpu().numpy();boxes=boxes[boxes[:,5]==0]
        features=[patch_at(im,(b[0]+b[2])/2,(b[1]+b[3])/2) for b in boxes]
        scores=score_batches(model,np.asarray(features))
        for k,(b,lr) in enumerate(zip(boxes,scores)):
            bb=[max(0,int(np.floor(b[0]))),max(0,int(np.floor(b[1]))),min(2048,int(np.ceil(b[2]))+1),min(2048,int(np.ceil(b[3]))+1)]
            predictions.append(dict(source_id=source,subset=row['subset'],prediction_id=f'{source}:prediction-{k}',x1=float(b[0]),y1=float(b[1]),x2=float(b[2]),y2=float(b[3]),yolo_score=float(b[4]),log_likelihood_ratio=float(lr),**owner(bb,cell_info)))
        if (index+1)%10==0:print(f'Inference {index+1}/{len(manifest)} source images',flush=True)
    pred=pd.DataFrame(predictions);pred.to_csv(out/'all_cytosolic_candidates.csv',index=False)
    thresholds={'yolo_default':('yolo_score',.25)}
    # This is a reference-unmatched rate, not a biological FPR when missing labels are possible.
    calibration=pred[pred.subset=='calibration'];calgt=gt[gt.subset=='calibration'];n_cells=int((cells_table.subset=='calibration').sum());allowed=int(np.floor(.1*n_cells))
    for method,score in [('yolo_rate_matched','yolo_score'),('filament_rate_matched','log_likelihood_ratio')]:
        tp,_=match_predictions(calibration,calgt,score);false=calibration.loc[(~tp)&(calibration.owner_status=='reliable'),score].sort_values(ascending=False).to_numpy()
        threshold=float(np.nextafter(false[allowed],np.inf)) if len(false)>allowed else float(false.min()-1) if len(false) else 0.
        thresholds[method]=(score,threshold)
    summary={'interpretation':'Development-source diagnostic only. Previously used validation files; reference-unmatched proposals are not confirmed false positives.',
             'candidate_generator':f'YOLOv10n checkpoint, whole source images at input size {args.imgsz}; cytosolic proposals with score >= .001, max 300 detections across classes.',
             'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
             'rate_matching':'Thresholds fixed using 20 calibration source images to allow <=0.1 reference-unmatched, reliably owned proposals per annotated cell. These images were not used for likelihood fitting, but the earlier detector had trained on them.',
             'methods':{},'restricted_negative_experiment':restricted}
    heldout=pred[pred.subset=='heldout_development'];test_gt=gt[gt.subset=='heldout_development'];cell_rows=[];found_by_method={}
    for method,(score,threshold) in thresholds.items():
        result,found=detection_summary(heldout,test_gt,score,threshold);found_by_method[method]=found;summary['methods'][method]=result
        pred['accepted_'+method]=pred[score]>=threshold
        chosen=heldout[heldout[score]>=threshold]
        for cell in cells_table[cells_table.subset=='heldout_development'].to_dict('records'):
            pcell=chosen[chosen.source_id==cell['source_id']]
            low=int((pcell.cell_id==cell['cell_id']).sum());uncertain=int(sum(cell['cell_id'] in str(s).split('|') for s in pcell.loc[pcell.owner_status=='ambiguous','candidate_cells']))
            cell_rows.append(dict(method=method,source_id=cell['source_id'],cell_id=cell['cell_id'],reference_lower=cell['certain_annotation_count'],reference_upper=cell['upper_annotation_count'],ownership_resolved=cell['ownership_resolved'],predicted_lower=low,predicted_upper=low+uncertain,ambiguous_predictions=uncertain))
    counts=pd.DataFrame(cell_rows);counts.to_csv(out/'heldout_per_cell_counts.csv',index=False)
    keys=['source_id','cell_id'];common=counts.groupby(keys).agg(resolved=('ownership_resolved','all'),ambiguous=('ambiguous_predictions','max'));common=common[common.resolved&(common.ambiguous==0)].reset_index()[keys]
    for method in thresholds:
        c=counts[counts.method==method].merge(common,on=keys);error=c.predicted_lower-c.reference_lower
        summary['methods'][method]['per_cell_annotation_count_error']={'paired_ownership_resolved_cells':len(c),'MAE':float(np.abs(error).mean()),'bias':float(error.mean()),'RMSE':float(np.sqrt(np.mean(error**2)))}
    # Source-cluster bootstrap for the paired faint recall difference.
    faint=test_gt[test_gt.contrast_group=='low'];ids=sorted(test_gt.source_id.unique());rng=np.random.default_rng(2026);diff=[]
    source_counts={s:(len(faint[faint.source_id==s]),sum(i in found_by_method['filament_rate_matched'] for i in faint[faint.source_id==s].index),sum(i in found_by_method['yolo_rate_matched'] for i in faint[faint.source_id==s].index)) for s in ids}
    for _ in range(1000):
        chosen=rng.choice(ids,len(ids),replace=True);n,a,b=np.asarray([source_counts[s] for s in chosen]).sum(axis=0)
        if n:diff.append(float((a-b)/n))
    summary['faint_recall_difference_cluster_bootstrap_95CI']=np.quantile(diff,[.025,.975]).tolist()
    pred['review_required']=(pred.owner_status!='reliable')|(pred.accepted_yolo_rate_matched!=pred.accepted_filament_rate_matched)
    pred.to_csv(out/'all_cytosolic_candidates.csv',index=False)
    test_gt=test_gt.copy()
    for method,found in found_by_method.items():test_gt['detected_'+method]=[i in found for i in test_gt.index]
    test_gt.to_csv(out/'heldout_detection_failures.csv',index=False)
    # Review contexts are selected from fit images only; no automatic new-positive decisions.
    fitpred=pred[pred.subset=='fit'];_,matched=match_predictions(fitpred,gt[gt.subset=='fit'],'yolo_score')
    candidates=fitpred.loc[~fitpred.index.isin(matched.keys())&(fitpred.owner_status=='reliable')].sort_values('log_likelihood_ratio',ascending=False)
    selected=[];per_source={}
    for idx,r in candidates.iterrows():
        if per_source.get(r.source_id,0)>=2:continue
        current=gt[gt.source_id==r.source_id]
        if len(current) and ious([r[['x1','y1','x2','y2']]],current[['annotation_x1','annotation_y1','annotation_x2','annotation_y2']]).max()>.05:continue
        im,_=scalar_image(root/'channel2_input2'/f'{r.source_id}.tif');j=json.loads((root/'channel2_input2/json'/f'{r.source_id}.json').read_text());bb=list(r[['x1','y1','x2','y2']].astype(int));ct=contrast(im,bb,exclusion_mask(j))
        if not np.isfinite(ct['contrast']) or ct['contrast']>json.loads((out/'audit_summary.json').read_text())['contrast_cutpoints_fit_only'][0]:continue
        selected.append(dict(r,annotation_id=r.prediction_id,contrast=ct['contrast'],decision='ambiguous',reviewer='',reason='Needs visual and biological confirmation; not an automatic positive.'))
        per_source[r.source_id]=per_source.get(r.source_id,0)+1
        if len(selected)>=8:break
    if selected:
        pd.DataFrame(selected).to_csv(out/'missing_label_review.csv',index=False)
        write_contact_sheet(selected,root,out,'missing_faint_label_review.png')
    summary['missing_label_audit']={'candidate_contexts':len(selected),'fit_source_images':len(per_source),'verified_missing_positives':0,'correction_activated':False,
        'status':'Candidates remain ambiguous pending biological review. Unmatched does not mean false positive or missing true label.'}
    (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
