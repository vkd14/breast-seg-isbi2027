#!/usr/bin/env python3
"""Evaluate every requested checkpoint, retaining failures and all native-grid masks."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from breastseg.core import Net,features,metrics,sha256
from breastseg.external import load_external,legacy_features

ROOT=Path(__file__).resolve().parents[1]

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def legacy_spec(name):
    if name.startswith('sota_'): return name[5:],False
    return {'ablation_no_gabor':('proposed',False),'ablation_no_scse':('ablation_no_scse',True),
        'ablation_unet_not_unetpp':('ablation_unet_eb7',True),
        'ablation_no_weighted_sampling':('proposed',True),'ablation_standard_loss':('proposed',True)}[name]

@torch.inference_mode()
def evaluate(model,cs,xs,out,checkpoint,metadata):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): print('SKIP',out,flush=True);return
    (out/'predictions').mkdir(exist_ok=True);model.cuda().eval();rows=[];start=time.time()
    for first in range(0,len(cs),8):
        x=torch.stack(xs[first:first+8]).cuda()
        # Float32 for checkpoint re-evaluation; new training used bfloat16.
        probs=model(x).sigmoid().cpu().numpy()[:,0]
        for c,p in zip(cs[first:first+8],probs):
            g=c['mask'];pred=cv2.resize(p,(g.shape[1],g.shape[0]),interpolation=cv2.INTER_LINEAR)>.5
            rows.append(dict(id=c['id'],group=c['group'],**metrics(pred,g)))
            if not cv2.imwrite(str(out/'predictions'/f"{c['id']}.png"),pred.astype(np.uint8)*255): raise IOError('Prediction write failed')
    with (out/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(**metadata,checkpoint_sha256=sha256(checkpoint),n=len(rows),seconds=time.time()-start,
        precision='float32',threshold=.5,tta=False,mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('id','group')})
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',json.dumps(summary),flush=True)
    model.cpu();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--legacy-a',type=Path);p.add_argument('--legacy-b',type=Path)
    p.add_argument('--new-only',action='store_true');p.add_argument('--legacy-only',action='store_true');a=p.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(8)
    if hasattr(cv2,'setLogLevel'): cv2.setLogLevel(0)
    failures=[]
    if not a.legacy_only:
        cs=load_external(a.data,'tnbc');xs=[features(c['image'],True) for c in cs]
        x_sobel=[features(c['image'],True,edge_kind='sobel') for c in cs]
        for checkpoint in sorted((ROOT/'checkpoints').glob('public*/*/best.pt')):
            state=torch.load(checkpoint,map_location='cpu',weights_only=True);model=Net(state['variant'])
            model.load_state_dict(state['model_state_dict'],strict=True)
            xx=x_sobel if 'sobel' in state['variant'] else (xs if model.use_edge else [x[:3] for x in xs])
            evaluate(model,cs,xx,ROOT/'results/tnbc_transfer'/checkpoint.parent.name,checkpoint,
                dict(dataset='tnbc',variant=state['variant'],seed=state['seed'],training='BBBC039 official train only',selection='BBBC039 validation only',
                    note='Cross-domain transfer, not in-domain TNBC training; no tuning on TNBC'))
            del model
    if not a.new_only:
        for dataset in ('bbbc039','bbbc038','tnbc'):
            cs=load_external(a.data,dataset);xs=[legacy_features(c['image'],True) for c in cs]
            for label,source in (('latest',a.legacy_a),('older',a.legacy_b)):
                if source is None: continue
                factory=module(source/'src/models.py','archived_models_'+label)
                for checkpoint in sorted((source/'results').glob('*/best_*.pt')):
                    name=checkpoint.parent.name;key,edge=legacy_spec(name)
                    try:
                        model,_=factory.build_model(key,in_channels=4 if edge else 3,encoder_weights=None)
                        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                        model.load_state_dict(state['model_state_dict'],strict=True)
                        evaluate(model,cs,xs if edge else [x[:3] for x in xs],ROOT/'results/legacy_external'/dataset/(label+'_'+name),checkpoint,
                            dict(dataset=dataset,source_version=label,experiment=name,checkpoint_epoch=state.get('epoch'),
                                note='Frozen historical private-trained checkpoint; training budgets and private splits not independently verified; not a matched-budget SOTA comparison'))
                        del model,state;torch.cuda.empty_cache()
                    except Exception as e:
                        failures.append(dict(dataset=dataset,version=label,experiment=name,error=repr(e)))
                        print('FAILED',failures[-1],flush=True)
    (ROOT/'results/external_failures.json').write_text(json.dumps(failures,indent=2))
    if failures: raise RuntimeError(f'{len(failures)} checkpoint evaluations failed; see external_failures.json')

if __name__=='__main__': main()
