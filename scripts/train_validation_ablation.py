#!/usr/bin/env python3
"""Validation-only Gabor and sampling sensitivity study; never decodes test cases."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from breastseg.core import (Net, SegmentationLoss, features, metrics,
                            sampling_weights, sha256)
from breastseg.data import load_bbbc039

ROOT=Path(__file__).resolve().parents[1]
CODE_PATHS=(Path(__file__),ROOT/'src/breastseg/core.py',ROOT/'src/breastseg/data.py')


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def protocol_record(config,data):
    files={str(p.relative_to(ROOT)):sha256(p) for p in CODE_PATHS}
    archives={p.name:sha256(p) for p in sorted(data.glob('*.zip'))}
    record=dict(config=json.loads(config.read_text()),code_sha256=files,archive_sha256=archives)
    digest=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return digest,record


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def state_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Cached(Dataset):
    def __init__(self,cases,key,augment=False):self.cases,self.key,self.augment=cases,key,augment
    def __len__(self):return len(self.cases)
    def __getitem__(self,index):
        case=self.cases[index];x,y=case[self.key],case['y']
        if self.augment:
            k=random.randrange(4);x,y=torch.rot90(x,k,(-2,-1)),torch.rot90(y,k,(-2,-1))
            if random.random()<.5:x,y=x.flip(-1),y.flip(-1)
        return x,y,index


@torch.inference_mode()
def evaluate(model,loader,cases,full=False):
    model.eval();rows=[]
    for x,_,indices in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x.cuda(non_blocking=True))
        probabilities=logits.float().sigmoid().cpu().numpy()[:,0]
        for probability,index in zip(probabilities,indices.tolist()):
            case=cases[index];reference=case['mask'].astype(bool)
            prediction=cv2.resize(probability,(reference.shape[1],reference.shape[0]),interpolation=cv2.INTER_LINEAR)>.5
            if full:score=metrics(prediction,reference)
            else:
                denominator=prediction.sum()+reference.sum()
                score={'dice':float(2*(prediction&reference).sum()/denominator) if denominator else 1.}
                score['empty_reference']=int(not reference.any())
            rows.append(dict(id=case['id'],group=case['group'],**score))
    if len(rows)!=len(cases):raise RuntimeError('Incomplete validation evaluation')
    return rows


def condition_features(case,condition,size):
    kind=condition['edge_kind']
    params=None
    if kind=='gabor':
        frequencies=condition['frequencies']
        params=dict(orientations=condition['orientations'],scales=len(frequencies),
                    frequencies=frequencies,sigma=condition['sigma'])
    return features(case['image'],True,size,kind,params)


def run_one(cfg,condition,seed,cases,protocol_hash):
    study=cfg['study'];name=condition['name'];run=f'{name}_s{seed}'
    out=ROOT/'results'/study/run;checkpoint_dir=ROOT/'checkpoints'/study/run
    out.mkdir(parents=True,exist_ok=True);checkpoint_dir.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        old=json.loads((out/'summary.json').read_text())
        if old['protocol_sha256']!=protocol_hash:raise RuntimeError(f'Protocol differs for {run}')
        print('SKIP completed',run,flush=True);return old
    if (out/'history.json').exists():raise RuntimeError(f'Partial run exists: {out}; inspect before restarting')
    seed_all(seed);started=time.time();model=Net('gabor_b7',pretrained=True).cuda();initial=state_digest(model)
    train=[c for c in cases if c['split']=='train'];validation=[c for c in cases if c['split']=='val']
    generator=torch.Generator().manual_seed(seed)
    weights=sampling_weights([c['mask'] for c in train],condition['weighted_sampling'])
    sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(train),replacement=True,generator=generator) if weights is not None else None
    train_loader=DataLoader(Cached(train,name,True),batch_size=cfg['batch_size'],sampler=sampler,
        shuffle=sampler is None,generator=generator if sampler is None else None,num_workers=0,pin_memory=True,drop_last=False)
    validation_loader=DataLoader(Cached(validation,name),batch_size=cfg['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
    loss_function=SegmentationLoss();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['learning_rate']*.01)
    history=[];best=-1.;best_epoch=None
    for epoch in range(1,cfg['epochs']+1):
        model.train();losses=[];epoch_started=time.time()
        for x,y,_ in train_loader:
            optimizer.zero_grad(set_to_none=True);x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=loss_function(model(x),y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
            losses.append((loss.item(),len(x)))
        rows=evaluate(model,validation_loader,validation)
        foreground=[r['dice'] for r in rows if not r['empty_reference']]
        if not foreground:raise RuntimeError('Validation partition has no foreground-containing images')
        score=float(np.mean(foreground));allcase_score=float(np.mean([r['dice'] for r in rows]))
        if score>best:
            best,best_epoch=score,epoch
            torch.save(dict(model_state_dict=model.state_dict(),condition=condition,seed=seed,epoch=epoch,
                val_foreground_dice=best,protocol_sha256=protocol_hash),checkpoint_dir/'best.pt')
        record=dict(epoch=epoch,train_loss=sum(value*n for value,n in losses)/sum(n for _,n in losses),
            val_foreground_dice=score,val_allcase_dice=allcase_score,best_val_foreground_dice=best,
            learning_rate=optimizer.param_groups[0]['lr'],seconds=time.time()-epoch_started)
        history.append(record);save_json(out/'history.json',history)
        save_json(ROOT/'results'/study/'status.json',dict(state='training',run=run,**record))
        print(json.dumps(dict(run=run,**record)),flush=True);scheduler.step()
    saved=torch.load(checkpoint_dir/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['model_state_dict'],strict=True)
    rows=evaluate(model,validation_loader,validation,full=True)
    with (out/'validation_per_image.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(condition=condition,seed=seed,n_validation=len(rows),epochs=cfg['epochs'],best_epoch=best_epoch,
        best_val_foreground_dice=best,parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,
        checkpoint_sha256=sha256(checkpoint_dir/'best.pt'),protocol_sha256=protocol_hash,seconds=time.time()-started,
        test_images_accessed=0,mean={key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in ('id','group')})
    save_json(out/'summary.json',summary);print('COMPLETE '+json.dumps(summary),flush=True)
    del model,optimizer;torch.cuda.empty_cache();return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/public_validation_ablation.json')
    parser.add_argument('--conditions',nargs='+');parser.add_argument('--seeds',nargs='+',type=int);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());protocol_hash,record=protocol_record(args.config,args.data)
    cv2.setNumThreads(1);torch.set_num_threads(8)
    if hasattr(cv2,'setLogLevel'):cv2.setLogLevel(0)
    # Archive files are hashed for identity, but the loader decodes only train/validation cases.
    cases=load_bbbc039(args.data,splits=('train','val'))
    result_root=ROOT/'results'/cfg['study'];result_root.mkdir(parents=True,exist_ok=True)
    save_json(result_root/'protocol.json',dict(protocol_sha256=protocol_hash,**record))
    save_json(result_root/'split_manifest.json',[{k:v for k,v in c.items() if k not in ('image','mask')} for c in cases])
    selected=[c for c in cfg['conditions'] if args.conditions is None or c['name'] in args.conditions]
    if args.conditions is not None and len(selected)!=len(set(args.conditions)):raise ValueError('Unknown or duplicate condition name')
    print(f'Caching {len(cases)} train/validation images for {len(selected)} conditions; test cases are not decoded',flush=True)
    for condition in selected:
        name=condition['name']
        for case in cases:case[name]=condition_features(case,condition,cfg['image_size'])
    for case in cases:
        case['y']=torch.from_numpy(cv2.resize(case['mask'],(cfg['image_size'],cfg['image_size']),interpolation=cv2.INTER_NEAREST).astype(np.float32)[None])
    summaries=[]
    for seed in (args.seeds if args.seeds is not None else cfg['seeds']):
        for condition in selected:summaries.append(run_one(cfg,condition,seed,cases,protocol_hash))
    for seed in set(summary['seed'] for summary in summaries):
        hashes={summary['initial_state_sha256'] for summary in summaries if summary['seed']==seed}
        if len(hashes)!=1:raise RuntimeError(f'Initial states differ within seed {seed}')
    save_json(result_root/'status.json',dict(state='complete',protocol_sha256=protocol_hash,runs=len(summaries),test_images_accessed=0))


if __name__=='__main__':main()
