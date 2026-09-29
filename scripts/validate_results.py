#!/usr/bin/env python3
"""Recompute every new saved prediction's metrics and audit run contracts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from breastseg.core import metrics,validate_split,sha256
from breastseg.external import load_external

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(1)
    cases={ds:{c['id']:c for c in load_external(a.data,ds)} for ds in ('bbbc039','bbbc038','tnbc')}
    foreground_counts={ds:dict(images=len(cs),foreground=sum(bool(c['mask'].any()) for c in cs.values()),empty=sum(not c['mask'].any() for c in cs.values())) for ds,cs in cases.items()}
    tasks=[];nfiles=0
    for path in sorted((ROOT/'results').rglob('*per_image.csv')):
        if 'public_study' in path.parts or 'public_followup' in path.parts: ds='bbbc039'
        elif 'tnbc_transfer' in path.parts: ds='tnbc'
        elif 'legacy_external' in path.parts: ds=path.parent.parent.name
        else: continue
        with path.open(newline='') as f: rows=list(csv.DictReader(f))
        if len(rows)!=len(cases[ds]) or {r['id'] for r in rows}!=set(cases[ds]): raise AssertionError(f'Incomplete set {path}')
        summary=json.loads((path.parent/'summary.json').read_text())
        for metric in ('dice','iou','boundary_f1_2px','surface_dice_2px','hd95_px'):
            assert abs(np.mean([float(r[metric]) for r in rows])-summary['mean'][metric])<1e-10,(path,metric)
        tasks.extend((path.parent,r,cases[ds][r['id']]['mask']) for r in rows);nfiles+=1
    def check(task):
        parent,row,gt=task;p=cv2.imread(str(parent/'predictions'/f"{row['id']}.png"),-1)
        if p is None or not set(np.unique(p)).issubset({0,255}): raise AssertionError(f'Invalid mask {parent}/{row["id"]}')
        actual=metrics(p>0,gt)
        for k,v in actual.items():
            if abs(float(row[k])-v)>1e-9: raise AssertionError((parent,row['id'],k,row[k],v))
        return 1
    with ThreadPoolExecutor(max_workers=8) as pool: checked=sum(pool.map(check,tasks))
    completed=[]
    for study in ('public_study','public_followup'):
        config=json.loads((ROOT/'configs'/f'{study}.json').read_text());manifest=json.loads((ROOT/'results'/study/'split_manifest.json').read_text());validate_split(manifest)
        runs=[]
        for seed in config['seeds']:
            for variant in config['variants']:
                out=ROOT/'results'/study/f'{variant}_s{seed}'
                s=json.loads((out/'summary.json').read_text());h=json.loads((out/'history.json').read_text())
                assert len(h)==config['epochs'] and s['best_epoch']==max(h,key=lambda r:r['val_dice'])['epoch']
                assert s['protocol_sha256']==sha256(ROOT/'configs'/f'{study}.json')
                assert s['checkpoint_sha256']==sha256(ROOT/'checkpoints'/study/out.name/'best.pt')
                runs.append(s)
            same=('gabor_b7','gabor_boundary_b7') if study=='public_study' else ('residual_gabor_b7','residual_sobel_b7','residual_gabor_boundary_b7')
            hashes={r['initial_state_sha256'] for r in runs if r['seed']==seed and r['variant'] in same}
            assert len(hashes)==1,('Initialization differs',study,seed)
        completed.append(dict(study=study,completed_runs=len(runs),test_predictions=len(runs)*50,initialization_pairing_verified=True))
    result=dict(status='PASS',csv_files=nfiles,masks_recomputed=checked,studies=completed,dataset_counts=foreground_counts,
        note='Software consistency checks do not establish biological independence or clinical validity.')
    (ROOT/'results/audit/result_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
