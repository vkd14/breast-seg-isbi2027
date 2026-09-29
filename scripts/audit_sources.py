#!/usr/bin/env python3
"""Read-only provenance and arithmetic audit; never rewrites historical results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def inventory(root):
    return {str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}

def main():
    p=argparse.ArgumentParser();p.add_argument('source_a',type=Path);p.add_argument('source_b',type=Path);a=p.parse_args()
    ia,ib=inventory(a.source_a),inventory(a.source_b)
    common=ia.keys()&ib.keys();dif=[k for k in sorted(common) if ia[k]!=ib[k]]
    discrepancies=[];checks=[]
    for path in a.source_b.glob('results/*/test_metrics.json'):
        obj=json.loads(path.read_text())
        for metric,entry in obj.items():
            if not isinstance(entry,dict) or 'values' not in entry: continue
            computed=statistics.mean(entry['values']);stored=entry.get('mean')
            row=dict(experiment=path.parent.name,metric=metric,n=len(entry['values']),stored_mean=stored,recomputed_mean=computed,
                difference=stored-computed if stored is not None else None)
            checks.append(row)
            if stored is not None and abs(stored-computed)>1e-6: discrepancies.append(row)
    result=dict(source_a=str(a.source_a),source_b=str(a.source_b),files_a=ia,files_b=ib,identical=ia==ib,
        different_files=dif,only_a=sorted(ia.keys()-ib.keys()),only_b=sorted(ib.keys()-ia.keys()),
        metric_discrepancies=discrepancies,metric_checks=checks,
        checkpoints=[k for k in ib if k.endswith(('.pt','.pth'))],
        proposed_full_checkpoint_present=any('proposed_full/' in k and k.endswith(('.pt','.pth')) for k in ib))
    out=ROOT/'results/audit';out.mkdir(parents=True,exist_ok=True)
    (out/'source_inventory.json').write_text(json.dumps(result,indent=2))
    with (out/'historical_metric_checks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files_a','files_b','metric_checks')},indent=2))

if __name__=='__main__': main()
