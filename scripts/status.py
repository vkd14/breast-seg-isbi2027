#!/usr/bin/env python3
"""Print training progress and exact local checkpoint locations without loading torch."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for study in ('public_study','public_followup'):
    path=root/'results'/study/'status.json'
    print(study,json.loads(path.read_text()) if path.exists() else 'not started')
    for path in sorted((root/'results'/study).glob('*/summary.json')):
        s=json.loads(path.read_text())
        print(f"  {path.parent.name:35s} epoch {s['best_epoch']:2d} test Dice {s['mean']['dice']:.5f}")
        print('   ',root/'checkpoints'/study/path.parent.name/'best.pt')
print('External checkpoint/dataset evaluations:',len(list((root/'results/legacy_external').glob('*/*/summary.json'))),'/ 72')
print('New-model TNBC transfer evaluations:',len(list((root/'results/tnbc_transfer').glob('*/summary.json'))),'/ 21')
study='public_validation_ablation';path=root/'results'/study/'status.json'
print(study,json.loads(path.read_text()) if path.exists() else 'not started')
completed=sorted((root/'results'/study).glob('*/summary.json'))
print('Validation-only sensitivity runs:',len(completed),'/ 30; BBBC039 test cases decoded/evaluated: 0')
for path in completed:
    s=json.loads(path.read_text())
    print(f"  {path.parent.name:35s} epoch {s['best_epoch']:2d} foreground-val Dice {s['best_val_foreground_dice']:.5f}")
