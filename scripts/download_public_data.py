#!/usr/bin/env python3
"""Fetch official archives with hashes pinned to the inherited evaluation manifest."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import urllib.request
from breastseg.core import sha256

ROOT=Path(__file__).resolve().parents[1]
URLS={
 'bbbc039':{n:'https://data.broadinstitute.org/bbbc/BBBC039/'+n for n in ('images.zip','masks.zip','metadata.zip')},
 'bbbc038':{n:'https://data.broadinstitute.org/bbbc/BBBC038/'+n for n in ('stage1_test.zip','stage1_solution.csv','metadata.xlsx')},
 'tnbc':{'TNBC_NucleiSegmentation.zip':'https://zenodo.org/records/2579118/files/TNBC_NucleiSegmentation.zip?download=1'}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',choices=list(URLS),default=list(URLS));p.add_argument('--dest',type=Path,default=ROOT/'data');a=p.parse_args()
    for ds in a.datasets:
        expected=json.loads((ROOT/'legacy/external_validation_20260928/outputs'/f'{ds}_zero_shot/protocol.json').read_text())['files']
        dest=a.dest/ds;dest.mkdir(parents=True,exist_ok=True)
        for name,url in URLS[ds].items():
            final=dest/name
            if final.exists():
                if name in expected and sha256(final)!=expected[name]: raise ValueError(f'Hash mismatch; preserved {final}')
                print('Verified',final);continue
            with tempfile.NamedTemporaryFile(dir=dest,suffix='.download',delete=False) as f: temp=Path(f.name)
            urllib.request.urlretrieve(url,temp)
            if name in expected and sha256(temp)!=expected[name]: raise ValueError(f'Download checksum differs: {temp}')
            os.replace(temp,final);print('Downloaded',final)

if __name__=='__main__':main()
