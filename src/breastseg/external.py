"""Public external sets and locked legacy inference preprocessing."""
import csv
import zipfile
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from .data import load_bbbc039

def decode_rle(encoded,h,w):
    a=np.zeros(h*w,np.uint8)
    if encoded.strip():
        runs=np.array(encoded.split(),dtype=np.int64).reshape(-1,2)
        for start,length in runs:
            if start<1 or length<0 or start-1+length>len(a): raise ValueError('Invalid RLE')
            a[start-1:start-1+length]=1
    return a.reshape((h,w),order='F')

def load_external(root,dataset):
    root=Path(root)/dataset
    if dataset=='bbbc039': return [c for c in load_bbbc039(root) if c['split']=='test']
    cases=[]
    if dataset=='tnbc':
        with zipfile.ZipFile(root/'TNBC_NucleiSegmentation.zip') as z:
            names=sorted(n for n in z.namelist() if '/Slide_' in n and n.endswith('.png') and '__MACOSX' not in n)
            for n in names:
                rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(n),np.uint8),1),cv2.COLOR_BGR2RGB)
                mask=cv2.imdecode(np.frombuffer(z.read(n.replace('/Slide_','/GT_')),np.uint8),-1)>0
                sid=Path(n).stem
                cases.append(dict(id=sid,group=sid.split('_')[0],image=rgb,mask=mask))
        assert len(cases)==50 and len({c['group'] for c in cases})==11
    elif dataset=='bbbc038':
        labels=defaultdict(list)
        with (root/'stage1_solution.csv').open(newline='') as f:
            for r in csv.DictReader(f): labels[r['ImageId']].append(r)
        with zipfile.ZipFile(root/'stage1_test.zip') as z:
            for n in sorted(n for n in z.namelist() if '/images/' in n and n.endswith('.png')):
                sid=Path(n).stem;rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(n),np.uint8),1),cv2.COLOR_BGR2RGB)
                h,w=rgb.shape[:2];mask=np.zeros((h,w),np.uint8)
                if sid not in labels: raise ValueError('Missing labels '+sid)
                for r in labels[sid]:
                    assert (int(r['Height']),int(r['Width']))==(h,w)
                    mask|=decode_rle(r['EncodedPixels'],h,w)
                cases.append(dict(id=sid,group=sid,image=rgb,mask=mask))
        assert len(cases)==65
    else: raise ValueError(dataset)
    for c in cases:
        if c['mask'].ndim!=2 or c['image'].shape[:2]!=c['mask'].shape: raise ValueError('Invalid geometry')
    return cases

def legacy_features(image,edge):
    # Matches supplied cv2.IMREAD_COLOR + RGB + archived Gabor + A.Normalize.
    image=np.asarray(image)
    if image.dtype==np.uint16: image=(image//256).astype(np.uint8)
    if image.ndim==2: image=np.repeat(image[...,None],3,axis=2)
    if edge:
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        responses=[]
        for theta in np.linspace(0,np.pi,8,endpoint=False):
            for freq in np.linspace(.05,.25,3):
                k=cv2.getGaborKernel((31,31),5.,theta,1/freq,.5,0,ktype=cv2.CV_32F)
                k/=k.sum()+1e-8
                responses.append(np.abs(cv2.filter2D(gray,cv2.CV_32F,k)))
        e=np.maximum.reduce(responses);e=(e-e.min())/(e.max()-e.min()+1e-8)
        image=np.dstack([image,(e*255).astype(np.uint8)])
    x=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
    x=(x-np.array([.485,.456,.406]+([.5] if edge else []),np.float32))/np.array([.229,.224,.225]+([.25] if edge else []),np.float32)
    return torch.from_numpy(x.transpose(2,0,1).copy())
