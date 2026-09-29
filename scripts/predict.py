#!/usr/bin/env python3
"""Strict single-image inference from a new public-study checkpoint."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from breastseg.core import Net,features

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args()
    if a.output.exists():raise FileExistsError(f'Output exists; choose a new filename: {a.output}')
    image=cv2.imread(str(a.image),cv2.IMREAD_UNCHANGED)
    if image is None:raise ValueError(f'Unreadable input {a.image}')
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=Net(state['variant'])
    model.load_state_dict(state['model_state_dict'],strict=True);model.to(a.device).eval()
    x=features(image,model.use_edge,edge_kind='sobel' if 'sobel' in state['variant'] else 'gabor')[None].to(a.device)
    with torch.inference_mode():prob=model(x).sigmoid().cpu().numpy()[0,0]
    mask=(cv2.resize(prob,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)>.5).astype(np.uint8)*255
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(a.output),mask):raise IOError('Output write failed')
    print(f'Saved native-grid nuclear foreground mask: {a.output}; variant={state["variant"]}; threshold=0.5; no TTA')

if __name__=='__main__':main()
