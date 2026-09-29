import zipfile
from pathlib import Path
import hashlib
import cv2
import numpy as np
from .core import binary_mask, validate_split


def load_bbbc039(root):
    root=Path(root)
    iz,mz,metadata=(zipfile.ZipFile(root/name) for name in ("images.zip","masks.zip","metadata.zip"))
    cases=[]
    for split,filename in (("train","training.txt"),("val","validation.txt"),("test","test.txt")):
        for name in sorted(metadata.read("metadata/"+filename).decode().splitlines()):
            sid=Path(name.strip()).stem
            if not sid:
                continue
            raw=cv2.imdecode(np.frombuffer(iz.read("images/"+sid+".tif"),np.uint8),-1)
            mask=cv2.imdecode(np.frombuffer(mz.read("masks/"+sid+".png"),np.uint8),1)
            if raw is None or mask is None:
                raise ValueError("Unreadable image/mask "+sid)
            gt=binary_mask(mask[...,2])
            if raw.shape!=gt.shape:
                raise ValueError("Mismatched geometry "+sid)
            digest=hashlib.sha256(raw.tobytes()).hexdigest()
            cases.append(dict(id=sid,group=sid,split=split,image_sha256=digest,image=raw,mask=gt))
    validate_split(cases)
    assert {s:sum(c["split"]==s for c in cases) for s in ("train","val","test")}=={"train":100,"val":50,"test":50}
    return cases
