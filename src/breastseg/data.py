import zipfile
from pathlib import Path
import hashlib
import cv2
import numpy as np
from .core import binary_mask, validate_split


def load_bbbc039(root, splits=("train","val","test")):
    root=Path(root)
    iz,mz,metadata=(zipfile.ZipFile(root/name) for name in ("images.zip","masks.zip","metadata.zip"))
    cases=[]
    filenames={"train":"training.txt","val":"validation.txt","test":"test.txt"}
    splits=tuple(splits)
    if not splits or not set(splits).issubset(filenames) or len(set(splits))!=len(splits):
        raise ValueError("splits must be a nonempty unique subset of train/val/test")
    for split in splits:
        filename=filenames[split]
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
    expected={"train":100,"val":50,"test":50}
    counts={s:sum(c["split"]==s for c in cases) for s in splits}
    if counts!={s:expected[s] for s in splits}:
        raise ValueError(f"Unexpected BBBC039 split counts: {counts}")
    return cases
