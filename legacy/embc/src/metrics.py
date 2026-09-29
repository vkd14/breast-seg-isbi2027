"""
metrics.py — Evaluation metrics for segmentation.

All functions operate on torch tensors (B, 1, H, W) and return Python floats.
"""

import torch
import numpy as np


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """Per-sample Dice, averaged over batch."""
    p = (torch.sigmoid(pred) > threshold).float() if pred.requires_grad or pred.max() > 1 else (pred > threshold).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean().item()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    p = (torch.sigmoid(pred) > threshold).float() if pred.requires_grad or pred.max() > 1 else (pred > threshold).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def precision_recall_f1(pred, target, threshold=0.5, smooth=1e-6):
    """Returns (precision, recall, f1) averaged over batch."""
    p = (torch.sigmoid(pred) > threshold).float() if pred.requires_grad or pred.max() > 1 else (pred > threshold).float()
    tp = (p * target).sum(dim=(1, 2, 3))
    fp = (p * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * target).sum(dim=(1, 2, 3))
    prec = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * prec * rec / (prec + rec + smooth)
    return prec.mean().item(), rec.mean().item(), f1.mean().item()


def compute_all(pred_logits, target, threshold=0.5):
    """Return dict with all metrics for a single batch."""
    with torch.no_grad():
        d = dice_score(pred_logits, target, threshold)
        i = iou_score(pred_logits, target, threshold)
        p, r, f = precision_recall_f1(pred_logits, target, threshold)
    return dict(dice=d, iou=i, precision=p, recall=r, f1=f)


class MetricTracker:
    """Accumulates per-image metrics across batches for full-epoch stats."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._store = dict(dice=[], iou=[], precision=[], recall=[], f1=[])

    def update(self, pred_logits, target, threshold=0.5):
        m = compute_all(pred_logits, target, threshold)
        for k, v in m.items():
            self._store[k].append(v)

    def summary(self):
        out = {}
        for k, vals in self._store.items():
            a = np.array(vals)
            out[k] = dict(
                mean=float(a.mean()), std=float(a.std()),
                median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()),
                values=vals,
            )
        return out

    def mean(self, key="dice"):
        return float(np.mean(self._store[key]))
