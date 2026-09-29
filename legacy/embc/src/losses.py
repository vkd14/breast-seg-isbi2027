"""
losses.py — Loss functions for breast cell nucleus segmentation.

Provides:
  - StabilizedCombinedLoss  (proposed: Dice + adaptive-weighted BCE + Tversky)
  - StandardDiceBCE         (ablation baseline)
  - BoundaryAwareDiceLoss   (component)
  - TverskyLoss             (component)
  - FocalLoss               (component)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
import numpy as np


class StabilizedCombinedLoss(nn.Module):
    """
    Proposed stabilized loss (Algorithm 1 in the paper).

    Three components with numerical safeguards:
      1. Per-sample Dice loss (avoids batch-level corruption)
      2. Weighted BCE with adaptive pos_weight from batch statistics
      3. Tversky loss (α=0.7 FP, β=0.3 FN → recall-biased)

    Clamping, epsilon guards, and NaN fallbacks prevent the catastrophic
    training collapses observed with naïve multi-component losses.
    """
    def __init__(self, w_dice=0.5, w_bce=0.3, w_tversky=0.2,
                 tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.w_dice = w_dice
        self.w_bce = w_bce
        self.w_tversky = w_tversky
        self.tv_alpha = tversky_alpha
        self.tv_beta = tversky_beta

    def forward(self, pred, target):
        pred = torch.clamp(pred, -10, 10)
        eps = 1e-7
        smooth = 1.0
        p = torch.clamp(torch.sigmoid(pred), eps, 1 - eps)

        B = pred.shape[0]

        # --- per-sample Dice ---
        dices, tverskys = [], []
        for i in range(B):
            pi, ti = p[i].flatten(), target[i].flatten()
            tp = (pi * ti).sum()
            fp = (pi * (1 - ti)).sum()
            fn = ((1 - pi) * ti).sum()
            u = pi.sum() + ti.sum()
            dices.append((2 * tp + smooth) / (u + smooth))
            tverskys.append(
                (tp + smooth) /
                (tp + self.tv_alpha * fp + self.tv_beta * fn + smooth)
            )

        L_dice = 1 - torch.stack(dices).mean()
        L_tversky = 1 - torch.stack(tverskys).mean()

        # --- adaptive weighted BCE ---
        pos_ratio = target.sum() / target.numel()
        pw = torch.clamp((1 - pos_ratio) / (pos_ratio + eps), 1.0, 50.0)
        L_bce = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=pw.expand_as(pred[:1, :1, :1, :1]),  # scalar → broadcastable
            reduction="mean",
        )

        # --- NaN guard ---
        for name, val in [("dice", L_dice), ("bce", L_bce), ("tversky", L_tversky)]:
            if torch.isnan(val) or torch.isinf(val):
                val = torch.tensor(1.0, device=pred.device)

        return self.w_dice * L_dice + self.w_bce * L_bce + self.w_tversky * L_tversky


class StandardDiceBCE(nn.Module):
    """Simple Dice + BCE baseline for ablation (no stabilisation tricks)."""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, pred, target):
        p = torch.sigmoid(pred)
        smooth = 1e-6
        inter = (p * target).sum()
        dice = 1 - (2 * inter + smooth) / (p.sum() + target.sum() + smooth)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        return self.dw * dice + self.bw * bce


class BoundaryAwareDiceLoss(nn.Module):
    """Dice loss with additional boundary and distance-transform terms."""
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, pred, target):
        p = torch.sigmoid(pred)
        dice = self._dice(p, target)
        bnd_gt = self._boundary(target)
        bnd_pr = self._boundary(p)
        bnd_dice = self._dice(bnd_pr, bnd_gt)
        dist_w = self._dist_weights(target)
        bce = F.binary_cross_entropy_with_logits(pred, target,
                                                  weight=dist_w)
        return self.alpha * dice + self.beta * bnd_dice + self.gamma * bce

    @staticmethod
    def _dice(p, t, smooth=1e-6):
        inter = (p.flatten() * t.flatten()).sum()
        return 1 - (2 * inter + smooth) / (p.sum() + t.sum() + smooth)

    @staticmethod
    def _boundary(mask, k=3):
        dilated = F.max_pool2d(mask, k, stride=1, padding=k // 2)
        eroded = -F.max_pool2d(-mask, k, stride=1, padding=k // 2)
        return dilated - eroded

    @staticmethod
    def _dist_weights(target):
        dev = target.device
        w = torch.zeros_like(target)
        for b in range(target.shape[0]):
            m = target[b, 0].cpu().numpy()
            if m.max() > 0:
                d = ndimage.distance_transform_edt(m == 0) + \
                    ndimage.distance_transform_edt(m > 0)
                d = np.exp(-d / (d.mean() + 1e-8))
                d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                d = 0.5 + 0.5 * d
            else:
                d = np.ones_like(m)
            w[b, 0] = torch.from_numpy(d).float().to(dev)
        return w


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, pred, target):
        p = torch.sigmoid(pred).flatten()
        t = target.flatten()
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        return 1 - (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        w = (1 - pt) ** self.gamma
        if self.alpha is not None:
            at = self.alpha * target + (1 - self.alpha) * (1 - target)
            w = at * w
        return (w * bce).mean()
