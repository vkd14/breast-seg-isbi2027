"""
trainer.py — Training engine with all stabilisation techniques.

Features:
  - OneCycleLR scheduling
  - Gradient accumulation (effective batch = batch_size × accum_steps)
  - Gradient clipping
  - EMA of validation Dice for model selection
  - IQR-based validation outlier filtering
  - Automatic checkpointing
  - Training curve plotting
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from .metrics import dice_score, iou_score, MetricTracker


class Trainer:
    def __init__(self, model, criterion, device,
                 save_dir="./checkpoints",
                 lr=3e-4, weight_decay=1e-4,
                 accum_steps=2, patience=30,
                 ema_alpha=0.9, grad_clip=1.0):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8,
        )
        self.scheduler = None  # set in train()
        self.accum_steps = accum_steps
        self.grad_clip = grad_clip

        # tracking
        self.best_dice = 0.0
        self.patience = patience
        self.patience_ctr = 0
        self.ema_val_dice = None
        self.ema_alpha = ema_alpha
        self.val_history = []

        self.history = dict(
            train_loss=[], val_loss=[],
            train_dice=[], val_dice=[],
            train_iou=[], val_iou=[],
            lr=[], ema_val_dice=[],
        )

    # ------------------------------------------------------------------
    def train_epoch(self, loader):
        self.model.train()
        total_loss, total_dice, total_iou, n = 0, 0, 0, 0
        self.optimizer.zero_grad()

        for i, (imgs, masks) in enumerate(tqdm(loader, desc="Train", leave=False)):
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            if torch.isnan(imgs).any():
                continue

            out = self.model(imgs)
            if torch.isnan(out).any():
                continue

            loss = self.criterion(out, masks) / self.accum_steps
            if torch.isnan(loss):
                continue
            loss.backward()

            if (i + 1) % self.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            with torch.no_grad():
                total_loss += loss.item() * self.accum_steps
                total_dice += dice_score(out, masks)
                total_iou += iou_score(out, masks)
                n += 1

        return (total_loss / max(n, 1),
                total_dice / max(n, 1),
                total_iou / max(n, 1))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        losses, dices, ious = [], [], []

        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            out = self.model(imgs)
            if torch.isnan(out).any():
                continue
            loss = self.criterion(out, masks)
            if torch.isnan(loss):
                continue
            losses.append(loss.item())
            dices.append(dice_score(out, masks))
            ious.append(iou_score(out, masks))

        if not dices:
            return float("inf"), 0.0, 0.0

        # IQR outlier filter on dice
        d = np.array(dices)
        q1, q3 = np.percentile(d, [25, 75])
        iqr = q3 - q1
        mask = (d >= max(0.3, q1 - 1.5 * iqr)) & (d <= min(1.0, q3 + 1.5 * iqr))
        if mask.sum() > 0:
            return float(np.array(losses)[mask].mean()), float(d[mask].mean()), float(np.array(ious)[mask].mean())
        return float(np.median(losses)), float(np.median(d)), float(np.median(ious))

    # ------------------------------------------------------------------
    def train(self, train_loader, val_loader, epochs=200, tag="model"):
        steps = (len(train_loader) // self.accum_steps) * epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=3e-4, total_steps=max(steps, 1),
            pct_start=0.1, anneal_strategy="cos",
            div_factor=25, final_div_factor=1000,
        )

        for epoch in range(1, epochs + 1):
            tl, td, ti = self.train_epoch(train_loader)
            vl, vd, vi = self.validate(val_loader)

            # EMA
            self.ema_val_dice = vd if self.ema_val_dice is None else (
                self.ema_alpha * self.ema_val_dice + (1 - self.ema_alpha) * vd
            )
            self.val_history.append(vd)

            lr = self.optimizer.param_groups[0]["lr"]
            for k, v in [("train_loss", tl), ("train_dice", td), ("train_iou", ti),
                          ("val_loss", vl), ("val_dice", vd), ("val_iou", vi),
                          ("lr", lr), ("ema_val_dice", self.ema_val_dice)]:
                self.history[k].append(v)

            improved = ""
            if self.ema_val_dice > self.best_dice:
                self.best_dice = self.ema_val_dice
                self.patience_ctr = 0
                torch.save(dict(
                    epoch=epoch, model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    best_dice=self.best_dice, history=self.history,
                ), self.save_dir / f"best_{tag}.pt")
                improved = f" ★ NEW BEST {self.best_dice:.4f}"
            else:
                self.patience_ctr += 1

            print(f"[{epoch:3d}/{epochs}]  "
                  f"train dice={td:.4f}  val dice={vd:.4f}  "
                  f"ema={self.ema_val_dice:.4f}  lr={lr:.2e}"
                  f"{improved}")

            if self.patience_ctr >= self.patience and td >= 0.95:
                print("Early stopping.")
                break

        self._plot(tag)
        return self.best_dice

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader, desc="Test"):
        """Full evaluation returning MetricTracker summary."""
        self.model.eval()
        tracker = MetricTracker()
        for imgs, masks in tqdm(loader, desc=desc, leave=False):
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            out = self.model(imgs)
            # per-image metrics
            for j in range(imgs.shape[0]):
                tracker.update(out[j:j+1], masks[j:j+1])
        return tracker.summary()

    # ------------------------------------------------------------------
    def _plot(self, tag):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ep = range(1, len(self.history["train_loss"]) + 1)

        axes[0].plot(ep, self.history["train_loss"], "b-", label="Train")
        axes[0].plot(ep, self.history["val_loss"], "r-", label="Val")
        axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
        axes[0].legend(); axes[0].grid(True)

        axes[1].plot(ep, self.history["train_dice"], "b-", label="Train")
        axes[1].plot(ep, self.history["val_dice"], "r-", label="Val")
        axes[1].plot(ep, self.history["ema_val_dice"], "g--", label="EMA Val")
        axes[1].axhline(0.95, color="orange", ls="--", label="95% target")
        axes[1].set(title="Dice Score", xlabel="Epoch", ylabel="Dice")
        axes[1].set_ylim(0.4, 1.0); axes[1].legend(); axes[1].grid(True)

        axes[2].plot(ep, self.history["train_iou"], "b-", label="Train")
        axes[2].plot(ep, self.history["val_iou"], "r-", label="Val")
        axes[2].set(title="IoU", xlabel="Epoch", ylabel="IoU")
        axes[2].set_ylim(0.3, 1.0); axes[2].legend(); axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / f"curves_{tag}.png", dpi=200)
        plt.close()
