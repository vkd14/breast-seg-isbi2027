"""
models.py — Model architectures for breast cell nucleus segmentation.

Provides the proposed EfficientNet-B7/UNet++ with SCSE + Gabor edge channel,
plus all SOTA baselines needed for the comparison table.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
#  Input projection: 4-channel (RGB + edge) → 3-channel for pretrained encoder
# ---------------------------------------------------------------------------
class InputProjection(nn.Module):
    """Learnable 4→3 channel projection that preserves ImageNet pretraining."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 1),
        )

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
#  Proposed model
# ---------------------------------------------------------------------------
class ProposedModel(nn.Module):
    """
    EfficientNet-B7 / UNet++ with SCSE attention and optional 4-channel input.
    This is the full proposed pipeline from the paper.

    NOTE: attribute names (input_proj, use_input_proj, base_model) intentionally
    match the original training script so that existing .pt checkpoints load
    without key remapping.
    """
    def __init__(self, in_channels=4, num_classes=1, encoder_weights="imagenet"):
        super().__init__()
        self.use_input_proj = (in_channels == 4)

        if self.use_input_proj:
            # Matches original: nn.Sequential assigned directly to self.input_proj
            self.input_proj = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0),
            )

        # Attribute named 'base_model' to match original checkpoint keys
        self.base_model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
            decoder_attention_type="scse",
            decoder_channels=(256, 128, 64, 32, 16),
        )

    def forward(self, x):
        if self.use_input_proj:
            x = self.input_proj(x)
        return self.base_model(x)


# ---------------------------------------------------------------------------
#  SOTA / ablation baselines — all constructed via a single factory
# ---------------------------------------------------------------------------
# Each entry: (encoder, architecture_class, scse, description)
BASELINE_REGISTRY = {
    # ---- SOTA comparisons ----
    "unet_resnet34": dict(
        encoder="resnet34", arch="Unet", scse=False,
        desc="Vanilla U-Net (ResNet-34)",
    ),
    "unet_resnet50": dict(
        encoder="resnet50", arch="Unet", scse=False,
        desc="U-Net (ResNet-50)",
    ),
    "unetpp_resnet50": dict(
        encoder="resnet50", arch="UnetPlusPlus", scse=False,
        desc="UNet++ (ResNet-50)",
    ),
    "attention_unet_resnet50": dict(
        encoder="resnet50", arch="Unet", scse=True,
        desc="Attention U-Net (ResNet-50, SCSE)",
    ),
    "deeplabv3p_resnet50": dict(
        encoder="resnet50", arch="DeepLabV3Plus", scse=False,
        desc="DeepLabV3+ (ResNet-50)",
    ),
    "manet_resnet50": dict(
        encoder="resnet50", arch="MAnet", scse=False,
        desc="MA-Net (ResNet-50)",
    ),
    "fpn_resnet50": dict(
        encoder="resnet50", arch="FPN", scse=False,
        desc="FPN (ResNet-50)",
    ),
    # ---- Ablation variants (same arch, components removed) ----
    "ablation_no_scse": dict(
        encoder="efficientnet-b7", arch="UnetPlusPlus", scse=False,
        desc="Ours w/o SCSE attention",
    ),
    "ablation_unet_eb7": dict(
        encoder="efficientnet-b7", arch="Unet", scse=True,
        desc="Ours w/ U-Net (not UNet++)",
    ),
}


class BaselineModel(nn.Module):
    """
    Generic wrapper: builds any smp architecture with 3-ch or 4-ch input.
    For 4-ch the same Sequential projection is used so the comparison is fair.
    Uses 'base_model' attribute name for consistency with ProposedModel.
    """
    def __init__(self, key: str, in_channels=3, num_classes=1,
                 encoder_weights="imagenet"):
        super().__init__()
        cfg = BASELINE_REGISTRY[key]
        self.use_input_proj = (in_channels == 4)

        if self.use_input_proj:
            self.input_proj = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0),
            )

        arch_cls = getattr(smp, cfg["arch"])
        kwargs = dict(
            encoder_name=cfg["encoder"],
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        if cfg["arch"] in ("Unet", "UnetPlusPlus"):
            kwargs["decoder_attention_type"] = "scse" if cfg["scse"] else None
            if cfg["arch"] == "UnetPlusPlus":
                kwargs["decoder_channels"] = (256, 128, 64, 32, 16)

        self.base_model = arch_cls(**kwargs)
        self.desc = cfg["desc"]

    def forward(self, x):
        if self.use_input_proj:
            x = self.input_proj(x)
        return self.base_model(x)


def build_model(name: str, in_channels: int = 4, encoder_weights="imagenet"):
    """Factory: returns (model, description_string)."""
    if name == "proposed":
        m = ProposedModel(in_channels=in_channels,
                          encoder_weights=encoder_weights)
        return m, "Proposed (EfficientNet-B7 / UNet++ / SCSE / Gabor edge)"
    elif name in BASELINE_REGISTRY:
        m = BaselineModel(name, in_channels=in_channels,
                          encoder_weights=encoder_weights)
        return m, m.desc
    else:
        raise ValueError(f"Unknown model: {name}. "
                         f"Choose from: proposed, {list(BASELINE_REGISTRY)}")
