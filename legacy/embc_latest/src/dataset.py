"""
dataset.py — Dataset, multi-scale Gabor edge enhancement, augmentations,
              ROI cropping, and complexity-weighted sampling.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Multi-scale Gabor Edge Enhancement (renamed from "quantum-inspired")
# ---------------------------------------------------------------------------
class GaborEdgeEnhancement:
    """
    Compute a boundary-emphasising 4th channel using a bank of Gabor filters
    at multiple orientations and scales.  The maximum filter response across
    all 24 kernels (8 orientations × 3 scales) produces an edge map that
    highlights cell boundaries even when contrast is low.
    """
    def __init__(self, num_orientations=8, num_scales=3):
        self.filters = self._build_bank(num_orientations, num_scales)

    @staticmethod
    def _build_bank(n_orient, n_scale):
        filters = []
        ksize = 31
        for theta in np.linspace(0, np.pi, n_orient, endpoint=False):
            for freq in np.linspace(0.05, 0.25, n_scale):
                kern = cv2.getGaborKernel(
                    (ksize, ksize), sigma=5.0, theta=theta,
                    lambd=1.0 / freq, gamma=0.5, psi=0, ktype=cv2.CV_32F,
                )
                filters.append(kern / (kern.sum() + 1e-8))
        return filters

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return float32 edge map in [0, 1], same H×W as input."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        gray = gray.astype(np.float32) / 255.0
        responses = [np.abs(cv2.filter2D(gray, cv2.CV_32F, k)) for k in self.filters]
        edge = np.max(np.stack(responses, axis=-1), axis=-1)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        return edge.astype(np.float32)


# ---------------------------------------------------------------------------
#  ROI cropping
# ---------------------------------------------------------------------------
class ROICropper:
    """Extract bounding box around tissue with configurable padding."""
    def __init__(self, padding=50, target_size=512, min_tissue_ratio=0.05):
        self.padding = padding
        self.target_size = target_size
        self.min_tissue_ratio = min_tissue_ratio

    def crop(self, image, mask=None):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        _, binary = cv2.threshold(gray, thr * 0.5, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, mask, None

        min_area = image.shape[0] * image.shape[1] * self.min_tissue_ratio
        valid = [c for c in contours if cv2.contourArea(c) > min_area] or contours
        x0, y0, x1, y1 = image.shape[1], image.shape[0], 0, 0
        for c in valid:
            x, y, w, h = cv2.boundingRect(c)
            x0, y0 = min(x0, x), min(y0, y)
            x1, y1 = max(x1, x + w), max(y1, y + h)

        p = self.padding
        x0, y0 = max(0, x0 - p), max(0, y0 - p)
        x1 = min(image.shape[1], x1 + p)
        y1 = min(image.shape[0], y1 + p)
        for coord, dim in [(x0, x1), (y0, y1)]:
            if (dim - coord) < self.target_size:
                mid = (coord + dim) // 2
                coord = max(0, mid - self.target_size // 2)

        roi_img = image[y0:y1, x0:x1]
        roi_mask = mask[y0:y1, x0:x1] if mask is not None else None
        return roi_img, roi_mask, (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
#  Augmentations
# ---------------------------------------------------------------------------
def build_transforms(image_size=512, is_training=True, num_channels=3):
    if num_channels == 4:
        mean, std = [0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.25]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if is_training:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15,
                               rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=120, sigma=9, p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
#  Dataset
# ---------------------------------------------------------------------------
class BreastCellDataset(Dataset):
    """
    Loads image-mask pairs.  Optionally adds a 4th Gabor-edge channel and
    computes per-sample complexity weights for weighted sampling.
    """
    def __init__(self, image_paths, mask_paths, transforms=None,
                 is_training=True, use_gabor=True, use_roi=False):
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transforms = transforms
        self.is_training = is_training
        self.use_gabor = use_gabor
        self.use_roi = use_roi

        if use_gabor:
            self.gabor = GaborEdgeEnhancement()
        if use_roi:
            self.roi = ROICropper()

        self.sample_weights = (
            self._compute_weights() if is_training else None
        )

    # ---- weighting --------------------------------------------------------
    def _compute_weights(self):
        weights = []
        for mp in tqdm(self.mask_paths, desc="Computing sample weights"):
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                weights.append(1.0)
                continue
            ratio = np.sum(mask > 127) / mask.size
            if ratio > 0.001:
                w = 2.0 + ratio * 3.0
                if ratio < 0.05:
                    w *= 1.5
                edges = cv2.Canny(mask, 50, 150)
                w *= 1 + np.sum(edges > 0) / edges.size
            else:
                w = 0.3
            weights.append(w)
        w = np.array(weights)
        return (w / w.sum() * len(w)).tolist()

    # ---- data loading -----------------------------------------------------
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            ch = 4 if self.use_gabor else 3
            return torch.zeros(ch, 512, 512), torch.zeros(1, 512, 512)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.use_roi:
            image, mask, _ = self.roi.crop(image, mask)

        if self.use_gabor:
            edge = self.gabor(image)
            image = np.dstack([image, (edge * 255).astype(np.uint8)])

        mask = (mask > 127).astype(np.float32) if mask.max() > 1 else (mask > 0.5).astype(np.float32)

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image, mask


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def find_image_mask_pairs(image_dir, mask_dir):
    """Match images to masks by filename stem."""
    exts = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for e in exts:
        images.extend(Path(image_dir).glob(e))
    masks = {p.stem: p for e in exts for p in Path(mask_dir).glob(e)}

    pairs, unmatched = [], []
    for img in sorted(images):
        stem = img.stem
        candidates = [stem, stem.replace("image", "mask"),
                       stem.replace("img", "mask"),
                       f"mask_{stem}", f"{stem}_mask"]
        matched = False
        for c in candidates:
            if c in masks:
                pairs.append((img, masks[c]))
                matched = True
                break
        if not matched:
            unmatched.append(img)
    return pairs, unmatched


def make_sampler(dataset):
    """Create WeightedRandomSampler from dataset.sample_weights."""
    if dataset.sample_weights is None:
        return None
    return WeightedRandomSampler(dataset.sample_weights,
                                  len(dataset), replacement=True)
