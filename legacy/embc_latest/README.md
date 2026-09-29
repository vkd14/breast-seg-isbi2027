# Breast Cell Nucleus Segmentation — EMBC 2026

**Paper:** *Breast Cell Segmentation Under Extreme Data Constraints: Multi-Scale Edge Enhancement with Adaptive Loss Stabilisation*

**Authors:** Varun Kumar Dasoju, Qingsu Cheng, Zeyun Yu  
**Affiliation:** University of Wisconsin–Milwaukee  
**Target:** IEEE EMBC 2026 Late-Breaking Abstract (deadline: April 30, 2026)

---

## Repository Structure

```
embc2026-breast-seg/
│
├── src/                          # Core library
│   ├── models.py                 # All architectures (proposed + 7 SOTA + ablation variants)
│   ├── dataset.py                # Dataset, Gabor edge enhancement, augmentations, sampling
│   ├── losses.py                 # StabilizedCombinedLoss + baselines (Dice+BCE, Tversky, Focal)
│   ├── metrics.py                # Dice, IoU, Precision, Recall, F1 + MetricTracker
│   └── trainer.py                # Training engine (OneCycleLR, EMA, gradient clipping, plotting)
│
├── configs/
│   └── default.yaml              # All hyperparameters in one place
│
├── run_all_experiments.py        # ★ ONE COMMAND runs all 13 experiments
├── evaluate.py                   # Evaluate any .pt checkpoint on test set
├── generate_tables.py            # Produce LaTeX tables from results CSV
├── requirements.txt              # pip dependencies
├── train_cl_b7_opt.py            # Original monolithic script (reference only)
└── README.md                     # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Expected layout:
```
/path/to/dataset/
├── train/
│   ├── images/          # or enhanced_images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/          # or enhanced_images/
    └── labels/
```

Image-mask pairs are matched by filename stem (e.g., `img_001.tif` ↔ `img_001.png` in labels).

### 3. Run ALL experiments (SOTA + ablations) in one go

```bash
python run_all_experiments.py --data_root /path/to/dataset --epochs 150 --batch_size 4
```

This trains **13 models** sequentially and writes:
- Per-model checkpoints to `results/<model_name>/best_<name>.pt`
- Per-model test metrics to `results/<model_name>/test_metrics.json`
- Training curves to `results/<model_name>/curves_<name>.png`
- **Comparison table** to `results/comparison_table.csv`

### 4. If you already have the proposed model `.pt` file

Skip its training and only train baselines:
```bash
python run_all_experiments.py \
    --data_root /path/to/dataset \
    --existing_model /path/to/best_enhanced_model.pt \
    --epochs 150
```

### 5. Run a single experiment (for debugging)

```bash
python run_all_experiments.py --data_root /data --only sota_unet_resnet34 --epochs 50
```

### 6. Evaluate any checkpoint with full visualisations

```bash
python evaluate.py \
    --model_path results/proposed_full/best_proposed_full.pt \
    --data_root /path/to/dataset \
    --model_type proposed \
    --in_channels 4 \
    --save_vis
```

Outputs per-image CSV, aggregate JSON, and 6-panel visualisation PNGs.

### 7. Generate LaTeX tables

```bash
python generate_tables.py --csv results/comparison_table.csv
```

Produces `table_sota.tex` and `table_ablation.tex` ready to paste into the paper.

---

## What Each Experiment Does

### SOTA Comparisons (Table II in the paper)

| Experiment Name              | Architecture               | Encoder      | Edge Ch | Loss          |
|------------------------------|----------------------------|--------------|---------|---------------|
| `proposed_full`              | UNet++ + SCSE              | EfficientNet-B7 | ✓ Gabor | Stabilized    |
| `sota_unet_resnet34`         | U-Net                      | ResNet-34    | ✗       | Standard      |
| `sota_unet_resnet50`         | U-Net                      | ResNet-50    | ✗       | Standard      |
| `sota_unetpp_resnet50`       | UNet++                     | ResNet-50    | ✗       | Standard      |
| `sota_attention_unet_resnet50` | U-Net + SCSE             | ResNet-50    | ✗       | Standard      |
| `sota_deeplabv3p_resnet50`   | DeepLabV3+                 | ResNet-50    | ✗       | Standard      |
| `sota_manet_resnet50`        | MA-Net                     | ResNet-50    | ✗       | Standard      |
| `sota_fpn_resnet50`          | FPN                        | ResNet-50    | ✗       | Standard      |

### Ablation Study (Table III in the paper)

| Experiment Name                  | What's Removed                           |
|----------------------------------|------------------------------------------|
| `proposed_full`                  | Nothing (full pipeline)                  |
| `ablation_no_gabor`             | Gabor edge channel (3-ch input)          |
| `ablation_no_weighted_sampling` | Complexity-weighted sampling             |
| `ablation_standard_loss`        | Stabilised loss → standard Dice+BCE      |
| `ablation_no_scse`              | SCSE decoder attention                   |
| `ablation_unet_not_unetpp`     | UNet++ → plain U-Net decoder             |

---

## Time Estimates (NVIDIA RTX 3090, batch=4, 150 epochs)

| Model                 | ~Training Time |
|-----------------------|----------------|
| Proposed (EB7/UNet++) | 6–8 hours      |
| ResNet-50 variants    | 3–4 hours each |
| ResNet-34 variants    | 2–3 hours each |
| **Total (all 13)**    | **~48–60 hours** |

To run on multiple GPUs, run experiments in parallel on different GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 python run_all_experiments.py --only proposed_full ...
CUDA_VISIBLE_DEVICES=1 python run_all_experiments.py --only sota_unet_resnet34 ...
```

---

## Original Script

`train_cl_b7_opt.py` is the original ~2300-line monolithic training script used for the initial submission. It is kept for reference. All functionality has been refactored into the `src/` modules with the following improvements:

- Modular: models, losses, metrics, dataset, trainer are independent
- All SOTA baselines available via a single factory function
- Ablation variants are just config flags (no code duplication)
- Proper per-image metrics on the FULL test set (not 4 cherry-picked images)
- LaTeX table generation for the paper

---

## Paper Revision Plan — IEEE EMBC 2026 Late-Breaking Abstract

Below is the complete list of issues identified by both reviewers (#0283 and #0485) and the editors, with the exact work required for acceptance.

### CRITICAL FIXES (must do for April 30)

#### 1. Add SOTA Comparison Table ← Reviewer #0283, Editors
**Problem:** No comparison with existing methods. Table II only shows your own progressive development, not external baselines. Reviewer: "It does not compare with any state-of-the-art cell segmentation methods in the literature."

**Fix:** Run all 7 SOTA baselines in this repo on the exact same dataset splits. Build a proper comparison table (Table II replacement) showing Dice, IoU, Precision, Recall for each. The `run_all_experiments.py` script + `generate_tables.py` produces this automatically.

**Paper changes:**
- Replace Table II with the SOTA comparison table
- Add a new Table III for the ablation study
- In Section IV, add a subsection "Comparison with State-of-the-Art" discussing results

#### 2. Clarify Cell vs. Nucleus Segmentation ← Reviewer #0485, Editors
**Problem:** Abstract says "mammary epithelial nuclei datasets" but text says "breast cell" and figures show what looks like nuclei. Reviewer: "I am not sure whether the author is segmenting whole cells or cell nuclei."

**Fix:** Decide definitively. From the ground truth masks, this appears to be **cell nucleus segmentation**. Use "breast cell nucleus" or "mammary epithelial cell nucleus" consistently throughout:
- Title, abstract, all body text
- Figure captions
- Table headers
- Remove all ambiguous references to "breast cell" without "nucleus"

#### 3. Explain or Rename "Quantum-Inspired" ← Both Reviewers
**Problem:** The Gabor filter bank is classical signal processing. Calling it "quantum-inspired" without justification raised flags. Reviewer #0485: "The use of quantum enhancement is not clear to me."

**Fix (choose one):**
- **Option A (recommended for a 1–2 page abstract):** Rename to "Multi-Scale Gabor Edge Enhancement." Drop all quantum framing. This repo already uses the class name `GaborEdgeEnhancement`.
- **Option B:** Keep the quantum framing but add a proper paragraph explaining the analogy: Gabor filter responses at multiple orientations/scales are analogous to quantum superposition of states; the max-pooling collapse is analogous to measurement. Cite Domingo & Chehimi [21]. This is a harder sell.

**Paper changes:**
- Rename Section III-D from "Quantum-Inspired Edge Enhancement" to "Multi-Scale Gabor Edge Enhancement"
- Replace the word "quantum" everywhere with "multi-scale Gabor"
- Add 2–3 sentences explaining HOW Gabor filters enhance edges: each filter detects edges at a specific orientation and scale; the max response across all 24 filters produces an orientation-invariant edge map; this is concatenated as a 4th channel.
- Explicitly state the parameter choices (8 orientations = 22.5° spacing for full 180° coverage; 3 scales = wavelengths 4/10/20 px covering fine to coarse boundaries; σ=5.0 and γ=0.5 from literature defaults).

#### 4. Explain the Loss Function ← Reviewer #0283
**Problem:** Algorithm 1 has no explanation. Reviewer: "There is essentially no description and explanation whatsoever."

**Fix:** Add a paragraph after Algorithm 1 explaining each component:
1. **Clamping (lines 3-5):** Prevents sigmoid overflow, keeps gradients finite
2. **Per-sample Dice (lines 9-16):** Computing Dice per image prevents a single bad image from corrupting the batch-level metric
3. **Adaptive pos_weight (lines 20-21):** Dynamically scales BCE based on class ratio in each batch; clamped to [1, 50] to prevent extreme weighting
4. **Tversky term (line 16):** α=0.7 penalises false positives more, β=0.3 penalises false negatives less → the model learns conservative over-segmentation (clinically preferable)
5. **Weight combination (line 23):** Dice (0.5) for overlap, BCE (0.3) for pixel-level correctness, Tversky (0.2) for recall bias

#### 5. Report Aggregate Test Metrics ← Both Reviewers
**Problem:** Results reported on only 4 images. Reviewer #0283: "It seems too quick to jump to the conclusion that the method will necessarily achieve a Dice score of over 90% for other breast cell images."

**Fix:** Report mean ± std, median, min, max on ALL 100 test images. The `evaluate.py` script produces this automatically.

**Paper changes:**
- Add a quantitative results paragraph: "Across the full 100-image test set, our method achieves a Dice score of X.XX ± Y.YY (median X.XX, range [min, max])."
- Keep the qualitative analysis (Figures 3a-d) but present them as illustrative examples, not the primary evaluation.

#### 6. Dataset Provenance ← Reviewer #0485
**Problem:** "Did the authors collect the training, validation, and test datasets, or are these publicly available data?" No information on imaging modality, resolution, or annotation protocol.

**Fix:** Add a subsection or paragraph in Section III-B covering:
- **Source:** [State whether public or private. If private, who collected it and under what IRB.]
- **Imaging modality:** [Brightfield microscopy? H&E stained? Fluorescence?]
- **Resolution:** [Image dimensions, pixel size in µm if known]
- **Annotation:** [Who annotated? How many annotators? What tool? Was there inter-annotator agreement measured?]
- **Availability:** [Will you release it? If not, state why.]

#### 7. Fix Figure Issues ← Reviewer #0485
**Problem:** Multiple figure problems:
- Fig. 1 encoder block diagram is unreadable when zoomed
- Fig. 3 caption says (a), (b), (c) but sub-images aren't labelled
- Fig. 3 shows Dice/IoU on ground truth images (makes no sense)
- Figures may use copyrighted diagrams from EfficientNet/UNet papers

**Fix:**
- **Redraw Figure 1** from scratch as a clean block diagram (Tikz, draw.io, or Figma). Show: Input → ROI crop → Gabor edge → 4-ch → Projection → Encoder (5 blocks with channel dims) → UNet++ decoder with SCSE → Output. Remove the tiny unreadable EfficientNet diagram.
- **Label Figure 3** sub-images as (a), (b), (c), (d) explicitly
- **Fix caption:** Ground truth has Dice=1.0 and IoU=1.0 by definition; show metrics only on prediction panels
- **Add proper citations** if adapting any diagrams, or redraw entirely

### IMPORTANT IMPROVEMENTS (strongly recommended)

#### 8. Significance Argument ← Reviewer #0283
**Problem:** "If the missing 6% is considered an issue, would it still be an issue for the missing 4.5%?"

**Fix:** The 95.5% → 94% difference must be argued differently:
- Show the improvement IS statistically significant (paired t-test or Wilcoxon signed-rank on per-image Dice scores between proposed vs. best baseline)
- Frame it as: "Our 1.5 percentage point improvement over nnU-Net translates to X% fewer missed boundary pixels, which at 3-pixel inter-annotator variation represents clinically significant improvement"
- The ablation table shows each component contributes a measurable delta

#### 9. Cross-Validation or Confidence Intervals
**Problem:** Small dataset (599 train, 100 test) makes single-split results unreliable.

**Fix:** Ideally, run 5-fold cross-validation. If time doesn't permit, at minimum report 95% confidence intervals on test set metrics. For Dice with n=100 images:
```
CI = mean ± 1.96 × (std / √n)
```

#### 10. ROI Extraction Explanation ← Reviewer #0485
**Problem:** "How can ROIs be extracted by padding the mask only?"

**Fix:** Clarify that ROI extraction uses Otsu thresholding on the grayscale image (not just the mask) to find tissue regions, then adds 50px padding around the bounding box. The mask is cropped to the same region for alignment.

### NICE TO HAVE (for a stronger submission)

#### 11. Statistical Significance Tests
- Paired t-test: proposed vs. each baseline on per-image Dice scores
- Report p-values in the SOTA table

#### 12. Failure Case Analysis
- Categorise errors: boundary errors, small lesion misses, low-contrast regions, artifacts
- Report: "X% of errors from lesions <50px, Y% within 3px of boundaries, Z% in low-contrast regions"
- This is already partially in the paper but needs quantification on the full test set

#### 13. Computational Efficiency Table
- Report params, FLOPs, inference time (FPS) for each method
- Your model is ~66M params (EB7) vs ~25M (ResNet-50) — show it's worth the cost

### WRITING CHANGES (for the late-breaking abstract format)

Late-breaking abstracts are typically 1–2 pages. Restructure as:

1. **Introduction (3–4 sentences):** Problem, gap, contribution
2. **Method (1 paragraph):** Architecture, edge enhancement, loss function — all renamed consistently
3. **Results (1 paragraph + 2 tables):** SOTA comparison table + ablation table, with aggregate metrics on full test set
4. **Conclusion (2 sentences):** Summary + future work

Remove all semi-supervised/active learning history (progressive Table II), as it's irrelevant to the final method. Focus entirely on the final supervised pipeline.

---

## Checklist Before Submission

- [ ] All 7 SOTA baselines trained and evaluated on same splits
- [ ] All 5 ablation variants trained and evaluated
- [ ] Aggregate metrics (mean ± std) on full 100-image test set
- [ ] LaTeX tables generated and pasted into paper
- [ ] "Quantum" renamed to "Multi-Scale Gabor" everywhere
- [ ] Consistent terminology: "breast cell nucleus" throughout
- [ ] Algorithm 1 explained in text
- [ ] Figure 1 redrawn from scratch
- [ ] Figure 3 sub-images labelled (a)-(d), ground truth Dice fixed
- [ ] Dataset provenance paragraph added
- [ ] Statistical significance test on proposed vs. best baseline
- [ ] Paper reformatted as 1–2 page late-breaking abstract
- [ ] All co-authors reviewed final version
