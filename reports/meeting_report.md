# Professor meeting: 2D manuscript progress and evidence

29 September 2026. Working draft for discussion, not a claim of publication readiness or guaranteed acceptance.

## Executive summary

The original 2D paper remains the foundation: multiscale Gabor features, EfficientNet-B7/UNet++, SCSE attention, a learnable input projection, and the limited-annotation motivation. Both supplied code versions, the earlier corrected private-data study, the existing public evaluations, and the 3D progression are retained. See paper_lineage.md for the exact old-to-new mapping.

New work adds a corrected modular implementation, 21 completed public-data training runs (420 total epochs), native-grid boundary evaluation, three-seed controlled comparisons, an identity-preserving edge-fusion experiment, and 72 complete historical-checkpoint/dataset evaluations. All numbers below are generated from saved results. Numerical consistency checks and regression-test logs are included separately.

The model configuration with the highest mean selected validation Dice is **Gabor + boundary**, with test Dice **96.84%**. This is an exploratory development result on U2OS fluorescence, not evidence of cancer diagnosis. The matched vanilla model scores 96.60%, the Gabor bundle 96.85%, and residual Gabor fusion 96.57%. Added complexity must be justified by these controlled comparisons, not by the weak historical baselines.

The original Gabor family remains the stronger candidate: its matched gain over vanilla is about 0.25 percentage points. Residual fusion is a negative follow-up result relative to that bundle, and the explicit boundary term does not establish a Dice improvement over it. Breast transfer remains a major failure: the Gabor model reaches only 2.94% patient-macro TNBC Dice. Improving reproducibility is real progress, but these numbers do not prove breast-histology success or superiority over Cellpose-SAM.

## What changed, and why

| Item | Historical issue | Current treatment |
|---|---|---|
| Source versions | Two similar folders actually contain different checkpoints and some different code. | Both preserved and hashed; the nested breast-seg export is the newer one. |
| Reported mean | Proposed JSON says 95.13% Dice / 90.25% IoU, but its 128 saved values average 94.13% / 89.25%. | Arithmetic discrepancy documented; old values not promoted to validated new results. |
| Evaluation | Logits guessed from their range; low validation batches excluded. | Explicit logits contract and complete validation selection. |
| Data handling | 0/1 masks mishandled for sampling; uint16 images truncated; missing data silently replaced. | Nonzero-mask decoding, native-intensity preprocessing, failure on unreadable data, split/hash checks. |
| Boundary claim | Old overlap losses did not explicitly measure a boundary. | Added morphological boundary loss and BF1/surface Dice/HD95, with controlled loss ablations. |
| Input projection | A randomly initialized projection changes the input seen by the pretrained encoder. | New residual projection starts as an exact image identity; its empirical benefit is tested, not assumed. |
| Comparison fairness | Historical training and selection provenance uncertain. | New models share one public split, epochs, optimizer, schedule, loss family, augmentation and threshold. |

## 1. Existing private-data work retained

The previous corrected study covers 34 conditions, including architecture baselines, Gabor parameter sweeps, alternative edges, sampling/loss ablations and smaller encoders. Its complete summary is retained in results/previous_private_study.csv. These experiments are not being replaced or forgotten.

| Previous corrected configuration | Group-macro Dice % | 95% group-bootstrap interval % |
|---|---|---|
| Full configuration | 93.12 | 91.34–94.72 |
| Vanilla B7 UNet++ | 93.49 | 91.75–95.15 |
| U-Net ResNet-50 | 93.20 | 91.40–95.02 |
| Plain Dice+BCE loss ablation | 93.66 | 91.99–95.29 |

This audit used 855 valid foreground-containing images. Only 334 could be traced to eight source stacks; 521 lacked exact source matching. The 516/133/206 train/validation/test split used conservative shape-defined groups, including seven test groups, not verified independent biological volumes. Earlier corrected comparisons did not establish a Holm-adjusted significant advantage for the full model. These provenance limits must stay visible.

## 2. New matched public-data training

![Actual edge-filter responses](figures/actual_edge_features.png)

Real Gabor and Sobel responses on the median-ranked fluorescence example; this replaces the earlier synthetic/quantum-labelled illustration with an executed preprocessing example.

BBBC039 official split: 100 training fields, 50 validation fields, 50 test fields. Three seeds per condition, 20 epochs, ImageNet encoder, AdamW, fixed cosine schedule, 512-pixel inputs, uniform sampling, no TTA, threshold 0.5. Test probabilities are restored to the original 520 x 696 grid before scoring. Every validation/test image is included. All fifty test references contain nuclear foreground; empty-field specificity is therefore not established.

| Model | Dice % ± seed SD | IoU % | BF1@2px % | HD95 px |
|---|---|---|---|---|
| Vanilla B7 | 96.60 ± 0.07 | 93.44 | 98.53 | 1.32 |
| Gabor bundle | 96.85 ± 0.13 | 93.91 | 98.63 | 1.28 |
| Gabor + boundary | 96.84 ± 0.05 | 93.88 | 98.68 | 1.28 |
| SCSE-only B7 | 96.38 ± 0.53 | 93.03 | 98.48 | 1.35 |
| Residual Gabor | 96.57 ± 0.27 | 93.39 | 98.52 | 1.31 |
| Residual Sobel | 96.44 ± 0.48 | 93.14 | 98.47 | 1.35 |
| Residual Gabor + boundary | 96.61 ± 0.31 | 93.46 | 98.58 | 1.29 |

BF1 uses a 2-native-pixel tolerance. HD95 is in native pixels, not micrometres. ± is sample standard deviation across three seed-level means, not uncertainty over new acquisitions. The residual-Gabor change is -0.28 percentage points relative to the random-projection Gabor bundle and -0.03 points relative to vanilla B7. These are separate comparisons.

![Learning curves and measured results](figures/learning_and_results.png)

Initial study: vanilla, Gabor bundle, and Gabor plus boundary. Follow-up: SCSE-only, residual Gabor, residual Sobel, and residual Gabor plus boundary. The follow-up was designed after initial seed-0 results and is explicitly exploratory. No checkpoint or threshold is selected using test scores within a run. BBBC039 test results had already been inspected in earlier work, so these experiments are not a fresh confirmatory test.

### Controlled mechanism comparisons

Gabor bundle versus Vanilla B7: Dice difference +0.250 percentage points, paired image-bootstrap interval [+0.197, +0.299], Holm-adjusted p=7.335e-09.

Gabor + boundary versus Gabor bundle: Dice difference -0.015 percentage points, paired image-bootstrap interval [-0.050, +0.023], Holm-adjusted p=0.4027.

Residual Gabor versus SCSE-only B7: Dice difference +0.193 percentage points, paired image-bootstrap interval [+0.163, +0.223], Holm-adjusted p=1.954e-12.

Residual Gabor versus Residual Sobel: Dice difference +0.132 percentage points, paired image-bootstrap interval [+0.099, +0.168], Holm-adjusted p=1.634e-09.

Residual Gabor + boundary versus Residual Gabor: Dice difference +0.040 percentage points, paired image-bootstrap interval [+0.001, +0.081], Holm-adjusted p=0.06468.

These paired comparisons average seeds within each field, use 10,000 field-bootstrap resamples, and apply Holm correction to one 20-test family (ten comparisons by two endpoints) for BBBC039. They remain descriptive/exploratory because the dataset is one acquisition study. Statistical significance within this screen is not proof of clinical benefit or a unique Gabor mechanism.

![Representative public-data outputs](figures/bbbc039_examples.png)

Actual predictions, selected at the 10th/50th/90th percentile ranks of residual-Gabor seed-0 Dice. No manually selected showcase examples or AI-generated microscopy images.

## 3. Existing frozen-model transfer results retained

| Dataset | Released 2D Dice % | Revised 2D Dice % | Otsu Dice % | Cellpose-SAM Dice % |
|---|---|---|---|---|
| BBBC038 | 32.66 | 71.67 | 75.45 | 87.02 |
| BBBC039 | 92.04 | 91.58 | 94.79 | 96.95 |
| TNBC | 21.08 | 17.55 | 42.75 | 82.13 |

These are the preceding completed evaluations: BBBC038 n=65, BBBC039 n=50, TNBC n=50. They are frozen-checkpoint results, not numbers from today's public-data training. Otsu is a classical intensity reference. Cellpose-SAM uses a large pretrained model whose training overlap is unresolved; it is not an equal-data comparator. BBBC038 partially overlaps BBBC039 and must not be counted as independent confirmation of a BBBC039-trained model.

All 24 supplied historical checkpoints (12 in each export) were additionally evaluated on each of the three complete partitions. The 72-condition table is results/legacy_external_summary.csv; per-image scores and hashes are under results/legacy_external. Those legacy results are an audit, not a fair-training state-of-the-art leaderboard.

## 4. New-model transfer to breast histology

All newly trained models are evaluated on all 50 TNBC images, without TNBC training or threshold tuning. Images are aggregated within their 11 patient groups and then across patients. Seeds are averaged within a patient, not counted as independent patients.

| Model | TNBC patient-macro Dice % | Descriptive 95% CI % |
|---|---|---|
| Vanilla B7 | 8.07 | 6.40–9.71 |
| Gabor bundle | 2.94 | 2.06–3.90 |
| Gabor + boundary | 3.86 | 2.98–4.77 |
| SCSE-only B7 | 4.36 | 3.31–5.43 |
| Residual Gabor | 4.04 | 3.03–5.09 |
| Residual Sobel | 4.43 | 3.40–5.45 |
| Residual Gabor + boundary | 4.49 | 3.46–5.59 |

Residual Gabor's patient-macro Dice is 4.04%. The fluorescence training domain and grayscale preprocessing differ sharply from H&E histology. These measurements characterize that domain shift; they must not be represented as an in-domain breast-histology benchmark or cancer-detection accuracy.

![Median-ranked histology transfer example](figures/tnbc_transfer_example.png)

## 5. Existing 3D progression

The existing eight-stack, 334-annotated-plane evaluation is retained as exploratory work: fixed Cellpose-SAM Dice 91.81%, fixed-threshold distilled 3D U-Net 85.80%, nested-selected U-Net/fusion 91.37%, and nested Cellpose-plus 91.77%, with stack-macro averaging. These are selected-plane semantic scores, not dense 3D instance validation. The QC student's pseudo-label validation score is not human-reference evidence. No claim of beating Cellpose in 3D is supported here.

## Decision for the meeting

The project is in a better evidential position: real public training/evaluation, explicit lineage, fairer controls, and transparent failures. It is not yet a defensible clinical or broad state-of-the-art claim. Decide whether the main contribution should be a carefully controlled low-data edge-fusion study or a breast-histology adaptation study. The latter needs histology training and a genuinely independent patient/site test.

Before submission, confirm private source provenance, annotation protocol and agreements, ethics, author approval, funding/conflicts, inherited code/data licenses, public checkpoint release and AI disclosure. Complete public-data encoder and Gabor-sensitivity replication, true-empty evaluation and data-efficiency curves as appropriate. Existing private ablations remain useful but do not resolve source independence.

ISBI 2027 is the immediate target, with a 26 October 2026 deadline. Four pages must contain all technical content; an optional paid fifth page is restricted to references, ethics and acknowledgments. See the [official instructions](https://biomedicalimaging.org/2027/papers/). A first draft in the official template is provided, but author-completion items and scientific limitations remain.
