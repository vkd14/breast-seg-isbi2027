import json, csv, sys
from pathlib import Path
sys.path.insert(0,'scripts')
from build_reports import render_pdf
V=json.load(open('results/report_values.json'))
doc=f"""# List of changes: rejected submission to current state

29 September 2026. Every item is something that was actually done, with the artefact that
evidences it. Items still outstanding are listed separately at the end and are not counted as
changes. Numbers come from committed result files.

## A. Corrections to the evaluation protocol

| # | Change | Why | Evidence |
|---|---|---|---|
| A1 | Split changed from per-image to conservative inferred-source-group partitioning | Source-volume metadata are unavailable, but 50% of test images had a near-duplicate in training (median max NCC 0.950); likely source groups were reconstructed from geometry and correlation | `legacy/scientific_reports_revision`, leakage audit figure; max cross-split NCC is 0.61 after grouping, without proving biological independence |
| A2 | Moved controlled experiments to public data with official splits | Private source-volume identity cannot currently be certified | `results/public_study/split_manifest.json` |
| A3 | Confidence intervals recomputed with the inferred source group as resampling unit | Images are clustered within reconstructed groups; per-image bootstrap overstates precision | `results/previous_private_study.csv` (n_groups = 7) |
| A4 | TNBC statistics aggregated per patient | 11 patients, not 50 independent images | `results/paired_statistics.csv` |
| A5 | Paired Wilcoxon with Holm correction across the comparison family | Original reported a single t-test | `results/paired_statistics.csv` |
| A6 | Metrics computed at native resolution after resampling probabilities | Metrics had been computed on the 512x512 network grid | `src/breastseg`, regression tests |
| A7 | Boundary metrics added (BF1@2px, surface Dice, HD95) | Paper claimed boundary awareness with no boundary metric | all result tables |
| A8 | Equal budget for every architecture (20 epochs, same optimiser, schedule, augmentation, threshold) | Baselines had 150 epochs vs 200 for the proposed model | `results/public_study/status.json`, protocol hash |
| A9 | Fixed threshold 0.5, no test-time augmentation anywhere | Manuscript claimed 7-fold TTA; code implemented 3 and the eval script called none | protocol config |
| A10 | Test partition evaluated only after validation-based checkpoint selection | Validation had filtered low-scoring batches before selection | `train_public_study.py` |

## B. Code defects found and fixed

| # | Defect | Consequence | Fix |
|---|---|---|---|
| B1 | Masks thresholded at 127 | 508 of 855 binary 0/1 annotations read as empty, including in the sampler (8.9x under-weighting) | All nonzero labels are foreground; encoding-equivalence test |
| B2 | Gabor kernels normalised by signed sum | Near-zero divisor for 8 of 24 kernels | L1 normalisation; finite-output tests |
| B3 | IQR/floor filtering of validation batches plus metric smoothing | Checkpoint selection not based on the full validation set | Full validation set, unsmoothed native-grid mean Dice |
| B4 | Logits vs probabilities inferred from gradient state and maximum value | Low-magnitude positive logits thresholded incorrectly | Output contract is always logits; test at logit 0.1 |
| B5 | Unreadable images/masks silently replaced with zeros | Missing data could masquerade as true negatives | Loader fails on unreadable data or geometry mismatch |
| B6 | 16-bit images loaded through 8-bit conversion keeping the high byte | Undocumented contrast change; measured cost ~0.01 Dice | Native intensities retained to deterministic percentile normalisation |
| B7 | Weighted-sampling flag not passed to the loader in the older export | The no-weighted-sampling ablation in that export is invalid | Explicit sampler contract and regression test |
| B8 | Modular and monolithic loss definitions differed | Exported runs cannot be treated as sharing one objective | Single versioned Dice/BCE objective plus optional morphological boundary term |
| B9 | Silently skipped batches; partial gradients not stepped | Effective budgets differed without being reported | Every batch used; non-finite loss or gradient stops the run |

## C. Claims withdrawn or corrected

| # | Original claim | Current status |
|---|---|---|
| C1 | 95.1% Dice / 89.2% IoU headline | Withdrawn. Stored means exceed the recomputed means of their own 128 per-image values by exactly 0.01 for both metrics. Neither value is carried forward. |
| C2 | Ablations show each component is essential (up to −95 Dice points) | Not reproducible. Largest measured single-component effect is under one Dice point; several components are neutral or harmful. |
| C3 | Tversky term with recall bias; boundary-aware loss | Withdrawn. The released checkpoint used 0.7·Dice + 0.3·adaptive BCE, with neither term. New study uses an explicit Dice+BCE objective and a separately evaluated boundary term. |
| C4 | Baselines score 0.47–0.76 | Not reproduced under equal budgets. Standard baselines reach 0.931–0.938 on the corrected private protocol. |
| C5 | "60% of images are empty" | Incorrect; artefact of mask-encoding threshold. All 855 valid images contain nuclei. |
| C6 | EfficientNet-B7 capacity is necessary | Withdrawn. The encoder ladder is flat: B0 (6.6M params) matches B7 (68M). |
| C7 | Clinical-grade deployment | Removed everywhere. No clinical validation was performed. |
| C8 | "Quantum" edge enhancement; AI-generated method figure | Removed. Figures are computed by released code from real images; no quantum mechanism claimed; AI-assistance disclosure drafted. |

## D. New experiments run

| # | Experiment | Scale | Headline outcome |
|---|---|---|---|
| D1 | Corrected private-data study | 38 runs, inferred-group split, 3 seeds | All conditions overlap once clustered CIs are used |
| D2 | Private 4-fold leave-fields-out CV | 23 fields, all held out once | Pooled Dice 0.894; exposes two annotation conventions costing 0.05–0.10 Dice |
| D3 | Matched public study (BBBC039) | {V['TOTAL_RUNS']} runs, {V['TOTAL_EPOCHS']} epochs, 3 seeds each | Gabor bundle +{V['GABOR_DELTA']} Dice points vs vanilla ([{V['GABOR_CI_LOW']}, {V['GABOR_CI_HIGH']}], p < 0.001) |
| D4 | Mechanism follow-up (SCSE-only, residual Gabor, residual Sobel, residual Gabor+boundary) | 12 runs | SCSE-only and residual Sobel are worse than vanilla; residual Gabor is neutral |
| D5 | Untuned transfer to breast histology (TNBC) | 7 configurations x 11 patients | All near zero; every added component makes transfer worse |
| D6 | Independent re-evaluation of historical checkpoints | 24 checkpoints, {V['LEGACY_EVALUATIONS']} evaluations | 0.46–0.71 on BBBC038; several collapse to 0.00 on BBBC039; failures retained |
| D7 | Edge-detector controls (Sobel, Scharr, LoG, Canny, none) | private + public | No detector consistently better; network largely indifferent to the edge map |
| D8 | Gabor parameter sensitivity (orientations, scales, sigma) | 6 private runs | All within ±0.01 Dice |
| D9 | Encoder ladder (ResNet-34/50, EfficientNet-B0/B3/B5/B7) | 6 private runs | Flat across a tenfold parameter range |
| D10 | Public validation-only Gabor/sampling sensitivity | 30 runs, 600 epochs, 3 seeds, zero test cases evaluated | Foreground Dice: weighted sampling +0.0473 points, five scales +0.0307, sigma 7 +0.0250 versus the Gabor reference; effects remain practically tiny and require independent confirmation |

## E. Infrastructure and reporting

| # | Change |
|---|---|
| E1 | New corrected implementation under `src/breastseg`; historical exports preserved unedited as evidence under `legacy/` |
| E2 | New validation-study identity binds the frozen config, active training/core/data source files and archive hashes; partial runs stop for inspection. Older completed runs retain their earlier config-hash protocol and are not retroactively relabelled. |
| E3 | Scientific regression tests (`tests/`): mask encodings, empty-mask Dice conventions, logit handling, sampler behaviour, kernel finiteness |
| E4 | All reports and the manuscript draft generated from committed CSV/JSON, so abstract, tables and text cannot diverge |
| E5 | 6,060 evaluated masks retained locally; per-image scores and checkpoint hashes committed |
| E6 | Qualitative figures produced from real images, predictions and logs; cases selected by rank, not appearance |
| E7 | Dataset licensing and attribution documented (`reports/datasets_and_sources.md`) |
| E8 | Reviewer-response tracker mapping all 20 Reviewer 1 items and Reviewer 2's concerns to evidence or remaining work |

## F. Still outstanding — not changes, blockers

| # | Item | Why it matters |
|---|---|---|
| F1 | Private source-volume/acquisition identifiers | Field grouping is reconstructed from frame geometry, not metadata; biological independence is unproven |
| F2 | Annotation protocol, annotator count, inter-rater agreement | Reviewer 1 item 10; cannot be reconstructed retrospectively. CV shows two conventions exist, which makes this more urgent, not less |
| F3 | Public code and weight release, inherited-code licensing, data-release permission | Reviewer 1 item 3; the repository is currently private, which does not satisfy it |
| F4 | Confirmatory protocol on genuinely unexamined data | Current public results are exploratory: the test labels have now been inspected |
| F5 | Matched-data modern baselines, including a fine-tuned foundation model | Cellpose-SAM zero-shot is context, not an equal-data comparison |
| F6 | Author list, contributions, funding, ethics and AI-use disclosure | Required before any submission |
| F7 | A defensible contribution | If the edge branch does not improve matched baselines, a superiority paper is not justified |
"""
Path('reports/changes_log.md').write_text(doc)
render_pdf(Path('reports/changes_log.md'))
print("wrote reports/changes_log.md/.pdf")
