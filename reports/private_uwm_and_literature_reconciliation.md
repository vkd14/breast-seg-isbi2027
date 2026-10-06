# Private UWM and public-literature result reconciliation

## What is now visible in the ISBI manuscript

The manuscript now includes three items that were previously difficult to find:

1. Figure 1 gives the complete implemented path from 16-bit input through percentile normalization, the 24-filter Gabor branch, learned 4-to-3 projection, EfficientNet-B7, UNet++/SCSE, and native-grid output.
2. Table 1(a) reports the private UWM study and reconciles the rejected-paper headline with the saved score arrays and the corrected source-group evaluation.
3. Table 1(b) lists results reported by prior BBBC039 and TNBC papers, with each protocol and endpoint named explicitly.

## Private UWM MCF10A dataset

- Source: in-house UWM MCF10A mammary epithelial cultures.
- Acquisition: confocal z-stacks, Zeiss LSM, 40x/1.1 W objective, 405/461 nm, 0.2506 micrometre pixels, 1.0 micrometre z-step, 16-bit.
- Valid data: 855 image-mask plane pairs after exclusion of two corrupt labels.
- Annotation: one trained annotator with expert review; no repeat-annotation/inter-annotator study is available.
- Provenance: 334 pairs can be linked exactly to eight named stacks. The remaining 521 do not retain exact acquisition identifiers.
- Corrected split: 516/133/206 planes from 11/5/7 conservative geometry-defined groups. These groups reduce related-slice leakage but are not verified biological replicates.
- Mask audit: all 855 valid masks contain foreground. The earlier statement that 60% were empty resulted from applying a 127 threshold to masks encoded as 0/1.
- Distribution: private images are not included in the public-data or code archive, and no public-study claim requires them.

## Why the original 95.1% result is not used as the headline

The rejected manuscript reported 95.13% Dice and 90.25% IoU. The 128 per-image values stored in the corresponding result file average 94.13% Dice and 89.25% IoU—exactly one percentage point lower for both metrics. The old evaluation also used a slice-level split in which related planes crossed partitions; 50% of test images had a training near-duplicate above normalized cross-correlation 0.95. Baseline budgets were not controlled consistently.

The corrected audit therefore uses source-group macro statistics, equal 30-epoch budgets, three seeds for trained models, group-bootstrap confidence intervals, and paired group-level tests.

| Corrected UWM configuration | Dice | 95% group-bootstrap CI | IoU |
|---|---:|---:|---:|
| Full Gabor/projection/SCSE/sampling/stabilized pipeline | 0.9312 | 0.9134–0.9472 | 0.8739 |
| Vanilla EfficientNet-B7/UNet++ | 0.9349 | 0.9175–0.9515 | 0.8805 |
| Plain Dice+BCE objective | **0.9366** | 0.9199–0.9529 | **0.8831** |
| U-Net/ResNet-50 | 0.9320 | 0.9140–0.9502 | 0.8752 |
| Cellpose-SAM, frozen contextual reference | 0.9198 | 0.9058–0.9321 | 0.8539 |

None of the corrected full-pipeline comparisons is statistically significant after Holm correction. The defensible conclusion is therefore not that the complete proposed bundle is superior on the private set. The strongest UWM point estimate is the simpler plain Dice+BCE control, while the full model, vanilla B7, and conventional baselines have overlapping group-bootstrap intervals.

## Public-dataset comparison

The main controlled result is the matched BBBC039 experiment: all trained baselines use the same official 100/50/50 split, preprocessing, 20-epoch budget, augmentation, checkpoint rule, native-grid evaluation, threshold, and no test-time augmentation.

| Study | Data/protocol | Reported endpoint |
|---|---|---:|
| This work | BBBC039 official split, three matched seeds | semantic Dice 0.9685 |
| Caicedo et al. | BBBC039 official split | object F1 0.898 for U-Net |
| SegmentGraph | BBBC039 official split | pixel Dice 0.9482; AJI 0.8680 |
| ASW-Net | BBBC039 with interior expansion | DICE1 0.9645 |
| MA-Net | BBBC039; protocol differs | Dice 0.973 |
| NUSeg | BBBC039 fivefold CV with different selection | Dice 0.9693; IoU 0.9406 |
| This work | all 50 TNBC images, zero-shot plus fixed inversion | semantic Dice 0.5130 |
| CellTranspose | selected TNBC transfer setting, 3–10-shot adaptation | pixel-F1 0.670–0.757 |
| Cellpose in CellTranspose | selected transfer protocol | pixel-F1 0.5829 |

These values are contextual, not a single leaderboard. Object F1, pixel Dice, AJI, DICE1, and pixel-F1 are not interchangeable. Fivefold cross-validation, official held-out testing, selected tissue subsets, and few-shot adaptation also answer different questions. A superiority claim is made only for the matched local comparisons; published values are used to show where the result sits in the literature.

## Current scientific message

On BBBC039, the Gabor bundle improves over the matched vanilla B7 control by 0.250 Dice percentage points, but it is statistically indistinguishable from matched UNet++/Xception/SCSE. On the corrected private UWM analysis, the full bundle is not better than the simpler controls. On TNBC, grayscale conversion was already present; reversing contrast polarity improves patient-macro Dice from 8.07% to 51.30%, but the remaining gap demonstrates substantial stain, morphology, and tissue-context shift.

This is a stronger paper than the rejected version because it replaces an unsupported large superiority claim with a reproducible, multi-dataset analysis of when the edge prior helps, when it does not, and why external transfer fails.
