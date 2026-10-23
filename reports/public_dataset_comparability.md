# Public-dataset protocol and metric comparability

Status: interim three-seed results; the queued ten-seed studies will replace these point estimates.

## What is directly comparable

BBBC039 has an official 100/50/50 train/validation/test partition. We now evaluate saved predictions with the public metric definitions used by Caicedo et al. and ASW-Net: foreground DICE1, ensemble DICE2, AJI, detection quality (DQ), segmentation quality (SQ), panoptic quality (PQ), and average object F1 over IoU thresholds 0.50--0.90. The same outputs are also evaluated with symmetric best Dice. This resolves the former metric mismatch; statistical comparison across papers is still unavailable because their per-image predictions and repeated seeds are not published.

Interim BBBC039 means across seeds 0--2, using connected components of our semantic output:

| Model | DICE1 | DICE2 | AJI | DQ | SQ | PQ | Caicedo object-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla B7/UNet++ | .9661 | .7988 | .6764 | .8141 | .9064 | .7397 | .7494 |
| Gabor/projection/SCSE B7 | .9685 | .8022 | .6811 | .8192 | .9118 | .7485 | .7576 |
| Xception/UNet++/SCSE | .9687 | .8011 | .6795 | .8154 | .9120 | .7454 | .7542 |

Published official-split context:

| Study | DICE1/pixel Dice | DICE2 | AJI | DQ | SQ | PQ | Object F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Caicedo U-Net | -- | -- | -- | -- | -- | -- | .898 |
| ASW-Net | .8964 | .8752 | .7981 | .9167 | .8063 | .7406 | -- |
| ASW-Net + interior expansion | .9645 | .8480 | .9020 | .9443 | .9167 | .8665 | -- |
| PFFNet (BISSG paper table) | .9478 | -- | .8477 | -- | -- | .8331 | .9451 |
| BISSG | .9482 | -- | .8680 | -- | -- | .8629 | .9670 |

Interpretation: our semantic foreground overlap is competitive, but connected components merge touching nuclei. AJI, PQ, and object F1 therefore trail instance-aware methods. This is now a measured limitation, not a hidden metric mismatch. A queued three-class interior/boundary model directly targets this gap.

## Reconstructed BBBC039 handling

- Caicedo et al.: official 100/50/50 split; TIFF converted to 8-bit PNG; 256-pixel random patches; batch 10; 500 steps per epoch; 15 epochs; rotations, flips, and illumination scaling; three output classes (background/interior/boundary); minimum object size 16; average object F1 over IoU 0.50--0.90.
- ASW-Net: official 100/50/50 split; 128-pixel random crops; batch 10; 300 steps per epoch; 50 epochs; rotations, flips, and illumination augmentation; DICE1, DICE2, AJI, DQ, SQ, and PQ; an explicitly reported interior-expansion postprocessing variant.
- BISSG: official 100/50/50 split; instance-aware dual-decoder U-Net and superpixel graph; 200 U-Net pretraining epochs followed by 200 graph-training epochs; AJI, pixel Dice, pixel F1, and PQ.
- Current semantic study: official split; complete 512-pixel resized images; 20 epochs; identical model budgets; no test-time augmentation; test predictions restored to 520x696 before evaluation.

The official split and metric formats are aligned. Training preprocessing remains part of each method and is therefore described rather than falsely treated as identical.

## TNBC protocol comparability

TNBC has no official split. The most important published protocols are not equivalent:

- FRE-Net randomly divides the 50 images 7:1:2 into train/validation/test. Exact image IDs and random seed are not disclosed. Because images come from 11 patients, a random image split can place one patient's images in multiple partitions.
- MA-Net chooses one image from each of the 11 patient groups for test, uses the remaining 39 for training, and uses 30% of those 39 for validation. The paper does not disclose the selected image IDs. Its public code reads RGB images, scales by 255, resizes to 512, uses batch size 4, Adam at 0.001, 100 epochs, and no augmentation.
- PFFNet reports random three-fold cross-validation. Its public inference code leaves the fold directories blank, so the exact folds cannot be reconstructed from the repository.
- The two-stage stacked U-Net uses 384-pixel patches at four scales, RGB input, rotations and horizontal/vertical flips, SGD at 0.0001, batch size 4, and 20 epochs, but does not disclose a reproducible TNBC partition in the article.
- Our primary TNBC study uses five fixed patient-disjoint folds, one validation fold, and three training folds. Every patient is tested exactly once. This is stricter for leakage prevention but cannot be numerically identical to papers whose test IDs are unavailable.

We now express our outputs in the same metric formats. Interim patient-macro means across seeds 0--2 are:

| Model | DICE1 | AJI | PQ | Object F1 | Symmetric best Dice |
|---|---:|---:|---:|---:|---:|
| Vanilla B7/UNet++, grayscale | .7438 | .4979 | .4418 | .3482 | .5709 |
| Gabor/projection/SCSE B7, grayscale | .6718 | .4460 | .3592 | .2649 | .4999 |
| Xception/UNet++/SCSE, grayscale | .7346 | .4313 | .3923 | .2985 | .5314 |

Published values can now be compared by metric, with the protocol caveat retained: DIST reports F1 .824/AJI .585; two-stage stacked U-Net reports F1 .806/AJI .621; R-SNN reports Dice .8441/AJI .7332; FRE-Net reports Dice .8563 +/- .0323; and MA-Net reports Dice .7936.

The completed grayscale study discarded H&E stain information, whereas the cited methods use RGB. A queued ten-seed RGB study retains H&E color under the same patient-disjoint folds. Its result will determine whether the performance gap is caused primarily by input representation or model structure.

## Rules for the manuscript table

1. Mark BBBC039 comparisons as metric- and split-aligned, but do not claim cross-paper significance without per-image outputs.
2. Mark TNBC comparisons as metric-aligned but split-misaligned unless exact test IDs become available.
3. Report image-macro values when matching papers and patient-macro values as the primary leakage-resistant endpoint.
4. Keep semantic and instance claims separate.
5. Replace all interim three-seed values only after the ten-seed validator confirms 0--9 completeness and protocol hashes.
