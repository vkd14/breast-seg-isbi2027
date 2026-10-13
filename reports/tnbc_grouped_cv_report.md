# TNBC patient-grouped cross-validation report

## Protocol

The study trains three matched models under five fixed patient-disjoint outer folds and seeds 0, 1, and 2. Each run uses 20 epochs, the same augmentations and optimizer, a fixed threshold of 0.5, no test-time augmentation, and native-grid evaluation. This adds 45 runs and 900 epochs. The primary endpoint is patient-macro semantic Dice across 11 held-out patients.

## Results

| Model | Dice (95% patient-bootstrap CI) | Seed SD | IoU | Boundary F1 |
|---|---:|---:|---:|---:|
| Vanilla EfficientNet-B7/UNet++ | 0.7429 [0.7148, 0.7692] | 0.0052 | 0.6001 | 0.6306 |
| Gabor/projection/SCSE B7 | 0.6702 [0.6235, 0.7104] | 0.0418 | 0.5189 | 0.5266 |
| UNet++/Xception/SCSE | 0.7303 [0.6929, 0.7641] | 0.0139 | 0.5871 | 0.5915 |

The Gabor bundle is 0.0726 Dice below vanilla (95% CI -0.0998 to -0.0526; two-sided paired Wilcoxon Holm-adjusted p = 0.00390625). It is 0.0600 below Xception/SCSE (95% CI -0.0788 to -0.0425; adjusted p = 0.00390625). Its boundary F1 is 0.1040 below vanilla (95% CI -0.1314 to -0.0821; adjusted p = 0.00390625).

## Interpretation

Combined with the BBBC039 result, the evidence shows a modality interaction. The same Gabor bundle adds 0.250 Dice percentage points on fluorescence but loses 7.26 points on H&E. The edge prior is therefore acquisition-dependent, and it should not be presented as a universal segmentation improvement. The vanilla B7 model is the strongest of the three under the locked TNBC protocol.

## Scope

This TNBC study is exploratory because the labels were viewed during earlier zero-shot analysis. It uses only 11 patients and grayscale input. A confirmatory extension should lock RGB, stain-normalized RGB, HED, and grayscale preprocessing before evaluating an unseen patient cohort.
