# Audit of the supplied 2D project

29 September 2026. Historical exports are retained under legacy/embc (older top-level export) and legacy/embc_latest (newer nested breast-seg export); they are evidence, not the active training implementation. Originals have not been edited. Both sources contain 12 distinct baseline/ablation checkpoints; their checkpoint hashes differ. Neither contains the proposed_full checkpoint. The new implementation is under src/breastseg. Numerical audit details are machine-readable in results/audit.

## Findings that affect the interpretation of old results

| Finding | Consequence | New implementation / remaining work |
|---|---|---|
| The older export does not pass the weighted-sampling flag to the loader; the newer nested export fixes this and aligns baseline losses. | The older no-weighted-sampling comparison is invalid; the two versions must not be conflated. Both retain other evaluation problems. | Preserve the newer fix and both archives. Explicit sampler contract and regression test; the first public study uses uniform sampling for every method. A new matched weighted-sampling experiment remains necessary. |
| Legacy validation removes low-scoring batches with an IQR/floor filter and smooths the surviving metric for checkpoint selection. | Checkpoint selection is not based on the full validation set. | All validation images are retained; selection uses unsmoothed native-grid mean Dice. |
| Legacy metrics infer logits versus probabilities from gradient state and maximum value. | Low-magnitude positive logits can be thresholded incorrectly. | The model output contract is always logits; sigmoid then threshold 0.5. Regression test includes logit 0.1. |
| The sampling helper thresholds masks at 127. | Binary 0/1 annotations are incorrectly treated as empty. | All nonzero annotation labels map to foreground; mask-encoding equivalence is tested. |
| Legacy loaders silently replace unreadable images/masks with zeros. | Missing data can masquerade as true negative examples. | New public loader fails on unreadable data or mismatched geometry. |
| Default color loading discards low bits of native 16-bit fluorescence. | Contrast and cross-dataset transfer depend on an undocumented conversion. | Native intensities are retained until deterministic per-image percentile normalization. |
| Signed Gabor kernel sums can be close to zero. | Filter amplitude can be unstable. | L1 kernel normalization and finite-output tests; empirical parameter sensitivity is still required. |
| The modular and monolithic loss definitions differ. | The manuscript cannot treat all exported runs as having one exact loss. | A new explicit Dice/BCE objective and optional differentiable morphological boundary term are versioned. No Tversky or recall-bias claim in the new study. |
| Some legacy failures are silently skipped or replaced; partial accumulated gradients may not be stepped. | Effective training budgets may differ without being reported. | Every batch is used, including the final short batch; nonfinite loss/gradients stop the run. |
| The proposed_full directory contains stored scores but no checkpoint. | Those scores cannot be reproduced from that directory alone. | The original and revised checkpoints elsewhere in the project remain separately identified. No checkpoint identity is assumed. |

## Interpretation

The proposed_full JSON in the supplied exports stores mean Dice 0.9513145454600453 and mean IoU 0.9024580204766244. Recomputing their 128 saved per-image values gives Dice 0.9413145454600453 and IoU 0.8924580204766244, respectively: each stored mean is higher by exactly 0.01. The comparison CSV agrees with the recomputed values. Neither the inflated summaries nor the lower recomputed values are adopted as new validated results.

These are code-level and arithmetic findings. They do not establish why a historical number was entered, and they do not establish misconduct. Old aggregate comparisons should not be reused as evidence of superiority. The stored per-image metrics themselves also need verified split and prediction provenance before they can be considered reliable.

The new study changes preprocessing, loss, validation, and training protocol together relative to the oldest pipeline. A gain over that oldest checkpoint is not an isolated Gabor effect. Within the new study, Gabor+projection+SCSE is one bundled configuration versus vanilla B7; only the boundary-term comparison isolates a single added loss term. Three seeds improve repeatability but do not substitute for independent patient/acquisition groups.

## Publication blockers that software cannot resolve

The original private dataset still needs verified source-volume/acquisition identifiers, a defensible annotation protocol, annotator count and agreement/uncertainty measurements, and release/ethics permissions. Shape-derived groups are not proven independent biological specimens. No public-data experiment fixes those provenance gaps. Clinical deployment, cancer detection, instance-separation superiority, and guaranteed acceptance must not be claimed from the present semantic foreground-segmentation experiments.
