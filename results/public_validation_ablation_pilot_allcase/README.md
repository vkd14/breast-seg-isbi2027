# Superseded endpoint pilot

This directory preserves the first two completed runs and one interrupted run from 29 September
2026. The pilot selected checkpoints using mean Dice across all 50 BBBC039 validation images.
Inspection showed that the partition contains 49 foreground-containing images and one empty
reference image. Correctly predicting that one case changed the all-case mean by approximately
0.02 and could determine checkpoint selection.

The pilot was therefore stopped rather than silently reused. The full
`public_validation_ablation` protocol selects checkpoints by foreground-containing Dice and
reports the empty-reference case separately. These pilot results are excluded from inferential
comparisons and are retained solely for auditability.
