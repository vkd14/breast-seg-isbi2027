# Repository review and next-step decision

Date: 29 September 2026

## Outcome

The repository is a substantial improvement over the rejected manuscript: the metric contracts
are tested, failures are retained, public and private evidence are separated, and the central
BBBC039 comparison is paired across identical images and seeds. The measured result is narrower
than the old paper claimed. The Gabor-labelled bundle improves public BBBC039 foreground Dice from
0.96604 to 0.96854 (0.250 percentage points), while zero-shot transfer to TNBC H&E is poor for every
project checkpoint (patient-macro Dice at most 0.0807). This supports a small within-domain semantic
segmentation gain—not breast-cancer diagnosis, clinical readiness, instance-level superiority, or
broad histology generalization.

No experimental design can guarantee acceptance. The defensible route is to make the claim match
the evidence, expose limitations, and obtain genuinely independent confirmation.

## Findings requiring action

| Priority | Finding | Action |
|---|---|---|
| Critical | Private “field” labels were reconstructed from geometry and correlation; acquisition and source-volume identifiers are absent. | Describe them only as conservative inferred source groups. Recover original identifiers or keep private results secondary. |
| Critical | BBBC039 test results have already been inspected repeatedly. | Treat them as exploratory. Perform all immediate method selection on the official validation partition and reserve a genuinely unseen cohort for confirmation. |
| High | The 0.250-point result bundles Gabor input, a learned 4-to-3 projection, and SCSE attention. | Isolate edge type, neutral edge, Gabor parameters, attention/projection, sampling, and boundary loss. Do not attribute the whole gain to Gabor. |
| High | TNBC H&E transfer nearly collapses. | Do not present this as external success. Add H&E-specific training with patient/slide separation and preserve another breast cohort for one-time testing. |
| High | Current labels and metrics represent semantic foreground unions. | Narrow the claim to semantic foreground segmentation or add instance reconstruction and AJI/PQ/object-F1 evaluation. |
| High | Earlier completed studies identify the configuration but do not enforce active source-code identity when deciding whether a run is reusable. | Keep completed studies immutable. New studies bind configuration, active source files, and archive hashes into one protocol hash. |
| Medium | The repository is private and trained weights are local; author, ethics, funding, conflict, licensing, and AI-use metadata remain unresolved. | Resolve these before submission and create a permissible public reproducibility release. |

## Verification performed

- All 17 scientific contract and statistics tests pass after the neutral-edge correction.
- The existing result audit reports 114 CSV files, 6,060 recomputed masks, 21 public runs,
  1,050 new public test predictions, and 72 historical checkpoint-by-dataset evaluations.
- The current ISBI draft renders as four US-letter pages.
- Existing public test scores, TNBC patient grouping, saved failure records, checkpoints, and report
  provenance were inspected directly.

## Completed validation-only sensitivity study

`configs/public_validation_ablation.json` froze a 30-run, three-seed study before execution. It
uses only the 100 official BBBC039 training cases and 50 validation cases. Ten conditions compare
the reference Gabor setting with 4/12 orientations, 1/5 scales, sigma 3/7, Sobel, a true zero-valued
edge channel, and complexity-weighted sampling. All conditions receive the same model, loss,
augmentation, optimizer, epoch budget, threshold, and native-grid evaluation. The runner hashes
the frozen configuration, active code, and data archives, and it does not decode or evaluate test
cases. All 30 runs (600 epochs) completed under protocol hash
`424afeb2a9b32e3b53b938d02b074162c42bf3598e712ae80a6d24c66ddbe0df`. This is a
sensitivity study, not independent confirmation.

An initial two-run pilot selected checkpoints using all-case mean Dice. It exposed that BBBC039
validation contains only one empty reference image: correct classification of that single case can
move the mean by roughly two percentage points. Those outputs are preserved under
`results/public_validation_ablation_pilot_allcase/`. Before the full grid, the protocol was
corrected to select on the 49 foreground-containing images and report empty-case false positives
as a separate stratum, directly addressing Reviewer 1 item 1.

Across the 49 foreground-containing validation images, the reference Gabor bank scores
0.966840 Dice (seed SD 0.001701). Weighted sampling has the highest mean at 0.967313: a gain of
0.0473 percentage points with paired bootstrap interval [+0.0218, +0.0795] and Holm-adjusted
p=0.000411. Five scales score 0.967147 (+0.0307 points, interval [+0.0138, +0.0537]) and sigma 7
scores 0.967091 (+0.0250 points, interval [+0.0098, +0.0408]). These effects are statistically
detectable within the reused validation set but practically tiny.

The neutral-edge architecture scores 0.965864, 0.0977 points below the reference; this supports a
small contribution from edge information when architecture is held fixed. It does not establish
that the original 24-filter bank is optimal: four orientations are neutral, twelve are slightly
worse, Sobel's positive mean has a bootstrap interval crossing zero, and sigma 3 is unstable and
0.1435 points worse. The sole empty case is too small for inference. One-scale and sigma-7 models
predict it empty in all seeds; weighted sampling does so in only one of three seeds.

The evidence now supports only a narrow claim: edge information contributes a small
fluorescence-domain benefit, while exact filter-bank and sampling choices change Dice by less than
0.15 percentage points. The recommended next gate is an untouched acquisition or cohort. Freeze
either the five-scale setting (cleaner empty-case behavior) or a prespecified five-scale-by-sampling
factorial on validation data, then evaluate the chosen model once. Do not optimize further against
BBBC039 test labels.
