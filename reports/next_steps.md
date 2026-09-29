# Decisions and next experiments

## Immediate meeting deliverable

Use the measured report, first draft, actual qualitative examples, source audit, and full result CSVs. Discuss what failed as well as what improved. Do not claim the project is clinically ready or guaranteed acceptance. The public-data experiments can stand independently of the private-source provenance issue, but they currently establish nucleus foreground segmentation, not breast cancer diagnosis.

## Highest-value work before submission

1. Confirm the scientific target with the professor: nuclei in fluorescence, nuclei in H&E breast histology, or instance separation. These require different evaluation and dataset design. The current 2D network returns a binary union mask.

2. Resolve private data provenance. Recover source-volume/acquisition identifiers before any split, document biological samples and adjacent slices, crop counts, annotations and permissions. If unavailable, remove private-data superiority claims and build the main paper on public data.

3. Freeze a new confirmatory protocol. The current public results are exploratory. Reserve a genuinely unexamined acquisition/site dataset and specify primary endpoint, grouping, confidence intervals and comparison family before evaluating it. Do not repeatedly optimize against the BBBC039 or TNBC test labels already inspected.

4. Make the breast-histology experiment credible. Train or fine-tune on an appropriate labelled histology training set using patient/slide-level separation; retain an independent breast cohort as the confirmation test. Compare RGB/stain-aware preprocessing with grayscale instead of assuming fluorescence transfer should succeed. TNBC has only 11 patients; seeds cannot replace more patients.

5. Add matched-data baselines: smaller U-Net/UNet++ encoders, a well-trained modern instance method if instance masks are the target, and an appropriately fine-tuned foundation-model comparator. Document its pretraining overlap. Cellpose-SAM zero-shot is useful context but not equal-data evidence.

6. Complete the ablation grid with a bounded validation-only search: Gabor orientations 4/8/12, scales 1/3, sigma 3/5/7; Sobel and no-edge controls; attention, projection and boundary term separated; uniform versus weighted sampling. Report every planned condition, not only winners. Numerical filter stability is not performance robustness.

7. Add data-efficiency curves (for example 10/25/50/100 fields, three seeds, identical selected subsets), true empty-field false-positive measurements, robustness to realistic exposure/contrast changes, and per-object boundary/instance metrics if claimed. Preserve the same preprocessing across training and inference.

8. Finalize statistics at the real independent unit (patient, slide, source volume or acquisition), effect sizes and paired bootstrap intervals. Use Holm correction for the prespecified family. Do not treat three seeds times fifty images as 150 independent samples, and do not infer generality from one chemical screen.

9. Publish code, permissible weights, manifests and representative permissible data with licensing and dataset citations. Confirm author list, contributions, funding, conflicts, ethics, and AI disclosure. The new private working repository is not a public reproducibility release.

10. Choose submission on evidence. If controlled edge gains remain negligible or inconsistent, simplify the model and frame the study honestly; do not retrofit a state-of-the-art claim. Keep 3D results as explicitly exploratory context, not independent confirmation of the 2D claim.

## Venue timing

ISBI 2027 is the immediate fit: deadline 26 October 2026, four technical pages, with an optional paid fifth page for references/ethics/acknowledgments only. Official instructions: https://biomedicalimaging.org/2027/papers/ (checked 29 September 2026). Use the official template and respect the prohibition on substantially similar simultaneous submissions. Previously rejected/withdrawn submissions are not current simultaneous submissions, per the user's clarification.

MIDL 2027 is a possible later AI-focused alternative if methodological evidence becomes stronger; recheck its official dates before committing. Biomedical Signal Processing and Control or Computers in Biology and Medicine remain possible journal directions after the experimental and provenance gaps are resolved. A longer journal manuscript should not simply repeat the rejected claims with new formatting.
