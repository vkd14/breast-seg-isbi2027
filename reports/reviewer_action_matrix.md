# Reviewer-response action matrix

29 September 2026. Completed software work is distinguished from experiments and author information still needed. This is a remediation tracker, not a claim that all rejection reasons are resolved.

| Reviewer 1 item | Action / evidence | Status before submission |
|---|---|---|
| 1. Empty images | Explicit both-empty Dice=1, one-empty Dice=0; masks decoded by nonzero labels; numerical tests; foreground and empty counts in public reports. | Software complete. Public sets do not establish empty-field specificity; add independent true empty fields. |
| 2. Source-volume independence | Public official split manifest plus image/group/hash disjointness. | Public split checked. Private source-volume identifiers still unresolved; do not claim private leakage-proof validation. |
| 3. Reproducibility | New code repository, locked configs, download script, result CSVs, checkpoint hashes. | Local reproduction supported; repository initially private and weights local. Public code/weights release and inherited licensing still needed. |
| 4. Tversky interpretation | New study uses explicit Dice+BCE, no Tversky or unsubstantiated recall-bias statement. | Resolved for new study. Historical FP/FN coefficient interpretation documented in audit. |
| 5. Boundary claims | Explicit differentiable morphological boundary Dice term; native-grid BF1, surface Dice, HD95. | Implemented and tested; empirical usefulness must follow measured ablation results. |
| 6. Gabor robustness | Earlier corrected private-data runs already include orientation/scale/sigma sweeps and Canny/LoG/Scharr/Sobel controls. These are retained in the inherited 34-condition summary. New public follow-up includes Sobel. | Private sweep remains limited by uncertain source groups; repeat sensitivity tests on public data. Numerical kernel tests alone are not empirical robustness evidence. |
| 7. Unequal budgets | Same 100 training fields, 50 validation fields, 20 epochs, optimizer, schedule, augmentation and threshold for new comparisons. | Completed for new study; architecture-dependent compute differs and should be reported. Historical comparisons remain unequal. |
| 8. Sampling details | Formula in core.py, binary-encoding and enable/disable regression tests. Uniform sampling throughout the first public studies. | New weighted-versus-uniform matched ablation remains. Newer supplied export already fixes the older flag omission. |
| 9. Zero-positive batches | Safe optional adaptive weighting; tests on all-zero, all-one and extreme-logit batches. | Software complete. Main public objective uses unweighted BCE. |
| 10. Annotation quality | Public annotations attributed to providers. | Authors must establish original private annotators, protocol, inter-rater agreement and uncertainty; cannot fabricate retrospectively. |
| 11. Clinical claims | No clinical-grade, diagnostic or deployment claim in the new draft. | Resolved by narrowing scope; no clinical validation performed. |
| 12. Statistics | Per-image scores, descriptive bootstrap CIs, paired comparisons with Holm correction; patient aggregation for TNBC. | Exploratory inference; BBBC039 has one acquisition study, not 50 independent patients. New independent confirmation still needed. |
| 13. Quantum/AI figures | Actual image/prediction plots; no quantum mechanism claimed. AI assistance disclosure drafted. | Author review of disclosure and all figure attributions required. |
| 14. External validation | Complete frozen-checkpoint BBBC038, BBBC039 and TNBC partitions; all new models transferred to TNBC without tuning. | Measured evidence includes failures. BBBC038/039 overlap and foundation-model training overlap prevent broad claims. |
| 15. TTA | Fixed threshold 0.5, no TTA for any new neural-model comparison. | Complete; separate TTA experiments would need matched settings. |
| 16. Small data/overfit | Three seeds, full learning curves, ImageNet initialization, AdamW and geometrical augmentation. | Partial: official public holdout is not patient-level cross-validation; add independent data-efficiency curves and grouped CV. |
| 17. B7 contribution | Vanilla B7, SCSE-only B7 and controlled edge/loss variants. Earlier private encoder ladder (B0/B3/B5/ResNet34/50) is retained. | Public replication of smaller-encoder controls and modern matched-data baselines remains needed. Do not discard the already completed private study. |
| 18. Crops/ROI | No ROI or crop generation in new public study; complete fields resized to 512 and probabilities restored to native grid. | New protocol complete; private historical crop/source provenance still needed if reported. |
| 19. Inconsistent IoU | Historical arithmetic audit: stored JSON means exceed saved values by 0.01 for Dice and IoU. New tables generated from CSVs. | Do not reuse historical inflated means. Generated new numbers independently rechecked. |
| 20. Failed baselines | All 24 supplied historical checkpoints attempted, including both export versions; failures retained explicitly. New run outputs not discarded for poor results. | Check evaluation failure log; historical failures are not evidence of novel-method superiority. |

## Reviewer 2 and EMBC concerns

The draft focuses its introduction on the empirical question, adds directly relevant segmentation literature, defines the nucleus-foreground target, provides exact dataset/task descriptions, removes quantum framing and synthetic science illustrations, and distinguishes method components through controlled comparisons. Numerical tables come from measured outputs. Unrelated EEG, depression or temporal-text papers suggested in a review are not added as token citations: relevance must be justified scientifically. Histopathology classification papers may motivate the wider application but are not segmentation baselines.

The most important outstanding issue is a defensible contribution beyond combining established blocks. If the edge branch does not consistently improve matched baselines, a superiority paper is not justified. A transparent evaluation paper may still be useful, but a strong methods submission needs either a validated improvement, a distinctive data-efficiency/robustness result, or a reusable dataset/protocol contribution.
