# Reviewer evidence and remaining work

Prepared from the Scientific Reports rejection and both submitted PDFs, with the later manuscript/code audit checked against available result files. This is an internal planning document, not a claim that every concern has been resolved.

## Fundamental changes in the new draft

The contribution is accurate, reproducible **2D nucleus foreground segmentation and external transfer evaluation**. It is not breast-cancer detection, clinical deployment, quantum computation, or dense 3D instance segmentation. Gabor features are classical image filters. The implemented stabilized objective is 0.7 Dice loss plus 0.3 adaptively weighted BCE; it contains no explicit boundary or Tversky term. New external experiments quantify boundary agreement without relabeling the loss.

The source-aware internal evidence no longer supports the earlier 18.6-point advantage or the claim that each added component is essential. Across three seeds, the full configuration has source-group macro Dice 0.9312 and vanilla B7 UNet++ has 0.9349. None of the corrected comparisons survive Holm correction; the smallest adjusted p-value is 0.515625. These are existing corrected results, not newly trained experiments in this session.

## Reviewer 1

| Item | Evidence or action in this version | Remaining limitation/status |
| --- | --- | --- |
| 1. Empty images | Corrected private-data audit finds 855 valid foreground-containing pairs. External metric code explicitly assigns Dice 1 to both-empty and 0 to one-empty masks, and records empty predictions. | Original 60% empty-image claim was a mask-encoding error. Future background-only evaluation should report false-positive burden separately. |
| 2. Volume independence | Existing audit maps 334 pairs to eight named stacks; remaining images are grouped by geometry. Draft explicitly distinguishes shape groups from verified acquisitions. | 521 pairs lack source identifiers. Geometry grouping does not prove complete acquisition independence. Recover IDs or restrict primary private experiments to traceable stacks. |
| 3. Reproducibility | New evaluation code, exact dataset URLs, checksums, protocols, per-image CSVs, predictions, and plots are available locally. Public datasets have unrestricted research access under stated licenses. | Private-data release authorization and publicly downloadable model weights still need resolution; local files are not a public release. |
| 4. Tversky interpretation | Draft states the implemented Dice-BCE objective. For TP/(TP + alpha FP + beta FN), stronger false-negative penalty requires beta > alpha. | Do not claim the original checkpoint used Tversky. Existing explicit boundary/Tversky ablation is a separate model configuration. |
| 5. Boundary claims | External evaluation adds foreground boundary F1 and surface Dice at 2 native pixels, plus HD95 in native pixels. | Foreground union erases touching-instance boundaries. These metrics do not establish instance separation. |
| 6. Gabor sensitivity | Existing three-seed runs vary orientation count 4/8/16, scales 2/3/5, sigma 2/5/10; Sobel/Scharr/LoG/Canny comparisons exist. | No consistent superiority. External component-wise ablations remain to be run. |
| 7. Equal training budgets | Existing corrected encoder-decoder comparisons use the same 30-epoch training harness and three seeds. | Public pretrained Cellpose has different training exposure and compute; report as an off-the-shelf reference, not a matched training comparison. |
| 8. Weighted sampling | Formula and normalized probability are in the new draft, matching the corrected harness. Uniform-sampling ablation is retained. | Corrected internal uniform-sampling Dice 0.9277 versus full 0.9312 is not a significant superiority result. |
| 9. Zero-positive batches | Code inspection confirms r=0 gives positive weight 1. The new metric checks cover empty masks. | No new loss-gradient stability test was run here; avoid saying it was. A constant nonfinite-loss fallback is not proof of stable optimization. |
| 10. Annotation agreement | Unmeasured boundary-uncertainty claims are removed. | Annotator count, software, written rule, review protocol, and repeat-annotation evidence require author confirmation. Encoding differences cannot identify annotator sessions. |
| 11. Clinical-grade claims | Removed from the replacement draft. Cultured-cell and histology nuclear foreground scores do not establish screening or treatment utility. | Clinical validation is outside the present evidence. |
| 12. Statistical inference | Existing internal results use group bootstrap, paired group-level Wilcoxon, Holm correction. New TNBC results aggregate within 11 patient folders before paired comparisons and bootstrap. | BBBC intervals are descriptive field/image bootstrap because full independent-source linkage is unavailable. Small group counts and public-model exposure limit inference. |
| 13. Figures/Quantum labels | New figures are plotted directly from real public images, masks, predictions, and computed Gabor responses. | AI assistance to writing/coding must be disclosed under venue rules and reviewed by the authors. No AI-generated scientific image evidence is used. |
| 14. External validation | Completed four-method evaluations on BBBC038 (65 test images), BBBC039 (50 test images), and TNBC (all 50 images). | Generalization is limited; TNBC failure is reported. BBBC038 and BBBC039 partly overlap and are not independent cohorts. |
| 15. TTA fairness | New neural-network runs use no test-time augmentation and fixed thresholds; native-grid metrics for all methods. | Cellpose uses its own native-resolution inference pipeline; wall-clock observations are not a controlled speed benchmark. |
| 16. Overfitting/complexity | Existing corrected four-fold source-group cross-validation gives macro Dice 0.911; encoder ablations and seed variation exist. External shift is now measured. | Formal label-budget learning curves and public adaptation curves remain necessary for a data-efficiency claim. |
| 17. Vanilla B7 anchor | Existing vanilla B7 UNet++ comparison scores 0.9349 versus full 0.9312. Smaller-encoder comparisons exist. | Only the available revised seed-0 checkpoint was externally tested. New vanilla and multi-seed external checkpoints must be trained or recovered. |
| 18. Cropping | External tests use entire images, a 512x512 network resize, and native-size scoring; no ROI selection. Corrected private pipeline also omits ROI cropping. | Verify original export/crop provenance for untraceable private images before claiming no related crops cross partitions. |
| 19. Inconsistent IoU | New tables are generated from the same JSON/CSV outputs; mean Dice and mean IoU are separately computed per image. | Retire original 0.892/0.902 inconsistencies rather than selecting the more favorable number. |
| 20. Failed baselines | Existing corrected MAnet and ResNet50 UNet++ runs are retained; they converged with the standardized harness. All four attempted methods are reported on each new dataset. | Historical baseline-failure causes cannot be fully reconstructed; do not invent an explanation. |

## Reviewer 2

The abstract now states the tested question, exact external populations, numerical outcomes, and limitations. The introduction focuses on modality transfer and evaluation independence. A short related-work section covers encoder-decoder segmentation and current microscopy foundation models. HIC-net and histopathology texture-classification studies can supply historical context, but they are classification tasks and should not be represented as segmentation baselines. The suggested EEG-depression and medical-text ontology papers do not directly inform this imaging task and are not added solely to increase citations.

The expanded internal ablations and new external results address breadth of evaluation. The largest unresolved requirements are evidence of a distinct methodological contribution, a fair external comparison against a vanilla same-encoder model, annotation reliability, and complete dataset provenance. Adding more model names alone will not repair these.

## Inconsistencies found in existing revision documents

The older response-to-reviewers file still claims verified acquisition-field splits, per-image BCa intervals, completed public release, and inferred annotation sessions. Those claims conflict with the later `main_revised.tex` and `stats_clustered.py`, which use geometry-defined groups, group-level percentile bootstrap, and explicitly acknowledge missing provenance and annotation evidence. This package follows the latter evidence and does not reuse the older response verbatim. Neither prior manuscript is currently under review, according to the authors; the new document is a replacement submission, not a response submitted to the rejecting journal.

## Evidence locations

Internal source: `data&code/outputs/revision/stats_clustered/{summary.csv,tests.csv,numbers.json}`; model and loss implementation: `data&code/src/revision/train_revision.py`; corrected text: `Template_for_submissions_to_Scientific_Reports/main_revised.tex`; external results: `renowned_2d_paper_v1/outputs/*_zero_shot/`; exploratory 3D results: `renowned_3d_paper_v1/outputs/{nested_fusion,cellpose_plus}/`.
