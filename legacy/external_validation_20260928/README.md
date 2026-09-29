# 2D nucleus segmentation: external validation and replacement-paper draft

Created September 28, 2026 for the September 29 professor meeting.

Start with [the meeting brief](reports/meeting_brief.pdf), [the first manuscript draft](paper/first_draft.pdf), [venue and dataset recommendations](reports/venues_and_dataset_plan.pdf), and [the reviewer action matrix](reports/reviewer_action_matrix.pdf). Editable Markdown sources accompany each PDF. The draft is a discussion manuscript, not a completed four-page ISBI submission.

## Completed results

| Dataset / evaluated population | Released 2D Dice | Revised 2D Dice | Otsu Dice | Cellpose-SAM Dice |
| --- | --- | --- | --- | --- |
| BBBC038v1, 65 official stage-1 test images | 0.3266 | 0.7167 | 0.7545 | 0.8702 |
| BBBC039v1, 50 official test images | 0.9204 | 0.9158 | 0.9479 | 0.9695 |
| TNBC v1.1, all 50 images / 11 patients | 0.2108 | 0.1755 | 0.4275 | 0.8213 |

These are newly executed frozen-checkpoint evaluations, not new training. Each cell is an image-macro mean over the complete stated population. The two BBBC datasets partly overlap and must not be pooled as independent cohorts. Cellpose-SAM is an off-the-shelf reference; its public-data pretraining exposure was not audited. Only revised seed 0 was available for external testing. Metrics concern binary nuclear foreground, including foreground contour accuracy, not instance separation or cancer detection.

The corrected pipeline substantially improves the heterogeneous BBBC038 result but is not uniformly better. The fluorescence result is useful, while histology transfer fails. Nothing here supports a claim of beating Cellpose or clinical readiness. The four-week plan prioritizes controlled domain adaptation and equal-budget same-encoder comparisons.

## Artifacts

- `outputs/*_zero_shot/protocol.json`: populations, data/checkpoint hashes, and inference settings.
- `outputs/*_zero_shot/per_image.csv`: all 660 method-image measurements.
- `outputs/*_zero_shot/summary.json`: full-population means and descriptive image-bootstrap Dice intervals.
- `outputs/*_zero_shot/predictions/`: native-grid binary predictions, retained locally and excluded from Git.
- `reports/analysis.json`: patient-level TNBC summaries, confidence intervals, paired Holm-corrected tests, and explicitly exploratory intensity strata.
- `reports/figures/`: actual benchmark figures; `figure_selection.json` records every selected image and quantile.
- `outputs/internal_reference_summary.json`: snapshot of the earlier source-group statistics; not recomputed training results.
- `sources/`: local PDF-to-text extracts of the submitted manuscripts, excluded from Git.

## Reproduction

The current local interpreter is `/home/vdasoju/Desktop/breast-3d/breast_qubo/.venv/bin/python`. GPU inference used an RTX 5090, PyTorch 2.11.0+cu130, and the software versions in `outputs/environment.json`.

Run from the package directory with that interpreter (shown as `python` below):

```bash
python src/download_data.py
python src/evaluate_bbbc038.py --dataset bbbc038
python src/evaluate_bbbc038.py --dataset bbbc039
python src/evaluate_bbbc038.py --dataset tnbc
python src/check_evaluation.py
python src/analyze_and_render.py
```

The evaluator accepts `--checkpoint-released`, `--checkpoint-revised`, and `--checkpoint-cellpose` for another filesystem layout. Defaults locate the existing project checkpoints and local Cellpose cache. Loading is strict; no missing layers are silently accepted. New evaluation runs overwrite this package's result files for that dataset, so copy the package or archive results before experimenting with different checkpoints.

The two author checkpoints are not included in Git and public download links remain to be provided. Their hashes are recorded. This is an explicit reproducibility requirement before submission, not a completed release. The original training project is needed only for the optional private preprocessing cross-check; downloaded public data and recorded results suffice to regenerate the external figures and discussion PDFs with the included internal summary snapshot.

## Verification

Checks cover nonsquare column-major RLE decoding, perfect/empty-mask metric conventions, unique complete case coverage, recomputation of all metrics from all 660 saved predictions, and tensor-level agreement between revised preprocessing and the original training dataset implementation. Scores are calculated at native resolution with fixed thresholds. The PDFs and prediction figures have also been inspected for layout and mask/image alignment.

## Submission plan

Best one-month conference target: [ISBI 2027, October 26, 2026](https://biomedicalimaging.org/2027/papers/). Alternative: [MIDL 2027, November 30 abstract / December 4 paper](https://2027.midl.io/important-dates). See the venue report for IPMI, MICCAI, and journal fit, plus the distinction between nuclei, whole-cell, and cryo-EM particle annotations.

Remaining author work includes acquisition provenance, annotation agreement, data/weight release, ethics/funding statements, and approval of the scientific framing. No paper has been submitted automatically. The original PDFs and older project results remain unchanged.

## Attribution

BBBC038v1 and BBBC039v1 are provided by the Broad Bioimage Benchmark Collection under CC0: https://bbbc.broadinstitute.org/BBBC038 and https://bbbc.broadinstitute.org/BBBC039. TNBC v1.1 is by Naylor Peter Jack, Walter Thomas, Lae Marick, and Reyal Fabien, under CC BY 4.0: https://doi.org/10.5281/zenodo.2579118. Plotted TNBC panels reproduce original images/masks with our prediction and layout additions. Dataset terms are separate from repository code licensing. Raw downloads are not committed.
