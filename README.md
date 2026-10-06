# BreastSeg: audited 2D nucleus segmentation for ISBI 2027

New working repository, created 29 September 2026. This is an experimental research package, **not a submission-ready or clinically validated model**. It consolidates both supplied 2D exports, the corrected Scientific Reports revision, and the completed external-validation package. Original directories and checkpoints are preserved in place.

## Start here

- Detailed reconciliation: reports/private_uwm_and_literature_reconciliation.md records the exact old-versus-corrected UWM results and protocol-aware comparison with prior BBBC039/TNBC papers.
- `submission/isbi2027_overleaf/main.pdf`: current four-page ISBI 2027 manuscript, with the official linked style and measured results.
- `submission/isbi2027_overleaf/`: self-contained Overleaf sources, bibliography, figures and submission checklist.
- `reports/isbi2027_submission_audit.md`: literature benchmark, reviewer-closure audit, remaining blockers and risk-ranked experiments.
- `results/public_architecture_baselines/`: six matched architectures, three seeds each, under the same BBBC039 budget.
- `results/grayscale_transfer_audit/`: exact grayscale-equivalence check and fixed TNBC polarity-inversion stress test.
- `reports/consolidated_progress_report.pdf`: the full story from the Scientific Reports rejection to the current measurements, with every figure. **Start here.**
- `reports/changes_log.pdf`: itemised list of every protocol correction, code defect fixed, claim withdrawn, and experiment run, plus remaining blockers.
- `reports/prediction_samples.pdf`: qualitative predictions on the private confocal data, public BBBC039, and TNBC breast histology.
- `reports/meeting_report.pdf`: measured progress, comparisons, and decisions for the professor meeting.
- `paper/first_draft.pdf`: first research manuscript draft, with limitations and author-completion items.
- `reports/code_audit.md`: findings that affect the old manuscript's validity.
- `reports/reviewer_action_matrix.md`: every Scientific Reports concern mapped to evidence or remaining work.
- `reports/next_steps.md`: prioritized plan through the ISBI deadline.
- `results/public_study/` and `results/public_followup/`: locked protocols, split manifests, complete learning curves and per-image results. Status files show the active run.
- `results/public_validation_ablation/`: completed 30-run validation-only Gabor/sampling sensitivity study, including foreground/empty strata and a code/archive-hash audit. No test case was decoded or evaluated.
- `results/legacy_external/`: independent re-evaluation of both supplied sets of 12 historical checkpoints. These are not matched-training comparisons.

## Repository map

`src/breastseg/` is the corrected active implementation. `scripts/` contains training, audit, evaluation, validation and report generation. `tests/` holds scientific regression tests. `legacy/embc_latest/` is the newer nested export; `legacy/embc/` is the older top-level export. The archives are historical evidence and contain known problems; do not run them as the new default. `legacy/scientific_reports_revision/` and `legacy/external_validation_20260928/` preserve the preceding audit and public-data results.

Raw images, private data, checkpoints and full prediction-mask directories are excluded from Git. Public datasets are downloadable from their providers. Local checkpoints are under `checkpoints/`; their hashes are in run summaries. All 6,060 new evaluated masks are retained locally under each result directory's `predictions/` folder; per-image scores and representative figures are committed. Private training data are not republished. The repository is public; selected checkpoints still require a separate versioned archive because they exceed normal GitHub file limits.

## Reproduce

Use Python 3.10+ and a CUDA-compatible PyTorch installation. The recorded run environment is in each study's environment.json. The development machine used an RTX 5090 and PyTorch 2.11.0+cu130. Install this package with `pip install -e .`, then:

```bash
python -m unittest discover -s tests -v
python scripts/status.py
python scripts/download_public_data.py
python scripts/train_public_study.py --data data/bbbc039
python scripts/train_public_study.py --data data/bbbc039 --config configs/public_followup.json --study public_followup
python scripts/evaluate_external.py --data data --new-only
python scripts/validate_results.py --data data
python scripts/build_reports.py --data data
```

The completed 30-run sensitivity experiment is validation-only and did not decode or evaluate BBBC039 test cases:

```bash
PYTHONPATH=src python scripts/train_validation_ablation.py --data data/bbbc039
PYTHONPATH=src python scripts/summarize_validation_ablation.py
```

The matched architecture and grayscale/polarity studies are reproduced with:

```bash
PYTHONPATH=src python scripts/train_public_baselines.py --data data/bbbc039
PYTHONPATH=src python scripts/summarize_public_baselines.py
PYTHONPATH=src:scripts python scripts/evaluate_grayscale_transfer.py --data data
PYTHONPATH=src:scripts python scripts/make_submission_figures.py
```

Its protocol identity includes the configuration, active training/core/data source files, and
public archive hashes. See `reports/repository_review_20260929.md` for the scientific review and
decision gates.

Compile the official-template draft from the `paper/` directory using `tectonic first_draft_isbi.tex` or a suitable LaTeX installation. `bash scripts/finalize.sh /path/to/python data` runs tests, saved-mask validation and document generation in order. This script does not commit or submit anything. For a new image, use `python scripts/predict.py --checkpoint checkpoints/public_study/gabor_b7_s0/best.pt --image /path/to/image.tif --output /path/to/new_mask.png`. It refuses to overwrite an existing output.

Training downloads ImageNet encoder weights using segmentation-models-pytorch. Do not change a protocol under a completed run name. Completed runs are skipped only when the protocol hash matches; partial runs stop for inspection rather than silently restarting. Checkpoint loading is strict. The training script uses full validation sets, fixed epochs and a fixed threshold, and evaluates the test partition only after validation-based checkpoint selection. All results are exploratory because earlier test results had already been inspected.

The historical checkpoints are not included; to re-evaluate them use `evaluate_external.py --legacy-only --data data --legacy-a /path/to/newer/export --legacy-b /path/to/older/export`. Explicit paths are necessary because their source training provenance cannot be inferred from filenames.

## Scientific scope

The task is 2D **semantic nuclear foreground segmentation**, not whole-cell segmentation, cancer diagnosis, or validated 3D instance segmentation. BBBC039 is U2OS fluorescence, not breast tissue. TNBC is breast histology and is used as a transfer stress test. BBBC038 partly overlaps BBBC039; newly BBBC039-trained models are therefore not presented as independently validated on BBBC038. Cellpose-SAM is an off-the-shelf reference with unverified pretraining overlap, not an equal-data baseline.

Initial study: vanilla B7 UNet++, the Gabor+projection+SCSE bundle, and that bundle with an explicit boundary term, each with three seeds and 20 epochs. Follow-up: SCSE-only, zero-initialized residual Gabor fusion, residual Sobel fusion, and residual Gabor plus boundary loss, with the same budget. The follow-up is labelled exploratory and was designed after initial results; it is not an independent confirmatory experiment.

The six additional matched baselines are U-Net/ResNet-34, U-Net/ResNet-50, UNet++/EfficientNet-B0, UNet++/Xception/SCSE, DeepLabV3+/ResNet-50 and FPN/ResNet-50. UNet++/Xception/SCSE reaches 96.87% mean Dice and is statistically indistinguishable from the 96.85% Gabor bundle. Explicit grayscale is bit-identical to the existing RGB-entry path for all 115 audited TNBC/BBBC038 images. Fixed intensity inversion raises vanilla TNBC patient-macro Dice from 8.07% to 51.30%, but this post-hoc stress test is not independent external confirmation.

## Attribution and rights

BBBC038/039 are provided under CC0; TNBC v1.1 is CC BY 4.0. Cite the providers and originating papers; see `reports/datasets_and_sources.md`. Their raw data are not relicensed by this repository. The author order, DOE grant DE-SC0025403 and no-competing-interests statement were confirmed on 5 October 2026. A formal UWM determination for the public, de-identified TNBC reuse remains pending. No blanket software license is asserted for inherited code until the rights holder selects one. All result figures are drawn from actual images, predictions and numerical logs; no synthetic microscopy or AI-generated experimental illustrations are used.
