# ISBI 2027 submission and evidence audit

3 October 2026. This audit distinguishes measured evidence from claims that remain unsupported. No review process or acceptance can be guaranteed.

## Submission decision

The strongest defensible paper is a controlled empirical study of edge priors and domain shift, not a claim of a new state-of-the-art breast-cancer detector. The work is 2D semantic nuclear foreground segmentation. BBBC039 is U2OS fluorescence, not breast tissue; TNBC is breast histology used as a zero-shot stress test. Preliminary 3D work is mentioned only as future direction.

The short manuscript now also includes the private UWM MCF10A lineage as an explicitly labelled audit. Its original 95.13% Dice aggregate is reconciled against the 94.13% mean recomputed from the saved per-image scores and the corrected 93.12% source-group result. This private evidence is not required to reproduce the public-data claims and is not presented as independent external validation.

The prepared manuscript uses the official ISBI-linked IEEE style, is four pages, and places all technical content on pages 1–3. Page 4 contains only compliance, acknowledgments/disclosures and references. ISBI 2027 is single-blind; the current author names are therefore present. The official deadline shown on the ISBI author page is 26 October 2026 at 11:59 PM US Eastern time. Recheck the portal before submission: https://biomedicalimaging.org/2027/papers/ .

## What was newly tested

| Evidence block | Runs / units | Principal result | Interpretation |
|---|---:|---|---|
| Original public component study | 7 models x 3 seeds, 420 epochs | Gabor bundle 96.85% Dice vs vanilla B7 96.60%; +0.250 points, 95% CI +0.197 to +0.299 | Small within-screen gain |
| Matched architecture baselines | 6 models x 3 seeds, 360 epochs | UNet++/Xception/SCSE 96.87%; Gabor minus baseline -0.014 points, CI -0.047 to +0.018, Holm p=.378 | No evidence that Gabor beats the strongest matched comparator |
| Validation-only sensitivity | 10 conditions x 3 seeds, 600 epochs | 96.541–96.731% foreground Dice across filter/sampler choices | Gabor settings are not uniquely optimal; effects are small |
| TNBC explicit grayscale audit | 21 checkpoints x 50 images | RGB-entry and explicit-grayscale features match exactly; 115/115 TNBC+BBBC038 images pass | The low transfer score was not caused by omitted grayscale conversion |
| TNBC one-shot polarity stress test | 21 checkpoints, 11 patient groups | Vanilla 8.07% to 51.30% patient-macro Dice; +43.24 points, patient bootstrap CI +36.55 to +50.15 | Bright/dark polarity explains much, but not all, domain failure |
| Corrected private UWM audit | 855 planes; 206-plane/7-group test; 3 seeds | Full pipeline 93.12% vs vanilla B7 93.49% and plain Dice+BCE 93.66% group-macro Dice | The rejected-paper superiority claim does not survive corrected grouping and equal budgets |

Total new controlled training: 69 runs and 1,380 fixed-budget epochs. Every run, including weak models, is retained.

## External literature comparison

These figures are context, not a common leaderboard. Metric definitions, splits, instance postprocessing and model-selection practices differ.

| Study | Dataset/protocol | Reported result | Comparability to this work |
|---|---|---:|---|
| Caicedo et al., Cytometry A (2019) | BBBC039 official split; object F1 averaged over IoU 0.50–0.90 | U-Net 0.898; DeepCell 0.858 | Not semantic Dice; useful historical context |
| SegmentGraph, IJCAI (2022) | BBBC039 official split | pixel Dice 0.9482; AJI 0.8680; object F1 0.9670 | Pixel Dice is closer, but graph postprocessing and instance endpoints differ |
| ASW-Net (2022) | BBBC039 official split with interior expansion | DICE1 0.96452; DICE2 0.84798; AJI 0.90200 | DICE1 construction differs from native semantic Dice |
| MA-Net, PLOS ONE (2024) | BBBC039 | Dice 0.973; TNBC 0.809 | Public description does not establish a training/selection protocol identical to ours |
| NUSeg / pi-PhenoDrug (2024) | BBBC039 fivefold CV | Dice 0.9693; IoU 0.9406 | Paper describes per-epoch test evaluation and best-weight selection; not a held-out equivalent |
| CellTranspose, WACV (2023) | BBBC039 to selected TNBC tissue types with 3/5/10-shot adaptation | pixel-F1 0.6702/0.7377/0.7568; Cellpose 0.5829 | Different subset and adaptation setting; our 0.5130 is all-50 zero-shot after fixed inversion |

Primary sources: BBBC039 provider https://bbbc.broadinstitute.org/BBBC039 ; BBBC038 provider https://bbbc.broadinstitute.org/BBBC038 ; TNBC v1.1 https://zenodo.org/records/2579118 ; Caicedo evaluation https://pmc.ncbi.nlm.nih.gov/articles/PMC6771982/ ; SegmentGraph https://www.ijcai.org/proceedings/2022/0169.pdf ; ASW-Net https://pmc.ncbi.nlm.nih.gov/articles/PMC8950038/ ; MA-Net https://doi.org/10.1371/journal.pone.0308326 ; NUSeg https://doi.org/10.1002/aisy.202400635 ; CellTranspose https://pmc.ncbi.nlm.nih.gov/articles/PMC10760785/ .

## Reviewer-comment closure

Resolved computationally in the new paper: empty-mask definitions and strata; official split and SHA-256 leakage checks; public code structure; removal of the incorrect Tversky recall claim; explicit boundary loss and BF1/surface-Dice/HD95; orientation/frequency/sigma/Sobel/neutral-edge sensitivity; equal budgets; explicit sampler formula and ablation; safe zero-positive batches; removal of clinical-grade claims; paired bootstrap intervals and Holm-adjusted tests; real-data figures and AI disclosure; TNBC transfer; identical no-TTA evaluation; three seeds and complete learning curves; B7/encoder/architecture controls; no hidden crop generation; recomputed consistent metrics; and retention of failed baselines.

Partially resolved or inherently unavailable: public provider annotations replace undocumented private labels, but this does not create inter-annotator agreement; BBBC039 is one acquisition study, not biological multi-site replication; TNBC inversion was motivated and evaluated post hoc, so it needs confirmation on unseen histology; BBBC039 partly overlaps BBBC038; foundation-model pretraining overlap is unresolved; binary masks do not permit instance split/merge validation; three seeds do not characterize all training variation.

## Submission blockers and risk-ranked next experiments

1. **Ethics determination — blocking.** A ready-to-submit UWM determination request is included in `submission/UWM_IRB_DETERMINATION_REQUEST.md`. Obtain the formal non-human-subjects/exempt determination for reused TNBC human histology and insert its exact wording/identifier. Do not infer ethics status from CC BY licensing.
2. **Immutable reproducibility archive — partial.** The repository is public. Settle inherited-code licensing, upload the selected checkpoint to a GitHub release/Zenodo archive, and cite an immutable DOI or tagged commit.
3. **Unseen histology confirmation — highest scientific value.** Freeze the polarity rule without further TNBC label use; evaluate once on a patient-separated external cohort with compatible nuclear masks. Report patient bootstrap CIs and predeclare the primary endpoint.
4. **Patient-separated histology adaptation.** Compare zero-shot, stain normalization, polarity canonicalization, and limited-shot fine-tuning under grouped folds. Apply all preprocessing identically to every baseline.
5. **Instance evidence.** If the paper claims touching-nucleus separation, train/evaluate against instance IDs using AJI, PQ, object F1 and split/merge errors. Current binary foreground results cannot support that claim.
6. **Compute reporting.** Add per-model training time, inference throughput and GPU memory from the retained summaries if the final paper emphasizes efficiency. Parameter counts alone are insufficient.
7. **Independent seed replication.** Five seeds would narrow optimization uncertainty, but unseen acquisition-level validation is more important than adding seeds to the same chemical screen.

## Package integrity

The Overleaf ZIP contains `main.tex`, `references.bib`, official style files, actual-data figures, README and checklist. The code ZIP excludes raw data, predictions and 14 GB of checkpoints; it includes source, tests, locked configs, scripts, per-image CSVs, histories, manifests and manuscript/audit sources. The separate dataset ZIP contains provider archives with attribution, licenses/source links and checksums. Checkpoints should be distributed separately through a versioned release because individual B7 files exceed normal GitHub file limits.
