# Venue and dataset decision for September 29, 2026

Prepared September 28, 2026. The authors confirmed that both previous submissions were rejected or withdrawn. Recommendation: target **ISBI 2027, October 26, 2026**, if the evidence and reproducibility work below can be completed. The conference deadline is approximately four weeks away; acceptance and the conference itself occur later.

## Conference shortlist

| Priority | Venue | Official submission deadline | Fit and recommendation |
| --- | --- | --- | --- |
| 1 | IEEE ISBI 2027 | October 26, 2026 | Best match to the requested one-month window and biological image analysis. Focus on a compact, reproducible 2D study with external boundary evaluation. |
| 2 | MIDL 2027 | Abstract November 30; full paper December 4, 2026, 23:59 AoE | Strong fit for deep learning and rigorous external validation; allows more time to complete adaptation, ablations, and source provenance. |
| 3 | IPMI 2027 | Abstract December 7; full article December 14, 2026 | Strong imaging-methods audience, but a substantially stronger methodological contribution would be needed for this project. |
| 4 | MICCAI 2027 | Not yet announced on the MICCAI Society page checked today | Strong future target for a mature method; not a verified one-month submission option. |

Official sources: [ISBI instructions](https://biomedicalimaging.org/2027/papers/), [MIDL dates](https://2027.midl.io/important-dates), [MIDL scope](https://2027.midl.io/call-for-papers), [IPMI dates](https://2027.ipmi-conf.org/), [MICCAI Society](https://miccai.org/upcoming-conferences/).

ISBI limits technical material to four pages. A paid fifth page can contain references, acknowledgments, and ethical compliance information only. Its one-page abstract track is not included in the proceedings. The current PDF is a readable meeting draft, not the final ISBI template. Recheck the deadline time zone and submission portal when it opens; the checked author page states the date but not a time zone.

MIDL is the best alternative if the October deadline would force an incomplete experiment. A validation-focused framing is more credible than claiming that the present architectural additions outperform all alternatives. The clinical-translation track should only be chosen if the finished study satisfies its biomedical/operational validation requirements; cultured-cell experiments are not clinical validation. Avoid overlapping submissions while a manuscript is under review.

## Journals

The three venues mentioned in the request are journals, not conferences. Standard research-article submission is not tied to the conference dates above. Their fit is more useful here than a prestige ranking:

1. **Biomedical Signal Processing and Control**: strongest match among the three for image preprocessing, Gabor features, optimization, and boundary evaluation. It still needs a measurable methodological contribution and adequate validation. [Publisher scope](https://shop.elsevier.com/journals/biomedical-signal-processing-and-control/1746-8094).
2. **Computers in Biology and Medicine**: good target if the paper establishes biological utility, robust dataset partitions, and meaningful downstream morphometry. Its publisher explicitly flags unclear splits and imprecise segmentation-task analysis as unsuitable. [Publisher scope](https://shop.elsevier.com/journals/computers-in-biology-and-medicine/0010-4825).
3. **Expert Systems with Applications**: consider after demonstrating a broadly useful AI method across several domains; a breast-only architecture combination with no consistent gain is a weaker fit. [Publisher scope](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174).

These are judgments about scope and readiness, not acceptance forecasts. The existing evidence does not support a state-of-the-art or clinical-deployment claim.

## Dataset choices

| Dataset | What is labeled | Use in this package | Role in the paper |
| --- | --- | --- | --- |
| BBBC038v1 / DSB2018 | Individual nuclear masks across varied light microscopy | All 65 official stage-1 test images, original released RLE annotations | Main appearance-shift stress test; nucleus-mask union scored at native resolution |
| BBBC039v1 | Hoechst-stained U2OS nuclei, 16-bit fluorescence | All 50 official test images; 100 train and 50 validation images remain unused here | Closest tested image/label match to fluorescence nucleus segmentation; best next controlled adaptation benchmark |
| TNBC v1.1 | Annotated nuclei in breast histology; 50 images from 11 patients | All 50 images; patient-level summaries and paired tests | Breast-specific, larger modality shift; current failure is useful evidence, not a successful transfer claim |
| LIVECell | Whole-cell instances in phase-contrast imaging, including MCF7, BT474, SkBr3 | Researched, not downloaded or evaluated | Suitable only if the target is expanded from nuclei to whole cells; well/location/time identifiers support grouped splits |
| CryoPPP | Protein-particle coordinates in cryo-EM micrographs | Researched, not evaluated | A separate particle-detection task; coordinates do not establish accurate object contours |
| CryoNuSeg | Nuclear masks in cryosectioned H&E histology | Researched, not evaluated | Optional future histology test; cryosectioned histology is not cryo-EM |

Dataset sources and terms: [BBBC038, CC0](https://bbbc.broadinstitute.org/BBBC038), [BBBC039, CC0](https://bbbc.broadinstitute.org/BBBC039), [TNBC v1.1, CC BY 4.0](https://doi.org/10.5281/zenodo.2579118), [LIVECell, CC BY-NC 4.0](https://sartorius-research.github.io/LIVECell/), [CryoPPP authors' repository](https://github.com/BioinfoMachineLearning/cryoppp), [CryoNuSeg original paper](https://arxiv.org/abs/2101.00442).

BBBC039's authors explicitly report overlap with BBBC038. These two benchmark results must remain separate; they cannot be pooled as independent cohorts. Before any cross-dataset training, audit overlap at source/patch level, including resized and contrast-transformed duplicates. Hash matching alone cannot exclude these. No public benchmark data were used for training or parameter selection in the current runs. Public-pretraining exposure of Cellpose-SAM has not been audited here, so its results are an off-the-shelf reference rather than proof of unseen-domain generalization under matched data access.

For cryo-EM, use coordinate-matching precision/recall and localization error with a stated tolerance, and split by EMPIAR experiment/protein. Synthetic disks around coordinates are weak labels, not independently annotated segmentation truth. This direction would distract from a four-page nucleus-segmentation submission and is deferred.

## Four-week work plan

| Dates | Work | Deliverable and decision criterion |
| --- | --- | --- |
| September 29-October 2 | Agree the scientific claim; recover acquisition IDs; document annotation protocol; lock public train/validation/test records | Source-level manifest and an honest claim that all authors endorse |
| October 3-9 | Compare the revised model with vanilla B7 UNet++, smaller U-Net, and Cellpose-SAM; use BBBC039 official train/validation only for adaptation, and patient-grouped development for TNBC | Three seeds, equal learned-model budgets, learning curves, frozen validation selection; clearly mark exploratory use of benchmarks already inspected |
| October 10-15 | Isolate normalization, edge channel, polarity handling, and optional boundary/distance supervision; evaluate all planned ablations | Improvement must survive boundary metrics and appropriate paired source-level uncertainty; keep negative results |
| October 16-19 | Evaluate locked choices on an additional previously uninspected cohort; obtain repeat annotation of a predefined private subset | Independent confirmation plus inter-annotator Dice/surface distances; no invented annotation ceiling |
| October 20-23 | Assemble four-page ISBI paper and public release | Actual figures, accessible weights, exact environment, raw result tables, checked claims and references |
| October 24-26 | Professor/coauthor review, template and portal checks, final submission | All authors approve; switch to MIDL if core evidence or release terms remain unresolved |

The highest-priority methodological experiment is **controlled domain adaptation and intensity normalization**, because the new measurements show a large imaging-modality dependence. A distance or boundary auxiliary head is the next candidate, but its value must be isolated against plain Dice+BCE and the same encoder. Increasing network size is not supported by the current ablation evidence.

Decisions for the professor: approve the replacement-paper framing; choose ISBI versus the longer MIDL schedule; confirm stain/acquisition/annotation details; nominate a second annotator; approve private-data/weight release terms. Keep 3D to one short subsection or supplementary figure, describing selected-plane foreground evaluation rather than dense 3D instance accuracy.
