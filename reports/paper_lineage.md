# How the new paper retains and extends the existing work

The submitted Scientific Reports and IEEE manuscripts remain the starting point. This is a revision/extension of that 2D project, not an unrelated replacement. Both original code exports and the intervening Scientific Reports revision are preserved. The old manuscripts are rejected/withdrawn according to the author.

| Existing paper component | Treatment in the new draft |
|---|---|
| Motivation: limited annotation for mammary epithelial nuclear segmentation | Retained, with a precise distinction between nuclear foreground, whole cells and cancer diagnosis. |
| 24-filter multiscale Gabor fourth channel | Retained and mathematically specified. Signed-sum normalization is replaced by L1 normalization; the change is disclosed. |
| EfficientNet-B7, nested UNet++ decoder, SCSE | Retained as the original model family. Vanilla B7 and SCSE-only controls show whether additions are justified. |
| Learnable 4-to-3 projection | Original projection retained as a comparison; an identity-preserving residual variant is a clearly labelled new experiment. |
| Adaptive/stabilized loss and complexity-weighted sampling | Original rationale, implementation discrepancies and sampling formula documented. Their existing private-data ablations are preserved. New matched public studies deliberately use one common Dice+BCE objective and uniform sampling; this is a protocol change, not a claim that the old exact recipe was retrained. |
| Private breast-cell dataset and submitted result tables | Retained as historical context, not discarded or mixed into public scores. Corrected private results replace unsupported superiority statements. Missing source provenance is explicit. |
| Earlier corrected ablations and encoder/edge sweeps | Preserved in the inherited clustered summary, including 34 experimental conditions; shown separately from new public-data experiments. They do not establish fully verified volume independence. |
| Existing public zero-shot experiments | Retained in full: BBBC038, BBBC039 and TNBC, with all stated test images and failed transfers included. |
| Original figures | Real-data material can be retained after provenance checking; quantum-labelled and AI-generated experimental imagery is replaced by actual filter responses and predictions. |
| 2D-to-3D progression | A short exploratory section retains the eight-stack selected-plane results. No new claim of beating Cellpose or dense 3D validation. |
| Strong novelty/clinical/generalization claims | Rewritten to match measured evidence. Reproducing the old claim verbatim would not resolve the reviewers' concerns. |

Original authors carried forward in the draft: Varun Kumar Dasoju, Qingsu Cheng, and Zeyun Yu, University of Wisconsin–Milwaukee. Author order, affiliations, contributions and approval of this revised content must be confirmed before submission.

The code package contains more detail than can fit into a four-page ISBI paper. The main paper will retain the original method's identity, the most relevant controlled comparisons and the breast transfer result; the full history, historical baselines and extended ablations belong in the accompanying reproducibility/report package. Repository material is not assumed to be reviewed supplementary material unless the conference explicitly permits it.
