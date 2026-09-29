# Dataset and literature sources

## Public data

BBBC039v1: 200 U2OS fluorescence fields, one per compound, 520 x 696 native 16-bit images, from one chemical-screen acquisition study. Official partition: 100 train, 50 validation, 50 test. Provider: https://bbbc.broadinstitute.org/BBBC039 . License CC0. Nuclear semantic foreground is the decoded annotation's red channel > 0, consistent with the author's decoder at https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075 . This is not a breast-cell dataset and not a multi-site validation set.

BBBC038 / 2018 Data Science Bowl: complete 65-image public stage-1 test set with released RLE ground truth, used for frozen private-checkpoint transfer. Provider: https://bbbc.broadinstitute.org/BBBC038 . License CC0. BBBC039's provider explicitly notes partial overlap with BBBC038. Do not pool the two as independent evidence or claim clean BBBC038 generalization from a BBBC039-trained model without resolving overlap.

TNBC nuclei segmentation v1.1: 50 H&E images from 11 patient groups, released binary nuclear masks. Dataset DOI https://doi.org/10.5281/zenodo.2579118 ; provider https://zenodo.org/records/2579118 ; license CC BY 4.0. Cite Naylor et al., Segmentation of Nuclei in Histopathology Images by Deep Regression of the Distance Map, IEEE Transactions on Medical Imaging (2019), https://doi.org/10.1109/TMI.2018.2865709 . Patient prefixes are retained for grouping. Binary masks do not establish the count of touching instances.

Prior frozen evaluations use all 65 + 50 + 50 specified images. New training uses only the official BBBC039 training split; validation is for checkpoint selection. TNBC is a complete untuned cross-domain transfer evaluation, not a held-out test from a TNBC-trained model. All image-level confidence intervals on BBBC datasets are descriptive; acquisition-level independence is not established.

## Relevant method references

Ronneberger et al., U-Net, MICCAI 2015: https://arxiv.org/abs/1505.04597 .

Zhou et al., UNet++, DLMIA 2018: https://arxiv.org/abs/1807.10165 .

Tan and Le, EfficientNet, ICML 2019: https://proceedings.mlr.press/v97/tan19a.html .

Caicedo et al., Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl, Nature Methods 2019: https://doi.org/10.1038/s41592-019-0612-7 .

Cellpose-SAM, 2025 preprint: https://doi.org/10.1101/2025.04.28.651001 . The evaluated checkpoint is cpsam_v2, Cellpose package 4.2.1.1. Pretraining overlap is unresolved; do not label this comparison equal-data or provably zero-exposure.

Micro-SAM: https://doi.org/10.1038/s41592-024-02580-4 . CellSAM: https://doi.org/10.1038/s41592-025-02879-w . These are relevant candidate comparisons, not methods newly benchmarked in this package.

## Task compatibility

Cryo-EM particle coordinates or picking boxes are not nuclear boundary masks. A cryo-EM picking dataset cannot establish contour accuracy without suitable segmentation ground truth. Likewise LIVECell whole-cell targets differ from nuclei. Prioritize matched imaging and annotation tasks rather than accumulating unrelated test datasets.
