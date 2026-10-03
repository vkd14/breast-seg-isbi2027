# Public dataset archive for the ISBI 2027 study

This archive collects the exact provider downloads used locally. It is a convenience copy, not a new dataset or a relicensing claim.

## BBBC039v1

- Provider: https://bbbc.broadinstitute.org/BBBC039
- Contents: `images.zip`, `masks.zip`, `metadata.zip`
- License: CC0 as stated by the Broad Bioimage Benchmark Collection.
- Use: official 100/50/50 training/validation/test split for the 2D public study.
- Important: U2OS Hoechst fluorescence, not breast tissue; partly overlaps BBBC038.

## BBBC038 / 2018 Data Science Bowl

- Provider: https://bbbc.broadinstitute.org/BBBC038
- Contents: `stage1_test.zip`, `stage1_solution.csv`, `metadata.xlsx`
- License: CC0 as stated by the provider.
- Use: frozen historical-checkpoint evaluation and grayscale-equivalence audit only. It is not independent of BBBC039.

## TNBC nuclei segmentation v1.1

- Provider: https://zenodo.org/records/2579118
- DOI: https://doi.org/10.5281/zenodo.2579118
- Contents: `TNBC_NucleiSegmentation.zip`
- License: CC BY 4.0. Preserve provider attribution.
- Citation: P. Naylor et al., “Segmentation of Nuclei in Histopathology Images by Deep Regression of the Distance Map,” IEEE TMI 38(2), 448–459, 2019. DOI: 10.1109/TMI.2018.2865709.
- Use: all 50 images from 11 patient groups as a post-hoc zero-shot transfer stress test.

Files are stored under `data/<dataset>/`. Verify every file against `CHECKSUMS.sha256` from the archive root before use. Public licensing does not determine whether an institution requires an ethics/non-human-subjects determination for human histology reuse.
