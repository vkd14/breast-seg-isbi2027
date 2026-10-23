# ISBI manuscript priority and cut order

The four technical pages should preserve the scientific claim, its evidence, and enough implementation detail to reproduce it. Material should be removed in this order:

1. **TNBC low-performance and transfer material — remove completely.** It creates a second, negative cross-modality story and consumes space needed for the stronger BBBC039 instance result.
2. **Exploratory 3-D progress — remove completely.** It is a separate task without dense 3-D instance validation and weakens the focused 2-D contribution.
3. **BBBC038 compatibility result — remove completely.** BBBC038 overlaps BBBC039 and is not independent external evidence.
4. **Private UWM detail — compress to one result sentence.** Retain it only because it documents the project lineage; move microscopy settings, group construction, and full table to the repository report.
5. **Long ablation table — compress to one sentence/range.** Keep the matched vanilla, Gabor, Xception, and contact-aware rows; place all other architecture/filter/sampling results in the public repository.
6. **Non-aligned literature rows — remove.** Keep only BBBC039 papers using the official split and the same AJI/PQ/object-F1 definitions.
7. **Implementation minutiae — shorten.** Keep the exact contact target, loss, postprocessing, split, seed count, and endpoint definitions; move filter sensitivity grids and sampling formulas to code/config files.
8. **General textbook background — remove.** Use the space for the gap, contribution, and evidence.

Do not cut: the official split, train/validation/test counts, leakage checks, ten-seed protocol, validation-only threshold selection, direct instance metrics, uncertainty/statistical testing, matched controls, limitations, ethics, conflict/funding statement, and code availability.
