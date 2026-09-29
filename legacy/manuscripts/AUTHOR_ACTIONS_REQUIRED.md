# Information and decisions required from the authors

These items cannot be reconstructed reliably from the files. Fill them before the manuscript is submitted.

## Required study metadata

- Cell-line supplier/source, catalogue or donor information, authentication method/date, and mycoplasma-testing method/date.
- Full MCF10A culture and 3D matrix protocol: medium, supplements, matrix/vendor/lot if relevant, seeding density, culture duration, stiffness and treatment conditions represented in the annotated set.
- Nuclear stain/fluorophore, concentration, incubation, fixation/live-cell status, and wash protocol.
- Exact Zeiss LSM model, acquisition software/version, objective, numerical aperture, excitation wavelength, emission window/filter, pinhole, detector type, bit depth, pixel size, z-step, and whether laser power/gain/offset were fixed or optimized per stack.
- Annotation software/version, display settings, written boundary rule, whether touching nuclei were separated, annotator training, expert-review procedure, corrections/adjudication, and approximate annotation dates.
- Confirmation that the work used established cell lines only and did not require human-subject/animal approval; provide the institutional wording if an exemption or approval applies.

## Required provenance decisions

- Confirm whether a laboratory index exists that maps all 855 exported slices to their original LSM stacks. If it exists, provide it; it can replace the current conservative shape grouping.
- Confirm whether 0/1 and 0/255 encodings correspond to known annotation batches or tools. Without records, the manuscript will describe only an encoding association and will not infer annotator/session effects.
- Decide whether to run a blinded second-annotator study. Recommended minimum: at least 30 pre-specified slices sampled across verified stacks, with pixel Dice/IoU, surface distance, and instance agreement if instance claims are retained.

## Required release decisions

- Public repository URL.
- Permanent archive DOI (for example, Zenodo).
- License for code, annotations, images, and model weights.
- Confirmation that all 855 underlying images and masks may be released. If not, document the precise legal/ethical restriction and choose a journal whose data policy permits it.
- Corresponding author and contact email; ORCID IDs; final affiliations; funding statement; author-contribution approval; competing-interest confirmation.

## Journal decision

Recommended first choice: PLOS ONE **only with full underlying-data release**. Its scope is compatible with a rigorous negative/evaluation result. If full release is impossible, select a different journal only after checking its current data policy; do not submit a sample-only availability statement to PLOS ONE.

## Optional experiments that would materially strengthen the paper

- External validation on a separately acquired, manually annotated cell line/instrument cohort.
- Randomly initialized controls if the paper wishes to claim a benefit from ImageNet pretraining.
- Instance-level evaluation (AJI/PQ, object F1, split/merge errors) if claims about individual nuclei or morphometry are retained.
- Prospective timing measurements for annotation and inference if efficiency claims are retained.

