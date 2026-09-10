# Core A provenance and checksum policy

## Source anchors

The release is anchored to the following non-public evidence identities:

- project source commit recorded by the experiment protocol:
  `204698b45b3712590f06245fbfba32d3be539812`;
- G1 dataset revision:
  `6b6b599def75e3558b2736062e79bdbeec6cd224`;
- duplicate-safe v3 split manifest:
  `2883debbbe542bd4df0478ec7e896ed1468e647f93cb609041dd6411e7856997`;
- v3 label map:
  `057a66c50e4fd85198e4a59359e4fc31c79565f0a379f0be368a389bb4811c22`;
- dynamic feature manifest:
  `59d54576a19b45edbc7627c181005d1c797cca07d0460e24c2a7abad5020d785`;
- static feature manifest:
  `1d1c1ed242f8f71cdf233f92c457b98134271732c1bb26b321ae55f5d8a4139c`;
- external V-JEPA checkpoint:
  `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`.

These hashes identify source inputs; the data, features, and checkpoint are not
present in the public tree.

## Cell-level lineage

For each Full-120 Stage1 cell, `experiments/core_a/ACCEPTANCE_SUMMARY.csv`
records the cell, adaptation/head/input, seed, source status, prediction hash,
checkpoint artifact ID, config artifact ID, acceptance status, final-test guard,
and canonical inference mode. The corresponding aggregate values are in
`experiments/core_a/RUN_LEVEL_RESULTS.csv`; effects are in
`experiments/core_a/COMPARISON_EFFECTS.csv`.

The D-P-L row intentionally records `accepted_derived_unmerged`: the source
run's merged BF16 diagnostic failed and its source `RUN_METRICS.json` was
missing. The reported aggregate is the independent logits recomputation from
the strict-load unmerged base-plus-adapter path.

## Hash layers

- Source/archive SHA-256 relationships were checked against the archive copy
  index and unified-analysis source manifest.
- Public file SHA-256 values are in `PUBLICATION_MANIFEST.csv` and
  `SHA256SUMS` at the publication root.
- A change to any public table or figure requires regenerating both public hash
  files and rerunning the public QA.

No server absolute path is required to interpret the public tables. A
`server-only://` value is a non-downloadable artifact ID, not a URL containing
credentials or a hidden filesystem path.
