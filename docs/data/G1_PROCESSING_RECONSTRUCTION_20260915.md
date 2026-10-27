# G1 data processing reconstruction (2026-09-15)

This record summarizes the server-side reconstruction of the initial G1 processing path. It describes metadata and eligibility accounting; raw videos, Parquet files, labels, and source manifests are not copied here.

| Processing stage | Episodes | Meaning |
| --- | ---: | --- |
| Raw inventory | 4,064 | Readable episode records with metadata/source pairing |
| 4-second eligibility | 4,060 | Four readable short episodes were excluded before split assignment |
| Duplicate-safe primary population | 3,976 | Conflicting duplicate groups and low-sample/inactive records are excluded |
| Primary train | 2,757 | Training split |
| Primary development/validation | 617 | Development split |
| Primary final/test | 602 | Final population; later access is disclosed in the Core A V2 analysis |

The processing reconstruction keeps the 4 short-episode exclusions separate from the 84-row reduction between the eligible and primary populations. The G1 decoder/clip path and split/label metadata remain bound to the public Core A contract. The later Core A V2 result uses a corrected class mapping and must be read with its access disclosure.

Reproducibility is limited by server-only raw media, Parquet data, feature caches, and model weights. Their artifact IDs and retrieval boundary are recorded in the release artifact registry; the registry is not a backup of those files.
