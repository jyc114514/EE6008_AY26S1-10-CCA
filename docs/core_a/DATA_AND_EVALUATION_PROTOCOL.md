# Core A data and evaluation protocol

## Dataset and split identity

- Dataset line: G1 Humanoid Everyday.
- Dataset revision: `6b6b599def75e3558b2736062e79bdbeec6cd224`.
- Eligible four-second episodes: 4,060.
- Duplicate audit: 40 exact duplicate groups, 80 conflicting duplicate
  episodes excluded; the earlier split is retired as duplicate-unsafe.
- Primary population: 120 classes, 2,757 train episodes and 617
  development/validation episodes.
- Final-test metrics: not computed; `final_test_accessed=0`.

The public repository contains task metadata and a sanitized split summary, not
episode-level split files, raw Parquet, video, or final-test identifiers. The
split manifest identity is recorded as SHA-256
`2883debbbe542bd4df0478ec7e896ed1468e647f93cb609041dd6411e7856997`.

## Input conditions

- Dynamic input: the sampled RGB prefix.
- Static input: the same first frame repeated over the temporal window.
- Observation: 4.0 seconds, 8 sampled frames per second, 32 frames.
- V-JEPA 2.1-B/384 contract: tubelet size 2, `ema_encoder`, 384-pixel input.
- Pooled head: pooled representation followed by the project probe.
- Attentive head: depth 4, 16 heads, final-token input.
- LoRA: top blocks 10 and 11, `qkv` and `proj`, rank 8, alpha 16, dropout
  0.05, zero-initialized `lora_B`.

The V-JEPA source tree, checkpoint, and feature caches are external inputs.
The checkpoint is identified by SHA-256
`848a77c33cc9e6649ed2119cbea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`; it is not
distributed here.

## Metrics and selection

Top-1, Top-5, and macro-F1 are primary. NLL, ECE, Brier, and mean class recall
are secondary. Stage1 selects the development/validation epoch using macro-F1,
then lower NLL, then earlier epoch. The 10,000-resample uncertainty analysis
uses paired episodes; it does not treat time steps or trajectory dimensions as
independent observations. Exact two-sided McNemar tests and the registered Holm
correction are retained in the effect table.

Historical baseline rows may have seed-level sample SD, while current Stage1
rows have one seed and episode-bootstrap comparisons. Those uncertainty types
must not be pooled.

## Protocol deviations retained in the release

The locked Full-120 LoRA protocol specified physical batch 48 and accumulation
1. Recovery used physical batch 8 and accumulation 4 (effective batch 32).
This difference is documented and prevents an unconditional claim of exact
locked-protocol replication. D-P-L additionally uses an accepted unmerged
base-plus-adapter derived prediction because the BF16 merged export failed its
equivalence check.
