# Core A publication manifest

This branch is a curated public evidence export, not a dump of the server
workspace. `PUBLICATION_MANIFEST.csv` and `.json` enumerate the files actually
included in the commit with their public role, source family, and SHA-256.

## Release identity

- Repository: `jyc114514/EE6008_AY26S1-10-CCA`
- Base branch observed before editing: `main`
- Base commit observed before editing: `3146d09b360066d08cc25cad68dbf57a5180aa17`
- Publication branch: `codex/core-a-evidence-20260910T082843Z`
- Scope: Core A only; Core B result artifacts are excluded.
- Final-test access: `0`.
- New training started by this publication: `false`.
- No blanket open-source license is asserted by this branch; license decisions
  for project, data, model, and third-party materials remain separate.

## Mapping

| Public area | Included content | Source/selection rule |
| --- | --- | --- |
| `src/ee6008/` | Project-authored Core A package, including LoRA and top-block helpers | Current source package; archive copies were SHA-256 matched where available. |
| `scripts/core_a/` | Portable runners, split/data/metric audits, pilot/Batch utilities, figure generation and public QA | Current project scripts or the prior sanitized export when hashes matched; server launch wrappers and backups excluded. |
| `configs/core_a/` | v3, frozen attentive, Batch48 screening, Top-2 LoRA pilot, and locked Full-120 summaries | Paths are placeholders or environment-variable based; no server absolute path. |
| `data_metadata/core_a/` | Task labels, selection rule, sanitized split summary, train-only class-weight record | Metadata only; no episode-level final-test files. |
| `experiments/core_a/` | Catalog, nine-cell run table, effects, acceptance summary | Aggregate rows and artifact IDs; no prediction arrays/checkpoints. |
| `results/core_a/` | Aggregate results, plot data, six figures, figure manifest and pilot summary | Rebuilt from public plot inputs; no per-sample outputs. |
| `docs/core_a/` | Scope, protocol, conclusions, limitations, reproduction, and engineering boundary | Path-redacted, bilingual where supplied. |

The detailed file-level manifest covers the payload tree and intentionally
excludes the self-referential `PUBLICATION_MANIFEST.*`, `SOURCE_SELECTION.*`,
and root `SHA256SUMS` files. It is generated immediately before staging and
must be regenerated if the worktree changes after that point.
