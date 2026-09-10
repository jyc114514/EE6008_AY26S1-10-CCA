# Core A reproduction guide

## What this clone can reproduce

The public clone can reproduce the focused CPU unit/contract tests and rebuild
the six figures from the curated aggregate/plot data. It cannot retrain or
replay the reported models without the separately authorized G1 data, external
V-JEPA source, checkpoint, and feature caches.

## Environment

The checked project environment used Python 3.12.14, PyTorch 2.12.1+cu130,
NumPy 2.5.2, pandas 3.0.5, PyYAML 6.0.3, matplotlib 3.11.1, and Pillow 12.3.0.
Use a project-local environment; do not install into system Python.

```bash
conda env create -f environment.yml
conda activate ee6008-cca
export EE6008_PROJECT_ROOT=/path/to/ee6008-project
export PYTHONPATH="$PWD/src:$PWD/scripts/core_a${PYTHONPATH:+:$PYTHONPATH}"
```

`EE6008_PROJECT_ROOT` is only a placeholder for an authorized external runtime
layout. No data or weights are fetched by these instructions.

## CPU checks

```bash
python -m compileall -q src scripts/core_a tests/core_a
pytest -q tests/core_a
```

The test suite uses synthetic tensors and metadata fixtures. It does not open
the locked final-test population, allocate a GPU, start training, or download
weights.

## Figure regeneration

Run from the repository root:

```bash
python scripts/core_a/make_core_a_figures.py
python scripts/core_a/qa_public_figures.py
```

The script reads only `results/core_a/*.csv`, writes six PNG/SVG/PDF files and
updates the local figure manifest. The QA checks non-empty outputs, dimensions,
plot-data links, and script links. Visual review remains a human responsibility.

## External evidence replay

The `run_*`, `audit_*`, `verify_*`, and `build_*` entrypoints preserve the
project's server-side contracts, but they require external paths and, for many
commands, raw data or feature/checkpoint artifacts. Before any such use, obtain
separate authorization, configure an isolated output root, verify the exact
dataset/split/checkpoint hashes, and keep final-test access disabled. This
release task did not run those entrypoints.
