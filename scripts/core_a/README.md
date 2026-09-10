# Core A scripts

The scripts are copied from project-authored source and kept under a dedicated
namespace so they do not imply that the public repository contains the private
runtime inputs. CPU-safe utility and unit-test paths are reproducible from this
clone. Training, feature extraction, checkpoint replay, and any command that
requires raw data or server-only artifacts require an explicitly configured
external environment and are not run by the publication workflow.

The figure path is directly reproducible:

```bash
python scripts/core_a/make_core_a_figures.py
python scripts/core_a/qa_public_figures.py
```

Run from the repository root. The canonical scientific result is the sealed
aggregate table; regenerating plots does not re-run an experiment.
