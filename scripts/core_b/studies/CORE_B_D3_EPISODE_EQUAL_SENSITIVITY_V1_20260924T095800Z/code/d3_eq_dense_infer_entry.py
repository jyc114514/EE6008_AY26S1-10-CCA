"""Resolve shared V3 data root and launch the D3-EQ dense inference worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def load_dense_module():
    script = Path(__file__).with_name("d3_eq_dense_infer.py")
    spec = importlib.util.spec_from_file_location("d3_eq_dense_infer_module", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load dense inference worker: {script.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Inject the shared V3 root on the actual module object so the worker
    # function resolves it through its own __globals__ dictionary.
    module.V3_ROOT = V3_ROOT
    return module


def main() -> int:
    module = load_dense_module()
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
