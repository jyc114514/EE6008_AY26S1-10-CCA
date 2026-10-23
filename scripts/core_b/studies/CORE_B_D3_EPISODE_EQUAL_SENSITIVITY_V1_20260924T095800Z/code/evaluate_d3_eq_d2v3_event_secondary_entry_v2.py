"""D2_V3 event adapter with locked V3 scaler and legacy helper return contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EVENT_CODE = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement/evaluate_d2_v3_panel.py"


class EventLoaderAdapter:
    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: Any) -> None:
        self.loader.exec_module(module)
        normalizers = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
        module.SIGMA = np.asarray(normalizers["target_sigma"], dtype=np.float64)
        original = module.predict_events

        def adapted(*args: Any, **kwargs: Any):
            known, armed, forecasts = original(*args, **kwargs)
            return forecasts, (known, armed)

        module.predict_events = adapted

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def main() -> int:
    import importlib.util as util
    original_factory = util.spec_from_file_location

    def wrapped_factory(name, location, *args, **kwargs):
        spec = original_factory(name, location, *args, **kwargs)
        if spec is not None and spec.loader is not None and Path(location).resolve() == EVENT_CODE.resolve():
            spec.loader = EventLoaderAdapter(spec.loader)
        return spec

    util.spec_from_file_location = wrapped_factory
    import evaluate_d3_eq_d2v3_event_secondary as evaluator
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
