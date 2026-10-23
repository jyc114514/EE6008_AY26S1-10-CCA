"""Entry wrapper adapting the locked D2_V3 prediction-event helper return tuple."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
EVENT_CODE = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement/evaluate_d2_v3_panel.py"


class EventLoaderAdapter:
    """Expose forecasts plus a compact state tuple without changing D2_V3 code."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: Any) -> None:
        self.loader.exec_module(module)
        original = module.predict_events

        def adapted(*args: Any, **kwargs: Any):
            known, armed, forecasts = original(*args, **kwargs)
            return forecasts, (known, armed)

        module.predict_events = adapted

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)


def main() -> int:
    original_factory = importlib.util.spec_from_file_location

    def wrapped_factory(name, location, *args, **kwargs):
        spec = original_factory(name, location, *args, **kwargs)
        if spec is not None and spec.loader is not None and Path(location).resolve() == EVENT_CODE.resolve():
            spec.loader = EventLoaderAdapter(spec.loader)
        return spec

    importlib.util.spec_from_file_location = wrapped_factory
    import evaluate_d3_eq_d2v3_event_secondary as evaluator
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
