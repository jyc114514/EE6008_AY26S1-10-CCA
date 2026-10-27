"""CPU synthetic adapter check for the locked D2_V3 event helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from evaluate_d3_eq_d2v3_event_secondary_entry_v2 import EVENT_CODE, EventLoaderAdapter, V3_ROOT


EQ_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    spec = importlib.util.spec_from_file_location("locked_d2v3_event_synthetic_test", EVENT_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(EVENT_CODE)
    module = importlib.util.module_from_spec(spec)
    adapter = EventLoaderAdapter(spec.loader)
    adapter.exec_module(module)
    normalizers = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float64)
    threshold_lock = json.loads((EQ_ROOT.parents[2] / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/d2_v3_measurement/D2_V3_PROTOCOL_LOCKED.json").read_text(encoding="utf-8"))
    thresholds = np.asarray([[threshold_lock["thresholds"][body]["theta_on"], threshold_lock["thresholds"][body]["theta_off"]] for body in module.GROUPS], dtype=np.float64)
    action = np.zeros((120, 28), dtype=np.float64)
    timestamps = np.arange(120, dtype=np.float64) / 30.0
    q = module.q_signal(action, sigma)
    moving = np.convolve(q[:, 0], np.ones(8, dtype=np.float64) / 8.0, mode="valid")
    state = module.state_trace(moving, 8, len(action), float(thresholds[0, 0]), float(thresholds[0, 1]))
    forecasts, known_armed = module.predict_events(
        np.zeros((30, 28), dtype=np.float32), action, timestamps, q, 30, state,
        float(thresholds[0, 0]), float(thresholds[0, 1]),
    )
    if not isinstance(known_armed, tuple) or len(known_armed) != 2 or not known_armed[0] or not known_armed[1] or forecasts:
        raise AssertionError("D2_V3 adapter return/state semantics failed")
    result = {
        "status": "PASS",
        "synthetic_state_known": bool(known_armed[0]),
        "synthetic_state_armed": bool(known_armed[1]),
        "zero_motion_forecast_events": len(forecasts),
        "locked_smoothing_window": 8,
        "dev_targets_loaded": False,
        "thresholds_refit": False,
        "final_test_access": 0,
    }
    out = EQ_ROOT / "qa/D3_EQ_D2V3_EVENT_ADAPTER_TESTS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
