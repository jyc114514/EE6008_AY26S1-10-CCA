from __future__ import annotations
import os

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'envs/jyc/bin/python'))
WATCHER = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'))
SEED = 20260924
PROTOCOL = "core_b_postfreeze_gated_residual_factorial_v1_20260913"

# Four matched blocks; each physical GPU receives two STATE/two FUSION,
# two shared/two split and two gated/two ungated cells across the schedule.
SCHEDULE = [
    {"wave": 1, "physical_gpu": 2, "cell_id": "E-S-SH-U"},
    {"wave": 1, "physical_gpu": 3, "cell_id": "E-S-SH-G"},
    {"wave": 2, "physical_gpu": 2, "cell_id": "E-S-SP-G"},
    {"wave": 2, "physical_gpu": 3, "cell_id": "E-S-SP-U"},
    {"wave": 3, "physical_gpu": 2, "cell_id": "E-F-SH-G"},
    {"wave": 3, "physical_gpu": 3, "cell_id": "E-F-SH-U"},
    {"wave": 4, "physical_gpu": 2, "cell_id": "E-F-SP-U"},
    {"wave": 4, "physical_gpu": 3, "cell_id": "E-F-SP-G"},
]


def main() -> None:
    rows = []
    for item in SCHEDULE:
        cell = item["cell_id"]
        gpu = item["physical_gpu"]
        job_id = f"jy02_postfreeze_{cell.lower().replace('-', '_')}_20260913T061201Z_gpu{gpu}"
        job = {
            "job_id": job_id,
            "physical_gpu": gpu,
            "protocol_id": PROTOCOL,
            "output_root": str(ROOT),
            "command": [str(PYTHON), str(ROOT / "code/train_gated_cell.py"), "--cell", cell, "--root", str(ROOT), "--seed", str(SEED)],
            "log_path": str(ROOT / f"logs/{cell}_child.log"),
            "environment": {"PYTHONPATH": str(ROOT / "code"), "PYTHONDONTWRITEBYTECODE": "1"},
            "resource_estimate": {"kind": "cached_feature_head_training", "max_optimizer_steps": 1720, "max_gpu_children": 1, "raw_video_io": False, "final_test_access": 0},
        }
        path = ROOT / f"logs/{cell}_job.json"
        path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({**item, "job_id": job_id, "job_path": str(path), "command_hash_pending": True})
    (ROOT / "RUN_SCHEDULE.json").write_text(json.dumps({"status": "LOCKED_BEFORE_RESULTS", "seed": SEED, "max_concurrent_children": 2, "actual_watcher_allowlist": [2, 3], "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (ROOT / "RUN_SCHEDULE.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("wave,physical_gpu,cell_id,job_id,job_path,command_hash_pending\n")
        for row in rows:
            handle.write(f"{row['wave']},{row['physical_gpu']},{row['cell_id']},{row['job_id']},{row['job_path']},true\n")
    print(json.dumps({"status": "PASS", "jobs": len(rows), "waves": 4, "gpus": [2, 3]}, sort_keys=True))


if __name__ == "__main__":
    main()
