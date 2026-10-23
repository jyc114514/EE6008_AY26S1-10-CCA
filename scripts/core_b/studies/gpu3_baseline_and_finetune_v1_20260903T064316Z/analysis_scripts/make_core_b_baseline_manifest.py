"""Create the fixed GPU3 Core-B B1 baseline queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LANES = (
    "B-STATE",
    "B-D-P-F",
    "B-S-P-F",
    "B-D-A-F",
    "B-S-A-F",
    "B-D-FUS-P-F",
    "B-S-FUS-P-F",
)
SEED = 20260903
GPU = 3
ROLE = "core_b"
PROTOCOL = "core_b_gpu3_native_v1_1p0s_batch8_accum4_b1"


def command_hash(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--core-b-root", type=Path, required=True)
    parser.add_argument("--feasibility-root", type=Path, required=True)
    parser.add_argument("--fix-root", type=Path, required=True)
    parser.add_argument("--coordinator-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--exclude-lane", action="append", default=[])
    args = parser.parse_args()
    project = args.project_root.resolve()
    repo = args.repo_root.resolve()
    root = args.core_b_root.resolve()
    feasibility = args.feasibility_root.resolve()
    fix = args.fix_root.resolve()
    coordinator = args.coordinator_root.resolve()
    target = args.output_manifest.resolve()
    if target.exists():
        raise FileExistsError(f"refusing existing manifest: {target}")
    input_root = (
        args.input_root.resolve()
        if args.input_root is not None
        else root / "inputs_1p0s_20260903T064316Z"
    )
    runs_root = (
        args.runs_root.resolve() if args.runs_root is not None else root / "runs"
    )
    log_root = (
        args.log_root.resolve() if args.log_root is not None else root / "logs"
    )
    excluded = set(args.exclude_lane)
    unknown_excluded = excluded - set(LANES)
    if unknown_excluded:
        raise ValueError(f"cannot exclude unknown lanes: {sorted(unknown_excluded)}")
    runner = root / "analysis_scripts/run_core_b_baseline.py"
    python = project / "envs/jyc/bin/python"
    source_root = project / "repo/third_party/vjepa2"
    jobs = []
    selected_lanes = [lane for lane in LANES if lane not in excluded]
    for lane in selected_lanes:
        slug = lane.lower().replace("-", "_")
        run_root = runs_root / f"{slug}_seed{SEED}_gpu3"
        command = [
            str(python),
            str(runner),
            "--root",
            str(input_root),
            "--project-root",
            str(project),
            "--feasibility-root",
            str(feasibility),
            "--fix-root",
            str(fix),
            "--source-root",
            str(source_root),
            "--run-root",
            str(run_root),
            "--lane",
            lane,
            "--physical-gpu",
            str(GPU),
            "--seed",
            str(SEED),
        ]
        attentive = lane in {"B-D-A-F", "B-S-A-F"}
        jobs.append(
            {
                "job_id": f"core_b_b1_{slug}_gpu3_{SEED}",
                "role": ROLE,
                "physical_gpu": GPU,
                "protocol_id": PROTOCOL,
                "lane": lane,
                "output_root": str(run_root),
                "command": command,
                "command_hash": command_hash(command),
                "status_path": str(log_root / f"{slug}_watcher_status.json"),
                "log_path": str(log_root / f"{slug}.log"),
                "raw_video_io": False,
                "preload_rss_mib": 52000 if attentive else 0,
                "environment": {
                    "PYTHONPATH": f"{fix}:{repo / 'src'}",
                    "LD_LIBRARY_PATH": f"{project / 'envs/jyc/lib'}:/usr/local/cuda-12.8/lib64",
                    "DUAL_TRACK_ROLE": ROLE,
                },
                "resource_estimate": {
                    "gpu_memory_mib": 23000,
                    "preload_rss_mib": 52000 if attentive else 0,
                    "raw_video_io": False,
                    "no_artificial_memory_filler": True,
                    "mode": "core_b_gpu3_full_data_b1_fresh_child",
                },
            }
        )
    manifest = {
        "status": "locked",
        "role": ROLE,
        "fixed_physical_gpu": GPU,
        "global_max_gpu_children": 2,
        "raw_video_io_capacity": 1,
        "gpu0_or_gpu1_new_jobs_started": False,
        "global_default_allowlist": [2, 3],
        "task_local_gpu_pool": [3],
        "seed": SEED,
        "horizon": "1.0s",
        "target_physical_unit_status": "unresolved",
        "physical_batch": 8,
        "gradient_accumulation": 4,
        "nominal_effective_batch": 32,
        "jobs": jobs,
        "coordinator_root": str(coordinator),
        "final_test_target_accessed": 0,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "created",
                "role": ROLE,
                "physical_gpu": GPU,
                "jobs": len(jobs),
                "excluded_lanes": sorted(excluded),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
