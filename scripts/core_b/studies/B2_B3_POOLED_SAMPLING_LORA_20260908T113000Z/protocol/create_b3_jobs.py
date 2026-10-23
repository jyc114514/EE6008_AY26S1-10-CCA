#!/usr/bin/env python3
"""Create task-local job records for the remaining formal dense B3 cells."""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
COORD = REPO / "reports/coordination/CORE_A_CORE_B_POST_OUTAGE_20260907T055509Z"
PYTHON = PROJECT / "envs/jyc/bin/python"
COMMON_PYTHONPATH = ":".join([str(COORD / "code/locked_source/src"), str(COORD / "code/core_a"), str(COORD / "code/core_a/analysis_scripts"), str(REPO / "src")])
GPU_UUIDS = {2: "GPU-00465dc6-e96e-67f1-285a-84ed61565ac2", 3: "GPU-b9c07d60-9a04-0693-af83-6976ba6d4dfb"}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def env(gpu: int, role: str) -> dict[str, str]:
    return {"CUDA_DEVICE_ORDER":"PCI_BUS_ID","CUDA_VISIBLE_DEVICES":str(gpu),"DUAL_TRACK_ROLE":role,"LD_LIBRARY_PATH":str(PROJECT/"envs/jyc/lib")+":/usr/local/cuda-12.8/lib64","PYTHONPATH":COMMON_PYTHONPATH}


def raw_job(condition: str, gpu: int, branch: str, video_input: str) -> dict[str, object]:
    input_name = "B2-V-0125" if branch == "video_only" else "B2-FUS-0125"
    run_root = EXP / "runs" / "corrected" / condition
    command = [str(PYTHON),str(EXP/"protocol/run_b3_raw_lora.py"),"--project-root",str(PROJECT),"--repo-root",str(REPO),"--input-root",str(EXP/"inputs"/input_name),"--manifest",str(REPO/"reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_FULL_DATA_MANIFEST.csv"),"--fix-root",str(COORD/"code/core_a"),"--source-experiment-root",str(REPO/"reports/benchmarks/CORE_A_FULL120_LORA_FACTORIAL_V1_20260902T061327Z"),"--input-normalization",str(REPO/"reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_INPUT_NORMALIZATION_TRAIN_ONLY.json"),"--run-root",str(run_root),"--condition-id",condition,"--branch",branch,"--video-input",video_input,"--physical-gpu",str(gpu),"--seed","20260903"]
    return {"job_id":f"core_b_{condition.lower().replace('-', '_')}_corrected_20260909_gpu{gpu}","physical_gpu":gpu,"gpu_uuid":GPU_UUIDS[gpu],"protocol_id":"core_b_b3_raw_lora_20260908_batch8_accum4_corrected_feature_norm","output_root":str(run_root),"log_path":str(EXP/"logs"/f"{condition}_corrected_gpu{gpu}.log"),"command":command,"environment":env(gpu,"core_b_b3_raw_lora"),"resource_estimate":{"physical_batch":8,"gradient_accumulation":4,"nominal_effective_batch":32,"precision":"BF16","raw_video_io":True,"host_ram_reservation_mib":52000,"gpu_memory_mib":22000,"feature_normalization":"train_only"}}


def frozen_job(condition: str, gpu: int, branch: str) -> dict[str, object]:
    input_name = "B2-FUS-0125" if branch == "fusion" else "B2-V-0125"
    lane = "B-S-FUS-P-F" if branch == "fusion" else "B-S-P-F"
    run_root = EXP / "runs" / condition
    command = [str(PYTHON),str(EXP/"protocol/run_b2_pooled.py"),"--root",str(EXP/"inputs"/input_name),"--project-root",str(PROJECT),"--feasibility-root",str(REPO/"reports/core_b/feasibility_and_smoke_20260901T160933Z"),"--fix-root",str(COORD/"code/core_a"),"--source-root",str(REPO),"--run-root",str(run_root),"--lane",lane,"--condition-id",condition,"--protocol-id","core_b_b3_pooled_dense_20260909_batch8_accum4","--physical-gpu",str(gpu),"--seed","20260903"]
    return {"job_id":f"core_b_{condition.lower().replace('-', '_')}_20260909_gpu{gpu}","physical_gpu":gpu,"gpu_uuid":GPU_UUIDS[gpu],"protocol_id":"core_b_b3_pooled_dense_20260909_batch8_accum4","output_root":str(run_root),"log_path":str(EXP/"logs"/f"{condition}_gpu{gpu}.log"),"command":command,"environment":env(gpu,"core_b_b3_frozen"),"resource_estimate":{"physical_batch":8,"gradient_accumulation":4,"nominal_effective_batch":32,"precision":"BF16","raw_video_io":False,"host_ram_reservation_mib":4000,"gpu_memory_mib":4000}}


def main() -> int:
    jobs = {
        "B3-V-D-L_corrected_gpu3.json": raw_job("B3-V-D-L",3,"video_only","dynamic"),
        "B3-V-S-L_corrected_gpu3.json": raw_job("B3-V-S-L",3,"video_only","static"),
        "B3-FUS-D-L_corrected_gpu2.json": raw_job("B3-FUS-D-L",2,"fusion","dynamic"),
        "B3-FUS-S-L_corrected_gpu3.json": raw_job("B3-FUS-S-L",3,"fusion","static"),
        "B3-FUS-S-F_gpu2.json": frozen_job("B3-FUS-S-F",2,"fusion"),
    }
    for name, value in jobs.items():
        path = EXP / "jobs" / name
        if not path.exists(): atomic_json(path, value)
    atomic_json(EXP / "B3_REUSE_MAP.json", {"status":"locked","strategy":"dense_0.125s_32_pixel_frames","reuse":{"B3-V-D-F":"B2-V-0125","B3-FUS-D-F":"B2-FUS-0125","B3-V-S-F":"existing_runs/B3-V-S-F"},"new":{"B3-V-D-L":"runs/corrected/B3-V-D-L","B3-V-S-L":"runs/corrected/B3-V-S-L","B3-FUS-D-L":"runs/corrected/B3-FUS-D-L","B3-FUS-S-L":"runs/corrected/B3-FUS-S-L","B3-FUS-S-F":"pooled_static_frozen_runner"},"feature_normalization":"train_only dynamic/static key applied after raw encoder pooling","final_test_access":0})
    print(json.dumps({"status":"pass","created_or_existing":len(jobs)},sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
