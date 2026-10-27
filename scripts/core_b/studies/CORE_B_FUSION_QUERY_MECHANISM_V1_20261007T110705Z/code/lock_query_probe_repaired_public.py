# Repaired public copy of EXCL-CODE-004; source SHA-256: ed398c7e6685bb62e55fe75fbfead4d2528e05bd8d873d4b8bb4539d84e37f06
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(os.environ.get("EE6008_FUSION_QUERY_STUDY_ROOT", ".")).expanduser().resolve()
PROJECT = Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
LEARN = PROJECT / "reports/benchmarks/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
SCHED = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"

def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

def source_item(path: Path, role: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"role": role, "path": str(path), "bytes": path.stat().st_size,
            "observed_sha256": sha(path), "read_only": True}

def json_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AssertionError("prelock CPU gate requires CUDA_VISIBLE_DEVICES empty")
    now = utc_now()
    source_reg_path = ROOT / "SOURCE_REGISTRY.json"
    source_accept_path = ROOT / "SOURCE_ACCEPTANCE.json"
    registry = read_json(source_reg_path)
    accept = read_json(source_accept_path)
    audit = read_json(ROOT / "forensics/FUSION_QUERY_PATH_AUDIT.json")
    audit_accept = read_json(ROOT / "forensics/A_AUDIT_ACCEPTANCE.json")
    freeze = read_json(ROOT / "DEVELOPMENT_REFERENCE_FREEZE.json")
    query_meta = read_json(ROOT / "protocol/QUERY_TABLES_AND_HASHES.json")
    tests_log = (ROOT / "logs/CPU_TESTS_ATTEMPT_002.log").read_text(encoding="utf-8")
    if "6 passed" not in tests_log:
        raise AssertionError("six CPU contract tests have not passed")
    if audit_accept.get("status") != "PASS_READ_ONLY_AUDIT":
        raise AssertionError("A audit acceptance missing")
    if audit_accept.get("strict_baseline_replay", {}).get("count") != 4:
        raise AssertionError("four baseline strict replays not accepted")
    if query_meta.get("status") != "PASS_QUERY_TABLES_LOCKED":
        raise AssertionError("Hadamard query table gate did not pass")
    if not audit_accept.get("stop_gate", {}).get("query_component_independently_replaceable", False):
        raise AssertionError("A audit did not accept isolated query replacement")

    learn_sa = read_json(LEARN / "SOURCE_ACCEPTANCE.json")
    source_lock_path = LEARN / "protocol/B_EXECUTION_SOURCE_LOCK.json"
    old_source_lock_sha = sha(source_lock_path)
    panel_lock_path = LEARN / "protocol/B_PANEL_SELECTION_LOCK.json"
    panel_lock_sha = sha(panel_lock_path)
    panel_lock = read_json(panel_lock_path)
    target_manifest = EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"
    target_cache = EQ / "d3/train_targets_trainonly.npz"
    cache_manifest = BENCH / "d3/cache/D3_CACHE_MANIFEST.json"
    video_normalizer = BENCH / "d3/cache/D3_VIDEO_NORMALIZER.json"
    target_normalizer = V3 / "data/V3_NORMALIZERS.json"
    v3_inputs = V3 / "data/V3_MAIN_INPUTS.npz"
    init_path = Path(learn_sa["seed_20261002_init"]["path"])
    init_sha = sha(init_path)
    if init_sha != learn_sa["seed_20261002_init"]["sha256"]:
        raise AssertionError("accepted INIT file hash mismatch")
    init_registry = read_json(EQ / "protocol/D3_EQ_SEED_REGISTRY.json")
    init_state_hash = init_registry["seeds"]["20261002"]["init_state_hash"]
    baseline_replays = list(csv.DictReader((ROOT / "forensics/BASELINE_REPLAY_ACCEPTANCE.csv").open()))
    if len(baseline_replays) != 4 or not all(x["replay_status"] == "PASS" for x in baseline_replays):
        raise AssertionError("four matched baseline strict replays are not accepted")

    query_h30_sha = query_meta["candidate_table_hashes"]["H30"]
    query_h60_sha = query_meta["candidate_table_hashes"]["H60"]
    panel_rows = panel_lock["rows"]
    by_panel = {
        p: sorted((x for x in panel_rows if x["panel"] == p), key=lambda x: int(x["panel_order"]))
        for p in ("P1", "P8")
    }
    baseline_by_key = {(x["config"], x["panel"]): x for x in baseline_replays}
    matrix_rows = []
    baseline_rows = []
    model_order = ("TS11-D", "STATE-CTRL")
    panel_order = ("P1", "P8")
    for config in model_order:
        for panel in panel_order:
            base = baseline_by_key[(config, panel)]
            old_run_id = base["run_id"]
            old_meta = read_json(LEARN / "runs/formal" / old_run_id / "RUN_COMPLETE.json")
            new_run_id = f"QUERY_GEOMETRY_TRAIN_FIT_PROBE-{config}-H30-FULL_LOSS-{panel}-S20261002"
            rows = by_panel[panel]
            matrix_rows.append({
                "run_id": new_run_id, "config": config, "horizon_steps": 30,
                "loss": "FULL_LOSS", "panel": panel, "seed": 20261002,
                "device": "cpu", "dtype": "float32", "updates": 2000,
                "stage0_updates": 4, "optimizer": "AdamW", "learning_rate": 0.001,
                "betas": "[0.9,0.999]", "eps": 1e-8, "weight_decay": 0.0,
                "grad_clip_norm": 1.0, "dropout": 0.0, "eval_interval_updates": 100,
                "sampler": "entire locked panel every update; no shuffle; equal row weight 1",
                "query_variant": "Sylvester H128 rows 1..30; per-step RMS matched to old fixed query",
                "query_table_sha256": query_h30_sha,
                "old_baseline_run_id": old_run_id,
                "panel_lock_sha256": panel_lock_sha,
                "init_file_sha256": init_sha, "init_state_hash": init_state_hash,
                "model_source_sha256": learn_sa["source_file_hashes"]["locked_model_source"]["sha256"],
                "historical_source_lock_sha256": old_source_lock_sha,
                "target_cache_manifest_sha256": sha(target_manifest),
                "target_cache_sha256": sha(target_cache),
                "target_normalizer_sha256": sha(target_normalizer),
                "video_normalizer_sha256": sha(video_normalizer),
                "feature_cache_manifest_sha256": sha(cache_manifest),
                "v3_input_archive_sha256": sha(v3_inputs),
                "anchor_ids_json": json_cell([x["anchor_id"] for x in rows]),
                "source_group_ids_json": json_cell([x["source_group_id"] for x in rows]),
                "target_signatures_json": json_cell([x["target_signature"] for x in rows]),
                "effective_input_hashes_json": json_cell([
                    {"state":x["state_input_sha256"],"TS11-D":x["TS11-D_effective_input_sha256"]}
                    for x in rows
                ]),
                "new_benchmark_candidate": False,
                "dev_targets_opened": False,
                "final_or_heldout_access": 0,
            })
            baseline_rows.append({
                "new_run_id": new_run_id, "old_run_id": old_run_id,
                "config": config, "panel": panel, "horizon_steps": 30,
                "loss": "FULL_LOSS", "seed": 20261002,
                "old_run_status": "PASS", "old_checkpoint_strict_load": True,
                "old_train_prediction_replay": "PASS exact; atol=1e-5 rtol=1e-5",
                "old_prediction_max_abs_replay_difference": base["prediction_max_abs_difference"],
                "old_checkpoint_sha256": base["final_checkpoint_sha256"],
                "old_prediction_sha256": base["prediction_sha256"],
                "init_file_sha256": init_sha,
                "init_state_hash": init_state_hash,
                "panel_lock_sha256": old_meta["panel_lock_sha256"],
                "source_lock_sha256": old_meta["source_lock_sha256"],
                "old_parameter_count": base["parameter_count"],
                "optimizer": "AdamW", "learning_rate": 0.001, "betas": "[0.9,0.999]",
                "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 1.0,
                "dropout": 0.0, "dtype": "float32", "device": "cpu",
                "updates": 2000, "full_panel_every_update": True,
                "only_locked_difference": "fixed future-query table",
                "comparison_status": "PASS_SOURCE_AND_REPLAY; query intervention is the only planned change",
            })

    matrix_path = ROOT / "protocol/MATRIX_4.csv"
    baseline_path = ROOT / "protocol/MATCHED_BASELINE_4.csv"
    write_csv(matrix_path, matrix_rows)
    write_csv(baseline_path, baseline_rows)

    # Refresh the source registry with the exact B implementation and locked query artifacts.
    source_files = registry["source_files"]
    seen = {x["path"] for x in source_files}
    new_sources = [
        (ROOT/"code/orthogonal_query.py","B fixed query construction"),
        (ROOT/"code/future_query_model.py","B isolated future-query wrapper"),
        (ROOT/"code/verify_query_checkpoint.py","fresh-process query checkpoint verifier"),
        (ROOT/"code/verify_query_checkpoint.py.pre-full-fit-path-20261007T115900Z.bak","pre-formal verifier source snapshot"),
        (ROOT/"code/train_query_fit.py","B CPU train-only runner"),
        (ROOT/"code/train_query_fit.py.pre-resource-and-output-guards-20261007T120205Z.bak","initial bounded runner snapshot"),
        (ROOT/"code/train_query_fit.py.pre-final-resource-timer-20261007T120320Z.bak","runner before final eval/wall guard"),
        (ROOT/"code/build_query_tables.py","query table builder"),
        (ROOT/"code/build_query_tables.py.pre-row-mean-fix-20261007T115910Z.bak","failed query-table gate source snapshot"),
        (ROOT/"code/lock_query_probe.py","pre-Stage0 protocol lock builder"),
        (ROOT/"tests/test_orthogonal_query.py","CPU numerical and fresh-process contract tests"),
        (ROOT/"logs/CPU_TESTS_ATTEMPT_002.log","previous passing CPU test result log"),
        (ROOT/"logs/CPU_TESTS_ATTEMPT_003.log","final passing CPU test result log"),
        (ROOT/"protocol/QUERY_TABLES_AND_HASHES.json","locked query table geometry and hashes"),
        (ROOT/"protocol/HADAMARD_QUERY_TABLES.npz","saved old and candidate query tables"),
        (ROOT/"protocol/HADAMARD_QUERY_TABLES.csv","human-readable old and candidate tables"),
        (ROOT/"forensics/A_AUDIT_ACCEPTANCE.json","accepted read-only A audit"),
    ]
    added = []
    for path, role in new_sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(path) not in seen:
            item = source_item(path, role)
            source_files.append(item)
            seen.add(str(path))
            added.append(item)
    registry["source_files"] = source_files
    registry["integrity"]["cpu_query_tests"] = "PASS_6_OF_6"
    registry["integrity"]["hadamard_table_gate"] = "PASS"
    registry["integrity"]["b_code_sources_registered"] = len(new_sources)
    registry["integrity"]["b_implementation_added_sources"] = len(added)
    write_json(ROOT/"SOURCE_REGISTRY.json", registry)
    write_json(ROOT/"protocol/SOURCE_REGISTRY.json", registry)
    registry_sha = sha(ROOT/"SOURCE_REGISTRY.json")

    accept = read_json(ROOT/"SOURCE_ACCEPTANCE.json")
    accept["status"] = "PASS_AUDIT_AND_B_PRELOCK_SOURCE_ACCEPTANCE"
    accept["source_registry_sha256"] = registry_sha
    accept["checks"]["cpu_query_contract_tests"] = "PASS_6_OF_6"
    accept["checks"]["hadamard_table_gate"] = "PASS_H30_AND_H60"
    accept["checks"]["candidate_trainable_parameters_added"] = 0
    accept["checks"]["strict_baseline_replay_gate"] = "PASS_4_OF_4"
    accept["checks"]["new_training_started"] = False
    write_json(ROOT/"SOURCE_ACCEPTANCE.json", accept)
    acceptance_sha = sha(ROOT/"SOURCE_ACCEPTANCE.json")

    # CPU-only resource preflight: no NVIDIA/CUDA inventory is queried.
    usage = shutil.disk_usage(PROJECT / "reports/benchmarks")
    free_fraction = usage.free / usage.total
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    scheduler_vars = [k for k in ("SLURM_JOB_ID","PBS_JOBID","LSB_JOBID") if os.environ.get(k)]
    host = socket.gethostname()
    baseline_seconds = []
    baseline_p95 = []
    for row in baseline_rows:
        meta = read_json(LEARN/"runs/formal"/row["old_run_id"]/"RUN_COMPLETE.json")
        baseline_seconds.append(float(meta["elapsed_seconds"]))
        baseline_p95.append(float(meta["p95_update_seconds"]))
    baseline_total = float(sum(baseline_seconds))
    fit_projection = baseline_total * 2.0
    stage0_projection = max(15.0, float(max(baseline_p95)) * 4 * 4 * 2.0)
    startup_and_input = 240.0
    post_run_overhead = 600.0
    projected_total = stage0_projection + fit_projection + startup_and_input + post_run_overhead
    max_cpu_wall = 3600
    max_output_bytes = 5 * 1024**3
    estimated_output_bytes = 512 * 1024**2
    current_root_bytes = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
    runtime_snapshot = {
        "observed_utc": now, "hostname": host,
        "node_role_inference": "Interactive server host inferred from hostname gpu33; scheduler allocation variables are absent. No GPU inventory was queried.",
        "scheduler_environment_detected": bool(scheduler_vars), "scheduler_variables_present": scheduler_vars,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "logical_cpu_count": os.cpu_count(), "process_affinity_cpu_count": affinity,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "torch_threads_locked": 1, "torch_interop_threads_locked": 1,
        "cuda_visible_devices": "", "gpu_query_use_lease": 0,
        "disk_total_bytes": usage.total, "disk_free_bytes": usage.free,
        "disk_free_fraction": free_fraction, "output_root_bytes_before_stage0": current_root_bytes,
        "memory_gpu_state": "not queried by task instruction",
        "cpu_policy_sources": [
            str(PROJECT/"AGENTS.md"),
            str(LEARN/"protocol/B_PROTOCOL_LOCKED.json"),
            str(ROOT/"CONTEXT_RESTORED.zh-CN.md"),
        ],
        "standalone_project_cpu_policy_document": "not found in the specified project source paths; current AGENTS and locked CPU protocol govern",
    }
    cpu_preflight_path = ROOT/"protocol/CPU_PREFLIGHT.json"
    write_json(cpu_preflight_path, runtime_snapshot)
    resource_pre = {
        "schema_version":1, "status":"PASS" if free_fraction >= 0.10 and projected_total <= max_cpu_wall and estimated_output_bytes <= max_output_bytes else "FAIL",
        "created_utc":now,"device":"cpu","torch_threads":1,"torch_interop_threads":1,
        "cuda_visible_devices":"","gpu_queries":0,"gpu_use":0,"gpu_lease":0,
        "max_cpu_wall_seconds":max_cpu_wall,"projected_cpu_wall_seconds":projected_total,
        "projection_components":{"historical_four_run_elapsed_seconds":baseline_total,
          "formal_projection_multiplier":2.0,"formal_projection_seconds":fit_projection,
          "stage0_projection_seconds":stage0_projection,"startup_and_input_seconds":startup_and_input,
          "post_run_analysis_and_figures_seconds":post_run_overhead},
        "max_new_output_bytes":max_output_bytes,"estimated_new_output_bytes":estimated_output_bytes,
        "current_new_root_bytes":current_root_bytes,
        "disk_free_bytes":usage.free,"disk_total_bytes":usage.total,"disk_free_fraction":free_fraction,
        "disk_free_fraction_minimum":0.10,
        "host_name":host,"scheduler_variables_present":scheduler_vars,
        "baseline_elapsed_seconds_by_run":dict(zip([x["old_run_id"] for x in baseline_rows],baseline_seconds)),
        "prelock_contract_tests":"6 passed",
        "gpu_state":"not queried; no inference about availability",
        "status_reason":"PASS when CPU projection is under 3600 seconds, output estimate under 5 GiB, and filesystem free fraction at least 10%.",
    }
    if resource_pre["status"] != "PASS":
        raise SystemExit("pre-Stage0 cost/disk gate failed")
    resource_path = ROOT/"protocol/RESOURCE_GATE_PRE_STAGE0.json"
    write_json(resource_path, resource_pre)

    source_hash_paths = {x["path"] for x in registry["source_files"]}
    explicit = [
        ROOT/"SOURCE_ACCEPTANCE.json",ROOT/"SOURCE_REGISTRY.json",
        ROOT/"CONTEXT_RESTORED.zh-CN.md",ROOT/"OUTCOME_EXPOSURE.md",
        ROOT/"DEVELOPMENT_REFERENCE_FREEZE.json",ROOT/"forensics/A_AUDIT_ACCEPTANCE.json",
        ROOT/"protocol/A_AUDIT_LOCK.json",ROOT/"protocol/CPU_PREFLIGHT.json",
        matrix_path,baseline_path,ROOT/"protocol/QUERY_TABLES_AND_HASHES.json",
        ROOT/"protocol/HADAMARD_QUERY_TABLES.npz",ROOT/"protocol/HADAMARD_QUERY_TABLES.csv",
        resource_path,
    ]
    for path in explicit:
        if not path.is_file(): raise FileNotFoundError(path)
        source_hash_paths.add(str(path))
    source_hashes = {p:sha(Path(p)) for p in sorted(source_hash_paths)}
    task_attachment = Path(os.environ["EE6008_FUSION_QUERY_TASK_ATTACHMENT"]).expanduser().resolve()
    learn_protocol = read_json(LEARN/"protocol/B_PROTOCOL_LOCKED.json")
    query_h30_sha = query_meta["candidate_table_hashes"]["H30"]
    protocol = {
        "schema_version":1,"protocol_id":"CORE_B_FUSION_QUERY_MECHANISM_V1",
        "status":"LOCKED_BEFORE_STAGE0","locked_utc":now,"new_root":str(ROOT),
        "task_attachment_sha256":sha(task_attachment),
        "development_reference_freeze_sha256":sha(ROOT/"DEVELOPMENT_REFERENCE_FREEZE.json"),
        "source_registry_sha256":registry_sha,"source_acceptance_sha256":acceptance_sha,
        "outcome_exposure_sha256":sha(ROOT/"OUTCOME_EXPOSURE.md"),
        "A_audit_acceptance_sha256":sha(ROOT/"forensics/A_AUDIT_ACCEPTANCE.json"),
        "A_audit_status":"PASS_READ_ONLY_AUDIT","hard_implementation_error_found":False,
        "device":"cpu","dtype":"float32","torch_threads":1,"torch_interop_threads":1,
        "cuda_visible_devices":"","gpu_queries":0,"gpu_use":0,"gpu_lease":0,
        "horizon_steps":30,"seed":20261002,"models":["TS11-D","STATE-CTRL"],
        "panels":["P1","P8"],"loss":"FULL_LOSS","matrix_cells":4,
        "matrix_sha256":sha(matrix_path),"matched_baseline_sha256":sha(baseline_path),
        "query_tables_json_sha256":sha(ROOT/"protocol/QUERY_TABLES_AND_HASHES.json"),
        "query_tables_npz_sha256":sha(ROOT/"protocol/HADAMARD_QUERY_TABLES.npz"),
        "query_table_sha256_h30":query_h30_sha,
        "historical_source_lock_sha256":old_source_lock_sha,
        "init_state_file_sha256":init_sha,"init_state_hash":init_state_hash,
        "parameter_count":890780,
        "panel_lock_sha256":panel_lock_sha,
        "target_cache_sha256":sha(target_cache),
        "target_cache_manifest_sha256":sha(target_manifest),
        "target_normalizer_sha256":sha(target_normalizer),
        "video_normalizer_sha256":sha(video_normalizer),
        "feature_cache_manifest_sha256":sha(cache_manifest),
        "v3_input_archive_sha256":sha(V3/"data/V3_MAIN_INPUTS.npz"),
        "training_conditions":{
            "optimizer":"AdamW","learning_rate":0.001,"betas":[0.9,0.999],"eps":1e-8,
            "weight_decay":0.0,"grad_clip_norm":1.0,"dropout":0.0,
            "updates_per_formal_cell":2000,"stage0_updates_per_cell":4,
            "eval_interval_updates":100,"batch":"entire locked P1/P8 panel each update",
            "panel_row_weights":"all equal 1; one row per source group","scheduler":None,
            "init":"strict-load same accepted original INIT; every Stage0 and formal cell starts from INIT",
            "loss":"L_pos + 0.1 * L_delta","native_actions":"numeric action indices only",
        },
        "query_intervention":{
            "only_changed_component":"fixed future-query table input to decoder",
            "construction":"canonical Sylvester H128; rows 1..H; exclude row0 all-ones DC row; columns in canonical order",
            "per_step_scale":"RMS of old source sinusoidal query at absolute step t, old time t/30 seconds",
            "H30_fit_table_sha256":query_h30_sha,
            "H60":"mathematical/interface test only; no H60 training",
            "learned_query_parameters_added":0,
            "other_embeddings_and_masks":"unchanged",
            "interpretation_limit":"Hadamard changes time distances, spectrum, smoothness and conditioning together; any train-fit signal is not benchmark gain or a unique rank effect.",
        },
        "matched_baselines":baseline_rows,
        "stage0":{"cells":4,"updates_each":4,"fresh_process_reload_required":True,
                  "stage0_weights_not_used_for_formal_initialization":True},
        "resource_budget":{
            "max_cpu_wall_seconds":max_cpu_wall,"max_new_output_bytes":max_output_bytes,
            "minimum_disk_free_fraction":0.10,"projected_total_cpu_wall_seconds":projected_total,
            "post_run_overhead_seconds":post_run_overhead,
            "projected_formal_seconds":fit_projection+post_run_overhead,
            "resource_preflight_path":str(resource_path),"resource_preflight_sha256":sha(resource_path),
            "single_process_at_a_time":True,
        },
        "prohibitions":{"dev_targets_opened":False,"final_or_heldout_access":0,
            "new_dev_model_selection":False,"new_full_data_benchmark_training":False,
            "D5_training":False,"benchmark_candidate":False,"p_values":False,"new_CI":False},
        "source_hashes":source_hashes,
    }
    protocol_path=ROOT/"protocol/PROTOCOL_LOCKED.json"
    write_json(protocol_path,protocol)
    print(json.dumps({
        "status":resource_pre["status"],"matrix_cells":len(matrix_rows),
        "matrix_sha256":protocol["matrix_sha256"],
        "matched_baseline_sha256":protocol["matched_baseline_sha256"],
        "source_hashes":len(source_hashes),"source_registry_entries":len(registry["source_files"]),
        "projected_total_cpu_wall_seconds":projected_total,
        "disk_free_fraction":free_fraction,"estimated_new_output_bytes":estimated_output_bytes,
        "protocol_sha256":sha(protocol_path),"new_training_started":False,
    },ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())

