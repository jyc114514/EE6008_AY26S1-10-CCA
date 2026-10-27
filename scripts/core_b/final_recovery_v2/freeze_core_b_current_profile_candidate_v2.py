#!/usr/bin/env python3
"""Freeze CURRENT_PROFILE_V2 before full development feature extraction/scoring."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,os,platform,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import torch
import torchvision

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT=PROJECT/"reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1_OUT=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
V1_RUN=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
V9=PROJECT/"reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z"
V9_ROOT=PROJECT/"reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z"
V9_MATRIX=V1_OUT/"protocol/FINAL_MODEL_MATRIX.csv"
V9_CODE_LOCK=V9/"protocol/CODE_LOCK.json"
V9_RECIPE=V9/"protocol/RECIPE_LOCK.json"
V9_SOURCE_MAP=V9/"inventory/SOURCE_MAP.json"
V1_STUDY=V1_OUT/"protocol/FINAL_STUDY_LOCK.json"
V1_ACCESS=V1_OUT/"protocol/FINAL_ACCESS_OPENED.json"
Q0=PROJECT/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json"
D3=PROJECT/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/code/d3_extract_grid.py"
PREPROCESS=PROJECT/"public_exports/core_a_github_publication_20260910T082843Z/clone/src/ee6008/data/clips.py"
CHECKPOINT=PROJECT/"checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
NORMALIZER=PROJECT/"reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z/normalizers/NORMALIZER_FORMAL.json"
METRIC=PROJECT/"reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/protocol/METRIC_SCALE_LOCK.json"
V9_DEV_TARGET=PROJECT/"reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/DEV_TARGETS_SCORER_ONLY.npz"
DEV_PREFIX=RUN/"replay/core_b/DEV_PREFIXES_CURRENT_PROFILE_V2.npz"
DEV_MANIFEST=OUT/"core_b/DEV_CURRENT_PROFILE_INPUTS.csv"
DEV_RECEIPT=OUT/"core_b/DEV_CURRENT_PROFILE_INPUTS_RECEIPT.json"
FINAL_MANIFEST=OUT/"core_b/FINAL_CURRENT_PROFILE_INPUTS.csv"
FINAL_MANIFEST_RECEIPT=OUT/"core_b/FINAL_CURRENT_PROFILE_INPUTS_RECEIPT.json"
FINAL_MAP=V1_OUT/"protocol/CORE_B_FINAL_VIDEO_MAP.csv"
FINAL_POP=V1_OUT/"protocol/CORE_B_FINAL_POPULATION_METADATA.csv"
FINAL_BOOT_NPZ=OUT/"core_b/CORE_B_FINAL_BOOTSTRAP_DRAWS_V2.npz"
FINAL_BOOT_JSON=OUT/"core_b/CORE_B_FINAL_BOOTSTRAP_DRAWS_V2.json"
FINAL_BOOT_RECEIPT=OUT/"core_b/CORE_B_FINAL_BOOTSTRAP_DRAWS_V2_RECEIPT.json"
DEV_BOOT_NPZ=OUT/"core_b/CORE_B_DEV_BOOTSTRAP_DRAWS_V2.npz"
DEV_BOOT_JSON=OUT/"core_b/CORE_B_DEV_BOOTSTRAP_DRAWS_V2.json"
DEV_BOOT_RECEIPT=OUT/"core_b/CORE_B_DEV_BOOTSTRAP_DRAWS_V2_RECEIPT.json"
PROBE=OUT/"core_b/CORE_B_GPU_DIFFERENTIAL_PROBES_ATTEMPT_02.json"
PROBE_ARCHIVE=RUN/"core_b/gpu_probes/CURRENT_PROFILE_BRIDGE_FEATURES_ATTEMPT_02.npz"
PROBE_TMUX=OUT/"core_b/CORE_B_GPU3_TRAIN_PROBE_TMUX_ATTEMPT_03.json"
CPU_AUDIT=OUT/"core_b/CORE_B_TRAIN_BRIDGE_CPU_AUDIT.json"
CROSS_CORE=OUT/"core_b/CROSS_CORE_INPUT_EXPOSURE.csv"
CROSS_CORE_SUMMARY=OUT/"qa/CROSS_CORE_INPUT_EXPOSURE_SUMMARY.json"
ROUTE=OUT/"core_b/ROUTE_DECISION.json"
PROFILE_TABLE=OUT/"core_b/PROFILE_AND_SOURCE_COMPARISON.csv"
HYP_CSV=OUT/"core_b/CORE_B_CAUSAL_HYPOTHESES.csv"
HYP_MD=OUT/"core_b/CORE_B_DIFFERENTIAL_HARNESS_AND_HYPOTHESES.md"
METRICS_CODE=V1_OUT/"code/analysis_metrics.py"
DEV_SCORER=V1_OUT/"code/score_dev_replay.py"
RAW_ROOT=PROJECT/"data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
VJEPA_SOURCE=PROJECT/"repo/third_party/vjepa2"
PUBLIC_SRC=PROJECT/"public_exports/core_a_github_publication_20260910T082843Z/clone/src"
TARGET=OUT/"core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json"
PROTOCOL_ID="EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
TASKS=(119,94,90,141,230)
SEEDS=(202610131,202610132,202610133)
PROGRAMS=("A0","AD","V0","VD","S_OPT")


def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def tree_hash(root, suffixes=(".py",)):
 root=Path(root);files=sorted(p for p in root.rglob("*") if not p.is_symlink() and p.is_file() and p.suffix in suffixes)
 digest=hashlib.sha256()
 for p in files:
  rel=p.relative_to(root).as_posix().encode();digest.update(len(rel).to_bytes(8,"little"));digest.update(rel);digest.update(bytes.fromhex(sha(p)))
 return {"root":str(root),"file_count":len(files),"sha256":digest.hexdigest(),"files":[{"path":str(p.relative_to(root)),"sha256":sha(p)} for p in files]}

def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.{os.getpid()}.part")
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,path)

def code_hashes():
 code=OUT/"code"
 files={
  "dev_input_builder":code/"build_core_b_current_profile_dev_inputs.py",
  "final_metadata_builder":code/"build_core_b_current_profile_final_manifest.py",
  "bootstrap_draw_builder":code/"build_core_b_bootstrap_draws_v2.py",
  "dev_extractor":code/"extract_core_b_current_profile_v2.py",
  "final_extractor":code/"extract_core_b_current_profile_final_v2.py",
  "predictor":code/"core_b_predict_v2.py",
  "runner":code/"run_core_b_current_profile_v2.py",
  "sealer":code/"seal_core_b_predictions_v2.py",
  "scorer":code/"score_core_b_v2.py",
  "final_prefix_builder":code/"prepare_core_b_final_prefix_v2.py",
  "contract_tests":code/"test_core_b_v2_contracts.py",
  "probe_analysis_updater":code/"update_core_b_probe_analysis_v2.py",
  "profile_lock_builder":Path(__file__),
 }
 missing=[name for name,p in files.items() if not p.is_file()]
 if missing:raise FileNotFoundError(f"missing candidate lock code files: {missing}")
 return {name:sha(path) for name,path in files.items()}

def check_bootstrap(npz,jsonp,receipt,split):
 m=json.loads(jsonp.read_text());r=json.loads(receipt.read_text())
 expected_status="LOCKED_BEFORE_CURRENT_PROFILE_DEV_SCORE" if split=="dev" else "LOCKED_BEFORE_FINAL_SCORE"
 if m.get("status")!=expected_status or m.get("draws")!=5000 or m.get("seed")!=2026101902 or r.get("scores_or_targets_read") is not False:
  raise ValueError(f"{split} bootstrap plan is not pre-score locked")
 if sha(npz)!=r.get("draw_npz_sha256") or sha(jsonp)!=r.get("draw_metadata_sha256"):
  raise ValueError(f"{split} bootstrap draw hashes differ from the receipt")
 return {"draw_archive_path":str(npz),"draw_archive_sha256":sha(npz),
         "draw_metadata_path":str(jsonp),"draw_metadata_sha256":sha(jsonp),
         "draw_receipt_path":str(receipt),"draw_receipt_sha256":sha(receipt),
         "seed":m["seed"],"draws":m["draws"],"episode_support_by_task":m["episode_support_by_task"],
         "resampling_unit":m["resampling_unit"],"metric_reduction":m["metric_reduction"]}

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--revise-from-sha256")
 parser.add_argument("--revision-reason")
 args=parser.parse_args()
 prior_lock=None
 prior_sha=None
 if TARGET.exists():
  prior_sha=sha(TARGET)
  if not args.revise_from_sha256 or args.revise_from_sha256!=prior_sha or not args.revision_reason:
   raise FileExistsError("CURRENT_PROFILE_CANDIDATE_LOCK.json exists; supply the exact prior SHA and a revision reason to create a pre-feature revision")
  prior_lock=json.loads(TARGET.read_text(encoding="utf-8"))
 else:
  if args.revise_from_sha256 or args.revision_reason:
   raise FileNotFoundError("cannot revise a candidate lock that does not exist")
 # No final/current-profile feature or score is allowed to exist before this lock.
 prohibited=[OUT/"core_b/CURRENT_PROFILE_V2_DEV_FEATURE_MANIFEST.json",OUT/"core_b/CURRENT_PROFILE_DEV_RESULTS.csv",OUT/"core_b/CURRENT_PROFILE_DEV_ACCEPTANCE.json",OUT/"core_b/CURRENT_PROFILE_V2_FINAL_FEATURE_MANIFEST.json",OUT/"core_b/FINAL_PREFIX_EXTRACTION_RECEIPT.json"]
 if any(p.exists() for p in prohibited):raise ValueError("current-profile dev features/scores already exist before candidate lock")
 if not all(p.is_file() for p in (DEV_PREFIX,DEV_MANIFEST,DEV_RECEIPT,FINAL_MANIFEST,FINAL_MANIFEST_RECEIPT,FINAL_MAP,FINAL_POP)):
  raise FileNotFoundError("fixed dev/final metadata inputs are incomplete")
 if sha(DEV_PREFIX)!=json.loads(DEV_RECEIPT.read_text())["outputs"]["prefix_sha256"] or sha(DEV_MANIFEST)!=json.loads(DEV_RECEIPT.read_text())["outputs"]["input_manifest_sha256"]:
  raise ValueError("dev no-label input receipt/hash mismatch")
 if sha(FINAL_MANIFEST)!=json.loads(FINAL_MANIFEST_RECEIPT.read_text())["output_sha256"]:
  raise ValueError("final metadata-only map receipt/hash mismatch")
 final_map_receipt=json.loads(FINAL_MANIFEST_RECEIPT.read_text())
 if final_map_receipt.get("final_payload_access") is not False or final_map_receipt.get("future_target_values_materialized")!=0:
  raise ValueError("final metadata map receipt reports payload/target access")

 route=json.loads(ROUTE.read_text())
 if route.get("route")!="R2" or route.get("status")!="R2_SELECTED_BEFORE_CURRENT_PROFILE_DEV_FEATURES_OR_SCORES":
  raise ValueError("R2 route decision is missing or not pre-score")
 probe=json.loads(PROBE.read_text())
 child=json.loads((RUN/"core_b/gpu_probes/CORE_B_GPU_PROBE_ATTEMPT_02.json").read_text())
 if (probe.get("status")!="COMPLETED_TRAIN_ONLY_CONTROLLED_GPU3_PROBES" or
     child.get("status")!="COMPLETED" or child.get("probe_configurations_completed")!=6 or
     len(child.get("completed_probe_ids",[]))!=6 or probe.get("physical_gpu")!=3 or
     probe.get("final_payload_read") or probe.get("future_targets_materialized")):
  raise ValueError("six train-only controlled GPU3 probe evidence incomplete")

 # Verify that the V2 predictor still imports the exact accepted V9 source closure.
 sys.path.insert(0,str(OUT/"code"))
 import core_b_predict_v2 as predictor
 accepted_sources=predictor.load_accepted_sources()
 v9_lock=json.loads(V9_CODE_LOCK.read_text())
 if sha(V9_CODE_LOCK)!=predictor.ACCEPTED_CODE_LOCK_SHA256:
  raise ValueError("accepted V9 CODE_LOCK SHA differs from the predictor pin")

 matrix=list(csv.DictReader(V9_MATRIX.open(encoding="utf-8-sig",newline="")))
 model_rows=[r for r in matrix if r["core"]=="B" and r["program"] in PROGRAMS and r["status"].startswith("accepted_source_bound")]
 if len(model_rows)!=15:raise ValueError(f"expected 15 accepted neural model cells, got {len(model_rows)}")
 cells={p:{} for p in PROGRAMS}
 for r in model_rows:
  p=str(r["program"]);seed=str(int(r["seed"]))
  for field,path_field in (("checkpoint_sha256","checkpoint_path"),("model_manifest_sha256","model_manifest_path"),("normalizer_sha256","normalizer_path")):
   path=Path(r[path_field])
   if not path.is_file() or sha(path)!=r[field]:raise ValueError(f"accepted model asset hash mismatch {p}/{seed}/{field}")
  if p == "S_OPT":
   if r["config_path"] != "V2 formal accepted model receipt; see model manifest" or r["config_sha256"] != "bound through accepted model receipt and source map":
    raise ValueError(f"S_OPT formal config pointer differs from accepted source map: {p}/{seed}")
   config_identity_kind="accepted formal model manifest and source map pointer"
  else:
   path=Path(r["config_path"])
   if not path.is_file() or sha(path)!=r["config_sha256"]:raise ValueError(f"accepted model config hash mismatch {p}/{seed}")
   config_identity_kind="file_sha256"
  cells[p][seed]={
   "status":r["status"],"mode":r["mode"],"role":r["role"],
   "checkpoint_path":r["checkpoint_path"],"checkpoint_sha256":r["checkpoint_sha256"],
   "model_manifest_path":r["model_manifest_path"],"model_manifest_sha256":r["model_manifest_sha256"],
   "model_state_hash":r["model_state_hash"],"config_path":r["config_path"],"config_sha256":r["config_sha256"],
   "normalizer_path":r["normalizer_path"],"normalizer_sha256":r["normalizer_sha256"],
   "selected_epoch":r["selected_epoch"],"selected_on":r["selected_on"],
  }
 if any(set(cells[p])!=set(map(str,SEEDS)) for p in PROGRAMS):raise ValueError("model matrix program/seed coverage mismatch")
 if any(cells["S_OPT"][str(seed)]["checkpoint_sha256"]==cells["A0"][str(seed)]["checkpoint_sha256"] for seed in SEEDS):
  raise ValueError("S_OPT checkpoint unexpectedly aliases A0")

 metric=json.loads(METRIC.read_text());normalizer=json.loads(NORMALIZER.read_text())
 if sha(METRIC)!="dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32" or metric.get("status")!="LOCKED":
  raise ValueError("accepted V9 metric-scale lock is not intact")
 if sha(NORMALIZER)!="22f66a54f6106de68c1a0bf0aa3a46ca3340c3077b4121c6a0fdf9d83528d65d":
  raise ValueError("accepted residual normalizer is not intact")
 old_access=json.loads(V1_ACCESS.read_text())
 if old_access.get("core_b_access_authorized") is not False or old_access.get("core_b_final_payload_opened") is not False:
  raise ValueError("V1 Core B access boundary no longer records the expected closed state")

 dev_receipt=json.loads(DEV_RECEIPT.read_text())
 probe_rows={r["anchor_id"]:r for r in probe["rows"]}
 q0_archive={
  "path":str(Q0),"sha256":sha(Q0),
  "row_count":20079,"shard_count":314,"feature_shape":"[N,2,16,4,768] float32",
  "reader_source_sha256":predictor.PINNED_SOURCE_HASHES["prepare_v2_data.py"],
  "reader": "accepted Q0ShardedGrid; exact direct shard slice, no read-time cast",
  "old_bridge_status":"FAIL_UNRESOLVED",
 }
 vje_src_git={}
 try:
  rev=subprocess.run(["git","-C",str(VJEPA_SOURCE),"rev-parse","HEAD"],text=True,capture_output=True,check=True).stdout.strip()
  status=subprocess.run(["git","-C",str(VJEPA_SOURCE),"status","--short"],text=True,capture_output=True,check=True).stdout.splitlines()
  vje_src_git={"head":rev,"status_short":status}
 except Exception as exc:
  vje_src_git={"status":"unavailable","reason":type(exc).__name__}
 vje_tree=tree_hash(VJEPA_SOURCE)
 public_tree=tree_hash(PUBLIC_SRC)
 input_paths=[
  "build_core_b_current_profile_dev_inputs.py","build_core_b_current_profile_final_manifest.py",
  "build_core_b_bootstrap_draws_v2.py","extract_core_b_current_profile_v2.py",
  "extract_core_b_current_profile_final_v2.py","core_b_predict_v2.py","run_core_b_current_profile_v2.py",
  "seal_core_b_predictions_v2.py","score_core_b_v2.py","prepare_core_b_final_prefix_v2.py",
  "test_core_b_v2_contracts.py","update_core_b_probe_analysis_v2.py","freeze_core_b_current_profile_candidate_v2.py",
 ]
 code=code_hashes()
 dev_draw=check_bootstrap(DEV_BOOT_NPZ,DEV_BOOT_JSON,DEV_BOOT_RECEIPT,"dev")
 final_draw=check_bootstrap(FINAL_BOOT_NPZ,FINAL_BOOT_JSON,FINAL_BOOT_RECEIPT,"final")
 final_population_rows=list(csv.DictReader(FINAL_POP.open(encoding="utf-8-sig",newline="")))
 dev_input_rows=list(csv.DictReader(DEV_MANIFEST.open(encoding="utf-8-sig",newline="")))
 final_input_rows=list(csv.DictReader(FINAL_MANIFEST.open(encoding="utf-8-sig",newline="")))
 if len(dev_input_rows)!=185 or len(final_input_rows)!=162 or len(final_population_rows)!=162:raise ValueError("locked dev/final input population counts changed")
 dev_tasks={str(t):sum(int(r["task_id"])==t for r in dev_input_rows) for t in TASKS}
 final_tasks={str(t):sum(int(r["task_id"])==t for r in final_input_rows) for t in TASKS}
 if len({r["episode_id"] for r in dev_input_rows})!=47 or len({r["episode_id"] for r in final_input_rows})!=41:
  raise ValueError("locked dev/final episode support changed")
 raw_probe_archive=PROBE_ARCHIVE
 expected_runtime={
  "matmul_allow_tf32":False,"cudnn_allow_tf32":False,"cudnn_benchmark":False,"cudnn_deterministic":False,
  "deterministic_algorithms":False,"float32_matmul_precision":"highest",
  "flash_sdp_enabled":True,"mem_efficient_sdp_enabled":True,"math_sdp_enabled":True,
 }
 if probe["candidate_profile_v2"].get("batch_size")!=1 or not probe["candidate_profile_repeatability"]["repeated_profile_allclose_1e-5"]:
  raise ValueError("candidate profile's registered train-only repeatability failed")

 lock={
  "schema_version":1,"protocol_id":PROTOCOL_ID,
  "lock_revision":int(prior_lock.get("lock_revision", 0)) + 1 if prior_lock else 1,
  "supersedes_lock_sha256":prior_sha,
  "revision_reason":args.revision_reason if prior_lock else "initial pre-feature candidate lock",
  "status":"LOCKED_BEFORE_CURRENT_PROFILE_DEV_FEATURES",
  "route":"R2","profile_id":"CURRENT_PROFILE_V2",
  "locked_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
  "authorization_source":"current user request and attached V2 recovery runbook; physical GPU3 separately authorized by user; no supervisor approval asserted",
  "decision":{
   "old_cached_q0_dev_replay":"PASS_ACCEPTED_V9_DEVELOPMENT_ONLY",
   "old_q0_fresh_feature_bridge":"FAIL_UNRESOLVED",
   "root_cause":"UNKNOWN; historical Q0 generation source/numeric profile is incomplete; no physical GPU-number attribution",
   "candidate_reason":"full-precision FP32, explicit batch/backend/TF32 policy, pinned preprocessing/pooling/input identities, and exact two-row train repeatability; not selected by full dev or final performance",
   "profile_is_a_new_evaluation":"true; does not restore or rename Q0 and does not alter the historical 1e-5 gate",
   "r1_status":"NOT_ESTABLISHED",
  },
  "sources":{
   "v1_out_root":str(V1_OUT),"v1_study_lock_sha256":sha(V1_STUDY),"v1_access_receipt_sha256":sha(V1_ACCESS),
   "v1_model_matrix_path":str(V9_MATRIX),"v1_model_matrix_sha256":sha(V9_MATRIX),
   "v9_root":str(V9),"v9_code_lock_path":str(V9_CODE_LOCK),"v9_code_lock_sha256":sha(V9_CODE_LOCK),
   "v9_recipe_lock_path":str(V9_RECIPE),"v9_recipe_lock_sha256":sha(V9_RECIPE),
   "v9_source_map_path":str(V9_SOURCE_MAP),"v9_source_map_sha256":sha(V9_SOURCE_MAP),
   "accepted_v9_source_closure_validated_by":"core_b_predict_v2.load_accepted_sources",
   "accepted_v9_source_hashes":predictor.PINNED_SOURCE_HASHES,
   "historical_d3_extractor_path":str(D3),"historical_d3_extractor_sha256":sha(D3),
   "q0_cache_manifest":q0_archive,
   "historical_d3_gpu1_build_receipt_path":str(PROJECT/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/logs/D3-GRID-FULL.json"),
   "historical_d3_gpu1_build_receipt_sha256":sha(PROJECT/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/logs/D3-GRID-FULL.json"),
   "preprocess_module_path":str(PREPROCESS),"preprocess_module_sha256":sha(PREPROCESS),
   "vjepa_source_git":vje_src_git,"vjepa_source_tree":vje_tree,"public_model_loader_source_tree":public_tree,
   "encoder_checkpoint_path":str(CHECKPOINT),"encoder_checkpoint_sha256":sha(CHECKPOINT),
  },
  "encoder_profile":{
   "source_pipeline":"current D3-based frozen V-JEPA 2.1-B EMA encoder; no encoder/head training",
   "input_selection":"32 frame indices floor(linspace(anchor_frame-119, anchor_frame, 32)); selected maximum equals anchor; no selected future frame",
   "decoder":"OpenCV VideoCapture, BGR-to-RGB, decode through last selected anchor frame",
   "preprocess":"pinned preprocess_vjepa module; 384 resolution, shortest-side resize/center crop and ImageNet normalization; output float32 [3,32,384,384]",
   "encoder":"checkpoint SHA-pinned; eval mode; 32 frames; 384 input; raw output [9216,768] float32 after output cast",
   "token_grid":"reshape [16,24,24,768]; four disjoint 12x12 spatial area means; output [16,4,768] per mode",
   "mode_order":["dynamic","static"],"static_input":"repeat the observed anchor RGB frame 32 times",
   "feature_archive_shape":"[N,2,16,4,768] float32",
   "numeric_profile":{
    "precision":"FP32","autocast":"disabled","encoder_batch_size":1,
    "matmul_tf32":False,"cudnn_tf32":False,"float32_matmul_precision":"highest",
    "sdpa":"force torch.nn.attention.sdpa_kernel(SDPBackend.MATH)",
    "cudnn_benchmark":False,"cudnn_deterministic":False,"deterministic_algorithms":False,
    "other_backend_flags":"record and require actual P5/P6 values below",
    "required_runtime_flags":expected_runtime,
   },
   "observed_candidate_probe":{
    "probe_ids":["P5_CURRENT_PROFILE_V2_REPEAT_0","P6_CURRENT_PROFILE_V2_REPEAT_1"],
    "probe_result_path":str(PROBE),"probe_result_sha256":sha(PROBE),
    "feature_archive_path":str(PROBE_ARCHIVE),"feature_archive_sha256":sha(PROBE_ARCHIVE),
    "physical_gpu":3,"logical_device":"cuda:0","device":"NVIDIA GeForce RTX 3090",
    "repeatability":"bitwise equal raw-token and pooled-grid hashes over 2 fixed train rows × dynamic/static",
    "q0_bridge":"fails original 1e-5 tolerance; preserved as failure",
   },
  },
  "predictor_contract":{
   "model_source_root":str(V9/"code"),"code_lock_sha256":sha(V9_CODE_LOCK),
   "input_tensors":["state","history_mask","relative_delta","anchor_command","dynamic","static"],
   "state_history":"same V9 state/history/task/time semantics and accepted residual normalizer as the fixed checkpoints",
   "video_grid":"dynamic/static CURRENT_PROFILE_V2 [N,2,16,4,768], identity bound by local cache_index + anchor + episode",
   "inference":"eval/full-context; training-time numeric-context dropout disabled; batch 64; accepted V9 per-cell inference flags",
   "native_reconstruction":"anchor_native + target_sigma * predicted_residual",
   "action_shape":"30 future rows × 28 action dimensions; physical units/action clock remain unresolved",
   "normalizer_path":str(NORMALIZER),"normalizer_sha256":sha(NORMALIZER),
   "metric_scale_path":str(METRIC),"metric_scale_sha256":sha(METRIC),
   "normalizer_and_metric_scale_are_distinct_assets":True,
   "model_cells":cells,
   "primary_single_model_reference":"V0",
   "dropout_secondary_condition":"VD",
   "mixtures":{"MIX0":"0.5*S_OPT_s + 0.5*V0_s in native float64","MIXD":"0.5*S_OPT_s + 0.5*VD_s in native float64"},
   "persistence":"repeat the observed anchor_native for 30 future rows; one fixed reference, not three model seeds",
  },
  "inputs":{
   "dev_split":"accepted fixed 185-window V9 dev population; no target values in prefix",
   "dev_prefix_path":str(DEV_PREFIX),"dev_prefix_sha256":sha(DEV_PREFIX),
   "dev_manifest_path":str(DEV_MANIFEST),"dev_manifest_sha256":sha(DEV_MANIFEST),
   "dev_input_receipt_path":str(DEV_RECEIPT),"dev_input_receipt_sha256":sha(DEV_RECEIPT),
   "dev_canonical_id_order_sha256":dev_receipt["canonical_id_order_sha256"],
   "dev_anchor_id_order_sha256":dev_receipt["anchor_id_order_sha256"],
   "dev_rows":185,"dev_episode_support":47,"dev_task_support":dev_tasks,
   "dev_source_metadata":{
    "allowlist_sha256":dev_receipt["v3_identity_sources"]["allowlist_sha256"],
    "anchor_manifest_sha256":dev_receipt["v3_identity_sources"]["anchor_manifest_sha256"],
    "q0_identity_manifest_sha256":dev_receipt["v3_identity_sources"]["q0_manifest_sha256"],
    "q0_feature_arrays_loaded_for_current_input_build":False,
   },
   "final_input_manifest_path":str(FINAL_MANIFEST),"final_input_manifest_sha256":sha(FINAL_MANIFEST),
   "final_manifest_receipt_path":str(FINAL_MANIFEST_RECEIPT),"final_manifest_receipt_sha256":sha(FINAL_MANIFEST_RECEIPT),
  },
  "final_inputs":{
   "final_test_access_still_closed_at_candidate_lock":True,
   "final_video_map_path":str(FINAL_MAP),"final_video_map_sha256":sha(FINAL_MAP),
   "final_population_metadata_path":str(FINAL_POP),"final_population_metadata_sha256":sha(FINAL_POP),
   "cross_core_input_exposure_path":str(CROSS_CORE),"cross_core_input_exposure_sha256":sha(CROSS_CORE),
   "cross_core_summary_path":str(CROSS_CORE_SUMMARY),"cross_core_summary_sha256":sha(CROSS_CORE_SUMMARY),
   "expected_rows":162,"episode_support":41,"task_support":final_tasks,
   "final_video_map_materialized_rgb":False,"final_targets_materialized":False,
   "final_prefix_builder_code_sha256":code["final_prefix_builder"],
   "final_prefix_policy":"after new V2 access receipt only; state/action history rows through each legal anchor, no H30 future rows",
  },
  "metrics":{
   "metric_source_path":str(METRICS_CODE),"metric_source_sha256":sha(METRICS_CODE),
   "dev_scorer_reference_path":str(DEV_SCORER),"dev_scorer_reference_sha256":sha(DEV_SCORER),
   "metric_scale_lock_path":str(METRIC),"metric_scale_lock_sha256":sha(METRIC),
   "residual_normalizer_path":str(NORMALIZER),"residual_normalizer_sha256":sha(NORMALIZER),
   "definitions":{
    "position_mae":"30×28 absolute native position error standardized by locked training target_sigma; two 14D blocks equally weighted per anchor",
    "safe_fd_l1":"29 within-future differences on the 30 H30 output rows, scaled by locked safe_fd_scale; two 14D blocks equally weighted per anchor",
    "seam_safe_fd_l1":"first future row versus observed anchor increment error, scaled by safe_fd_scale; two 14D blocks equally weighted per anchor",
    "aggregation":"anchor mean within episode/source group; equal episode mean within each of five tasks; equal macro mean across fixed task order",
    "unit_interpretation":"standardized errors; do not invent physical units or seconds",
   },
   "training_seeds":list(SEEDS),"main_matrix":["A0","AD","V0","VD"] ,
   "additional_reference":["S_OPT"],"derived":["MIX0","MIXD"],"fixed_reference":["PERSISTENCE"],
   "primary_reference":"V0","dropout_secondary":"VD",
   "comparisons":["A0−V0","Persistence−V0","V0−VD","A0−AD","AD−VD","MIX0−MIXD","V0−MIX0"],
   "gain_direction":"reference minus candidate; positive favors candidate lower error",
   "seed_summary":"per-seed task macro; then mean and sample SD across three fixed model seeds",
   "bootstrap":[dev_draw,final_draw],
   "p_values_and_equivalence_claims":"not planned",
  },
  "source_evidence":{
   "route_decision_path":str(ROUTE),"route_decision_sha256":sha(ROUTE),
   "profile_source_table_path":str(PROFILE_TABLE),"profile_source_table_sha256":sha(PROFILE_TABLE),
   "causal_hypotheses_csv_path":str(HYP_CSV),"causal_hypotheses_csv_sha256":sha(HYP_CSV),
   "differential_hypotheses_md_path":str(HYP_MD),"differential_hypotheses_md_sha256":sha(HYP_MD),
   "cpu_train_bridge_audit_path":str(CPU_AUDIT),"cpu_train_bridge_audit_sha256":sha(CPU_AUDIT),
   "gpu_probe_result_path":str(PROBE),"gpu_probe_result_sha256":sha(PROBE),
   "gpu_probe_feature_archive_path":str(PROBE_ARCHIVE),"gpu_probe_feature_archive_sha256":sha(PROBE_ARCHIVE),
   "gpu_probe_tmux_record_path":str(PROBE_TMUX),"gpu_probe_tmux_record_sha256":sha(PROBE_TMUX),
   "dev_target_archive_path":str(V9_DEV_TARGET),"dev_target_archive_sha256_expected":"233a0cdb31d1ddd7cea6ecfaa5e9d02e9c42e9822af7fe488c7fe5a40febee27",
   "dev_target_archive_bytes_expected":435083,
   "old_bridge_failure_preserved":True,
  },
  "code_sha256":code,
  "environment":{
   "host":platform.node(),"python":sys.version.split()[0],"platform":platform.platform(),
   "numpy":np.__version__,"torch":str(torch.__version__),"torch_cuda_build":torch.version.cuda,
   "torchvision":str(torchvision.__version__),"opencv":cv2.__version__,"pyarrow":pa.__version__,
   "physical_gpu":3,"gpu_model":"NVIDIA GeForce RTX 3090","logical_gpu":"cuda:0",
   "gpu_probe_runtime":probe["runtime"],"no_environment_upgrade":True,
   "vjeps_source_git":vje_src_git,
  },
  "technical_failure_criteria":[
   "any source/episode/task/anchor/cache-index mismatch",
   "any selected encoder frame later than its observation anchor",
   "preprocess/checkpoint/model/manifest/normalizer/metric-scale/code hash mismatch",
   "candidate P5/P6 repeatability fails or full feature/prediction output is nonfinite/incomplete/misaligned",
   "prediction/derived recipe/metric scorer contract fails or any target is read before all prediction bundles are sealed",
   "a future target or target/label key appears in any predictor input",
  ],
  "performance_based_failure_or_profile_selection":False,
  "scope_at_lock":{
   "current_profile_full_dev_features_extracted":False,"current_profile_dev_scores_revealed":False,
   "core_b_final_rgb_extracted_by_v2":False,"core_b_final_future_targets_materialized_by_v2":False,
   "new_training_runs":0,"supervisor_approval_asserted":False,
  },
 }
 # Ignore the deliberately present manifest/language-only final target frame indices; values remain unopened.
 if prior_lock:
  backup=TARGET.with_name(TARGET.name+".pre-revision-"+str(prior_lock.get("lock_revision",1))+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".bak")
  shutil.copy2(TARGET,backup)
 else:
  backup=None
 atomic_json(TARGET,lock)
 lock_sha=sha(TARGET)
 if prior_lock:
  log_path=OUT/"core_b/CURRENT_PROFILE_CANDIDATE_LOCK_REVISION_LOG.json"
  history=json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"schema_version":1,"revisions":[]}
  history["revisions"].append({"revision":lock["lock_revision"],"previous_lock_sha256":prior_sha,"new_lock_sha256":lock_sha,
    "reason":args.revision_reason,"backup_path":str(backup.resolve()),"created_utc":lock["locked_utc"],
    "dev_features_extracted_before_revision":False,"current_profile_dev_scores_revealed_before_revision":False})
  atomic_json(log_path,history)
 print(json.dumps({"status":lock["status"],"path":str(TARGET),"sha256":lock_sha,"lock_revision":lock["lock_revision"],
                   "backup":str(backup.resolve()) if backup else None,
                   "model_cells":sum(len(v) for v in cells.values()),"dev_rows":len(dev_input_rows),
                   "final_rows":len(final_input_rows),"code_files":len(code),
                   "full_dev_features_before_lock":False,"final_targets_before_lock":False},indent=2,sort_keys=True))
 return 0

if __name__=="__main__":raise SystemExit(main())
