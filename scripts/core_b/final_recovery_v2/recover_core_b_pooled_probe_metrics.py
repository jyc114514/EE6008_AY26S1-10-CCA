#!/usr/bin/env python3
"""Recover pooled-grid comparisons from the partial train-only P1-P6 feature archive."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT=PROJECT/"reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1RUN=PROJECT/os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
D3ROOT=PROJECT/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
MANIFEST=D3ROOT/"d3/cache/D3_CACHE_MANIFEST.json"
PARTIAL=RUN/"core_b/gpu_probes/CURRENT_PROFILE_BRIDGE_FEATURES.npz"
BRIDGE=V1RUN/"replay/core_b/core_b_bridge_manifest.csv"
OUTPUT=OUT/"core_b/CORE_B_GPU_POOLED_PROBE_RECOVERY_FROM_PARTIAL.json"
IDS=("e001726_a01_f000149","e001769_a00_f000119")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a,b):
 x=np.asarray(a,dtype=np.float32);y=np.asarray(b,dtype=np.float32);d=np.abs(x.astype(np.float64)-y.astype(np.float64));
 return {"shape":list(x.shape),"max_abs_difference":float(d.max()),"mean_abs_difference":float(d.mean()),"p95_abs_difference":float(np.quantile(d,.95)),"p99_abs_difference":float(np.quantile(d,.99)),"relative_l2_difference":float(np.linalg.norm(d.ravel())/max(float(np.linalg.norm(y.astype(np.float64).ravel())),1e-30)),"allclose_atol_1e-5_rtol_1e-5":bool(np.allclose(x,y,atol=1e-5,rtol=1e-5))}
def main():
 bridge=list(csv.DictReader(BRIDGE.open(encoding="utf-8-sig",newline="")));by={r["anchor_id"]:r for r in bridge if r["dataset_role"]=="train" and r["anchor_id"] in IDS}
 if set(by)!=set(IDS):raise RuntimeError("fixed train bridge IDs missing")
 manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));q0=[]
 for anchor in IDS:
  row=by[anchor];idx=int(row["cache_index"]);shard=next(x for x in manifest["shards"] if int(x["start_row"])<=idx<int(x["end_row"]))
  values=np.load(Path(shard["array"]),mmap_mode="r",allow_pickle=False)[idx-int(shard["start_row"])]
  meta=json.loads(Path(shard["meta"]).read_text(encoding="utf-8"));local=idx-int(shard["start_row"])
  if str(meta["anchor_ids"][local])!=anchor or int(meta["episode_index"][local])!=int(row["episode_id"]):raise RuntimeError(f"Q0 row identity mismatch {anchor}")
  q0.append(np.asarray(values,dtype=np.float32))
 q0=np.stack(q0).reshape(4,16,4,768)
 with np.load(PARTIAL,allow_pickle=False) as z:
  names=list(z.files);features={k:np.asarray(z[k],dtype=np.float32) for k in names}
 if set(names)!={"P1_D3_BF16_BATCH1_DEFAULT","P2_FP32_BATCH1_DEFAULT","P3_FP32_BATCH1_MATH_SDPA","P4_FP32_BATCH1_TF32_OFF","P5_CURRENT_PROFILE_V2_REPEAT_0","P6_CURRENT_PROFILE_V2_REPEAT_1"}:
  raise RuntimeError(f"partial feature keys differ: {names}")
 if any(x.shape!=(4,16,4,768) or not np.isfinite(x).all() for x in features.values()):raise RuntimeError("partial feature array shape/finite check failed")
 p1=features["P1_D3_BF16_BATCH1_DEFAULT"];out={}
 for name,arr in features.items():
  out[name]={"pooled_vs_q0":[stats(arr[i],q0[i]) for i in range(4)],"pooled_vs_p1":[stats(arr[i],p1[i]) for i in range(4)],"feature_sha256_by_input":[hashlib.sha256(np.ascontiguousarray(arr[i]).tobytes()).hexdigest() for i in range(4)],"all_finite":True}
 repeat=features["P6_CURRENT_PROFILE_V2_REPEAT_1"]
 stable=[stats(repeat[i],features["P5_CURRENT_PROFILE_V2_REPEAT_0"][i]) for i in range(4)]
 receipt={"schema_version":1,"status":"PASS_POOLED_METRICS_RECOVERED_FROM_PARTIAL_GPU3_ARCHIVE","created_utc":datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z"),"scope":"two fixed train rows; CPU comparison of already-produced pooled grids and existing Q0 slices; no new GPU run, final media, targets, labels, or training.","partial_gpu_archive_path":str(PARTIAL),"partial_gpu_archive_sha256":sha(PARTIAL),"q0_manifest_sha256":sha(MANIFEST),"fixed_anchor_ids":list(IDS),"variant_results":out,"candidate_profile_repeatability":{"P5_vs_P6_pooled_grid":stable,"allclose_all_inputs":all(x["allclose_atol_1e-5_rtol_1e-5"] for x in stable)},"raw_token_metrics_persisted":False,"raw_token_metrics_recovery":"Raw token arrays/comparison summaries were not serialized in the failed attempt; the corrected receipt retry is required to record them.","result_json_from_attempt01_written":False,"new_training_runs":0,"final_payload_read":False,"future_targets_materialized":False}
 OUTPUT.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"status":receipt["status"],"candidate_repeatable":receipt["candidate_profile_repeatability"]["allclose_all_inputs"],"output":str(OUTPUT)},ensure_ascii=False))
if __name__=="__main__":main()
