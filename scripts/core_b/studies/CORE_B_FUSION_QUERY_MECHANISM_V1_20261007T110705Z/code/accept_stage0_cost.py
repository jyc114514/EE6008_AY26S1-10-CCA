from __future__ import annotations
import os
import csv
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'))
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
LEARN=PROJECT/"reports/benchmarks/CORE_B_COLLAPSE_LEARNABILITY_V1_20261007T060954Z"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(p.read_text(encoding="utf-8"))

def main():
 protocol=read(ROOT/"protocol/PROTOCOL_LOCKED.json")
 stage0=read(ROOT/"protocol/B_STAGE0_ACCEPTANCE.json")
 pre=read(ROOT/"protocol/RESOURCE_GATE_PRE_STAGE0.json")
 if protocol.get("status")!="LOCKED_BEFORE_STAGE0":raise AssertionError("protocol is not Stage0 locked")
 if stage0.get("status")!="PASS" or stage0.get("accepted_cells")!=4 or stage0.get("all_four_cells_accepted") is not True:
  raise AssertionError("Stage0 is not accepted 4/4")
 if len(stage0.get("results",[]))!=4 or any(x.get("optimizer_updates")!=4 or not x.get("fresh_process_prediction_bitwise_equal") for x in stage0["results"]):
  raise AssertionError("one or more Stage0 cells lack exact four-update and reload acceptance")
 if pre.get("status")!="PASS":raise AssertionError("pre-Stage0 resource gate failed")
 p95_total=sum(float(x["p95_update_seconds"]) for x in stage0["results"])
 mean_total=sum(float(x["mean_update_seconds"]) for x in stage0["results"])
 p95_formal_projection=p95_total*2000*1.25
 baseline_total=sum(read_baselines())
 baseline_projection=baseline_total*2.0
 formal_train_projection=max(p95_formal_projection,baseline_projection)
 post=float(protocol["resource_budget"]["post_run_overhead_seconds"])
 projected_formal=formal_train_projection+post
 stage0_elapsed=float(stage0["elapsed_seconds_sum"])
 startup=float(pre["projection_components"]["startup_and_input_seconds"])
 projected_total=stage0_elapsed+projected_formal+startup
 max_wall=float(protocol["resource_budget"]["max_cpu_wall_seconds"])
 usage=shutil.disk_usage(PROJECT/"reports/benchmarks")
 free_fraction=usage.free/usage.total
 root_bytes=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
 max_bytes=int(protocol["resource_budget"]["max_new_output_bytes"])
 estimated_future=int(pre["estimated_new_output_bytes"])
 projected_root_bytes=root_bytes+estimated_future
 passed=(projected_total<=max_wall and free_fraction>=float(protocol["resource_budget"]["minimum_disk_free_fraction"])
         and projected_root_bytes<=max_bytes)
 result={
  "schema_version":1,"status":"PASS" if passed else "FAIL",
  "created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
  "gate_script_path":str(Path(__file__).resolve()),"gate_script_sha256":sha(Path(__file__).resolve()),
  "protocol_sha256":sha(ROOT/"protocol/PROTOCOL_LOCKED.json"),
  "matrix_sha256":sha(ROOT/"protocol/MATRIX_4.csv"),
  "query_h30_sha256":protocol["query_table_sha256_h30"],
  "stage0_acceptance_sha256":sha(ROOT/"protocol/B_STAGE0_ACCEPTANCE.json"),
  "stage0_cells":4,"stage0_updates_each":4,"stage0_elapsed_seconds_sum":stage0_elapsed,
  "stage0_fresh_process_reloads":"PASS_4_OF_4",
  "formal_projection":{
   "stage0_observed_update_p95_sum_seconds":p95_total,
   "stage0_p95_projected_2000_update_total_seconds":p95_formal_projection,
   "historical_four_baseline_elapsed_sum_seconds":baseline_total,
   "historical_projection_multiplier":2.0,
   "historical_projected_formal_training_seconds":baseline_projection,
   "selected_formal_training_projection_seconds":formal_train_projection,
   "post_run_analysis_and_figure_seconds":post,
   "startup_and_input_reserve_seconds":startup,
   "projected_total_cpu_wall_seconds":projected_total,
   "limit_cpu_wall_seconds":max_wall,
  },
  "disk":{
   "free_bytes":usage.free,"total_bytes":usage.total,"free_fraction":free_fraction,
   "minimum_fraction":float(protocol["resource_budget"]["minimum_disk_free_fraction"]),
   "current_new_root_bytes":root_bytes,"estimated_future_output_bytes":estimated_future,
   "projected_new_root_bytes":projected_root_bytes,"max_new_output_bytes":max_bytes,
  },
  "runtime":{"device":"cpu","torch_threads":1,"gpu_queries":0,"gpu_use":0,"gpu_lease":0},
  "final_or_heldout_access":0,"dev_targets_opened":False,
  "formal_runs_authorized_by_gate":passed,
 }
 out=ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json"
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+chr(10),encoding="utf-8")
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if not passed: raise SystemExit(2)
 return 0

def read_baselines():
 matrix=list(csv.DictReader((ROOT/"protocol/MATCHED_BASELINE_4.csv").open(newline="",encoding="utf-8")))
 out=[]
 for row in matrix:
  run=LEARN/"runs/formal"/row["old_run_id"]
  record=json.loads((run/"RUN_COMPLETE.json").read_text(encoding="utf-8"))
  if record.get("status")!="PASS" or record.get("optimizer_updates")!=2000:
   raise AssertionError(f"old baseline runtime source is not accepted: {row['old_run_id']}")
  out.append(float(record["elapsed_seconds"]))
 return out

if __name__=="__main__":
 raise SystemExit(main())

