from __future__ import annotations
import os
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'))

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,default=str)+chr(10),encoding="utf-8")

def source_item(path,role):
 p=Path(path)
 if not p.is_file():raise FileNotFoundError(p)
 return {"role":role,"path":str(p),"bytes":p.stat().st_size,"observed_sha256":sha(p),"read_only":True}

def main():
 now=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 protocol_path=ROOT/"protocol/PROTOCOL_LOCKED.json"
 protocol=read(protocol_path)
 amendment_path=ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_04.json"
 amendment=read(amendment_path)
 if protocol.get("status")!="LOCKED_BEFORE_FORMAL":raise AssertionError("expected Amendment 02 preformal lock")
 if sha(protocol_path)!=amendment["previous_protocol_sha256"]:raise AssertionError("Amendment 03 previous protocol SHA mismatch")
 if sha(ROOT/"code/train_query_fit.py")!=amendment["updated_runner_sha256"]:raise AssertionError("runner SHA differs from Amendment 03")
 for path,key in [
  (ROOT/"protocol/MATRIX_4.csv","matrix_sha256_unchanged"),
  (ROOT/"protocol/MATCHED_BASELINE_4.csv","matched_baseline_sha256_unchanged"),
  (ROOT/"protocol/QUERY_TABLES_AND_HASHES.json","query_tables_json_sha256_unchanged"),
  (ROOT/"protocol/HADAMARD_QUERY_TABLES.npz","query_tables_npz_sha256_unchanged"),
 ]:
  if sha(path)!=amendment[key]:raise AssertionError(f"Amendment 03 locked content changed: {path}")
 stage0=read(ROOT/"protocol/B_STAGE0_ACCEPTANCE.json")
 gate=read(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json")
 if stage0.get("status")!="PASS" or stage0.get("accepted_cells")!=4 or stage0.get("all_four_cells_accepted") is not True:
  raise AssertionError("Stage0 4/4 acceptance is missing")
 if gate.get("status")!="PASS" or gate.get("formal_runs_authorized_by_gate") is not True:
  raise AssertionError("post-Stage0 resource gate is not PASS")
 if gate.get("protocol_sha256")!=stage0.get("protocol_sha256"):
  raise AssertionError("Stage0 acceptance/resource gate protocol hashes differ")
 if gate["formal_projection"]["projected_total_cpu_wall_seconds"]>protocol["resource_budget"]["max_cpu_wall_seconds"]:
  raise AssertionError("Stage0 cost projection exceeds locked CPU wall budget")
 if any(p.is_dir() for p in (ROOT/"runs/formal").iterdir()):
  raise AssertionError("formal run directory exists; refusing preformal relock")
 failed=(ROOT/"logs/FORMAL_ATTEMPT_003.log").read_text(encoding="utf-8")
 if "startup_and_input_seconds" not in failed or "KeyError" not in failed:
  raise AssertionError("expected formal preflight failure log missing")
 stage0_files=[]
 for result in stage0["results"]:
  run=ROOT/"runs/stage0"/result["run_id"]
  record=read(run/"STAGE0_COMPLETE.json")
  if record.get("status")!="PASS" or record.get("optimizer_updates")!=4:
   raise AssertionError("Stage0 run record changed or is incomplete")
  checkpoint=run/"checkpoint_stage0.pt"
  prediction=run/"PREDICTIONS_STEP_0004.npz"
  if sha(checkpoint)!=result["final_checkpoint_sha256"] or sha(prediction)!=result["prediction_sha256"]:
   raise AssertionError("Stage0 checkpoint or prediction hash mismatch")
  reload=read(run/"FRESH_PROCESS_RELOAD.json")
  if reload.get("status")!="PASS" or not reload.get("prediction_bitwise_equal"):
   raise AssertionError("Stage0 fresh process reload failed")
  for f in ["STAGE0_COMPLETE.json","checkpoint_stage0.pt","PREDICTIONS_STEP_0004.npz",
            "METRICS.csv","SAMPLE_METRICS.csv","FRESH_PROCESS_RELOAD.json","FRESH_PROCESS_RELOAD_COMMAND.json"]:
   p=run/f
   if not p.is_file():raise FileNotFoundError(p)
   stage0_files.append({"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size})

 registry_path=ROOT/"SOURCE_REGISTRY.json"
 registry=read(registry_path)
 source_files=registry["source_files"]
 index={x["path"]:i for i,x in enumerate(source_files)}
 extra=[
  (ROOT/"code/train_query_fit.py","corrected formal CPU runner"),
  (ROOT/"code/train_query_fit.py.pre-resource-projection-path-fix-20261007T122523Z.bak","runner source before Stage0 resource projection fix"),
  (ROOT/"code/train_query_fit.py.pre-stage0-elapsed-field-fix-20261007T122034Z.bak","runner source before first Stage0 elapsed field fix"),
  (ROOT/"code/accept_stage0_cost.py","post-Stage0 CPU resource gate implementation"),
  (ROOT/"code/accept_stage0_cost.py.pre-baseline-sum-fix-20261007T121905Z.bak","cost gate before baseline summation fix"),
  (ROOT/"code/accept_stage0_cost.py.pre-old-baseline-runtime-20261007T121850Z.bak","cost gate before historical runtime pointer fix"),
  (ROOT/"code/lock_before_formal_amendment_02.py","Stage0 acceptance relocker"),
  (ROOT/"code/lock_before_formal_amendment_03.py","Amendment 03 formal relocker"),
  (ROOT/"code/lock_before_formal_amendment_03.py","Amendment 03 relocker source"),
  (ROOT/"code/lock_before_formal_amendment_04.py","Amendment 04 preformal relocker"),
  (ROOT/"code/lock_before_formal_amendment_04.py.pre-source-list-fix-20261007T123020Z.bak","Amendment 04 relocker snapshot before final source list"),
  (ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_03.json","formal runner field amendment"),
  (ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_04.json","formal resource field amendment"),
  (ROOT/"code/train_query_fit.py.pre-startup-reserve-key-fix-20261007T123006Z.bak","runner source before startup reserve field fix"),
  (ROOT/"logs/FORMAL_ATTEMPT_003.log","third formal preflight failure before run creation"),
  (ROOT/"protocol/PROTOCOL_LOCKED.json.pre-amendment-04-20261007T123020Z.bak","previous protocol before Amendment 04"),
  (ROOT/"SOURCE_REGISTRY.json.pre-amendment-04-20261007T123020Z.bak","source registry before Amendment 04"),
  (ROOT/"SOURCE_ACCEPTANCE.json.pre-amendment-04-20261007T123020Z.bak","source acceptance before Amendment 04"),
  (ROOT/"protocol/SOURCE_REGISTRY.json.pre-amendment-04-20261007T123020Z.bak","protocol source registry before Amendment 04"),
  (ROOT/"logs/FORMAL_ATTEMPT_001.log","first formal preflight failure before run creation"),
  (ROOT/"logs/FORMAL_ATTEMPT_002.log","second formal preflight failure before run creation"),
  (ROOT/"logs/PREFORMAL_RELOCK_ATTEMPT_001.log","first preformal relock source-path failure"),
  (ROOT/"protocol/PROTOCOL_LOCKED.json.pre-amendment-03-20261007T122554Z.bak","prior locked protocol preserved before Amendment 03"),
  (ROOT/"SOURCE_REGISTRY.json.pre-amendment-03-20261007T122554Z.bak","prior source registry preserved before Amendment 03"),
  (ROOT/"SOURCE_ACCEPTANCE.json.pre-amendment-03-20261007T122554Z.bak","prior acceptance preserved before Amendment 03"),
 ]
 extra.extend((Path(x["path"]),"accepted Stage0 artifact") for x in stage0_files)
 added=0;refreshed=0
 for path,role in extra:
  item=source_item(path,role)
  if str(path) in index:
   i=index[str(path)]
   prior=source_files[i]
   expected=prior.get("manifest_expected_sha256")
   if expected is not None and expected!=item["observed_sha256"]:
    raise AssertionError("preexisting source manifest hash changed: "+str(path))
   source_files[i]=item
   refreshed+=1
  else:
   index[str(path)]=len(source_files);source_files.append(item);added+=1
 registry["source_files"]=source_files
 registry["integrity"]["stage0_acceptance"]="PASS_4_OF_4"
 registry["integrity"]["stage0_resource_gate"]="PASS"
 registry["integrity"]["formal_preflight_optimizer_updates"]=0
 registry["integrity"]["preformal_amendment_04"]="PASS; runner reads startup_and_input_reserve_seconds"
 registry["integrity"]["formal_relock_sources_added"]=added
 registry["integrity"]["formal_relock_sources_refreshed"]=refreshed
 for path in [ROOT/"SOURCE_REGISTRY.json",ROOT/"protocol/SOURCE_REGISTRY.json"]:
  write(path,registry)
 registry_sha=sha(ROOT/"SOURCE_REGISTRY.json")

 accept=read(ROOT/"SOURCE_ACCEPTANCE.json")
 accept["status"]="PASS_STAGE0_AND_FORMAL_PRELOCK_ACCEPTANCE"
 accept["source_registry_sha256"]=registry_sha
 accept["checks"]["stage0_acceptance"]="PASS_4_OF_4"
 accept["checks"]["stage0_fresh_process_reloads"]="PASS_4_OF_4_BITWISE"
 accept["checks"]["stage0_resource_gate"]="PASS"
 accept["checks"]["formal_attempt_001_updates"]=0
 accept["checks"]["formal_attempt_002_updates"]=0
 accept["checks"]["formal_run_directories_before_amendment_04"]=0
 accept["checks"]["formal_optimizer_updates_before_amendment_04"]=0
 accept["checks"]["pre_formal_lock_amendment"]="AMENDMENT_04"
 accept["checks"]["matrix_baseline_query_hashes_unchanged"]=True
 accept["checks"]["dev_targets_opened"]=False
 accept["checks"]["final_or_heldout_access"]=0
 accept["checks"]["gpu_queries_use_lease"]=0
 accept["checks"]["new_formal_training_started"]=False
 accept.setdefault("notes",[]).append("Amendment 03 corrected the pre-formal resource gate field path. The formal preflight invocation stopped before training while reading a resource reserve key; Stage0 4/4 and its cost gate remain accepted. The runner now reads the exact reserve field under Amendment 04.")
 write(ROOT/"SOURCE_ACCEPTANCE.json",accept)
 acceptance_sha=sha(ROOT/"SOURCE_ACCEPTANCE.json")

 source_paths={x["path"] for x in registry["source_files"]}
 explicit=[
  ROOT/"SOURCE_REGISTRY.json",ROOT/"SOURCE_ACCEPTANCE.json",
  ROOT/"CONTEXT_RESTORED.zh-CN.md",ROOT/"OUTCOME_EXPOSURE.md",
  ROOT/"DEVELOPMENT_REFERENCE_FREEZE.json",ROOT/"forensics/A_AUDIT_ACCEPTANCE.json",
  ROOT/"forensics/FUSION_QUERY_PATH_AUDIT.json",ROOT/"protocol/A_AUDIT_LOCK.json",
  ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_01.json",ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_02.json",
  ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_03.json",
  ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_04.json",
  ROOT/"protocol/RESOURCE_GATE_PRE_STAGE0.json",ROOT/"protocol/B_STAGE0_ACCEPTANCE.json",
  ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json",ROOT/"protocol/MATRIX_4.csv",
  ROOT/"protocol/MATCHED_BASELINE_4.csv",ROOT/"protocol/QUERY_TABLES_AND_HASHES.json",
  ROOT/"protocol/HADAMARD_QUERY_TABLES.npz",ROOT/"protocol/HADAMARD_QUERY_TABLES.csv",
  ROOT/"protocol/CPU_PREFLIGHT.json",
 ]
 for p in explicit:
  if not p.is_file():raise FileNotFoundError(p)
  source_paths.add(str(p))
 source_hashes={p:sha(Path(p)) for p in sorted(source_paths)}
 amendment_sha=sha(ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_04.json")
 protocol=dict(protocol)
 protocol["status"]="LOCKED_BEFORE_FORMAL"
 protocol["locked_utc"]=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 protocol["source_registry_sha256"]=registry_sha
 protocol["source_acceptance_sha256"]=acceptance_sha
 protocol["source_hashes"]=source_hashes
 protocol["pre_formal_lock_amendment_04"]={
  "id":amendment["amendment_id"],"path":str(ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_04.json"),
  "sha256":amendment_sha,"previous_protocol_sha256":amendment["previous_protocol_sha256"],
  "corrected_runner_sha256":amendment["updated_runner_sha256"],
  "stage0_acceptance_sha256":sha(ROOT/"protocol/B_STAGE0_ACCEPTANCE.json"),
  "stage0_resource_gate_sha256":sha(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json"),
  "matrix_baseline_query_unchanged":True,"formal_optimizer_updates_before_relock":0,
 }
 protocol["stage0_acceptance_sha256"]=sha(ROOT/"protocol/B_STAGE0_ACCEPTANCE.json")
 protocol["stage0_resource_gate_sha256"]=sha(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json")
 protocol["resource_budget"]["stage0_elapsed_seconds_sum"]=gate["stage0_elapsed_seconds_sum"]
 protocol["resource_budget"]["post_stage0_projected_formal_seconds"]=gate["formal_projection"]["projected_total_cpu_wall_seconds"]
 protocol["resource_budget"]["post_stage0_resource_gate_sha256"]=sha(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json")
 if protocol["matrix_sha256"]!=sha(ROOT/"protocol/MATRIX_4.csv") or protocol["matched_baseline_sha256"]!=sha(ROOT/"protocol/MATCHED_BASELINE_4.csv"):
  raise AssertionError("Amendment 03 changed matrix/baseline")
 write(ROOT/"protocol/PROTOCOL_LOCKED.json",protocol)
 print(json.dumps({"status":protocol["status"],"protocol_sha256":sha(ROOT/"protocol/PROTOCOL_LOCKED.json"),
  "matrix_sha256":protocol["matrix_sha256"],"matched_baseline_sha256":protocol["matched_baseline_sha256"],
  "source_registry_sha256":registry_sha,"source_acceptance_sha256":acceptance_sha,
  "source_hash_count":len(source_hashes),"stage0_files":len(stage0_files),"formal_dirs":0,"formal_updates":0,
  "matrix_query_baseline_unchanged":True},ensure_ascii=False,indent=2))

if __name__=="__main__":
 main()

