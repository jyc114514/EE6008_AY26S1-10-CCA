from __future__ import annotations
import os
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'))
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True,default=str)+chr(10),encoding="utf-8")

def item(path,role):
 path=Path(path)
 if not path.is_file():raise FileNotFoundError(path)
 return {"role":role,"path":str(path),"bytes":path.stat().st_size,"observed_sha256":sha(path),"read_only":True}

def main():
 now=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 protocol_path=ROOT/"protocol/PROTOCOL_LOCKED.json"
 protocol=read(protocol_path)
 amendment_path=ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_02.json"
 amendment=read(amendment_path)
 if protocol.get("status")!="LOCKED_BEFORE_STAGE0":raise AssertionError("expected previous pre-Stage0 lock")
 if sha(protocol_path)!=amendment["previous_protocol_sha256"]:raise AssertionError("Amendment 02 previous protocol hash mismatch")
 if sha(ROOT/"code/train_query_fit.py")!=amendment["updated_runner_sha256"]:raise AssertionError("corrected runner hash mismatch")
 if sha(ROOT/"protocol/MATRIX_4.csv")!=amendment["matrix_sha256_unchanged"]:raise AssertionError("matrix changed since amendment")
 if sha(ROOT/"protocol/MATCHED_BASELINE_4.csv")!=amendment["matched_baseline_sha256_unchanged"]:raise AssertionError("matched baseline changed since amendment")
 if sha(ROOT/"protocol/QUERY_TABLES_AND_HASHES.json")!=amendment["query_tables_json_sha256_unchanged"]:raise AssertionError("query table metadata changed")
 if sha(ROOT/"protocol/HADAMARD_QUERY_TABLES.npz")!=amendment["query_tables_npz_sha256_unchanged"]:raise AssertionError("query table file changed")
 stage0=read(ROOT/"protocol/B_STAGE0_ACCEPTANCE.json")
 gate=read(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json")
 if stage0.get("status")!="PASS" or stage0.get("accepted_cells")!=4 or stage0.get("all_four_cells_accepted") is not True:
  raise AssertionError("Stage0 acceptance is not PASS 4/4")
 if gate.get("status")!="PASS" or gate.get("formal_runs_authorized_by_gate") is not True:
  raise AssertionError("post-Stage0 resource gate is not PASS")
 if gate.get("protocol_sha256")!=sha(protocol_path) or gate.get("matrix_sha256")!=sha(ROOT/"protocol/MATRIX_4.csv"):
  raise AssertionError("Stage0 resource gate does not match the preceding locked protocol")
 formal_root=ROOT/"runs/formal"
 if any(p.is_dir() for p in formal_root.iterdir()):
  raise AssertionError("formal run directory already exists; refusing preformal relock")
 stage0_artifacts=amendment["stage0_files"]
 for record in stage0_artifacts:
  if sha(Path(record["path"]))!=record["sha256"]:raise AssertionError("accepted Stage0 artifact changed: "+record["path"])

 registry_path=ROOT/"SOURCE_REGISTRY.json"
 registry=read(registry_path)
 source_files=registry["source_files"]
 path_index={x["path"]:i for i,x in enumerate(source_files)}
 sources=[
  (ROOT/"code/train_query_fit.py","corrected preformal CPU runner"),
  (ROOT/"code/train_query_fit.py.pre-stage0-elapsed-field-fix-20261007T122034Z.bak","runner source before Stage0 elapsed field fix"),
  (ROOT/"code/train_query_fit.py.pre-interop-duplicate-fix-20261007T121149Z.bak","runner source before interop initialization fix"),
  (ROOT/"code/accept_stage0_cost.py","post-Stage0 CPU resource gate"),
  (ROOT/"code/accept_stage0_cost.py.pre-old-baseline-runtime-20261007T121850Z.bak","resource gate source before baseline runtime pointer fix"),
  (ROOT/"code/accept_stage0_cost.py.pre-baseline-sum-fix-20261007T121905Z.bak","resource gate source before elapsed sum fix"),
  (ROOT/"code/lock_before_formal_amendment_02.py","preformal Amendment 02 lock implementation"),
  (ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_02.json","preformal runner hash amendment"),
  (ROOT/"logs/FORMAL_ATTEMPT_001.log","preserved formal preflight failure before run creation"),
  (ROOT/"logs/STAGE0_RESOURCE_GATE.log","preserved failed resource gate calculation"),
  (ROOT/"logs/STAGE0_RESOURCE_GATE_ATTEMPT_002.log","accepted post-Stage0 resource gate run"),
  (ROOT/"protocol/B_STAGE0_ACCEPTANCE.json","accepted 4/4 Stage0 gate"),
  (ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json","accepted post-Stage0 CPU/disk gate"),
  (ROOT/"protocol/PROTOCOL_LOCKED.json.pre-preformal-amendment-02-20261007T122034Z.bak","pre-Stage0 lock snapshot before Amendment 02"),
  (ROOT/"SOURCE_ACCEPTANCE.json.pre-preformal-amendment-02-20261007T122034Z.bak","source acceptance snapshot before Amendment 02"),
  (ROOT/"SOURCE_REGISTRY.json.pre-preformal-amendment-02-20261007T122034Z.bak","source registry snapshot before Amendment 02"),
  (ROOT/"protocol/SOURCE_REGISTRY.json.pre-preformal-amendment-02-20261007T122034Z.bak","protocol source registry snapshot before Amendment 02"),
  (ROOT/"logs/FORMAL_ATTEMPT_001.log","formal attempt preflight failure"),
 ]
 for record in stage0_artifacts:
  sources.append((Path(record["path"]),"accepted Stage0 output artifact"))
 added=0;refreshed=0
 for path,role in sources:
  new=item(path,role)
  if str(path) in path_index:
   index=path_index[str(path)]
   old=source_files[index]
   expected=old.get("manifest_expected_sha256")
   if expected is not None and expected!=new["observed_sha256"]:raise AssertionError("preexisting expected source hash changed: "+str(path))
   source_files[index]=new;refreshed+=1
  else:
   path_index[str(path)]=len(source_files);source_files.append(new);added+=1
 registry["source_files"]=source_files
 registry["integrity"]["stage0_acceptance"]="PASS_4_OF_4"
 registry["integrity"]["stage0_resource_gate"]="PASS"
 registry["integrity"]["preformal_amendment"]="CORE_B_FUSION_QUERY_MECHANISM_PRE_FORMAL_AMENDMENT_02"
 registry["integrity"]["formal_optimizer_updates"]=0
 registry["integrity"]["formal_run_dirs"]=0
 registry["integrity"]["preformal_sources_added"]=added
 registry["integrity"]["preformal_sources_refreshed"]=refreshed
 for p in (ROOT/"SOURCE_REGISTRY.json",ROOT/"protocol/SOURCE_REGISTRY.json"):write(p,registry)
 registry_sha=sha(ROOT/"SOURCE_REGISTRY.json")

 accept=read(ROOT/"SOURCE_ACCEPTANCE.json")
 accept["status"]="PASS_STAGE0_AND_PREFORMAL_SOURCE_ACCEPTANCE"
 accept["source_registry_sha256"]=registry_sha
 accept["checks"]["stage0_acceptance"]="PASS_4_OF_4"
 accept["checks"]["stage0_fresh_process_reloads"]="PASS_4_OF_4_BITWISE"
 accept["checks"]["stage0_resource_gate"]="PASS"
 accept["checks"]["formal_run_directories_before_amendment_02"]=0
 accept["checks"]["formal_optimizer_updates_before_amendment_02"]=0
 accept["checks"]["pre_formal_runner_hash_fix"]="PASS"
 accept["checks"]["pre_formal_lock_amendment"]="AMENDMENT_02"
 accept["checks"]["matrix_baseline_query_hashes_unchanged"]=True
 accept["checks"]["dev_targets_opened"]=False
 accept["checks"]["final_or_heldout_access"]=0
 accept["checks"]["gpu_queries_use_lease"]=0
 accept["checks"]["new_training_started"]=False
 accept.setdefault("notes",[]).append("Stage0 passed 4/4. The first formal invocation stopped before data loading because it referenced an incorrect Stage0 elapsed field; no formal run directory or optimizer update was created. Amendment 02 corrects the runner field and preserves the Stage0 result and unchanged matrix/query/baseline hashes.")
 write(ROOT/"SOURCE_ACCEPTANCE.json",accept)
 acceptance_sha=sha(ROOT/"SOURCE_ACCEPTANCE.json")

 # Refresh CPU preflight and cost gate are immutable inputs to this preformal relock.
 pre=read(ROOT/"protocol/RESOURCE_GATE_PRE_STAGE0.json")
 if pre.get("status")!="PASS":raise AssertionError("pre-Stage0 resource gate no longer passes")
 resource=read(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json")
 if resource.get("status")!="PASS":raise AssertionError("Stage0 resource gate no longer passes")
 source_hash_paths={x["path"] for x in registry["source_files"]}
 explicit=[
  ROOT/"SOURCE_REGISTRY.json",ROOT/"SOURCE_ACCEPTANCE.json",
  ROOT/"CONTEXT_RESTORED.zh-CN.md",ROOT/"OUTCOME_EXPOSURE.md",
  ROOT/"DEVELOPMENT_REFERENCE_FREEZE.json",ROOT/"forensics/A_AUDIT_ACCEPTANCE.json",
  ROOT/"forensics/FUSION_QUERY_PATH_AUDIT.json",ROOT/"protocol/A_AUDIT_LOCK.json",
  ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_01.json",ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_02.json",
  ROOT/"protocol/RESOURCE_GATE_PRE_STAGE0.json",ROOT/"protocol/B_STAGE0_ACCEPTANCE.json",
  ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json",ROOT/"protocol/MATRIX_4.csv",
  ROOT/"protocol/MATCHED_BASELINE_4.csv",ROOT/"protocol/QUERY_TABLES_AND_HASHES.json",
  ROOT/"protocol/HADAMARD_QUERY_TABLES.npz",ROOT/"protocol/HADAMARD_QUERY_TABLES.csv",
  ROOT/"protocol/CPU_PREFLIGHT.json",
 ]
 for p in explicit:
  if not p.is_file():raise FileNotFoundError(p)
  source_hash_paths.add(str(p))
 source_hashes={p:sha(Path(p)) for p in sorted(source_hash_paths)}
 if source_hashes[str(ROOT/"code/train_query_fit.py")]!=amendment["updated_runner_sha256"]:
  raise AssertionError("runner source hash in Amendment 02 differs from registry source map")
 protocol=dict(protocol)
 protocol["status"]="LOCKED_BEFORE_FORMAL"
 protocol["locked_utc"]=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
 protocol["source_registry_sha256"]=registry_sha
 protocol["source_acceptance_sha256"]=acceptance_sha
 protocol["source_hashes"]=source_hashes
 protocol["pre_formal_lock_amendment"]={
  "id":amendment["amendment_id"],"path":str(ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_02.json"),
  "sha256":sha(ROOT/"protocol/PROTOCOL_LOCK_AMENDMENT_02.json"),
  "previous_protocol_sha256":amendment["previous_protocol_sha256"],
  "corrected_runner_sha256":amendment["updated_runner_sha256"],
  "stage0_acceptance_sha256":sha(ROOT/"protocol/B_STAGE0_ACCEPTANCE.json"),
  "stage0_resource_gate_sha256":sha(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json"),
  "matrix_baseline_query_unchanged":True,"formal_optimizer_updates_before_relock":0,
 }
 protocol["stage0_acceptance_sha256"]=sha(ROOT/"protocol/B_STAGE0_ACCEPTANCE.json")
 protocol["stage0_resource_gate_sha256"]=sha(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json")
 protocol["resource_budget"]["stage0_elapsed_seconds_sum"]=resource["stage0_elapsed_seconds_sum"]
 protocol["resource_budget"]["post_stage0_projected_formal_seconds"]=resource["formal_projection"]["projected_total_cpu_wall_seconds"]
 protocol["resource_budget"]["post_stage0_resource_gate_sha256"]=sha(ROOT/"protocol/B_RESOURCE_GATE_STAGE0.json")
 if protocol["matrix_sha256"]!=sha(ROOT/"protocol/MATRIX_4.csv") or protocol["matched_baseline_sha256"]!=sha(ROOT/"protocol/MATCHED_BASELINE_4.csv"):
  raise AssertionError("preformal relock changed the four-cell matrix or baseline ledger")
 write(ROOT/"protocol/PROTOCOL_LOCKED.json",protocol)
 print(json.dumps({
  "status":protocol["status"],"protocol_sha256":sha(ROOT/"protocol/PROTOCOL_LOCKED.json"),
  "matrix_sha256":protocol["matrix_sha256"],"matched_baseline_sha256":protocol["matched_baseline_sha256"],
  "source_registry_sha256":registry_sha,"source_acceptance_sha256":acceptance_sha,
  "source_hash_count":len(source_hashes),"stage0_cells":4,"formal_run_dirs":0,
  "formal_optimizer_updates_before_relock":0,"matrix_query_baseline_unchanged":True,
 },ensure_ascii=False,indent=2))

if __name__=="__main__":
 main()

