#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-012; source SHA-256: a0163213886fd8e7f6d1d06ed715cccd8cde0e7432c732d6ba30a6788d3dc187
"""Seal new-root code/config inputs while keeping run ledgers and results mutable."""
from __future__ import annotations
import csv,hashlib,json,shutil
from datetime import datetime,timezone
import os
from pathlib import Path
ROOT=Path(os.environ.get("EE6008_QUERY_GEOMETRY_STUDY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT=Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
ATTACH=Path(os.environ.get("EE6008_QUERY_GEOMETRY_TASK_ATTACHMENT", ".")).expanduser().resolve()
SERVER_POLICY=Path(os.environ.get("EE6008_SERVER_CODEX_POLICY", ".")).expanduser().resolve()

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 start=json.loads((ROOT/'inventory/REGISTERED_SOURCE_HASHES_AT_START.json').read_text(encoding='utf-8'))
 external=[]
 for x in start['source_rows']:
  external.append({k:x.get(k,'') for k in ['source_root','role','path','expected_sha256','observed_sha256','hash_match','bytes']})
 policy_sources=[(ATTACH,'task_authorization_attachment'),(PROJECT/'AGENTS.md','project_operating_policy'),(SERVER_POLICY,'server_operating_policy'),(PROJECT/'repo/ee6008-humanoid-vjepa/docs/operations/GPU_WATCHER_POLICY.md','shared_gpu_watcher_policy'),(PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py','shared_gpu_watcher_source')]
 seen={x['path']:x.get('observed_sha256') for x in external}
 for p,role in policy_sources:
  h=sha(p);key=str(p)
  if key in seen:
   if seen[key]!=h:raise AssertionError(f'conflicting existing external hash for {key}')
   continue
  external.append({'source_root':'OPERATING_POLICY','role':role,'path':key,'expected_sha256':h,'observed_sha256':h,'hash_match':'true','bytes':p.stat().st_size})
 local=[]
 for base in [ROOT/'code',ROOT/'tests']:
  for p in sorted(base.rglob('*.py')):
   rel=p.relative_to(ROOT).as_posix();h=sha(p)
   local.append({'source_root':'EXP_ROOT','role':'local_execution_code','path':rel,'expected_sha256':h,'observed_sha256':h,'hash_match':'true','bytes':p.stat().st_size})
 static=[
  'CONTEXT_RESTORED.zh-CN.md','CONTEXT_RESTORED.zh-CN.json','coordination/resource-snapshot.json',
  'coordination/jobs/STAGE0.jsonl','coordination/jobs/INFER_SMOKE.jsonl',
  'coordination/jobs/FORMAL_TEMPLATE_PREFERRED.jsonl','coordination/jobs/INFERENCE_TEMPLATE_PREFERRED.jsonl',
  'protocol/MATRIX12.csv','protocol/STAGE0_MATRIX.csv','protocol/REUSE_ELIGIBILITY.csv','protocol/SEED_INIT_MANIFEST.csv',
  'protocol/QUERY_TABLES_AND_HASHES.json','protocol/HADAMARD_QUERY_TABLES.npz','protocol/HADAMARD_QUERY_TABLES.csv',
  'protocol/INPUT_IDENTITIES.json','protocol/ANALYSIS_LOCKED.json','protocol/OUTCOME_EXPOSURE.json','protocol/TRAJECTORY_SELECTION_LOCKED.json',
  'protocol/MOTION_ACTIVITY_THRESHOLD.json','protocol/MOTION_THRESHOLD_AUDIT.csv','protocol/MOTION_STRATA_SUPPORT.csv',
  'protocol/SCHEMA_VERSION.json','protocol/RESOURCE_ESTIMATE_PRE_STAGE0.json','protocol/RESOURCE_GATE_HOST.json',
  'protocol/RESOURCE_POLICY_RESOLUTION.json','protocol/RESOURCE_POLICY_RESOLUTION.md','protocol/RECONCILIATION.csv','protocol/FAILURE_REGISTER.jsonl',
  'inventory/DATA_ALLOWLIST.json','inventory/SOURCE_GROUP_MAP.csv','inventory/ANCHOR_MANIFEST.csv','inventory/POPULATION_MANIFEST.csv',
  'inventory/REGISTERED_SOURCE_HASHES_AT_START.json','receipts/REJECTED_WATCHER_COPY_20261008.snapshot',
  'qa/CPU_CONTRACT_QA.json','qa/AST_QA.json','qa/CPU_PRELOCK_QA_SUMMARY.json','qa/GPU_QUEUE_DRYRUN_002.json','qa/GPU_QUEUE_DRYRUN_003.json','qa/WATCHER_ADAPTER_QA.json','qa/CONTROLLER_DRYRUN.json','qa/PRELOCK_AUDIT.json']
 for rel in static:
  p=ROOT/rel
  if not p.is_file():raise FileNotFoundError(p)
  h=sha(p);role='historical_unexecuted_source_snapshot' if rel=='receipts/REJECTED_WATCHER_COPY_20261008.snapshot' else 'frozen_local_config_or_qa_input';local.append({'source_root':'EXP_ROOT','role':role,'path':rel,'expected_sha256':h,'observed_sha256':h,'hash_match':'true','bytes':p.stat().st_size})
 reg=ROOT/'inventory/SOURCE_REGISTRY.csv';acc=ROOT/'inventory/SOURCE_ACCEPTANCE.json'
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 for p in (reg,acc):shutil.copyfile(p,p.with_name(p.name+f'.pre-final-local-register-{stamp}.bak'))
 rows=external+local
 with reg.open('w',newline='',encoding='utf-8') as f:
  fields=['source_root','role','path','expected_sha256','observed_sha256','hash_match','bytes']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 status=json.loads(acc.read_text(encoding='utf-8'))
 status['local_execution_code_and_frozen_config_entries']=len(local)
 status['local_execution_code_and_frozen_config_hashes']='PASS'
 status['source_registry_total_entries']=len(rows)
 status['operating_policy_and_authorization_entries']=len(policy_sources)
 status['resource_policy_resolution_status']='RESOLVED_KEEP_SHARED_WATCHER_TWO_PROJECT_CHILDREN'
 status['source_registry_sha256']=sha(reg)
 status['created_before_formal_protocol_lock']=True
 status['resource_policy_resolution_status']='RESOLVED_KEEP_SHARED_WATCHER_TWO_PROJECT_CHILDREN'
 status['shared_watcher_modified']=False
 status['local_relaxed_watcher_used']=False
 status['project_gpu_child_limit']=2
 status['mutable_runtime_outputs_excluded_from_source_registry']=['coordination/JOB_LEDGER.jsonl','coordination/GPU_LEASES.jsonl','coordination/RUN_STATE.json','runs/','predictions/','results/','figures/','logs/']
 acc.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':'PASS_LOCAL_SOURCE_REGISTRATION','external_entries':len(external),'local_entries':len(local),'total_entries':len(rows),
                   'source_registry_sha256':sha(reg),'acceptance_sha256':sha(acc)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
