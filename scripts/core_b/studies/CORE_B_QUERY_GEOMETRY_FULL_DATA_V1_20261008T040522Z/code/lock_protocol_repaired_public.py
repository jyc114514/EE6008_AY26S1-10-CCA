#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-014; source SHA-256: 1fca72c7a552de733e4ba5514589732d06fdcf1b07f53372ce9d7e4d3fe8b093
"""Seal the full-data H30 query-geometry protocol once, before GPU training."""
from __future__ import annotations
import csv,hashlib,json
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
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 path=ROOT/'protocol/PROTOCOL_LOCKED.json'
 if path.exists():raise FileExistsError('protocol is already locked; do not rewrite')
 if not ATTACH.is_file():raise FileNotFoundError(ATTACH)
 matrix=ROOT/'protocol/MATRIX12.csv';stage0=ROOT/'protocol/STAGE0_MATRIX.csv'
 with matrix.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 if len(rows)!=12 or sum(x['action']=='reuse' for x in rows)!=6 or sum(x['action']=='train_had' for x in rows)!=6:raise AssertionError('locked matrix must be exactly 6 reuse + 6 Had training cells')
 if len({x['run_id'] for x in rows})!=12 or any(int(x['horizon_steps'])!=30 for x in rows):raise AssertionError('MATRIX12 IDs/H30 contract failed')
 with stage0.open(newline='',encoding='utf-8') as f:s0=list(csv.DictReader(f))
 if len(s0)!=2 or any(int(x['updates'])!=4 or int(x['horizon_steps'])!=30 for x in s0):raise AssertionError('Stage0 matrix contract failed')
 q=readj(ROOT/'protocol/QUERY_TABLES_AND_HASHES.json');source=readj(ROOT/'inventory/SOURCE_ACCEPTANCE.json')
 if source.get('corrected_H30_controls_reuse_pass')!='6/6' or source.get('old_control_repairs')!=0:raise AssertionError('Q0 controls not accepted for reuse')
 if q.get('candidate_table_sha256_h30')!='44d05e4bbe9d8e645adfb00f60881db2619a2fd733237d7157559c2154165b57':raise AssertionError('candidate H30 table changed')
 if q.get('candidate_H60_training') is not False:raise AssertionError('H60 must remain untrained')
 srcsnap=readj(ROOT/'inventory/REGISTERED_SOURCE_HASHES_AT_START.json')
 external={}
 for item in srcsnap['source_rows']:
  p=item['path'];observed=item['observed_sha256']
  if observed=='MISSING':raise FileNotFoundError(p)
  if p in external and external[p]!=observed:raise AssertionError(f'conflicting observed source hashes {p}')
  external[p]=observed
 verified_external=0
 for source_path,observed_hash in external.items():
  source_file=Path(source_path)
  if not source_file.is_file() or sha(source_file)!=observed_hash:raise AssertionError(f'external source changed after start snapshot: {source_path}')
  verified_external+=1
 for extra in [ATTACH,PROJECT/'AGENTS.md',SERVER_POLICY,
               PROJECT/'repo/ee6008-humanoid-vjepa/docs/operations/GPU_WATCHER_POLICY.md',
               PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py']:
  external[str(extra)]=sha(extra)
 for source_path,observed_hash in external.items():
  qpath=Path(source_path)
  if not qpath.is_file() or sha(qpath)!=observed_hash:raise AssertionError(f'external source changed before lock: {source_path}')
 verified_external=len(external)
 task_code={}
 for base in [ROOT/'code',ROOT/'tests']:
  for p in sorted(base.rglob('*.py')):
   if any('.bak' in part for part in p.parts):continue
   rel=p.relative_to(ROOT).as_posix();task_code[rel]=sha(p)
 expected_code=['code/train_full_query.py','code/infer_full_query.py','code/infer_stage0_smoke.py','code/build_gpu_jobs.py',
  'code/run_orchestrator.py','code/accept_stage0.py','code/schema_adapter.py','code/preflight_full_query.py',
  'code/lock_dev_cases.py','code/analyze_full_query.py','code/generate_figures.py','code/prepare_protocol.py','code/lock_protocol.py','tests/test_shared_watcher_policy.py']
 if any(k not in task_code for k in expected_code):raise AssertionError('required task code missing from hash map')
 local_artifacts=[
  'CONTEXT_RESTORED.zh-CN.md','CONTEXT_RESTORED.zh-CN.json','receipts/REJECTED_WATCHER_COPY_20261008.snapshot',
  'protocol/RESOURCE_POLICY_RESOLUTION.json','protocol/RESOURCE_POLICY_RESOLUTION.md','protocol/RECONCILIATION.csv','protocol/FAILURE_REGISTER.jsonl',
  'protocol/MATRIX12.csv','protocol/STAGE0_MATRIX.csv',
  'protocol/REUSE_ELIGIBILITY.csv','protocol/SEED_INIT_MANIFEST.csv','protocol/QUERY_TABLES_AND_HASHES.json',
  'protocol/HADAMARD_QUERY_TABLES.npz','protocol/HADAMARD_QUERY_TABLES.csv','protocol/INPUT_IDENTITIES.json',
  'protocol/ANALYSIS_LOCKED.json','protocol/OUTCOME_EXPOSURE.json','protocol/TRAJECTORY_SELECTION_LOCKED.json',
  'protocol/MOTION_ACTIVITY_THRESHOLD.json','protocol/MOTION_THRESHOLD_AUDIT.csv','protocol/MOTION_STRATA_SUPPORT.csv',
  'protocol/SCHEMA_VERSION.json','protocol/RESOURCE_ESTIMATE_PRE_STAGE0.json','protocol/RESOURCE_GATE_HOST.json',
  'inventory/SOURCE_REGISTRY.csv','inventory/SOURCE_ACCEPTANCE.json','inventory/DATA_ALLOWLIST.json',
  'inventory/SOURCE_GROUP_MAP.csv','inventory/ANCHOR_MANIFEST.csv','inventory/POPULATION_MANIFEST.csv',
  'inventory/REGISTERED_SOURCE_HASHES_AT_START.json',
  'protocol/TRAIN_MONITOR_PANEL.csv','coordination/resource-snapshot.json',
  'coordination/jobs/STAGE0.jsonl','coordination/jobs/INFER_SMOKE.jsonl',
  'coordination/jobs/FORMAL_TEMPLATE_PREFERRED.jsonl','coordination/jobs/INFERENCE_TEMPLATE_PREFERRED.jsonl',
  'qa/CPU_CONTRACT_QA.json','qa/AST_QA.json','qa/CPU_PRELOCK_QA_SUMMARY.json','qa/GPU_QUEUE_DRYRUN_002.json','qa/GPU_QUEUE_DRYRUN_003.json',
  'qa/WATCHER_ADAPTER_QA.json','qa/CONTROLLER_DRYRUN.json','qa/PRELOCK_AUDIT.json','protocol/DEVIATIONS.jsonl']
 missing=[rel for rel in local_artifacts if not (ROOT/rel).is_file()]
 if missing:raise FileNotFoundError(f'prelock artifacts missing: {missing}')
 now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
 seedinit=[]
 for x in csv.DictReader((ROOT/'protocol/SEED_INIT_MANIFEST.csv').open(newline='',encoding='utf-8')):
  seedinit.append({'seed':int(x['seed']),'init_state_hash':x['init_state_hash'],'init_state_file_sha256':x['init_state_file_sha256'],'path':x['init_state_path']})
 lock={
  'schema_version':'1.0','protocol_id':'CORE_B_QUERY_GEOMETRY_FULL_DATA_H30_V1','study_id':'H30_FULL_DATA_QUERY_GEOMETRY_EXPLORATORY',
  'status':'LOCKED','locked_utc':now,'task_attachment_path':str(ATTACH),'task_attachment_sha256':sha(ATTACH),
  'new_root':str(ROOT),'matrix12_path':'protocol/MATRIX12.csv','matrix12_sha256':sha(matrix),'stage0_matrix_path':'protocol/STAGE0_MATRIX.csv','stage0_matrix_sha256':sha(stage0),
  'reuse_eligibility_path':'protocol/REUSE_ELIGIBILITY.csv','reuse_eligibility_sha256':sha(ROOT/'protocol/REUSE_ELIGIBILITY.csv'),
  'query_table_archive_path':'protocol/HADAMARD_QUERY_TABLES.npz','query_archive_sha256':sha(ROOT/'protocol/HADAMARD_QUERY_TABLES.npz'),
  'query_tables_json_path':'protocol/QUERY_TABLES_AND_HASHES.json','query_tables_json_sha256':sha(ROOT/'protocol/QUERY_TABLES_AND_HASHES.json'),
  'query_table_sha256_h30':q['candidate_table_sha256_h30'],'query_shape':[30,128],'query_buffer_key':'future_query_table',
  'query_construction':'canonical Sylvester H128 rows 1..30, fixed column order, row RMS matched to old sin(step/30); only future-query component is replaced',
  'candidate_H60_training':False,'candidate_trainable_parameters_added':0,'parameter_count':890780,
  'models':['TS11-D','STATE-CTRL'],'query_conditions':['old_fixed_sinusoidal_time_query','canonical_Hadamard_H30_RMS_matched'],
  'horizon_steps':30,'seeds':[20261001,20261002,20261003],'full_matrix_rows':12,'new_hadamard_training_cells':6,
  'old_control_reuse_cells':6,'old_control_repair_cells':0,'stage0_cells':2,'stage0_updates_each':4,
  'seed_init_manifest_path':'protocol/SEED_INIT_MANIFEST.csv','seed_init_manifest_sha256':sha(ROOT/'protocol/SEED_INIT_MANIFEST.csv'),
  'seed_inits':{str(x['seed']):x for x in seedinit},
  'input_identities_path':'protocol/INPUT_IDENTITIES.json','input_identities_sha256':sha(ROOT/'protocol/INPUT_IDENTITIES.json'),
  'source_registry_path':'inventory/SOURCE_REGISTRY.csv','source_registry_sha256':sha(ROOT/'inventory/SOURCE_REGISTRY.csv'),
  'source_acceptance_path':'inventory/SOURCE_ACCEPTANCE.json','source_acceptance_sha256':sha(ROOT/'inventory/SOURCE_ACCEPTANCE.json'),
  'source_acceptance_status':source['status'],'registered_source_start_snapshot_sha256':sha(ROOT/'inventory/REGISTERED_SOURCE_HASHES_AT_START.json'),
  'external_source_observed_hashes':external,'external_source_hashes_rechecked_at_lock':verified_external,'task_code_hashes':task_code,
  'local_artifact_hashes':{rel:sha(ROOT/rel) for rel in local_artifacts},
  'train_population':{'anchors':17757,'episodes':2712,'source_groups':2712},
  'dev_population':{'anchors':2322,'episodes':603,'source_groups':603,'prior_exposure':True,'evaluation':'exploratory_development'},
  'input_contract':'32x71 standardized history plus mask/relative delta; native 28D anchor; frozen dynamic 16x4x768 V-JEPA grid; residual output native=anchor+sigma*residual',
  'loss_contract':'Full28 position + 0.1 safe-FD; exactly 29 future-internal differences; no anchor-to-first seam in FD; episode-equal weights W/(E*n_episode), weighted sum / actual batch size',
  'training_conditions':{'epochs':40,'updates_per_epoch':278,'updates_total':11120,'effective_batch':64,'physical_batch':64,'accumulation':1,
    'dtype':'FP32','dropout':'source default 0.1','optimizer':'AdamW','learning_rate':0.001,'betas':[0.9,0.999],'eps':1e-8,
    'weight_decay':0.0001,'gradient_clip_norm':1.0,'scheduler':'update-driven cosine after each successful optimizer.step; T_max=11120; eta_min=1e-5',
    'selection':'epoch40/last only; no best-dev or early-stop'},
  'monitor_panel_path':'protocol/TRAIN_MONITOR_PANEL.csv','monitor_panel_sha256':sha(ROOT/'protocol/TRAIN_MONITOR_PANEL.csv'),
  'analysis_lock_path':'protocol/ANALYSIS_LOCKED.json','analysis_lock_sha256':sha(ROOT/'protocol/ANALYSIS_LOCKED.json'),
  'outcome_exposure_path':'protocol/OUTCOME_EXPOSURE.json','outcome_exposure_sha256':sha(ROOT/'protocol/OUTCOME_EXPOSURE.json'),
  'trajectory_selection_algorithm_path':'protocol/TRAJECTORY_SELECTION_LOCKED.json','trajectory_selection_algorithm_sha256':sha(ROOT/'protocol/TRAJECTORY_SELECTION_LOCKED.json'),
  'resource_estimate_pre_stage0_path':'protocol/RESOURCE_ESTIMATE_PRE_STAGE0.json','resource_estimate_pre_stage0_sha256':sha(ROOT/'protocol/RESOURCE_ESTIMATE_PRE_STAGE0.json'),
  'resource_gate_host_path':'protocol/RESOURCE_GATE_HOST.json','resource_gate_host_sha256':sha(ROOT/'protocol/RESOURCE_GATE_HOST.json'),
  'resource_policy_resolution_path':'protocol/RESOURCE_POLICY_RESOLUTION.json','resource_policy_resolution_sha256':sha(ROOT/'protocol/RESOURCE_POLICY_RESOLUTION.json'),
  'reconciliation_path':'protocol/RECONCILIATION.csv','reconciliation_sha256':sha(ROOT/'protocol/RECONCILIATION.csv'),
  'failure_register_path':'protocol/FAILURE_REGISTER.jsonl','failure_register_sha256_at_lock':sha(ROOT/'protocol/FAILURE_REGISTER.jsonl'),
  'gpu_policy':{'pool':'dynamic_all_visible','artificial_gpu_hour_ceiling':None,'artificial_disk_gib_ceiling':None,
    'watcher_path':str(PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'),
    'watcher_source_sha256':sha(PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'),
    'lease_dir':str(PROJECT/'repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease'),
    'account_child_slots_current_watcher':2,'one_child_per_physical_gpu':True,'utilization_is_admission_gate':False,
    'resource_policy_resolution':'KEEP_SHARED_WATCHER_TWO_PROJECT_CHILDREN','shared_watcher_modified':False,'local_relaxed_watcher_used':False,
    'gpu_selection':'dynamic memory-contract capacity; all currently authorized visible GPUs eligible; no fixed physical GPU allowlist'},
  'prediction_policy':{'train_rows':17757,'dev_rows':2322,'target_blind_inference':True,'target_timestamp_formula':'anchor_timestamp+step/30; frame=anchor_frame+step; step=1..30',
    'all_12_model_systems_train_dev_bundles_required_before_scoring':True,'fresh_process_replay_atol':1e-5,'fresh_process_replay_rtol':1e-5},
  'statistics':{'position_and_safe_fd_co_primary':True,'group_bootstrap_replicates':10000,'prng_seed':20261008,'cluster_unit':'source_group',
    'shared_draws_across_cells_seeds_and_metrics':True,'bootstrap_I_directly':True,'p_values':'not planned','seed_population_uncertainty':'not implied by three fixed seeds'},
  'prohibitions':{'final_test_access':0,'heldout_access':0,'D5_training':False,'H60_training':False,'LoRA':False,
    'encoder_change':False,'dev_model_selection':False,'new_query_scale_frequency_lr_or_seed_sweep':False},
  'deviations_path':'protocol/DEVIATIONS.jsonl','deviations_sha256':sha(ROOT/'protocol/DEVIATIONS.jsonl'),'status_note':'Protocol locked before Stage0; Stage0 weights are discarded and formal cells must load their original seed INIT.'}
 path.write_text(json.dumps(lock,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({'status':'LOCKED','protocol_sha256':sha(path),'matrix12_sha256':sha(matrix),'task_code_count':len(task_code),'external_source_count':len(external),'artifact_count':len(local_artifacts),'new_had_cells':6,'q0_reuse_cells':6},indent=2))
if __name__=='__main__':main()
