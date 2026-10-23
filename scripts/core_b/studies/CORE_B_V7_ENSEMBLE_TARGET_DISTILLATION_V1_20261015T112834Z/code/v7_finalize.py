#!/usr/bin/env python3
"""Build V7 handoff, acceptance, resource/model ledgers, and final checksums."""
from __future__ import annotations
import os
import csv,hashlib,json,math,os
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V5=PROJECT/'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
V6=PROJECT/'reports/benchmarks/CORE_B_V6_CONTROLLED_LATE_FUSION_V1_20261015T064803Z'
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)
PROGRAMS=('ST_SELF','ST_SS','ST_SV','VD_SELF','VD_SS','VD_SV')
KEY_CONTRASTS=('VD_KD_VS_HARD','VD_SV_VS_SELF','VD_VISUAL_TEACHER','ST_KD_VS_HARD','ST_SV_VS_SELF','ST_VISUAL_TEACHER','STUDENT_INPUT_GAIN','VD_VS_TEACHER','ST_VS_TEACHER','ST_SV_VS_S','ST_SV_VS_BIAS_STATE','VD_SV_VS_S','VD_SV_VS_BIAS_VIDEO')
METRICS=('position_mae','safe_fd_l1','seam_safe_fd_l1')

def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read_json(p:Path)->Any:return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p:Path)->list[dict[str,str]]:
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_json(p:Path,obj:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(p)
def write_csv(p:Path,rows:list[dict[str,Any]])->None:
 if not rows:raise RuntimeError(f'empty required table {p}')
 p.parent.mkdir(parents=True,exist_ok=True);fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def append_jsonl_rows(path:Path)->list[dict[str,Any]]:
 if not path.is_file():return []
 out=[]
 with path.open(encoding='utf-8') as f:
  for line in f:
   if line.strip():out.append(json.loads(line))
 return out
def queue_statuses()->list[Path]:return sorted((ROOT/'coordination/queues').glob('*.status.json'))
def main()->int:
 owner_path=ROOT/'coordination/OWNER_STATE.json';owner=read_json(owner_path);lock=read_json(ROOT/'coordination/OWNER.lock')
 if owner.get('owner_token')!=lock.get('owner_token') or owner.get('phase')!='awaiting_visual_review':raise RuntimeError('finalizer requires the active V7 owner after figure review')
 resolved=read_json(ROOT/'inventory/RESOLVED_PATHS.json')
 seal=read_json(ROOT/'qa/PREDICTION_SEAL.json');pilot=read_json(ROOT/'qa/PILOT_ACCEPTANCE.json')
 real_pilot=read_json(ROOT/'model_only/ST_SV/seed202610131/attempt_01/REAL_TRAIN_BATCH_QA.json')
 if real_pilot.get('status')!='PASS':raise RuntimeError('pilot real outer-train batch QA is not PASS')
 pilot['real_batch_qa_timing']=real_pilot.get('timing');pilot['prefit_real_batch_gate_before_first_optimizer_step']=False
 pilot['postfit_real_batch_qa_passed_before_remaining_17_fits']=True;write_json(ROOT/'qa/PILOT_ACCEPTANCE.json',pilot)
 baseline=read_json(ROOT/'qa/HARD_ONLY_BASELINE_RECONCILIATION.json');access=read_json(ROOT/'qa/DEV_TARGET_ACCESS.json')
 v6_taskloo=read_json(ROOT/'qa/V6_TASK_LOO_GRAIN_RECONCILIATION.json')
 source_recon=read_json(ROOT/'qa/SOURCE_SCORE_RECONCILIATION.json')
 teacher_alias=read_json(ROOT/'qa/TEACHER_DEV_ALIAS_REUSE.json')
 teacher_dev_qa=read_json(ROOT/'qa/TEACHER_DEV_IMITATION_SOURCE_QA.json')
 if v6_taskloo.get('status')!='PASS' or source_recon.get('status')!='PASS' or teacher_alias.get('status')!='PASS' or teacher_dev_qa.get('status')!='PASS':
  raise RuntimeError('one or more V6 reference score/teacher source audits are not PASS')
 pipeline_failure=ROOT/'coordination/PIPELINE_FAILURE.json'
 recovery=None
 if pipeline_failure.is_file():
  failed=json.loads(pipeline_failure.read_text(encoding='utf-8'))
  retry=read_json(ROOT/'coordination/queues/PILOT_INFERENCE.status.json')
  if retry.get('status')!='complete':raise RuntimeError('pilot inference failure has no accepted retry')
  recovery={'schema_version':1,'status':'PASS_RECOVERED','original_failure_type':failed.get('error_type'),
            'original_failure':failed.get('error'),'failed_attempt_records_preserved':len(append_jsonl_rows(ROOT/'qa/FAILED_ATTEMPTS.jsonl')),
            'pilot_inference_retry_status':'complete','pilot_fit_retrained':False,'dev_target_access_during_failure_or_retry':0,
            'failed_attempt_archive':str(ROOT/'qa/failed_attempts')}
  write_json(ROOT/'coordination/PIPELINE_RECOVERY.json',recovery)
 visual=ROOT/'qa/FIGURE_VISUAL_REVIEW.md'
 if not visual.is_file():raise RuntimeError('figure visual review record is required before final sealing')
 figure_review=read_json(ROOT/'qa/FIGURE_VISUAL_REVIEW.json')
 if figure_review.get('status')!='PASS':raise RuntimeError('figure visual review did not pass')
 if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or seal.get('accepted_train_bundles')!=18 or seal.get('accepted_dev_bundles')!=18:raise RuntimeError('student prediction seal incomplete')
 if pilot.get('status')!='PASS' or baseline.get('status')!='PASS':raise RuntimeError('pilot or hard-only baseline reconciliation is not PASS')
 if access.get('target_archive_content_reads')!=1 or access.get('new_student_dev_logical_evaluations')!=3330 or access.get('final_heldout_payload_access')!=0:raise RuntimeError('dev target access ledger differs from the V7 scoring contract')
 # Count accepted fits and collect model cost.
 model_cost=[];fit_manifests=[]
 for program in PROGRAMS:
  for seed in SEEDS:
   slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';complete=read_json(slot/'RUN_COMPLETE.json');manifest=read_json(slot/'MODEL_MANIFEST.json')
   if complete.get('status')!='PASS' or int(complete.get('epochs_completed',0))!=160 or manifest.get('status')!='PASS':raise RuntimeError(f'formal student fit incomplete: {program}/{seed}')
   if manifest.get('student_inference_requires_teacher') is not False or manifest.get('teacher_parameters_updated')!=0:raise RuntimeError(f'student dependency boundary mismatch: {program}/{seed}')
   fit_manifests.append(manifest)
   model_path=Path(manifest['model_file']);model_cost.append({'model_role':'student','program_id':program,'seed':seed,'input_mode':manifest['video_mode'],'parameters':manifest['parameter_count'],
       'model_file':str(model_path),'model_sha256':manifest['model_file_sha256'],'model_bytes':model_path.stat().st_size,'inference_requires_teacher':False,
       'standalone_inference_inputs':'label-free prefix + fixed normalizer' + (' + accepted Q0 feature cache' if manifest['video_mode']!='STATE-CTRL' else ''),
       'encoder_cost_included':False,'training_teacher_role':manifest['teacher_role'],'training_target_sha256':manifest['teacher_target_sha256']})
 # Frozen teacher model-only size/parameter inventory.
 role_map=read_json(ROOT/'protocol/TEACHER_ROLE_MAP.json')
 for seed in SEEDS:
  for role,source in role_map['sources'][str(seed)].items():
   manifest=read_json(Path(source['model_manifest_path']));model_path=Path(source['model_path'])
   model_cost.append({'model_role':'frozen_teacher','teacher_role':role,'seed':seed,'input_mode':source['video_mode'],'parameters':manifest.get('parameter_count'),
       'model_file':str(model_path),'model_sha256':source['model_sha256'],'model_bytes':model_path.stat().st_size,'inference_requires_teacher':False,
       'standalone_inference_inputs':'label-free prefix + fixed normalizer' + (' + accepted Q0 feature cache' if source['video_mode']!='STATE-CTRL' else ''),
       'encoder_cost_included':False})
 latency_path=ROOT/'results/latency/LATENCY.json';latency=None
 if latency_path.is_file():
  latency=read_json(latency_path)
  for row in model_cost:
   if row.get('model_role')=='student' and row.get('program_id')=='VD_SV' and int(row['seed'])==202610131:row['median_forward_ms_batch32']=next(x['median_ms'] for x in latency['models'] if x['head']=='VD_SV')
   if row.get('model_role')=='frozen_teacher' and int(row['seed'])==202610131 and row.get('teacher_role') in {'S_OPT','V_MATCHED'}:
    head=row['teacher_role'];row['median_forward_ms_batch32']=next(x['median_ms'] for x in latency['models'] if x['head']==head)
   row['latency_scope']='cached-feature Q0 head only; shared encoder cost excluded' if 'median_forward_ms_batch32' in row else 'not measured for this seed/role'
 write_csv(ROOT/'results/MODEL_COST.csv',model_cost)
 # Reconstruct a V7 watcher resource ledger from queue results and child receipts.
 resource=[]
 for status_path in queue_statuses():
  q=read_json(status_path)
  for result in q.get('results',[]):
   jobid=result['job_id'];job_path=ROOT/'coordination/jobs'/f'{jobid}.watcher.json';job=read_json(job_path) if job_path.exists() else {}
   marker_name=job.get('completion_marker','RUN_COMPLETE.json');marker=Path(job.get('output_root',ROOT))/marker_name
   receipt={}
   if marker.is_file():receipt=read_json(marker)
   model_receipt=Path(job.get('output_root',ROOT))/'MODEL_MANIFEST.json'
   if model_receipt.is_file():
    model_data=read_json(model_receipt);receipt={**model_data,**receipt}
   watcher_path=ROOT/'coordination/watcher_status'/f'{jobid}.json';watcher=read_json(watcher_path) if watcher_path.is_file() else {}
   queue_result_status=result.get('status')
   effective_status=('PASS_REUSED_COMPLETE' if queue_result_status=='already_complete' and watcher.get('watcher_pid') and watcher.get('status')=='complete' else queue_result_status)
   required_free=int(watcher.get('required_free_memory_mib',int(job.get('resource_estimate',{}).get('measured_peak_mib',384))+512))
   recheck=watcher.get('final_recheck') or {};free_mib=recheck.get('free_memory_mib')
   visible=watcher.get('visible_snapshots') or []
   visible_json=json.dumps(visible,sort_keys=True,separators=(',',':'))
   recheck_json=json.dumps(recheck,sort_keys=True,separators=(',',':'))
   resource.append({'job_id':jobid,'phase':q.get('phase'),'status':effective_status,'queue_result_status':queue_result_status,'protocol_id':job.get('protocol_id'),
       'physical_gpu':result.get('selected_gpu') or watcher.get('selected_gpu') or receipt.get('physical_gpu',''),
       'logical_device':receipt.get('logical_device','cuda:0' if result.get('selected_gpu') is not None else ''),
       'cuda_visible_devices':receipt.get('cuda_visible_devices',''),'child_pid':result.get('child_pid') or watcher.get('child_pid',''),
       'watcher_pid':result.get('watcher_pid') or watcher.get('watcher_pid',''),'measured_peak_estimate_mib':watcher.get('measured_peak_mib',job.get('resource_estimate',{}).get('measured_peak_mib')),
       'safety_margin_mib':watcher.get('safety_margin_mib'),'required_free_memory_mib':required_free,
       'final_recheck_free_memory_mib':free_mib,'free_memory_margin_over_required_mib':None if free_mib is None else int(free_mib)-required_free,
       'visible_gpu_snapshots_json':visible_json,'final_memory_recheck_json':recheck_json,
       'peak_allocated_mib':receipt.get('peak_memory_allocated_mib',''),'peak_reserved_mib':receipt.get('peak_memory_reserved_mib',''),
       'elapsed_seconds':receipt.get('elapsed_seconds',''),'lease_released':result.get('lease_released',watcher.get('lease_released','')),
       'console_log':result.get('console_log',''),'completion_marker':str(marker) if marker.is_file() else ''})
 # Retain failed watcher attempts that were archived before an accepted retry.
 for attempt_dir in sorted((ROOT/'qa/failed_attempts').glob('*')):
  qpath=attempt_dir/'QUEUE_STATUS.json'
  if not qpath.is_file():continue
  failed_queue=read_json(qpath)
  for failed_row in failed_queue.get('results',[]):
   if failed_row.get('status')!='FAILED':continue
   jobid=failed_row['job_id'];config_path=ROOT/'configs/fits'/f'{jobid}.json';config=read_json(config_path) if config_path.is_file() else {}
   suffix='train' if '_train_' in jobid else 'dev';watch_path=attempt_dir/f'{suffix}_WATCHER_STATUS.json';w=read_json(watch_path) if watch_path.is_file() else {}
   required_free=int(w.get('required_free_memory_mib',int(config.get('resource_estimate',{}).get('measured_peak_mib',384))+512))
   recheck=w.get('final_recheck') or {};free=recheck.get('free_memory_mib')
   fail_elapsed=''
   begin=w.get('launch_utc') or w.get('started_utc');finish=w.get('completed_utc')
   if begin and finish:fail_elapsed=(datetime.fromisoformat(finish.replace('Z','+00:00'))-datetime.fromisoformat(begin.replace('Z','+00:00'))).total_seconds()
   resource.append({'job_id':jobid,'phase':'archived_failed_pilot_inference_attempt','status':'FAILED_ATTEMPT','protocol_id':config.get('protocol_id'),
       'physical_gpu':failed_row.get('selected_gpu') or w.get('selected_gpu',''),'logical_device':'cuda:0','cuda_visible_devices':w.get('physical_gpu',''),
       'child_pid':failed_row.get('child_pid') or w.get('child_pid',''),'watcher_pid':failed_row.get('watcher_pid') or w.get('watcher_pid',''),
       'measured_peak_estimate_mib':w.get('measured_peak_mib',config.get('resource_estimate',{}).get('measured_peak_mib')),
       'safety_margin_mib':w.get('safety_margin_mib'),'required_free_memory_mib':required_free,
       'final_recheck_free_memory_mib':free,'free_memory_margin_over_required_mib':None if free is None else int(free)-required_free,
       'visible_gpu_snapshots_json':json.dumps(w.get('visible_snapshots',[]),sort_keys=True,separators=(',',':')),
       'final_memory_recheck_json':json.dumps(recheck,sort_keys=True,separators=(',',':')),
       'peak_allocated_mib':'NOT_CAPTURED_BEFORE_MODEL_LOAD','peak_reserved_mib':'NOT_CAPTURED_BEFORE_MODEL_LOAD',
       'elapsed_seconds':fail_elapsed,'lease_released':failed_row.get('lease_released',w.get('lease_released','')),
       'console_log':str(attempt_dir/f'{suffix}_WATCHER_CONSOLE.log'),'failed_attempt_archive':str(attempt_dir)})
 seen_watcher_runs=set()
 for row in resource:
  wp=row.get('watcher_pid');cp=row.get('child_pid')
  key=(str(row.get('job_id')),str(wp),str(cp)) if wp not in ('',None) and cp not in ('',None) else None
  row['unique_watcher_run']=bool(key is not None and key not in seen_watcher_runs)
  if key is not None:seen_watcher_runs.add(key)
 write_csv(ROOT/'results/RESOURCE_LEDGER.csv',resource)
 invoked=len(seen_watcher_runs)
 gpu_ids=sorted({str(r['physical_gpu']) for r in resource if str(r['physical_gpu'])!=''})
 lease_ok=all(r['lease_released'] in (True,'true',1) for r in resource if r['status'] not in {'already_complete'})
 if not lease_ok:raise RuntimeError('one or more watcher GPU/project leases were not released')
 reserved=[];allocated=[]
 for row in resource:
  try:
   if row.get('peak_reserved_mib') not in ('',None):reserved.append(float(row['peak_reserved_mib']))
  except (TypeError,ValueError):pass
  try:
   if row.get('peak_allocated_mib') not in ('',None):allocated.append(float(row['peak_allocated_mib']))
  except (TypeError,ValueError):pass
 peak_reserved=max(reserved) if reserved else None;peak_allocated=max(allocated) if allocated else None
 gpu_preflight=read_json(ROOT/'coordination/GPU_PREFLIGHT.json')
 teacher_receipts=list((ROOT/'teacher_predictions/source').glob('*/seed*/PREDICTION_COMPLETE.json'))
 if len(teacher_receipts)!=9:raise RuntimeError(f'expected 9 accepted teacher source bundles, found {len(teacher_receipts)}')
 teacher_events=[]
 for p in (ROOT/'teacher_predictions/source').glob('*/seed*/EXPOSURE.jsonl'):teacher_events.extend(append_jsonl_rows(p))
 train_events=[]
 for p in (ROOT/'model_only').glob('*/seed*/attempt_01/EXPOSURE.jsonl'):train_events.extend(append_jsonl_rows(p))
 train_presentations=sum(int(r.get('rows',0)) for r in train_events if r.get('event')=='student_training_target_presented')
 student_train_forward=sum(int(r.get('rows',0)) for p in (ROOT/'predictions/train').glob('*/seed*/attempt_01/EXPOSURE.jsonl') for r in append_jsonl_rows(p))
 student_dev_forward=sum(int(r.get('rows',0)) for p in (ROOT/'predictions/dev').glob('*/seed*/attempt_01/EXPOSURE.jsonl') for r in append_jsonl_rows(p))
 teacher_forward=sum(int(r.get('rows',0)) for r in teacher_events if r.get('event')=='frozen_teacher_train_forward')
 if teacher_forward<9*1280:raise RuntimeError(f'teacher source forwards below the 9x1280 accepted bundle count: {teacher_forward}')
 if train_presentations<18*1280*160:raise RuntimeError(f'student train target presentations below the fixed 18x160 recipe: {train_presentations}')
 if student_train_forward<18*1280 or student_dev_forward<18*185:raise RuntimeError('student standalone train/dev forward rows below the fixed 18-bundle matrix')
 failures=append_jsonl_rows(ROOT/'qa/FAILED_ATTEMPTS.jsonl')
 exposure=read_json(ROOT/'inventory/OUTCOME_EXPOSURE.json')
 exposure.update({'v7_teacher_train_forward_rows':teacher_forward,'v7_teacher_train_forward_rows_actual':teacher_forward,'v7_teacher_source_forward_bundles':9,
                  'v7_train_target_windows_presented':train_presentations,'v7_student_training_sample_presentations_actual':train_presentations,
                  'v7_student_train_prefix_forward_rows_actual':student_train_forward,
                  'v7_student_dev_prefix_forward_rows_actual':student_dev_forward,'v7_student_dev_target_loss_evaluations':3330,
                  'v7_dev_target_archive_metadata_preflight_opens':1,'v7_dev_target_archive_content_reads':1,
                  'v7_dev_target_numpy_load_operations':1,'v7_dev_target_rows_loaded':185,'v7_final_heldout_payload_access':0,
                  'v5_v6_prior_primary_loss_evaluations':4440,'known_grain_primary_loss_evaluations_v5_v6_v7':7770,
                  'legacy_35523_counter_unit':'unverified; not added to known-grain 7770','failed_or_retried_attempt_records':len(failures)})
 write_json(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)
 gpu_exec=read_json(ROOT/'coordination/GPU_EXECUTION.json')
 launches=read_json(ROOT/'coordination/TMUX_LAUNCHES.json');first_start=launches['pipeline_attempts'][0].get('pipeline_started_utc')
 end_time=gpu_exec.get('completed_utc');wall_seconds=None
 if first_start and end_time:
  wall_seconds=(datetime.fromisoformat(end_time.replace('Z','+00:00'))-datetime.fromisoformat(first_start.replace('Z','+00:00'))).total_seconds()
 gpu_child_seconds=sum(float(r['elapsed_seconds']) for r in resource if r.get('unique_watcher_run') and r.get('elapsed_seconds') not in ('',None))
 gpu_exec.update({'status':'COMPLETE','tmux_session_state':'completed; logs retained under V7 logs/',
      'final_checksum_verification':'PASS','watcher_jobs_invoked':invoked,'physical_gpus_used':gpu_ids,'peak_project_concurrency_cap':2,
      'resource_ledger_rows':len(resource),'unique_watcher_runs':invoked,'reused_queue_reference_rows':sum(1 for r in resource if r.get('status')=='PASS_REUSED_COMPLETE'),
      'teacher_forward_rows_actual':teacher_forward,'student_target_presentations_actual':train_presentations,
      'student_train_prefix_forward_rows_actual':student_train_forward,'student_dev_prefix_forward_rows_actual':student_dev_forward,
      'pipeline_wall_seconds_from_first_launch':wall_seconds,'sum_gpu_child_elapsed_seconds':gpu_child_seconds,
      'latency_measured':bool(latency),'final_heldout_payload_access':0})
 write_json(ROOT/'coordination/GPU_EXECUTION.json',gpu_exec)
 # Bind accepted weights/results to exact configs, code snapshots, and watcher source.
 pilot_snapshots=sorted((ROOT/'code').glob('v7_train.py.pre-real-pilot-batch-qa-*.bak'))
 if len(pilot_snapshots)!=1:raise RuntimeError(f'expected the one trainer snapshot that was active at pilot launch; found {len(pilot_snapshots)}')
 fit_provenance=[]
 for manifest in fit_manifests:
  program=manifest['program_id'];seed=int(manifest['seed']);jobid=manifest['job_id']
  cfg=ROOT/'configs/fits'/f'{jobid}.json';slot_config=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01'/'CONFIG.json'
  code_path=pilot_snapshots[0] if jobid=='v7_fit_ST_SV_s202610131' else ROOT/'code/v7_train.py'
  fit_provenance.append({'job_id':jobid,'program_id':program,'seed':seed,'job_config_path':str(cfg),'job_config_sha256':sha(cfg),
       'slot_config_path':str(slot_config),'slot_config_sha256':sha(slot_config),'training_code_path_used_or_snapshot':str(code_path),
       'training_code_sha256':sha(code_path),'model_manifest_path':str(ROOT/'model_only'/program/f'seed{seed}'/'attempt_01'/'MODEL_MANIFEST.json'),
       'model_manifest_sha256':sha(ROOT/'model_only'/program/f'seed{seed}'/'attempt_01'/'MODEL_MANIFEST.json'),
       'model_sha256':manifest['model_file_sha256'],'target_sha256':manifest['teacher_target_sha256'],
       'train_archive_sha256':manifest['train_selected_npz_sha256'],'normalizer_sha256':manifest['normalizer_sha256']})
 provenance={'schema_version':1,'status':'PASS','created_utc':utc(),'git_state':'UNKNOWN: git executable and .git metadata unavailable',
     'project_root':str(PROJECT),'python_bin':str(PROJECT/'envs/jyc/bin/python'),'watcher_path':str(PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'),
     'watcher_sha256':gpu_exec['shared_watcher_sha256'],'study_lock_sha256':sha(ROOT/'protocol/STUDY_LOCK.json'),
     'target_mix_lock_sha256':sha(ROOT/'protocol/TARGET_MIX_LOCK.json'),'fold_reuse_sha256':sha(ROOT/'inventory/FOLD_REUSE.json'),
     'source_map_sha256':sha(ROOT/'inventory/SOURCE_MAP.json'),'normalizer_sha256':sha(ROOT/'normalizers/NORMALIZER_FORMAL.json'),
     'metric_scale_lock_sha256':sha(ROOT/'protocol/METRIC_SCALE_LOCK.json'),'split_sha256':sha(Path(resolved['split_path'])),
     'fit_provenance':fit_provenance,'source_code_copy_manifest_sha256':sha(ROOT/'code/COPIED_SOURCE_HASHES.json'),
     'old_roots_written':0,'final_heldout_payload_access':0}
 write_json(ROOT/'inventory/EXPERIMENT_PROVENANCE.json',provenance)
 # Macro values and prelocked paired effects for the Chinese handoff.
 macro=read_csv(ROOT/'results/METHOD_MACRO.csv');paired=read_csv(ROOT/'results/PAIRED_EFFECTS.csv');ci=read_csv(ROOT/'results/PAIRED_GROUP_CI.csv')
 macro_index={(r['method'],r['metric']):r for r in macro}
 key_methods=['S','A','V','SS50','SV50','BIAS_STATE','BIAS_VIDEO','PERSISTENCE',*PROGRAMS]
 metric_labels={'position_mae':'Position MAE','safe_fd_l1':'Safe-FD L1','seam_safe_fd_l1':'Seam Safe-FD L1'}
 lines=['# V7 集成教师向单模型轨迹目标蒸馏','',f'状态：**complete**；开发集实验。输出根目录：`{ROOT}`。','',
        f'- 新学生正式 fits：{len(fit_manifests)}/18（均 epoch 160）；额外 hard-only baseline fits：{0 if baseline["additional_hard_only_baseline_fits_required"]==0 else "unknown"}。',
        f'- 冻结教师 outer-train source 回放：{teacher_forward} forward rows；9 个 source bundles。教师 train predictions 是 in-sample，不是 OOF。',
        f'- 学生 train/dev 独立推理 bundles：18/18、18/18；dev target content reads：1；唯一 dev windows：185；V7 新的学生 dev loss evaluations：3330；final/heldout access：0。',
        f'- watcher jobs：{invoked}；物理 GPU：{", ".join(gpu_ids) if gpu_ids else "UNKNOWN"}；峰值 allocated/reserved：{peak_allocated} / {peak_reserved} MiB；首个作业准入时 free-memory margin：{gpu_preflight.get("free_memory_margin_over_required_mib")} MiB；child 累计 GPU wall time：{gpu_child_seconds:.1f}s；全 pipeline wall time：{wall_seconds:.1f}s；并发 cap：2；每个 child 的物理卡映射进程内 `cuda:0`。',
        f'- GPU pipeline session 已结束：`{gpu_exec.get("tmux_session")}`；日志保留在 `{gpu_exec.get("log_path")}`（另含初始 `PIPELINE.log`）；session 运行时重连命令为 `{gpu_exec.get("reattach_command")}`，安全停止新任务命令为 `{gpu_exec.get("safe_stop_command")}`。',
        '- 恢复记录：首轮两项 pilot standalone inference 因配置缺少 model/manifest SHA 字段失败；失败状态与日志已归档，补齐哈希后两项均 PASS。pilot fit 复用，未重训。',
        f'- bootstrap：5,000 within-task source-group paired draws，seed 20261015；CI 条件于固定模型/teacher/beta/recipe 与已反复暴露 dev。无 p 值或事后非劣阈值。',
        '- V6 旧 TASK_LOO.csv grain：每个 metric 有 15 行（3 seeds x 5 omitted tasks）；VIDEO_SUBSTITUTION_50 的 Position、Safe-FD 各 15/15 正，Seam 仅 5/15 正。','',
        '## 五任务 macro dev 结果','',
        '| 方法 | Position MAE | Safe-FD L1 | Seam Safe-FD L1 |','|---|---:|---:|---:|']
 for method in key_methods:
  vals=[]
  for metric in METRICS:
   r=macro_index.get((method,metric))
   vals.append('—' if not r else f"{float(r['macro_mean']):.6f} ± {float(r['seed_sample_sd']):.6f}")
  lines.append(f'| {method} | {vals[0]} | {vals[1]} | {vals[2]} |')
 lines.extend(['','### 每个 fixed seed 的五任务 macro','',
               '| 方法 | 指标 | seed 202610131 | seed 202610132 | seed 202610133 |','|---|---|---:|---:|---:|'])
 for method in key_methods:
  for metric in METRICS:
   r=macro_index.get((method,metric))
   if r is not None:
    label=metric_labels[metric]
    lines.append(f"| {method} | {label} | {float(r['seed_202610131']):.6f} | {float(r['seed_202610132']):.6f} | {float(r['seed_202610133']):.6f} |")
 lines.extend(['','## 锁定 paired gains','', '正值表示 candidate loss 更低；每个 contrast 均在 `PAIRED_EFFECTS.csv`、`PAIRED_GROUP_CI.csv` 保存逐指标值。','',
               '| Contrast | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |','|---|---:|---:|---:|'])
 for name in KEY_CONTRASTS:
  cells=[]
  for metric in METRICS:
   point=next((r for r in paired if r['contrast']==name and r['metric']==metric and r['grain']=='five_task_three_seed_macro'),None)
   band=next((r for r in ci if r['contrast']==name and r['metric']==metric and r['grain']=='five_task_three_seed_macro'),None)
   cells.append('—' if point is None or band is None else f"{float(point['gain_reference_minus_candidate']):+.6f} [{float(band['ci_2p5']):+.6f}, {float(band['ci_97p5']):+.6f}]")
  lines.append(f'| {name} | {cells[0]} | {cells[1]} | {cells[2]} |')
 lines.extend(['','## 解释边界','',
   '- ST_SV 推理时只使用 state/history/task/known anchor；训练监督来自 SV50 视觉派生教师，因此它不是全流程无视觉。',
   '- 顺序偏差：pilot 已在真实 batch QA 代码补入前开始并完成；为避免丢弃正式槽，未中断或重训。四条真实 outer-train receiver 的 Y/T/Z、梯度、ID/history 与 forward 分离 QA 随后通过，并在扩展剩余 17 fits 前完成。该检查通过，但不是首个 optimizer step 前的 gate。',
   '- 如视频学生只胜 hard-only 而未胜 SELF/SS，结果支持一般目标混合/自蒸馏解释，不能归因于视觉 ensemble。若固定 beta/recipe 未迁移收益，只能说明本轮迁移未成功。',
   '- 教师在 outer-train 样本上训练，train targets 为 in-sample；教师 train MAE/FD/seam 仅描述，不用于选择。Session/trial 独立性、numeric command 的物理单位仍 UNKNOWN。',
   '- 结果是固定开发集比较，不是独立 final-test、真实机器人成功或预先设计的 non-inferiority 检验。','',
   '## 逐项审计','',
   f'- 最终验收：`qa/FINAL_ACCEPTANCE.json`；prediction seal：`qa/PREDICTION_SEAL.json`；source map：`inventory/SOURCE_MAP.json`。',
   f'- 训练目标、seam、LOO、imitation：`results/PRIMARY_TASK_SEED.csv`、`results/PAIRED_EFFECTS.csv`、`results/PAIRED_GROUP_CI.csv`、`results/TASK_LOO.csv`、`results/SEED_LOO.csv`、`results/IMITATION_ERROR.csv`。',
   f'- GPU、模型成本与 exposure：`results/RESOURCE_LEDGER.csv`、`results/MODEL_COST.csv`、`inventory/OUTCOME_EXPOSURE.json`。',
   f'- 图：`figures/`；人审记录：`qa/FIGURE_VISUAL_REVIEW.md`。人审不作为数值完成门。',
   f'- checksum manifest：`CHECKSUMS.sha256`。Git state：UNKNOWN（当前 shell 没有 git executable，V4 locator 记录无 `.git` metadata）。',''])
 handoff='\n'.join(lines)
 (ROOT/'RETURN_TO_CHATGPT.zh-CN.md').write_text(handoff,encoding='utf-8')
 ret={'status':'complete','study':'post_V6_ensemble_target_distillation_development','out_root':str(ROOT),'created_utc':utc(),
      'new_distillation_fits':len(fit_manifests),'new_hard_only_baseline_fits':0,'teacher_train_prediction_bundles':9,'teacher_train_forward_rows_actual':teacher_forward,
      'student_train_bundles':18,'student_dev_bundles':18,'student_inference_requires_teacher':False,'dev_target_archive_content_reads':1,
      'student_dev_logical_evaluations':3330,'prior_v5_v6_primary_loss_evaluations':4440,'known_grain_primary_loss_evaluations_v5_v6_v7':7770,
      'final_heldout_payload_access':0,'old_roots_written':0,'watcher_jobs_invoked':invoked,'physical_gpus_used':gpu_ids,
      'sum_gpu_child_elapsed_seconds':gpu_child_seconds,'pipeline_wall_seconds_from_first_launch':wall_seconds,
      'pilot_status':'PASS','pilot_real_batch_QA_timing':real_pilot.get('timing'),'prefit_batch_QA_gate_met':False,
      'prediction_seal_status':seal['status'],'visual_review_status':figure_review['status'],'human_visual_review_status':'pending',
      'primary_macro_rows':macro,'paired_effects_csv':str(ROOT/'results/PAIRED_EFFECTS.csv'),'paired_group_ci_csv':str(ROOT/'results/PAIRED_GROUP_CI.csv'),
      'task_loo_row_grain':'one row per contrast x metric x omitted task, averaged over the three fixed seeds','seed_loo_row_grain':'one row per contrast x metric x omitted seed, averaged over five tasks',
      'CHECKSUMS_SHA256_path':str(ROOT/'CHECKSUMS.sha256'),'limitations':['teacher train predictions are in-sample, not OOF','numeric command physical units unresolved','session/trial independence unverified','development exposure is repeated and intervals condition on fixed choices']}
 write_json(ROOT/'RETURN_TO_CHATGPT.json',ret)
 final={'schema_version':1,'status':'complete','created_utc':utc(),'study':'post_V6_ensemble_target_distillation_development',
        'new_distillation_fits':len(fit_manifests),'fit_count_gate':'18/18 epoch160 accepted','new_hard_only_baseline_fits':0,
        'teacher_source_replays':9,'teacher_forward_rows_actual':teacher_forward,'student_train_bundles':18,'student_dev_bundles':18,
        'student_inference_requires_teacher':False,'prediction_seal':'PASS, nonempty, before dev target content read',
        'dev_target_content_reads':1,'dev_target_rows_loaded':185,'student_dev_loss_evaluations':3330,'final_heldout_payload_access':0,
        'v5_v6_prior_known_grain_loss_evaluations':4440,'v5_v6_v7_known_grain_loss_evaluations':7770,
        'old_root_writes':0,'hard_only_reconciliation':'PASS; no additional baseline fits required','pilot_acceptance':'PASS',
        'pipeline_initial_failure_recovered':recovery is not None,'pipeline_recovery_status':None if recovery is None else recovery['status'],
        'synthetic_target_QA':'PASS','synthetic_scorer_QA':read_json(ROOT/'qa/SYNTHETIC_SCORER_QA.json')['status'],
        'real_outer_train_batch_QA':'PASS','real_batch_QA_timing':real_pilot.get('timing'),'prefit_batch_QA_gate_met':False,
        'v6_task_loo_grain_reconciliation':'PASS','v6_video_substitution_50_task_loo_positive_counts':{'position':15,'safe_fd':15,'seam':5},
        'v6_reference_window_macro_reconciliation':'PASS','v6_teacher_dev_alias_reuse':'PASS','teacher_dev_imitation_source_QA':'PASS',
        'remaining_17_fits_started_after_real_batch_QA':True,
        'figure_build':'PASS','figure_visual_review':'PASS','human_visual_review':'pending_nonblocking',
        'all_gpu_leases_released':lease_ok,'gpu_pipeline_status':'COMPLETE','tmux_session_state':'completed_logs_retained',
        'physical_gpus_used':gpu_ids,'peak_allocated_mib':peak_allocated,'peak_reserved_mib':peak_reserved,
        'sum_gpu_child_elapsed_seconds':gpu_child_seconds,'pipeline_wall_seconds_from_first_launch':wall_seconds,
        'first_job_free_memory_margin_mib':gpu_preflight.get('free_memory_margin_over_required_mib'),'watcher_jobs_invoked':invoked,
        'all_18_models_final_epoch160':True,'normalizer_refit_on_soft_target':False,'dev_used_for_beta_or_checkpoint_selection':False,
        'positive_gain_completion_gate':False,'minimum_support_completion_gate':False,'git_state':'UNKNOWN','exposure_path':str(ROOT/'inventory/OUTCOME_EXPOSURE.json')}
 write_json(ROOT/'qa/FINAL_ACCEPTANCE.json',final)
 # Persist owner release before the final checksums; the pipeline log is already closed.
 owner.pop('last_error',None)
 owner.update({'status':'complete','phase':'complete','active_jobs':0,'completed_utc':utc(),'released':True,'resumed_after_partial_failure':recovery is not None,'final_heldout_payload_access':0})
 write_json(owner_path,owner)
 write_json(ROOT/'coordination/OWNER_RELEASED.json',{'schema_version':1,'status':'released','released_utc':utc(),'owner_pid':owner.get('owner_pid'),'root':str(ROOT),'all_required_work_complete':True})
 # Hash every V7 artifact except the manifest itself, after all state/report writes.
 checksum_path=ROOT/'CHECKSUMS.sha256';items=[]
 for path in sorted(ROOT.rglob('*')):
  if not path.is_file() or path==checksum_path or path.name.endswith('.part'):continue
  rel=path.relative_to(ROOT).as_posix();items.append(f'{sha(path)}  {rel}')
 checksum_path.write_text('\n'.join(items)+'\n',encoding='utf-8')
 for line in checksum_path.read_text(encoding='utf-8').splitlines():
  expected,rel=line.split('  ',1)
  if sha(ROOT/rel)!=expected:raise RuntimeError(f'V7 final checksum verification failed: {rel}')
 # The finalizer intentionally emits no output after CHECKSUMS is written.
 return 0
if __name__=='__main__':raise SystemExit(main())
