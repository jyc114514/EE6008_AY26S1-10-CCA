#!/usr/bin/env python3
"""V7 owner pipeline: teacher replay, pilot, fixed matrix, seal, score, figures."""
from __future__ import annotations
import os
import hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=PROJECT/'envs/jyc/bin/python';SESSION='coreb-v7-20261015t112834z'
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p:Path,obj)->None:p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(p)
def owner(**updates):
 p=ROOT/'coordination/OWNER_STATE.json';d=json.loads(p.read_text(encoding='utf-8'));d.update(updates);d['updated_utc']=utc();d['owner_pid']=os.getpid();d['tmux_session']=SESSION;write(p,d)
def step(label:str,cmd:list[str])->None:
 if (ROOT/'coordination/STOP_REQUESTED').exists():raise RuntimeError('V7 stop sentinel requested; active watcher child was allowed to finish')
 owner(phase=label,status='running',active_jobs=0)
 print(f'[{utc()}] START {label}: {cmd}',flush=True)
 result=subprocess.run(cmd,cwd=str(PROJECT),check=False)
 print(f'[{utc()}] END {label}: rc={result.returncode}',flush=True)
 if result.returncode!=0:raise RuntimeError(f'{label} failed with rc={result.returncode}')
def queue(name:str)->None:
 step(f'queue_{name.lower()}',[str(PYTHON),str(ROOT/'code/v7_run_queue.py'),'--queue',str(ROOT/'coordination/queues'/name)])
def fail(exc:Exception)->None:
 record={'schema_version':1,'status':'partial','failed_utc':utc(),'phase':json.loads((ROOT/'coordination/OWNER_STATE.json').read_text()).get('phase'),
         'error_type':type(exc).__name__,'error':str(exc),'final_heldout_payload_access':0}
 write(ROOT/'coordination/PIPELINE_FAILURE.json',record)
 with (ROOT/'qa/FAILED_ATTEMPTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+'\n')
 owner(status='partial',phase='pipeline_failure',active_jobs=0,last_error=f'{type(exc).__name__}: {exc}',failed_utc=utc())
 print(f'[{utc()}] PARTIAL {type(exc).__name__}: {exc}',flush=True)
def main()->int:
 started=utc();lock=json.loads((ROOT/'coordination/OWNER.lock').read_text(encoding='utf-8'));state=json.loads((ROOT/'coordination/OWNER_STATE.json').read_text(encoding='utf-8'))
 if lock.get('owner_token')!=state.get('owner_token'):raise RuntimeError('V7 exclusive owner token mismatch')
 owner(status='running',phase='startup',active_jobs=0,started_utc=started,tmux_session=SESSION,main_pid=os.getpid())
 try:
  # CPU-only checks remain outside the dev target reader.
  step('baseline_recipe_reconciliation',[str(PYTHON),str(ROOT/'code/v7_reconcile_baselines.py')])
  step('synthetic_target_forward_qa',[str(PYTHON),str(ROOT/'code/v7_protocol_qa.py'),'--phase','synthetic'])
  teacher_complete=list((ROOT/'teacher_predictions/source').glob('*/seed*/PREDICTION_COMPLETE.json'))
  if len(teacher_complete)!=9:queue('TEACHER.json')
  cpu_preflight=json.loads((ROOT/'inventory/CPU_PREFLIGHT.json').read_text(encoding='utf-8'))
  teacher_queue=json.loads((ROOT/'coordination/queues/TEACHER.json').read_text(encoding='utf-8'))
  first_teacher=teacher_queue['jobs'][0]['job_id'];watcher_snapshot=json.loads((ROOT/'coordination/watcher_status'/f'{first_teacher}.json').read_text(encoding='utf-8'))
  visible=watcher_snapshot.get('visible_snapshots',[])
  if not visible or watcher_snapshot.get('selected_gpu') is None:raise RuntimeError('approved shared watcher did not provide a host GPU resource snapshot')
  teacher_receipt=json.loads((Path(teacher_queue['jobs'][0]['output_root'])/'PREDICTION_COMPLETE.json').read_text(encoding='utf-8'))
  recheck=watcher_snapshot.get('final_recheck') or {}
  free_mib=recheck.get('free_memory_mib')
  gpu_preflight={'schema_version':1,'status':'PASS','checked_utc':utc(),'hostname':cpu_preflight['hostname'],
       'node_role':'interactive GPU host (inferred from hostname configured shared GPU host, absent scheduler environment/tools, and watcher-visible GPU telemetry)',
       'scheduler_environment':cpu_preflight['scheduler_environment'],'scheduler_commands':cpu_preflight['scheduler_commands'],
       'module_command':cpu_preflight['module_command'],'visible_gpu_pool':watcher_snapshot.get('visible_gpu_pool'),
       'visible_gpu_count':len(visible),'visible_gpu_snapshots':visible,'first_job_id':first_teacher,
       'selected_physical_gpu':watcher_snapshot.get('selected_gpu'),'first_job_final_recheck':recheck,
       'first_real_job_id':first_teacher,'first_real_job_actual_peak_allocated_mib':teacher_receipt.get('peak_memory_allocated_mib'),
       'first_real_job_actual_peak_reserved_mib':teacher_receipt.get('peak_memory_reserved_mib'),
       'registered_peak_mib':384,'required_free_memory_mib':896,
       'final_recheck_free_memory_mib':free_mib,'free_memory_margin_over_required_mib':None if free_mib is None else int(free_mib)-896,
       'max_concurrent_project_gpu_children':2,
       'other_process_pid_or_command_enumeration':False,'non_child_signals':0}
  write(ROOT/'coordination/GPU_PREFLIGHT.json',gpu_preflight)
  context=json.loads((ROOT/'RUN_CONTEXT.json').read_text(encoding='utf-8'));context['node_role']=gpu_preflight['node_role'];context['gpu_preflight_path']=str(ROOT/'coordination/GPU_PREFLIGHT.json');context['gpu_preflight_status']='PASS';write(ROOT/'RUN_CONTEXT.json',context)
  targets_complete=list((ROOT/'teacher_targets').glob('*/seed*/TARGET_MANIFEST.json'))
  if len(targets_complete)!=18:step('teacher_target_composition',[str(PYTHON),str(ROOT/'code/v7_build_teacher_targets.py')])
  step('bind_target_hashes',[str(PYTHON),str(ROOT/'code/v7_bind_target_hashes.py')])
  queue('PILOT.json')
  step('bind_pilot_model_hashes',[str(PYTHON),str(ROOT/'code/v7_bind_model_hashes.py'),'--program','ST_SV','--seed','202610131'])
  queue('PILOT_INFERENCE.json')
  step('pilot_acceptance',[str(PYTHON),str(ROOT/'code/v7_protocol_qa.py'),'--phase','pilot'])
  step('pilot_resource_reconciliation',[str(PYTHON),str(ROOT/'code/v7_update_resource_estimates.py')])
  queue('FITS_REMAINDER.json')
  step('bind_all_student_model_hashes',[str(PYTHON),str(ROOT/'code/v7_bind_model_hashes.py'),'--all'])
  queue('TRAIN_INFERENCE.json')
  queue('DEV_INFERENCE.json')
  step('prediction_seal',[str(PYTHON),str(ROOT/'code/v7_seal_predictions.py')])
  # The scorer runs its synthetic preflight and header-only schema check before its single content read.
  step('dev_score_and_trajectory_export',[str(PYTHON),str(ROOT/'code/v7_score.py')])
  queue('LATENCY.json')
  step('figure_build',[str(PYTHON),str(ROOT/'code/v7_make_figures.py')])
  write(ROOT/'coordination/GPU_EXECUTION.json',{'schema_version':1,'status':'GPU_AND_CPU_PIPELINE_COMPLETE_AWAITING_VISUAL_REVIEW','started_utc':started,'completed_utc':utc(),
       'tmux_session':SESSION,'main_pid':os.getpid(),'pilot_fit':'v7_fit_ST_SV_s202610131','new_distillation_fits':18,
       'teacher_replay_bundles':9,'student_train_prediction_bundles':18,'student_dev_prediction_bundles':18,
       'python_bin':str(PYTHON),'full_pipeline_command':[str(PYTHON),str(ROOT/'code/v7_pipeline.py')],
       'working_directory':str(PROJECT),'log_path':str(ROOT/'logs/PIPELINE.log'),
       'reattach_command':f'tmux attach -t {SESSION}',
       'safe_stop_command':f'touch "{ROOT}/coordination/STOP_REQUESTED"; active watcher child is allowed to finish, then queued jobs stop',
       'shared_watcher_sha256':sha(PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'),'final_heldout_payload_access':0})
  owner(status='running',phase='awaiting_visual_review',active_jobs=0,gpu_pipeline_completed_utc=utc())
  print(f'[{utc()}] GPU and scoring pipeline complete; root={ROOT}; awaiting figure inspection before final checksum sealing',flush=True);return 0
 except Exception as exc:
  fail(exc);return 2
if __name__=='__main__':raise SystemExit(main())
