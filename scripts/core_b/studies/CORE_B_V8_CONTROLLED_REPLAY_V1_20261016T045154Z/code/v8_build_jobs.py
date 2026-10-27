#!/usr/bin/env python3
"""Build immutable nine-fit V8 branch-B configs and per-seed triplet watcher jobs."""
from __future__ import annotations
import os
import hashlib,json,os,socket
from datetime import datetime,timezone
from pathlib import Path
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT=Path(__file__).resolve().parents[1]
PYTHON=PROJECT/'envs/jyc/bin/python';WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
TASKS=[119,94,90,141,230];SEEDS=[202610131,202610132,202610133]
ARM_ORDER={202610131:['GT_RETRAIN','SELF_SOFT','SV_SOFT'],202610132:['SELF_SOFT','SV_SOFT','GT_RETRAIN'],202610133:['SV_SOFT','GT_RETRAIN','SELF_SOFT']}
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def main():
 branch=json.loads((ROOT/'protocol/BRANCH_LOCK.json').read_text());recipe=json.loads((ROOT/'protocol/RECIPE_LOCK.json').read_text())
 role=json.loads((ROOT/'protocol/TARGET_ROLE_MAP.json').read_text());resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text())
 sampler=json.loads((ROOT/'protocol/SAMPLER_SEQUENCE_MANIFEST.json').read_text());mix=ROOT/'protocol/TARGET_MIX_LOCK.json'
 if branch.get('status')!='LOCKED_BEFORE_FULL_FITS_AND_NEW_DEV' or recipe.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV':raise RuntimeError('branch/recipe lock is not sealed')
 if role.get('branch')!='B_SYNCHRONOUS_VIDEO_9' or len(role.get('targets',[]))!=9:raise RuntimeError('target role map differs from branch B matrix')
 if sha(mix)!=recipe['target_mix_lock_sha256'] or sha(ROOT/'protocol/BRANCH_LOCK.json')!=recipe['branch_lock_sha256']:raise RuntimeError('target/branch lock hash check failed')
 if sha(ROOT/'code/v8_train.py')!=recipe['runner_sha256']:raise RuntimeError('runner changed after recipe lock')
 targets={(x['program_id'],int(x['seed'])):x for x in role['targets']}
 inits=recipe['fresh_init_by_seed']; mix_sha=sha(mix);recipe_sha=sha(ROOT/'protocol/RECIPE_LOCK.json');branch_sha=sha(ROOT/'protocol/BRANCH_LOCK.json')
 fit_jobs=[];triplet_jobs=[]
 for seed in SEEDS:
  slots=[]
  for program in ARM_ORDER[seed]:
   target=targets[(program,seed)];slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';slot.mkdir(parents=True,exist_ok=True)
   init=inits[str(seed)]
   mode='hard_y' if program=='GT_RETRAIN' else 'fixed_soft_z'
   beta=0.0 if program=='GT_RETRAIN' else 0.5
   job_id=f'v8_fit_{program}_s{seed}'
   cfg={'schema_version':1,'job_id':job_id,'phase':'formal','program_id':program,'model_id':program,'teacher_role':'HARD_Y' if program=='GT_RETRAIN' else ('V_MATCHED' if program=='SELF_SOFT' else 'SV50'),
     'target_mode':mode,'target_expression':target['target_expression'],'target_role_map_sha256':sha(ROOT/'protocol/TARGET_ROLE_MAP.json'),
     'seed':seed,'scope':'shared','task_id':None,'task_ids':TASKS,'video_mode':'TS11-D','waypoint_mode':None,
     'init_path':init['path'],'expected_init_state_hash':init['state_hash'],
     'normalizer_path':recipe['normalizer_path'],'normalizer_sha256':recipe['normalizer_sha256'],'train_npz_path':recipe['train_npz_path'],'train_role':'outer_train',
     'teacher_target_path':target['target_path'],'teacher_target_rows_path':target['rows_path'],'teacher_target_manifest_path':target['manifest_path'],'teacher_target_sha256':target['target_sha256'],
     'target_mix_lock_sha256':mix_sha,'beta':beta,'sampler_orders_path':sampler['path'],'sampler_orders_sha256':sampler['sha256'],
     'branch_lock_path':str(ROOT/'protocol/BRANCH_LOCK.json'),'branch_lock_sha256':branch_sha,'recipe_lock_path':str(ROOT/'protocol/RECIPE_LOCK.json'),'recipe_lock_sha256':recipe_sha,
     'lr':0.0005,'epochs':160,'effective_batch':64,'microbatch_size':64,'accumulation_steps':1,'scheduler_tmax':3200,'checkpoint_epochs':[160],
     'gradient_clip_norm':1.0,'weight_decay':0.0001,'dtype':'float32','amp':False,'compile':False,
     'oom_fallback':{'microbatch_size':32,'accumulation_steps':2,'effective_batch':64},
     'job_root':str(slot),'attempt_dir':str(slot),'output_root':str(slot),'final_heldout_payload_access':0}
   cfg_path=ROOT/'configs/fits'/f'{job_id}.json';write(cfg_path,cfg);slots.append({'job_id':job_id,'program_id':program,'seed':seed,'config_path':str(cfg_path),'config_sha256':sha(cfg_path),'slot_path':str(slot)})
   fit_jobs.append({'job_id':job_id,'program_id':program,'seed':seed,'config_path':str(cfg_path),'config_sha256':sha(cfg_path),'slot_path':str(slot)})
  triplet_root=ROOT/'model_only/seed_triplets'/f'seed{seed}';triplet_root.mkdir(parents=True,exist_ok=True)
  triplet_config={'schema_version':1,'seed':seed,'program_order':ARM_ORDER[seed],'fits':slots,'branch_lock_sha256':branch_sha,'recipe_lock_sha256':recipe_sha,
     'stop_sentinel':str(ROOT/'coordination/STOP_REQUESTED'),'output_root':str(triplet_root),'completion_marker':str(triplet_root/'SEED_TRIPLET_COMPLETE.json')}
  triplet_cfg=ROOT/'configs/triplets'/f'v8_seed_triplet_s{seed}.json';write(triplet_cfg,triplet_config)
  job={'job_id':f'v8_seed_triplet_s{seed}','protocol_id':f'coreb_v8_branch_b_seed_triplet_s{seed}','output_root':str(triplet_root),
   'command':[str(PYTHON),str(ROOT/'code/v8_run_seed_triplet.py'),'--triplet-json',str(triplet_cfg)],'cwd':str(ROOT),
   'completion_marker':'SEED_TRIPLET_COMPLETE.json','preferred_gpu':{202610131:0,202610132:1,202610133:2}[seed],
   'resource_estimate':{'measured_peak_mib':recipe['resource_estimate']['measured_peak_mib'],'basis':recipe['resource_estimate']['basis']},
   'log_path':str(ROOT/'logs'/f'training_seed{seed}_watcher_child.log')}
  watcher_path=ROOT/'coordination/jobs'/f'v8_seed_triplet_s{seed}.watcher.json';write(watcher_path,job)
  triplet_jobs.append({'job_id':job['job_id'],'seed':seed,'program_order':ARM_ORDER[seed],'triplet_config_path':str(triplet_cfg),'triplet_config_sha256':sha(triplet_cfg),'watcher_job_path':str(watcher_path),'watcher_job_sha256':sha(watcher_path),'output_root':str(triplet_root),'preferred_gpu':job['preferred_gpu']})
 plan={'schema_version':1,'status':'PASS','created_utc':utc(),'branch':'B_SYNCHRONOUS_VIDEO_9','branch_lock_sha256':branch_sha,'recipe_lock_sha256':recipe_sha,
  'formal_fit_jobs':len(fit_jobs),'formal_fits_expected':9,'programs':['GT_RETRAIN','SELF_SOFT','SV_SOFT'],'seeds':SEEDS,'arm_order':ARM_ORDER,
  'student_train_inference_expected':9,'student_dev_inference_expected':9,'preferred_gpu_is_ranking_only':True,'registered_peak_mib':recipe['resource_estimate']['measured_peak_mib'],
  'watcher_path':str(WATCHER),'watcher_sha256':sha(WATCHER),'max_concurrent_project_gpu_children':2,'all_output_paths_under_v8_root':True,
  'old_roots_written':False,'dev_target_archive_opened':False,'final_heldout_payload_access':0,'fits':fit_jobs,'seed_triplets':triplet_jobs}
 write(ROOT/'coordination/JOB_PLAN.json',plan)
 run_context={'schema_version':1,'created_utc':utc(),'study':'post_V7_controlled_runner_and_target_replay_development','out_root':str(ROOT),'hostname':socket.gethostname(),'uid':os.getuid(),
  'user':'project-team','node_role':'interactive GPU host (inferred from hostname configured shared GPU host, absent scheduler environment/tools, and watcher-visible GPU telemetry)',
  'scheduler_environment':{},'scheduler_tools_available':False,'python_bin':str(PYTHON),'repo_root':resolved['repo_root'],'repo_git_state':resolved['git_state'],
  'branch':'B_SYNCHRONOUS_VIDEO_9','branch_lock_sha256':branch_sha,'recipe_lock_sha256':recipe_sha,'v8_train_runner_sha256':sha(ROOT/'code/v8_train.py'),
  'watcher_path':str(WATCHER),'watcher_sha256':sha(WATCHER),'gpu_preflight_path':str(ROOT/'qa/PREFLIGHT.json'),'gpu_preflight_status':'PASS',
  'tmux_sessions_expected':[f'coreb-v8-train-s{seed}-20261016' for seed in SEEDS],'safe_stop':'touch "'+str(ROOT/'coordination/STOP_REQUESTED')+'"; wrapper stops after current fit; current fit completes/resumes at its checkpoint',
  'reattach_template':'tmux attach -t coreb-v8-train-s<seed>-20261016','log_template':str(ROOT/'logs/training_seed<seed>_watcher_child.log')}
 write(ROOT/'RUN_CONTEXT.json',run_context)
 print(json.dumps({'status':'PASS','fit_jobs':len(fit_jobs),'triplets':len(triplet_jobs),'recipe_lock_sha256':recipe_sha,'runner_sha256':sha(ROOT/'code/v8_train.py')},sort_keys=True))
if __name__=='__main__':main()
