#!/usr/bin/env python3
"""Build standalone train/dev inference configs after all nine models are accepted."""
from __future__ import annotations
import os
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''));ROOT=Path(__file__).resolve().parents[1]
PYTHON=PROJECT/'envs/jyc/bin/python';WATCHER=PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'
PROGRAMS=('GT_RETRAIN','SELF_SOFT','SV_SOFT');SEEDS=(202610131,202610132,202610133)
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def main():
 recipe=json.loads((ROOT/'protocol/RECIPE_LOCK.json').read_text());branch=json.loads((ROOT/'protocol/BRANCH_LOCK.json').read_text());resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text())
 if branch.get('branch')!='B_SYNCHRONOUS_VIDEO_9' or recipe.get('formal_fits_expected')!=9:raise RuntimeError('inference matrix differs from branch/recipe locks')
 jobs=[]
 for seed in SEEDS:
  entries=[]
  for dataset,prefix,expected in [('train',Path(recipe['train_prefix_path']),1280),('dev',Path(resolved['dev_prefix_npz']),185)]:
   for program in PROGRAMS:
    slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01';manifest_path=slot/'MODEL_MANIFEST.json';model_path=slot/'model.safetensors'
    if not manifest_path.is_file() or not model_path.is_file():raise RuntimeError(f'model not accepted before inference config creation: {program}/{seed}')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status')!='PASS' or int(manifest.get('seed',-1))!=seed or manifest.get('video_mode')!='TS11-D':raise RuntimeError(f'model manifest rejected: {program}/{seed}')
    if sha(model_path)!=manifest.get('model_file_sha256'):raise RuntimeError(f'model hash mismatch: {program}/{seed}')
    job_id=f'v8_pred_{dataset}_{program}_s{seed}';out=ROOT/'predictions'/dataset/program/f'seed{seed}'/'attempt_01'
    cfg={'schema_version':1,'job_id':job_id,'program_id':program,'seed':seed,'video_mode':'TS11-D','dataset_role':dataset,
      'model_path':str(model_path),'model_sha256':sha(model_path),'model_manifest_path':str(manifest_path),'model_manifest_sha256':sha(manifest_path),
      'normalizer_path':recipe['normalizer_path'],'normalizer_sha256':recipe['normalizer_sha256'],
      'prefix_path':str(prefix),'prefix_sha256':sha(prefix),'expected_rows':expected,'batch_size':64,'output_root':str(out),
      'branch_lock_sha256':sha(ROOT/'protocol/BRANCH_LOCK.json'),'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),
      'final_heldout_payload_access':0}
    cp=ROOT/'configs/inference'/f'{job_id}.json';write(cp,cfg)
    entries.append({'job_id':job_id,'program_id':program,'dataset_role':dataset,'config_path':str(cp),'config_sha256':sha(cp),'output_root':str(out)})
  triplet_root=ROOT/'predictions/seed_triplets'/f'seed{seed}';triplet_root.mkdir(parents=True,exist_ok=True)
  triplet={'schema_version':1,'seed':seed,'program_order':list(PROGRAMS),'dataset_order':['train','dev'],'jobs':entries,
    'branch_lock_sha256':sha(ROOT/'protocol/BRANCH_LOCK.json'),'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),
    'stop_sentinel':str(ROOT/'coordination/STOP_REQUESTED'),'output_root':str(triplet_root)}
  tp=ROOT/'configs/inference_triplets'/f'v8_infer_triplet_s{seed}.json';write(tp,triplet)
  watcher={'job_id':f'v8_infer_triplet_s{seed}','protocol_id':f'coreb_v8_branch_b_inference_s{seed}','output_root':str(triplet_root),
   'command':[str(PYTHON),str(ROOT/'code/v8_run_infer_seed_triplet.py'),'--triplet-json',str(tp)],'cwd':str(ROOT),
   'completion_marker':'SEED_INFERENCE_COMPLETE.json','preferred_gpu':{202610131:0,202610132:1,202610133:2}[seed],
   'resource_estimate':{'measured_peak_mib':recipe['resource_estimate']['measured_peak_mib'],'basis':'conservative V8 Stage0 replay/gradient measured peak reserved'},
   'log_path':str(ROOT/'logs'/f'inference_seed{seed}_watcher_child.log')}
  wp=ROOT/'coordination/jobs'/f'v8_infer_triplet_s{seed}.watcher.json';write(wp,watcher)
  jobs.append({'seed':seed,'job_id':watcher['job_id'],'program_order':list(PROGRAMS),'bundle_count':6,'triplet_config_path':str(tp),'triplet_config_sha256':sha(tp),'watcher_job_path':str(wp),'watcher_job_sha256':sha(wp),'preferred_gpu':watcher['preferred_gpu']})
 plan={'schema_version':1,'status':'PASS','created_utc':utc(),'branch':'B_SYNCHRONOUS_VIDEO_9','branch_lock_sha256':sha(ROOT/'protocol/BRANCH_LOCK.json'),
  'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),'train_bundles_expected':9,'dev_bundles_expected':9,'logical_bundles_expected':18,
  'dev_target_archive_opened':False,'final_heldout_payload_access':0,'seed_triplet_jobs':jobs}
 write(ROOT/'coordination/INFERENCE_JOB_PLAN.json',plan)
 print(json.dumps({'status':'PASS','train_bundles':9,'dev_bundles':9,'seed_triplets':3},sort_keys=True))
if __name__=='__main__':main()
