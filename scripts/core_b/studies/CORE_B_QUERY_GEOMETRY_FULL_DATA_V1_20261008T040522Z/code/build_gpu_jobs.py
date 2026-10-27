#!/usr/bin/env python3
"""Build hashable dynamic-pool watcher jobs from the locked matrix."""
from __future__ import annotations
import os
import argparse,csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=str(PROJECT/'envs/jyc/bin/python')
WATCHER=str(PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py')
PROTOCOL='CORE_B_QUERY_GEOMETRY_FULL_DATA_H30_V1'
BASE_ENV={'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}

def q1_rows():
 with (ROOT/'protocol/MATRIX12.csv').open(newline='',encoding='utf-8') as f:return [r for r in csv.DictReader(f) if r['action']=='train_had']
def preferred_gpu(config,seed):
 # Ranking preference balances paired model/seed conditions across the cool/capacity-rich observed GPUs; it is not an allowlist.
 return 2 if ((config=='TS11-D' and int(seed) in (20261001,20261003)) or (config=='STATE-CTRL' and int(seed)==20261002)) else 3

def job(job_id,run_id,phase,command,outroot,peak,preferred=None):
 value={'job_id':job_id,'protocol_id':PROTOCOL,'output_root':str(outroot),'command':[PYTHON,*command],
  'resource_estimate':{'measured_peak_mib':int(peak),'peak_basis':'pre-registered prior H30 runner peak; replaced by accepted Stage0/inference-smoke peak for formal queues','source':'protocol/RESOURCE_ESTIMATE_PRE_STAGE0.json or qa/CANARY_ACCEPTANCE.json'},
  'environment':dict(BASE_ENV),'log_path':str(ROOT/'logs'/f'{job_id}.log'),'max_oom_requeues':0,
  'run_id':run_id,'phase':phase}
 if preferred is not None:value['preferred_gpu']=int(preferred)
 return value
def write_jobs(path,jobs):
 p=ROOT/path
 if p.exists():raise FileExistsError(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in jobs),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('initial','formal','inference'),required=True);ap.add_argument('--peak-mib',type=int,default=1024);args=ap.parse_args()
 rows=q1_rows();jobs=[]
 if args.phase=='initial':
  stage_jobs=[];smoke_jobs=[];formal_template=[];infer_template=[]
  for model in ('TS11-D','STATE-CTRL'):
   row=next(x for x in rows if x['config']==model and int(x['seed'])==20261001)
   run=row['run_id'];pref=preferred_gpu(model,20261001);out=ROOT/'runs/stage0'/run
   cmd=[str(ROOT/'code/train_full_query.py'),'--config',model,'--horizon','30','--seed','20261001','--run-id',run,'--run-root',str(out),'--stage0']
   stage_jobs.append(job(f'stage0-{model}-H30-S20261001',run,'stage0',cmd,out,args.peak_mib,pref))
   icmd=[str(ROOT/'code/infer_stage0_smoke.py'),'--run-id',run,'--batch-size','64']
   smoke_jobs.append(job(f'infer-smoke-{model}-H30-S20261001',run,'infer_smoke',icmd,out,args.peak_mib,pref))
  # Freeze the two dependent phases separately so inference never starts before Stage0 checkpoint acceptance.
  write_jobs('coordination/jobs/STAGE0.jsonl',stage_jobs)
  write_jobs('coordination/jobs/INFER_SMOKE.jsonl',smoke_jobs)
  for row in rows:
   model=row['config'];seed=row['seed'];run=row['run_id'];pref=preferred_gpu(model,seed)
   out=ROOT/'runs/formal'/run
   fcmd=[str(ROOT/'code/train_full_query.py'),'--config',model,'--horizon','30','--seed',str(seed),'--run-id',run,'--run-root',str(out)]
   formal_template.append(job(f'train-{model}-H30-S{seed}',run,'formal',fcmd,out,args.peak_mib,pref))
   iout=ROOT/'predictions'/run
   infcmd=[str(ROOT/'code/infer_full_query.py'),'--run-id',run,'--split','both','--device','cuda','--batch-size','64','--threads','4','--resume']
   infer_template.append(job(f'infer-{model}-H30-S{seed}',run,'inference',infcmd,iout,args.peak_mib,pref))
  write_jobs('coordination/jobs/FORMAL_TEMPLATE_PREFERRED.jsonl',formal_template)
  write_jobs('coordination/jobs/INFERENCE_TEMPLATE_PREFERRED.jsonl',infer_template)
  jobs=stage_jobs+smoke_jobs+formal_template+infer_template
 elif args.phase=='formal':
  qa=json.loads((ROOT/'qa/CANARY_ACCEPTANCE.json').read_text())
  if qa.get('status')!='PASS_2_OF_2_TRAIN_AND_INFER_SMOKE':raise RuntimeError('Stage0 and inference smoke acceptance missing')
  measured=max(int(x['training_peak_mib']) for x in qa['cells'])
  for row in rows:
   model=row['config'];seed=row['seed'];run=row['run_id'];rid=f'train-{model}-H30-S{seed}'
   out=ROOT/'runs/formal'/run
   cmd=[str(ROOT/'code/train_full_query.py'),'--config',model,'--horizon','30','--seed',str(seed),'--run-id',run,'--run-root',str(out)]
   jobs.append(job(rid,run,'formal',cmd,out,measured,preferred_gpu(model,seed)))
  write_jobs('coordination/jobs/FORMAL.jsonl',jobs)
 elif args.phase=='inference':
  qa=json.loads((ROOT/'qa/CANARY_ACCEPTANCE.json').read_text())
  if qa.get('status')!='PASS_2_OF_2_TRAIN_AND_INFER_SMOKE':raise RuntimeError('Stage0 and inference smoke acceptance missing')
  formal=json.loads((ROOT/'qa/FORMAL_TRAINING_ACCEPTANCE.json').read_text())
  if formal.get('status')!='PASS_6_OF_6':raise RuntimeError('six formal H30 Hadamard training cells must pass before inference')
  measured=max(int(x['inference_peak_mib']) for x in qa['cells'])
  for row in rows:
   model=row['config'];seed=row['seed'];run=row['run_id'];rid=f'infer-{model}-H30-S{seed}'
   out=ROOT/'predictions'/run
   cmd=[str(ROOT/'code/infer_full_query.py'),'--run-id',run,'--split','both','--device','cuda','--batch-size','64','--threads','4','--resume']
   jobs.append(job(rid,run,'inference',cmd,out,measured,preferred_gpu(model,seed)))
  write_jobs('coordination/jobs/INFERENCE.jsonl',jobs)
 print(json.dumps({'phase':args.phase,'job_count':len(jobs),'jobs':[x['job_id'] for x in jobs],'pool':'dynamic_all_visible','current_watcher_child_slots':2,'gpu_preferences_are_not_allowlists':True},indent=2))
if __name__=='__main__':main()
