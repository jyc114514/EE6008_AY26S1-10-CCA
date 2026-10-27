#!/usr/bin/env python3
"""Recover paired results from sealed saved losses, then measure latency and build figures."""
from __future__ import annotations
import os
import hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
PYTHON=PROJECT/'envs/jyc/bin/python';SESSION='coreb-v7-score-recovery-20261015t1305z'
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def read_json(p:Path):return json.loads(p.read_text(encoding='utf-8'))
def write(p:Path,obj)->None:p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(f'.{p.name}.{os.getpid()}.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(p)
def owner(**updates):
 p=ROOT/'coordination/OWNER_STATE.json';d=json.loads(p.read_text());d.update(updates);d['updated_utc']=utc();d['owner_pid']=os.getpid();d['tmux_session']=SESSION;write(p,d)
def step(label:str,cmd:list[str]):
 owner(status='running',phase=label,active_jobs=0);print(f'[{utc()}] START {label}: {cmd}',flush=True)
 r=subprocess.run(cmd,cwd=str(PROJECT),check=False);print(f'[{utc()}] END {label}: rc={r.returncode}',flush=True)
 if r.returncode!=0:raise RuntimeError(f'{label} failed rc={r.returncode}')
def main()->int:
 lock=json.loads((ROOT/'coordination/OWNER.lock').read_text());state=json.loads((ROOT/'coordination/OWNER_STATE.json').read_text())
 if lock.get('owner_token')!=state.get('owner_token'):raise RuntimeError('V7 owner token mismatch before score recovery')
 seal=json.loads((ROOT/'qa/PREDICTION_SEAL.json').read_text())
 if seal.get('status')!='SEALED_BEFORE_TARGET_ACCESS' or seal.get('accepted_dev_bundles')!=18:raise RuntimeError('score recovery requires a full sealed V7 prediction matrix')
 started=utc();owner(status='running',phase='score_recovery_startup',active_jobs=0)
 try:
  step('recover_paired_results_from_saved_losses',[str(PYTHON),str(ROOT/'code/v7_recover_score.py')])
  step('cached_feature_latency_queue',[str(PYTHON),str(ROOT/'code/v7_run_queue.py'),'--queue',str(ROOT/'coordination/queues/LATENCY.json')])
  step('figure_build',[str(PYTHON),str(ROOT/'code/v7_make_figures.py')])
  gpu=read_json(ROOT/'coordination/GPU_EXECUTION.json') if (ROOT/'coordination/GPU_EXECUTION.json').exists() else {}
  gpu.update({'status':'GPU_AND_SCORING_PIPELINE_COMPLETE_AWAITING_VISUAL_REVIEW','completed_utc':utc(),'score_recovered_from_saved_per_window_losses':True,
              'additional_dev_target_reads_during_recovery':0,'tmux_session':SESSION,'recovery_main_pid':os.getpid(),
              'recovery_log_path':str(ROOT/'logs/RECOVERY_PIPELINE.log'),'recovery_reattach_command':f'tmux attach -t {SESSION}',
              'final_heldout_payload_access':0})
  write(ROOT/'coordination/GPU_EXECUTION.json',gpu)
  launches=read_json(ROOT/'coordination/TMUX_LAUNCHES.json');launches['pipeline_attempts'].append({'attempt':len(launches['pipeline_attempts'])+1,'pipeline_started_utc':started,
       'tmux_pane_pid':'recorded by recovery launch','result':'scorer recovery reused saved window losses; latency queue and figures complete','log_same_file':False,'log_path':str(ROOT/'logs/RECOVERY_PIPELINE.log')})
  write(ROOT/'coordination/TMUX_LAUNCHES.json',launches)
  owner(status='running',phase='awaiting_visual_review',active_jobs=0,recovered_score_completed_utc=utc())
  print(f'[{utc()}] COMPLETE score recovery; root={ROOT}; awaiting figure visual review',flush=True);return 0
 except Exception as exc:
  rec={'schema_version':1,'status':'partial','failed_utc':utc(),'error_type':type(exc).__name__,'error':str(exc),'final_heldout_payload_access':0}
  write(ROOT/'coordination/RECOVERY_FAILURE.json',rec)
  with (ROOT/'qa/FAILED_ATTEMPTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False,sort_keys=True)+'\n')
  owner(status='partial',phase='score_recovery_failure',active_jobs=0,last_error=f'{type(exc).__name__}: {exc}')
  print(f'[{utc()}] PARTIAL {type(exc).__name__}: {exc}',flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
