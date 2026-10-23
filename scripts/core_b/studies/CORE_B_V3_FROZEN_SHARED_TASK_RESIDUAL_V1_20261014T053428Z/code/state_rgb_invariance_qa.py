#!/usr/bin/env python3
"""Verify state-only base and trained nonzero R_STATE outputs ignore RGB blocks."""
from __future__ import annotations
import argparse,csv,json,os,time,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from v3_common import ROOT,V2,TASK_IDS,read_json,atomic_json,sha,load_prefix_archive,make_prefix,select_base,load_base,capture_forward,Q0ShardedGrid
from train_readout import Readout

def read_csv(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def find_fit(program,seed):
 root=ROOT/'model_only/formal'/program/f'seed{seed}'
 found=[]
 for p in root.glob('attempt_*/FIT_COMPLETE.json'):
  d=read_json(p)
  if d.get('status')=='PASS':found.append((p,d))
 if len(found)!=1:raise RuntimeError(f'expected one accepted readout fit {program}/{seed}, got {len(found)}')
 return found[0]
def set_donor(prefix,row_pos,raw,normalizer,device):
 vm=np.asarray(normalizer['video_mean'],dtype=np.float32)[None,None,:]
 vs=np.asarray(normalizer['video_safe_std'],dtype=np.float32)[None,None,:]
 prefix['dynamic'][row_pos]=torch.as_tensor((raw[0]-vm[0])/vs[0],device=device,dtype=torch.float32)
 prefix['static'][row_pos]=torch.as_tensor((raw[1]-vm[0])/vs[0],device=device,dtype=torch.float32)
def run(job_path):
 job=read_json(job_path);out=Path(job['output_root']);marker=out/'STATE_RGB_QA_COMPLETE.json'
 if marker.is_file() and read_json(marker).get('status')=='PASS':print(json.dumps({'status':'already_complete','job_id':job['job_id']}));return 0
 if out.exists() and any(out.iterdir()):raise RuntimeError(f'preserving incomplete state RGB QA: {out}')
 out.mkdir(parents=True,exist_ok=True);started=time.time();statusp=out/'RUN_STATUS.json'
 def status(s,**kw):atomic_json(statusp,{'schema_version':1,'job_id':job['job_id'],'status':s,'updated_utc':time.time(),**kw})
 status('running',gpu_watcher_physical=os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','unknown'))
 try:
  if not torch.cuda.is_available():raise RuntimeError('state RGB invariance QA requires a watcher-authorized CUDA device')
  seed=int(job['seed']);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);device=torch.device('cuda:0')
  base=select_base('formal','G-S',seed);model,before,_=load_base(base,device)
  norm=read_json(Path(base['normalizer_path']));data=load_prefix_archive(Path(job['dev_prefix_path']),'dev')
  grid=Q0ShardedGrid(read_json(Path(job['cache_manifest_path'])))
  donor_map={x['receiver_anchor_id']:x for x in read_csv(Path(job['donor_map_path']))}
  fixtures=[]
  for task in TASK_IDS:
   inds=np.flatnonzero(data['task_id']==task)
   if not len(inds):raise RuntimeError(f'no dev prefix fixture for task {task}')
   ix=min(inds.tolist(),key=lambda i:hashlib.sha256(('v3statefixture|'+str(data['anchor_id'][i])).encode()).hexdigest())
   fixture_data=load_prefix_archive(Path(job['dev_prefix_path']),'dev')
   idx=np.asarray([ix],dtype=np.int64)
   # Request the exact accepted normalized video input blocks while keeping the base STATE-CTRL.
   matched,task_rank=make_prefix(fixture_data,idx,norm,grid,'TS11-D',device)
   matched_h,matched_z=capture_forward(model,matched,task_rank)
   conds={'MATCHED':matched}
   zero={k:v.clone() for k,v in matched.items()};zero['dynamic'].zero_();zero['static'].zero_();conds['ZERO_NORMALIZED_BLOCK']=zero
   receiver=str(data['anchor_id'][ix]);dr=donor_map[receiver]
   if dr['donor_status']!='PASS' or int(dr['task_id'])!=task or int(dr['donor_episode_id'])==int(data['episode_id'][ix]):raise RuntimeError(f'invalid state fixture donor {receiver}')
   donor_i=np.flatnonzero(fixture_data['anchor_id'].astype(str)==dr['donor_anchor_id'])
   if len(donor_i)!=1:raise RuntimeError('donor anchor missing from V2 label-free prefix')
   donor_raw=grid.row(int(fixture_data['cache_index'][donor_i[0]]),str(dr['donor_anchor_id']),str(dr['donor_episode_id']))
   donor={k:v.clone() for k,v in matched.items()};set_donor(donor,0,donor_raw,norm,device);conds['SAME_TASK_DONOR']=donor
   shared_marker,shared_fit=find_fit('SHARED_RES_STATE',seed);task_marker,task_fit=find_fit('TASK_RES_STATE',seed)
   shared=Readout('SHARED_RES_STATE').to(device).eval();shared.load_state_dict(load_file(shared_fit['readout_path'],device='cpu'),strict=True)
   task_readout=Readout('TASK_RES_STATE').to(device).eval();task_readout.load_state_dict(load_file(task_fit['readout_path'],device='cpu'),strict=True)
   shared_match_output=shared(matched_h,task_rank)
   task_match_output=task_readout(matched_h,task_rank)
   shared_nonzero=float(shared_match_output.detach().abs().max().cpu())>0
   task_module=task_readout.heads[int(data['task_rank'][ix])]
   task_nonzero=float(task_match_output.detach().abs().max().cpu())>0
   rows=[]
   for name,prefix in conds.items():
    h,z=capture_forward(model,prefix,task_rank)
    rh_s=shared(h,task_rank);rh_t=task_readout(h,task_rank)
    base_delta=float((z-matched_z).abs().max().cpu());h_delta=float((h-matched_h).abs().max().cpu())
    # Fixture weights are deterministic and nonzero if a fitted R head is exactly zero.
    fixture=not (shared_nonzero and task_nonzero)
    if fixture:
     w=torch.arange(128*28,device=device,dtype=torch.float32).reshape(28,128)*1e-7
     b=torch.linspace(-1e-3,1e-3,28,device=device)
     synthetic=torch.nn.functional.linear(h,w,b)
     synthetic_match=torch.nn.functional.linear(matched_h,w,b)
     synthetic_delta=float((synthetic-synthetic_match).abs().max().cpu())
    else:synthetic_delta=0.0
    rows.append({'condition':name,'h_max_abs_change':h_delta,'base_output_max_abs_change':base_delta,
     'shared_R_max_abs_change':float((rh_s-shared_match_output).abs().max().cpu()),
     'task_R_max_abs_change':float((rh_t-task_match_output).abs().max().cpu()),
     'matched_shared_R_output_max_abs':float(shared_match_output.detach().abs().max().cpu()),
     'matched_task_R_output_max_abs':float(task_match_output.detach().abs().max().cpu()),
     'shared_R_nonzero':shared_nonzero,'task_R_head_nonzero':task_nonzero,'synthetic_nonzero_R_fixture_used':fixture,
     'synthetic_R_output_max_abs_change':synthetic_delta,'non_rgb_prefix_fields_unchanged':all(torch.equal(prefix[k],matched[k]) for k in ('state','history_mask','relative_delta','anchor_command')),
     'donor_episode_id':dr['donor_episode_id'],'donor_anchor_id':dr['donor_anchor_id']})
   if any(r['h_max_abs_change']!=0 or r['base_output_max_abs_change']!=0 or r['shared_R_max_abs_change']!=0 or r['task_R_max_abs_change']!=0 or r['synthetic_R_output_max_abs_change']!=0 or not r['non_rgb_prefix_fields_unchanged'] for r in rows):
    raise RuntimeError(f'state-only base/readout changed under RGB condition task={task}: {rows}')
   fixtures.append({'task_id':task,'anchor_id':receiver,'episode_id':int(data['episode_id'][ix]),'seed':seed,'conditions':rows,
    'matched_R_shared_norm':float(shared.shared.weight.detach().norm().cpu()),'matched_R_task_head_norm':float(task_module.weight.detach().norm().cpu())})
  if __import__('v3_common').state_hash(model)!=before:raise RuntimeError('state-only base changed in RGB invariance QA')
  torch.cuda.synchronize()
  receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'seed':seed,'base_model_id':'G-S',
   'base_model_sha256':base['model_sha256'],'readout_programs':['SHARED_RES_STATE','TASK_RES_STATE'],
   'fixture_tasks':list(TASK_IDS),'fixtures':fixtures,'device':'GPU','physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),
   'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':str(device),
   'peak_memory_allocated_mib':torch.cuda.max_memory_allocated(device)/(1024**2),'peak_memory_reserved_mib':torch.cuda.max_memory_reserved(device)/(1024**2),
   'elapsed_seconds':time.time()-started,'dev_targets_loaded':False,'final_heldout_payload_access':0}
  atomic_json(marker,receipt);status('PASS',completion_marker=str(marker));print(json.dumps(receipt,sort_keys=True));return 0
 except Exception as e:
  status('FAILED',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),dev_targets_loaded=False,final_heldout_payload_access=0);raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);raise SystemExit(run(ap.parse_args().job_json))
