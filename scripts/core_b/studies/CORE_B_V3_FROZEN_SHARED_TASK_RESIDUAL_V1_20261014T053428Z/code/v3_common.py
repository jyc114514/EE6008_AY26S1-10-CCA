"""Read-only V2 adapters and feature interface for the isolated V3 study."""
from __future__ import annotations
import os
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
LOCKED=ROOT/'code/locked_v2'
sys.path.insert(0,str(LOCKED/'locked_source'))
sys.path.insert(0,str(LOCKED))
from D3_model import WIDTH,ACTION_DIM  # noqa: E402
from d3_eq_contract import per_sample_loss,shuffled_epoch_order  # noqa: E402
from prepare_v2_data import Q0ShardedGrid  # noqa: E402
from v2_model import V2TaskConditionedPredictor,state_hash  # noqa: E402
TASK_IDS=(119,94,90,141,230)
SCREEN_SEED=202610130
FORMAL_SEEDS=(202610131,202610132,202610133)

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def read_json(p:Path): return json.loads(p.read_text(encoding='utf-8'))

def atomic_json(p:Path,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(f'.{p.name}.{os.getpid()}.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 os.replace(tmp,p)

def load_base_map():return read_json(ROOT/'inventory/BASE_STAGE_MAP.json')

def select_base(stage:str,model_id:str,seed:int):
 groups={'screening':load_base_map()['screening_bases'],'formal':load_base_map()['formal_bases'],'probe_control':load_base_map()['probe_control_bases']}
 found=[x for x in groups[stage] if x['model_id']==model_id and int(x['seed'])==int(seed)]
 if len(found)!=1:raise RuntimeError(f'expected exactly one {stage} base for {model_id}/{seed}, found {len(found)}')
 return found[0]

def load_prefix_archive(path:Path,kind:str)->dict[str,np.ndarray]:
 with np.load(path,allow_pickle=False) as a:
  if kind=='train':
   keys=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native','inner_role')
   if not {'target_native_h30'}<=set(a.files):raise RuntimeError('accepted train archive has no training labels')
  elif kind=='dev':
   keys=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')
   if 'target_native_h30' in a.files:raise RuntimeError('dev prefix archive contains target key')
  else:raise ValueError(kind)
  if not set(keys)<=set(a.files):raise RuntimeError(f'{kind} prefix missing fields: {sorted(set(keys)-set(a.files))}')
  return {k:a[k].copy() for k in keys}

def make_prefix(data:dict[str,np.ndarray],indices:np.ndarray,normalizer:dict,grid,video_mode:str,device):
 state_mean=np.asarray(normalizer['state_mean'],dtype=np.float32)
 state_std=np.asarray(normalizer['state_safe_std'],dtype=np.float32)
 rel_mean=np.float32(normalizer['relative_delta_mean']);rel_std=np.float32(normalizer['relative_delta_safe_std'])
 state=(data['history_native'][indices]-state_mean[None,None])/state_std[None,None]
 rel=(data['relative_delta'][indices]-rel_mean)/rel_std
 prefix={'state':torch.as_tensor(state,device=device,dtype=torch.float32),
  'history_mask':torch.as_tensor(data['history_mask'][indices],device=device,dtype=torch.float32),
  'relative_delta':torch.as_tensor(rel,device=device,dtype=torch.float32),
  'anchor_command':torch.as_tensor(data['anchor_native'][indices],device=device,dtype=torch.float32)}
 if video_mode!='STATE-CTRL':
  if grid is None:raise RuntimeError('video model lacks accepted Q0 reader')
  features=np.stack([grid.row(int(data['cache_index'][i]),str(data['anchor_id'][i]),str(data['episode_id'][i])) for i in indices]).astype(np.float32,copy=False)
  vmean=np.asarray(normalizer['video_mean'],dtype=np.float32)[None,None,None,:]
  vstd=np.asarray(normalizer['video_safe_std'],dtype=np.float32)[None,None,None,:]
  prefix['dynamic']=torch.as_tensor((features[:,0]-vmean)/vstd,device=device,dtype=torch.float32)
  prefix['static']=torch.as_tensor((features[:,1]-vmean)/vstd,device=device,dtype=torch.float32)
 task=torch.as_tensor(data['task_rank'][indices].astype(np.int64),device=device,dtype=torch.long)
 return prefix,task

def load_base(base:dict,device):
 manifest=read_json(Path(base['model_manifest_path']))
 model_path=Path(base['model_path'])
 if manifest.get('status')!='PASS' or sha(model_path)!=manifest.get('model_file_sha256') or sha(model_path)!=base.get('model_sha256'):
  raise RuntimeError(f'accepted V2 base weight hash check failed: {model_path}')
 model=V2TaskConditionedPredictor(seed=int(base['seed']),video_mode=str(base['video_mode']),waypoint_mode=None)
 state=load_file(str(model_path),device='cpu')
 missing,unexpected=model.load_state_dict(state,strict=True)
 if missing or unexpected:raise RuntimeError(f'strict V2 safe load mismatch: {missing}/{unexpected}')
 before=state_hash(model)
 if before!=manifest.get('final_state_hash'):raise RuntimeError('V2 reloaded state hash differs from accepted manifest')
 model=model.to(device).eval()
 for p in model.parameters():p.requires_grad_(False)
 if any(p.requires_grad for p in model.parameters()):raise RuntimeError('V2 base parameter not frozen')
 if model.output.in_features!=WIDTH or model.output.out_features!=ACTION_DIM:
  raise RuntimeError(f'output head changed: {model.output}')
 return model,before,manifest

def capture_forward(model,prefix,task_rank):
 captured={}
 def prehook(module,args):
  if len(args)!=1 or args[0].ndim!=3:raise RuntimeError('unexpected output readout activation')
  captured['h']=args[0].detach()
 hook=model.output.register_forward_pre_hook(prehook)
 try:
  with torch.no_grad():out=model(prefix,task_rank)
 finally:hook.remove()
 if 'h' not in captured:raise RuntimeError('final output projection prehook did not run')
 h=captured['h']
 if h.shape[1:]!=(30,128) or out.action_residual.shape!=(len(task_rank),30,28):
  raise RuntimeError(f'h/output shape mismatch {tuple(h.shape)}/{tuple(out.action_residual.shape)}')
 return h,out.action_residual.detach()

def episode_task_weights(data,indices):
 indices=np.asarray(indices,dtype=np.int64)
 if not len(indices):raise ValueError('empty training population')
 tasks=data['task_id'][indices].astype(np.int64);episodes=data['episode_id'][indices].astype(np.int64)
 unique_tasks=sorted(set(int(x) for x in tasks.tolist()))
 if set(unique_tasks)!=set(TASK_IDS):raise RuntimeError(f'V3 shared readout requires all fixed tasks; got {unique_tasks}')
 weights=np.empty(len(indices),dtype=np.float64);W=len(indices)
 for task in TASK_IDS:
  pos=np.flatnonzero(tasks==task);eps=episodes[pos]
  ue,inv,count=np.unique(eps,return_inverse=True,return_counts=True)
  E=len(ue)
  weights[pos]=W/(5.0*E*count[inv])
 if not np.isfinite(weights).all() or np.any(weights<=0) or abs(weights.sum()-W)>1e-8:
  raise RuntimeError('V3 sample weights failed exact full-population sum')
 return weights
