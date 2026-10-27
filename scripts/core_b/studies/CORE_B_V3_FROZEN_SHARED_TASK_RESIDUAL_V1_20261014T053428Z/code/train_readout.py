#!/usr/bin/env python3
"""Train only the frozen-base output residual readout."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from v3_common import ROOT,V2,TASK_IDS,FORMAL_SEEDS,read_json,atomic_json,sha,load_prefix_archive,episode_task_weights,per_sample_loss,shuffled_epoch_order

class Readout(torch.nn.Module):
 def __init__(self,program:str):
  super().__init__();self.program=program
  if program.startswith('SHARED_RES_'):
   self.shared=torch.nn.Linear(128,28,bias=True)
   with torch.no_grad():self.shared.weight.zero_();self.shared.bias.zero_()
   self.heads=None
  elif program.startswith('TASK_RES_'):
   self.shared=None;self.heads=torch.nn.ModuleList([torch.nn.Linear(128,28,bias=True) for _ in TASK_IDS])
   with torch.no_grad():
    for h in self.heads:h.weight.zero_();h.bias.zero_()
  else:raise ValueError(f'unknown readout program {program}')
 def forward(self,h:torch.Tensor,task_rank:torch.Tensor)->torch.Tensor:
  if self.shared is not None:return self.shared(h)
  ids=task_rank.detach().cpu().tolist()
  return torch.stack([self.heads[int(rank)](h[i]) for i,rank in enumerate(ids)],dim=0)

def write_csv(path:Path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 fields=list(rows[0]) if rows else []
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def read_csv(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def safe_save(path:Path,tensors):
 tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
 save_file({k:v.detach().cpu().contiguous() for k,v in tensors.items()},str(tmp));os.replace(tmp,path)
def episode_macro(pred_native,target_native,task_ids,episode_ids,metric_sigma,metric_fd):
 grouped=defaultdict(lambda:defaultdict(list))
 for i,(task,episode) in enumerate(zip(task_ids,episode_ids)):
  pe=np.abs(pred_native[i]-target_native[i])/metric_sigma[None,:]
  fd=np.abs(np.diff(pred_native[i],axis=0)-np.diff(target_native[i],axis=0))/metric_fd[None,:]
  grouped[int(task)][int(episode)].append((float(pe.mean()),float(fd.mean())))
 by_task={}
 for task,episodes in grouped.items():
  epvals=[np.asarray(episodes[e],dtype=np.float64).mean(axis=0) for e in sorted(episodes)]
  arr=np.stack(epvals)
  by_task[task]={'position_mae':float(arr[:,0].mean()),'safe_fd_l1':float(arr[:,1].mean()),'episode_count':len(epvals)}
 tasks_present=[t for t in TASK_IDS if t in by_task]
 if len(tasks_present)!=5:raise RuntimeError(f'tune evaluation missing fixed task(s): {tasks_present}')
 pos=float(np.mean([by_task[t]['position_mae'] for t in TASK_IDS]))
 fd=float(np.mean([by_task[t]['safe_fd_l1'] for t in TASK_IDS]))
 return {'macro_position_mae':pos,'macro_safe_fd_l1':fd,'tune_j':pos+0.1*fd,'task_metrics':by_task}

def atomic_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)

def run(job_path:Path):
 job=read_json(job_path);out=Path(job['output_root']);marker=out/'FIT_COMPLETE.json'
 if marker.is_file() and read_json(marker).get('status')=='PASS':print(json.dumps({'status':'already_complete','job_id':job['job_id']}));return 0
 if out.exists() and any(out.iterdir()):raise RuntimeError(f'preserving incomplete readout fit output: {out}')
 out.mkdir(parents=True,exist_ok=True);started=time.time();status_path=out/'RUN_STATUS.json'
 def status(x,**kw):atomic_json(status_path,{'schema_version':1,'job_id':job['job_id'],'status':x,'updated_utc':time.time(),**kw})
 try:
  if not torch.cuda.is_available():raise RuntimeError('readout fit requires watcher-authorized CUDA device')
  stage=str(job['stage']);program=str(job['program']);seed=int(job['seed']);lam=float(job['lambda']);epochs=int(job['epochs'])
  if stage not in {'screening','formal'}:raise ValueError(stage)
  if abs(float(job['lr'])-0.001)>1e-15 or epochs<=0:raise RuntimeError('readout LR/epoch is outside the locked protocol')
  if stage=='screening' and epochs!=80:raise RuntimeError('screening fit must be 80 epochs')
  if stage=='formal':
   selection_path=ROOT/'protocol/READOUT_SELECTION.json'
   locked=json.loads(selection_path.read_text())
   if job.get('selection_sha256')!=sha(selection_path):raise RuntimeError('formal job does not bind the current immutable inner-tune selection hash')
   choice=next(x for x in locked['choices'] if x['program']==program)
   if locked.get('status')!='LOCKED' or float(choice['selected_lambda'])!=lam or int(choice['selected_epoch'])!=epochs:
    raise RuntimeError('formal readout parameters differ from locked inner-tune selection')
   if seed not in FORMAL_SEEDS:raise RuntimeError(f'unlocked formal seed {seed}')
  if stage=='screening' and seed!=202610130:raise RuntimeError('screening seed must be V2 pilot seed')
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
  device=torch.device('cuda:0');model=Readout(program).to(device)
  if sum(p.numel() for p in model.parameters())!=(3612 if program.startswith('SHARED') else 18060):raise RuntimeError('readout parameter count mismatch')
  feature_path=Path(job['feature_path']);feature_receipt=read_json(Path(job['feature_complete_path']))
  if feature_receipt.get('status')!='PASS':raise RuntimeError('base hidden feature extraction not accepted')
  base_record=job['base'];
  # Match the base and role-specific feature file by anchor ID, never by filesystem order alone.
  tensors=load_file(str(feature_path),device='cpu');h_all=tensors['h'];z_all=tensors['z_base']
  rows=read_csv(Path(job['feature_rows_path']))
  data=load_prefix_archive(Path(job['train_prefix_path']),'train')
  if [str(x) for x in data['anchor_id'].tolist()]!=[str(x['anchor_id']) for x in rows]:raise RuntimeError('train hidden feature IDs/order differ from V2 archive')
  if h_all.shape!=(len(rows),30,128) or z_all.shape!=(len(rows),30,28):raise RuntimeError('hidden/base feature shapes invalid')
  fold_rows=read_csv(ROOT/'inventory/ACTUAL_FOLD_MEMBERSHIP.csv');fold={}
  for row in fold_rows:
   for ep in row['episode_ids'].split(';'):
    if ep:fold[(int(row['task_id']),int(ep))]=row['role']
  roles=np.asarray([fold.get((int(t),int(e)),'UNKNOWN') for t,e in zip(data['task_id'],data['episode_id'])])
  if 'UNKNOWN' in roles:raise RuntimeError('some train episodes are absent from actual V2 fold CSV')
  if not np.array_equal(roles,data['inner_role']):raise RuntimeError('V3 fold join differs from accepted V2 archive role')
  if stage=='screening':
   fit_idx=np.flatnonzero(roles=='inner_fit');tune_idx=np.flatnonzero(roles=='inner_tune')
   if len(fit_idx)!=1010 or len(tune_idx)!=270:raise RuntimeError('screen role sample counts changed')
  else:
   fit_idx=np.arange(len(rows),dtype=np.int64);tune_idx=np.asarray([],dtype=np.int64)
   if len(fit_idx)!=1280:raise RuntimeError('formal outer-train population changed')
  # Future labels are loaded from the train-only V2 archive after all prefix/hidden validation.
  with np.load(Path(job['train_prefix_path']),allow_pickle=False) as archive:target_native=archive['target_native_h30'].copy()
  if target_native.shape!=(1280,30,28) or not np.isfinite(target_native).all():raise RuntimeError('V2 train target population invalid')
  normalizer=read_json(Path(base_record['normalizer_path']))
  sigma=torch.as_tensor(np.asarray(normalizer['target_sigma'],dtype=np.float32),device=device)
  fdscale=torch.as_tensor(np.asarray(normalizer['safe_fd_scale'],dtype=np.float32),device=device)
  weights_np=episode_task_weights(data,fit_idx)
  W=len(fit_idx)
  if abs(float(weights_np.sum())-W)>1e-8:raise RuntimeError('fit weights fail W sum')
  opt=torch.optim.AdamW(model.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0)
  metric_lock=read_json(V2/'protocol/METRIC_SCALE_LOCK.json')
  metric_sigma=np.asarray(metric_lock['target_sigma'],dtype=np.float32)
  metric_fd=np.asarray(metric_lock['safe_fd_scale'],dtype=np.float32)
  if metric_lock.get('status')!='LOCKED':raise RuntimeError('canonical V2 metric scales are not locked')
  tune_feature=tensors['h'][tune_idx].to(device) if len(tune_idx) else None
  tune_z=tensors['z_base'][tune_idx].to(device) if len(tune_idx) else None
  tune_task=torch.as_tensor(data['task_rank'][tune_idx].astype(np.int64),device=device,dtype=torch.long) if len(tune_idx) else None
  tune_target=target_native[tune_idx] if len(tune_idx) else None
  tune_anchor=data['anchor_native'][tune_idx] if len(tune_idx) else None
  tune_rows=tune_idx
  curve=[];active_grad_counts=set();checkpoint_rows=[]
  status('running',stage=stage,program=program,lambda_value=lam,epochs=epochs,fit_rows=len(fit_idx),tune_rows=len(tune_idx),
    parameter_count_total=sum(p.numel() for p in model.parameters()),base_frozen=True,labels_passed_to_forward=False)
  for epoch in range(1,epochs+1):
   model.train();order=shuffled_epoch_order(len(fit_idx),seed,epoch)
   total_weighted=0.0;total_weight=0.0;last_loss=None
   for start in range(0,len(order),64):
    loc=order[start:start+64];idx=fit_idx[loc];B=len(idx)
    h=tensors['h'][idx].to(device);z=tensors['z_base'][idx].to(device)
    tasks=torch.as_tensor(data['task_rank'][idx].astype(np.int64),device=device,dtype=torch.long)
    target=torch.as_tensor(target_native[idx],device=device,dtype=torch.float32)
    anchor=torch.as_tensor(data['anchor_native'][idx],device=device,dtype=torch.float32)
    corr=model(h,tasks);pred=z+corr
    action=per_sample_loss(pred,target,anchor,sigma,fdscale)
    penalty=corr.square().mean(dim=(1,2))
    sw=torch.as_tensor(weights_np[loc],device=device,dtype=torch.float32)
    loss=torch.sum(sw*(action+lam*penalty))/float(B)
    if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite readout loss at epoch {epoch}')
    opt.zero_grad(set_to_none=True);loss.backward()
    for name,p in model.named_parameters():
     if p.grad is not None:
      if not torch.isfinite(p.grad).all():raise FloatingPointError(f'nonfinite readout grad: {name}')
      active_grad_counts.add(name)
    opt.step();last_loss=float(loss.detach().cpu())
    total_weighted+=float(torch.sum((action+lam*penalty).detach()*sw).cpu());total_weight+=float(sw.sum())
   row={'job_id':job['job_id'],'model_id':base_record['model_id'],'program':program,'stage':stage,'seed':seed,'lambda':lam,'epoch':epoch,
     'fit_objective_weighted_mean':total_weighted/W,'actual_weight_sum':total_weight,'fit_rows':len(fit_idx),'last_batch_loss':last_loss,
     'tune_J':'NA','tune_position_mae':'NA','tune_safe_fd_l1':'NA','tune_episodes':0}
   if stage=='screening' and epoch in (20,40,80):
    model.eval();preds=[]
    with torch.no_grad():
     for start in range(0,len(tune_idx),64):
      end=min(start+64,len(tune_idx));corr=model(tune_feature[start:end],tune_task[start:end])
      preds.append((tune_z[start:end]+corr).detach().cpu().numpy())
    pred_z=np.concatenate(preds,axis=0)
    tune_sigma=np.asarray(normalizer['target_sigma'],dtype=np.float32)
    pred_native=tune_anchor[:,None,:]+tune_sigma[None,None,:]*pred_z
    eval_result=episode_macro(pred_native,tune_target,tasks_for_rows(data,tune_rows),data['episode_id'][tune_idx],metric_sigma,metric_fd)
    row.update({'tune_J':eval_result['tune_j'],'tune_position_mae':eval_result['macro_position_mae'],
     'tune_safe_fd_l1':eval_result['macro_safe_fd_l1'],'tune_episodes':sum(x['episode_count'] for x in eval_result['task_metrics'].values())})
    cp=out/f'READOUT_EPOCH{epoch:03d}.safetensors';safe_save(cp,model.state_dict())
    checkpoint_rows.append({'job_id':job['job_id'],'program':program,'lambda':lam,'epoch':epoch,'tune_j':eval_result['tune_j'],
      'tune_position_mae':eval_result['macro_position_mae'],'tune_safe_fd_l1':eval_result['macro_safe_fd_l1'],
      'checkpoint_path':str(cp),'checkpoint_sha256':sha(cp),'valid_candidates':True,'tune_episodes':row['tune_episodes']})
   curve.append(row)
   if stage=='formal' and epoch==epochs:
    cp=out/'READOUT_FINAL.safetensors';safe_save(cp,model.state_dict())
  model.eval()
  if not active_grad_counts or any(p.requires_grad is False for p in model.parameters()):raise RuntimeError('readout training gradients/parameters invalid')
  write_csv(out/'EPOCH_METRICS.csv',curve)
  if checkpoint_rows:write_csv(out/'CANDIDATES.csv',checkpoint_rows)
  final_name='READOUT_FINAL.safetensors' if stage=='formal' else f'READOUT_EPOCH{epochs:03d}.safetensors'
  final_path=out/final_name
  if stage=='screening' and not final_path.is_file():safe_save(final_path,model.state_dict())
  for p in model.parameters():
   if not torch.isfinite(p).all():raise FloatingPointError('nonfinite final readout tensor')
  torch.cuda.synchronize()
  receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'stage':stage,'program':program,'model_id':base_record['model_id'],'seed':seed,
   'lambda':lam,'lr':0.001,'epochs':epochs,'checkpoint_epochs':[20,40,80] if stage=='screening' else [epochs],
   'base_model_path':base_record['model_path'],'base_model_sha256':base_record['model_sha256'],
   'base_feature_path':str(feature_path),'base_feature_sha256':sha(feature_path),'base_feature_complete_sha256':sha(Path(job['feature_complete_path'])),
   'normalizer_path':str(base_record['normalizer_path']),'normalizer_sha256':base_record['normalizer_sha256'],
   'train_archive_path':str(job['train_prefix_path']),'train_archive_sha256':sha(Path(job['train_prefix_path'])),
   'fit_role':'inner_fit' if stage=='screening' else 'outer_train','fit_rows':len(fit_idx),'fit_episodes':len(set((int(data['task_id'][i]),int(data['episode_id'][i])) for i in fit_idx)),
   'tune_role':'inner_tune' if stage=='screening' else 'not_used','tune_rows':len(tune_idx),
   'parameter_count_total':sum(p.numel() for p in model.parameters()),'parameter_count_active_per_sample':3612,
   'active_gradient_parameter_names':sorted(active_grad_counts),'base_parameters_in_optimizer':False,'base_forward_in_fit_job':False,
   'labels_passed_to_forward':False,'train_labels_used_for_loss_only':True,'readout_initialization':'all-zero weight and bias',
   'optimizer':'AdamW','weight_decay':0.0,'scheduler':'constant LR','effective_batch_size':64,
   'weighted_loss_denominator':'actual_batch_size','weights_sum_each_epoch':W,
   'readout_path':str(final_path),'readout_sha256':sha(final_path),'fit_config_path':str(job_path),'fit_config_sha256':sha(job_path),
   'gpu_physical':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),
   'logical_device':str(device),'peak_memory_allocated_mib':torch.cuda.max_memory_allocated(device)/(1024**2),
   'peak_memory_reserved_mib':torch.cuda.max_memory_reserved(device)/(1024**2),'elapsed_seconds':time.time()-started,'final_heldout_payload_access':0}
  atomic_json(marker,receipt);status('PASS',completion_marker=str(marker),readout_sha256=receipt['readout_sha256'])
  print(json.dumps(receipt,sort_keys=True));return 0
 except Exception as e:
  status('FAILED',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),final_heldout_payload_access=0);raise

def tasks_for_rows(data,rows):return data['task_id'][rows].astype(np.int64)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job-json',type=Path,required=True);raise SystemExit(run(p.parse_args().job_json))
