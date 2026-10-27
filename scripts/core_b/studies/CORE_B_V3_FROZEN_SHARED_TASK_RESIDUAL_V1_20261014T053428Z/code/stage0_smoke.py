#!/usr/bin/env python3
"""One non-counted Stage0 readout smoke on one accepted screening base."""
from __future__ import annotations
import argparse,csv,json,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from v3_common import ROOT,V2,TASK_IDS,SCREEN_SEED,Q0ShardedGrid,read_json,atomic_json,sha,load_prefix_archive,make_prefix,select_base,load_base,capture_forward,episode_task_weights,per_sample_loss,shuffled_epoch_order

def run(job_path):
 job=read_json(job_path);out=Path(job['output_root'])
 done=out/'SMOKE_COMPLETE.json'
 if done.exists():
  d=read_json(done)
  if d.get('status')=='PASS':print(json.dumps({'status':'already_complete','job_id':job['job_id']}));return 0
 if out.exists() and any(out.iterdir()):raise RuntimeError(f'preserving incomplete prior Stage0 output: {out}')
 out.mkdir(parents=True,exist_ok=True)
 started=time.time();status_path=out/'RUN_STATUS.json'
 def status(state,**extra):atomic_json(status_path,{'schema_version':1,'job_id':job['job_id'],'status':state,'updated_utc':time.time(),**extra})
 status('running',gpu_watcher_physical=os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','unknown'))
 try:
  if not torch.cuda.is_available():raise RuntimeError('Stage0 requires a watcher-authorized CUDA device')
  torch.manual_seed(int(job['seed']));torch.cuda.manual_seed_all(int(job['seed']))
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.use_deterministic_algorithms(True)
  device=torch.device('cuda:0')
  base=select_base('screening',str(job['model_id']),int(job['seed']))
  normalizer=read_json(Path(base['normalizer_path']))
  data=load_prefix_archive(Path(job['train_prefix_path']),'train')
  fit_indices=np.flatnonzero(data['inner_role']=='inner_fit')
  if len(fit_indices)!=1010:raise RuntimeError(f'expected 1010 V2 inner_fit anchors, got {len(fit_indices)}')
  shuffled=shuffled_epoch_order(len(fit_indices),int(job['seed']),1)
  chosen=fit_indices[shuffled[:64]]
  grid=None
  if base['video_mode']!='STATE-CTRL':grid=Q0ShardedGrid(read_json(Path(job['cache_manifest_path'])))
  model,base_hash_before,manifest=load_base(base,device)
  prefix,task=make_prefix(data,chosen,normalizer,grid,base['video_mode'],device)
  h,z=capture_forward(model,prefix,task)
  readout=torch.nn.Linear(128,28,device=device,dtype=torch.float32)
  with torch.no_grad():readout.weight.zero_();readout.bias.zero_()
  z_replay=model.output(h)
  r0=readout(h)
  if not torch.allclose(z_replay,z,rtol=1e-7,atol=1e-6):raise RuntimeError('output-prehook replay differs from V2 action output')
  if not torch.allclose(z+r0,z,rtol=0,atol=0) or torch.count_nonzero(r0).item()!=0:raise RuntimeError('zero readout did not exactly replay accepted V2 output')
  # Load future training labels only after the prefix/base forward; labels are a loss target, never a forward input.
  with np.load(Path(job['train_prefix_path']),allow_pickle=False) as archive:
   target_np=archive['target_native_h30'][chosen].copy()
  target=torch.as_tensor(target_np,device=device,dtype=torch.float32)
  anchor=torch.as_tensor(data['anchor_native'][chosen],device=device,dtype=torch.float32)
  sigma=torch.as_tensor(np.asarray(normalizer['target_sigma'],dtype=np.float32),device=device)
  fdscale=torch.as_tensor(np.asarray(normalizer['safe_fd_scale'],dtype=np.float32),device=device)
  full_weights=episode_task_weights(data,fit_indices)
  weight_by_row={int(ix):float(w) for ix,w in zip(fit_indices,full_weights)}
  weights=torch.as_tensor([weight_by_row[int(ix)] for ix in chosen],device=device,dtype=torch.float32)
  opt=torch.optim.AdamW(readout.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0)
  correction=readout(h.detach());pred=z+correction
  action=per_sample_loss(pred,target,anchor,sigma,fdscale)
  total=action+correction.square().mean(dim=(1,2))
  loss=torch.sum(weights*total)/len(chosen)
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite Stage0 readout loss')
  opt.zero_grad(set_to_none=True);loss.backward()
  if readout.weight.grad is None or not torch.isfinite(readout.weight.grad).all() or not torch.isfinite(readout.bias.grad).all():raise FloatingPointError('readout gradient missing/nonfinite')
  if any(p.grad is not None for p in model.parameters()):raise RuntimeError('frozen V2 base received gradient')
  grad_norm=float(torch.sqrt(sum((p.grad.detach().float().square().sum() for p in readout.parameters()))).cpu())
  opt.step()
  base_hash_after=__import__('v3_common').state_hash(model)
  if base_hash_after!=base_hash_before:raise RuntimeError('V2 base parameters or buffers changed during Stage0')
  tensor_path=out/'STAGE0_READOUT.safetensors'
  safe={'weight':readout.weight.detach().cpu().contiguous(),'bias':readout.bias.detach().cpu().contiguous()}
  save_file(safe,str(tensor_path))
  reloaded=load_file(str(tensor_path),device='cpu')
  if not torch.equal(reloaded['weight'],readout.weight.detach().cpu()) or not torch.equal(reloaded['bias'],readout.bias.detach().cpu()):raise RuntimeError('safe Stage0 readout reload mismatch')
  torch.cuda.synchronize()
  peak_alloc=int(torch.cuda.max_memory_allocated(device));peak_reserved=int(torch.cuda.max_memory_reserved(device))
  receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'model_id':base['model_id'],'seed':base['seed'],
   'base_job_id':base['job_id'],'base_model_sha256':base['model_sha256'],'base_state_hash_before':base_hash_before,'base_state_hash_after':base_hash_after,
   'base_unchanged':True,'all_base_parameters_frozen':all(not p.requires_grad for p in model.parameters()),'base_parameter_gradients_absent':all(p.grad is None for p in model.parameters()),
   'hidden_shape':list(h.shape),'hidden_dim':int(h.shape[-1]),'h_interface':'output Linear(128,28) pre-input activation',
   'readout_parameters':sum(p.numel() for p in readout.parameters()),'zero_readout_exact_replay':True,'replay_max_abs_error':float((z_replay-z).abs().max().cpu()),
   'smoke_optimizer_updates':1,'counted_as_screen_or_formal_fit':False,'chosen_batch_rows':len(chosen),'inner_fit_population':len(fit_indices),
   'batch_weight_sum':float(weights.double().sum().cpu()),'objective_lambda':1.0,'loss':float(loss.detach().cpu()),'readout_grad_norm':grad_norm,
   'safe_tensor_path':str(tensor_path),'safe_tensor_sha256':sha(tensor_path),'safe_reload_pass':True,
   'gpu_physical':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),
   'logical_device':str(device),'torch_version':torch.__version__,'torch_cuda_build':torch.version.cuda,
   'peak_memory_allocated_mib':peak_alloc/(1024**2),'peak_memory_reserved_mib':peak_reserved/(1024**2),
   'elapsed_seconds':time.time()-started,'train_future_labels_read_for_loss_only':True,'labels_passed_to_forward':False,
   'final_heldout_payload_access':0}
  atomic_json(out/'SMOKE_COMPLETE.json',receipt);status('PASS',smoke_complete=str(out/'SMOKE_COMPLETE.json'),peak_memory_allocated_mib=receipt['peak_memory_allocated_mib'],peak_memory_reserved_mib=receipt['peak_memory_reserved_mib'])
  print(json.dumps(receipt,sort_keys=True));return 0
 except Exception as e:
  status('FAILED',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),final_heldout_payload_access=0)
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job-json',type=Path,required=True);raise SystemExit(run(p.parse_args().job_json))
