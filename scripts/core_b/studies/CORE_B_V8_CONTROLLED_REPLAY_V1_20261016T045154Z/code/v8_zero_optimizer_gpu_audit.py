#!/usr/bin/env python3
"""Watcher-launched V8 same-GPU replay/loss/gradient/two-step audit."""
from __future__ import annotations
import os
import csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4=PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V7=PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
CODE=ROOT/'code'; LOCKED=CODE/'locked_source'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(LOCKED))
import v8_train as tr
from prepare_v2_data import Q0ShardedGrid
from v2_model import V2TaskConditionedPredictor, state_hash
from v2_scheduler import UpdateDrivenCosineAnnealingLR
SEEDS=(202610131,202610132,202610133); TASKS=(119,94,90,141,230)
LOSS_ATOL=2e-6; LOSS_RTOL=2e-5; STEP_ATOL=2e-6; STEP_RTOL=2e-5

def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read_json(path:Path): return json.loads(path.read_text(encoding='utf-8'))
def write_json(path:Path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
def load_pred(bundle:Path,expected_role:str,seed:int,data):
 receipt=read_json(bundle/'PREDICTION_COMPLETE.json')
 field='teacher_role' if 'teacher_role' in receipt else 'program_id'
 if receipt.get('status')!='PASS' or int(receipt.get('seed',-1))!=seed or receipt.get(field)!=expected_role: raise RuntimeError(f'prediction receipt mismatch {bundle}')
 pp=Path(receipt['prediction_path']);rp=Path(receipt['prediction_rows_path'])
 if sha(pp)!=receipt['prediction_sha256'] or sha(rp)!=receipt['prediction_rows_sha256']: raise RuntimeError(f'prediction payload checksum mismatch {bundle}')
 values=load_file(str(pp),device='cpu')['prediction_native'].numpy()
 if values.shape!=(1280,30,28) or values.dtype!=np.float64 or not np.isfinite(values).all(): raise RuntimeError(f'prediction schema mismatch {bundle}')
 with rp.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
 by={r['anchor_id']:r for r in rows}; ids=data['anchor_id'].astype(str)
 if len(by)!=1280 or set(by)!=set(ids): raise RuntimeError(f'prediction anchor IDs mismatch {bundle}')
 order=np.empty(1280,np.int64)
 for i,aid in enumerate(ids):
  r=by[str(aid)]
  if int(r['episode_id'])!=int(data['episode_id'][i]) or int(r['task_id'])!=int(data['task_id'][i]) or str(r['source_cluster'])!=str(data['source_cluster'][i]): raise RuntimeError(f'prediction row metadata mismatch {aid}')
  order[i]=int(r['row'])
 if not np.array_equal(np.sort(order),np.arange(1280)): raise RuntimeError(f'prediction row map non-bijective {bundle}')
 return values[order].copy(),receipt

def replay(model_path:Path,manifest_path:Path,seed:int,data,norm,grid,device,role:str):
 manifest=read_json(manifest_path)
 if manifest.get('status')!='PASS' or sha(model_path)!=manifest.get('model_file_sha256'): raise RuntimeError(f'accepted model manifest/hash failure {role}/{seed}')
 model=V2TaskConditionedPredictor(seed=seed,video_mode='TS11-D',waypoint_mode=None)
 model.load_state_dict(load_file(str(model_path),device='cpu'),strict=True)
 before=state_hash(model)
 expected_state=manifest.get('final_state_hash',manifest.get('model_state_hash'))
 if expected_state and before!=expected_state: raise RuntimeError(f'{role} strict-state hash mismatch')
 model=model.to(device).eval(); sigma=torch.as_tensor(np.asarray(norm['target_sigma'],dtype=np.float32),device=device)
 pred=[]
 with torch.no_grad():
  for start in range(0,1280,64):
   ix=np.arange(start,min(start+64,1280),dtype=np.int64)
   prefix,tasks,anchor,_,_,_=tr.input_batch(data,ix,norm,grid,'TS11-D',device)
   output=model(prefix,tasks)
   native=anchor[:,None,:]+sigma[None,None,:]*output.action_residual.float()
   pred.append(native.detach().to(torch.float64).cpu().numpy())
 arr=np.concatenate(pred,axis=0)
 after=state_hash(model)
 if before!=after: raise RuntimeError(f'{role} inference changed model state')
 return arr,before,after

def loss_np(pred,target,sigma,fdscale):
 pos_err=np.abs(pred-target)/sigma[None,None,:]
 pos=.5*(pos_err[...,:14].mean(axis=(1,2))+pos_err[...,14:].mean(axis=(1,2)))
 # 29 future-internal differences. The anchor-to-first-target seam is not included.
 fd_err=np.abs(np.diff(pred-target,axis=1)/fdscale[None,None,:])
 fd=.5*(fd_err[...,:14].mean(axis=(1,2))+fd_err[...,14:].mean(axis=(1,2)))
 return pos,fd

def independent_episode_task_weights(data):
 n=len(data['anchor_id']); weights=np.empty(n,dtype=np.float64)
 for task in sorted(int(x) for x in np.unique(data['task_id'])):
  task_ix=np.flatnonzero(data['task_id']==task); episodes=np.unique(data['episode_id'][task_ix]); e_count=len(episodes)
  for ep in episodes:
   ix=task_ix[data['episode_id'][task_ix]==ep]
   weights[ix]=n/(5*e_count*len(ix))
 return weights

def choose_batch(data,weights):
 selected=[]; quotas={119:12,94:13,90:13,141:13,230:13}
 for task in TASKS:
  task_rows=np.flatnonzero(data['task_id']==task)
  episodes=sorted(np.unique(data['episode_id'][task_rows]).tolist())
  quota=quotas[task]
  if len(episodes)<quota: raise RuntimeError(f'not enough task {task} episodes for deterministic batch')
  chosen=episodes[:quota]
  for ep in chosen:
   rows=task_rows[data['episode_id'][task_rows]==ep]
   rows=rows[np.argsort(data['anchor_frame'][rows],kind='stable')]
   selected.append(int(rows[0]))
 if len(selected)!=64 or len(set(selected))!=64: raise RuntimeError('preselected QA batch must have 64 unique rows')
 w=weights[np.asarray(selected)]
 for task in TASKS:
  if len(set(data['episode_id'][np.asarray(selected)[data['task_id'][selected]==task]].tolist()))<3: raise RuntimeError(f'batch lacks cross-episode support for {task}')
 if len(np.unique(np.round(w,12)))<2: raise RuntimeError('preselected QA batch does not contain unequal sample weights')
 return np.asarray(selected,np.int64)

def independent_native_loss(pred_resid,target_native,anchor,sigma,fdscale):
 pred_native=anchor[:,None,:]+sigma[None,None,:]*pred_resid
 pos_err=torch.abs(pred_native-target_native)/sigma[None,None,:]
 pos=.5*(pos_err[...,:14].mean(dim=(1,2))+pos_err[...,14:].mean(dim=(1,2)))
 pred_fd=torch.diff(pred_native,dim=1)/fdscale[None,None,:]
 target_fd=torch.diff(target_native,dim=1)/fdscale[None,None,:]
 fd_err=torch.abs(pred_fd-target_fd)
 fd=.5*(fd_err[...,:14].mean(dim=(1,2))+fd_err[...,14:].mean(dim=(1,2)))
 return pos,fd,pos+0.1*fd,pred_native

def make_step_model(seed,init_path,device):
 model=V2TaskConditionedPredictor(seed=seed,video_mode='TS11-D',waypoint_mode=None)
 model.load_state_dict(load_file(str(init_path),device='cpu'),strict=True)
 return model.to(device).train()

def compare_two_updates(seed,init_path,device,prefix,tasks,anchor,y,z,sigma,fdscale,weights,weights_ref,weight_formula_max_abs):
 records=[]
 for name,target in [('Y',y),('Z_SELF',z)]:
  target=torch.as_tensor(target,device=device,dtype=torch.float32)
  model_a=make_step_model(seed,init_path,device); model_b=make_step_model(seed,init_path,device)
  init_a=state_hash(model_a);init_b=state_hash(model_b)
  if init_a!=init_b: raise RuntimeError('temporary step clones did not start from identical fresh INIT')
  opt_a=torch.optim.AdamW(model_a.parameters(),lr=5e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
  opt_b=torch.optim.AdamW(model_b.parameters(),lr=5e-4,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
  ids_a={id(p) for g in opt_a.param_groups for p in g['params']}; ids_model={id(p) for p in model_a.parameters() if p.requires_grad}
  if ids_a!=ids_model or {id(p) for g in opt_b.param_groups for p in g['params']}!={id(p) for p in model_b.parameters() if p.requires_grad}: raise RuntimeError('optimizer parameter ownership differs from trainable model')
  sch_a=UpdateDrivenCosineAnnealingLR(opt_a,T_max=3200,eta_min=1e-5);sch_b=UpdateDrivenCosineAnnealingLR(opt_b,T_max=3200,eta_min=1e-5)
  torch.cuda.manual_seed_all(918273)
  rng=torch.cuda.get_rng_state(device)
  step_comparisons=[]
  for step in range(1,3):
   grads=[]; norms=[]; losses=[]; lrs=[]; hashes=[]; positions=[]; fds=[]; per_rows=[]; rng_after=[]
   for branch,model,opt,sch,kind in [('runner',model_a,opt_a,sch_a,'helper'),('independent_reference',model_b,opt_b,sch_b,'native')]:
    torch.cuda.set_rng_state(rng,device)
    opt.zero_grad(set_to_none=True)
    out=model(prefix,tasks)
    if kind=='helper':
     per=tr.per_sample_loss(out.action_residual,target,anchor,sigma,fdscale)
     position,fd,_,_=tr.metric_components(out.action_residual,target,anchor,sigma,fdscale)
     total=torch.sum(per*weights)/float(len(weights))
    else:
     position,fd,per,native=independent_native_loss(out.action_residual,target,anchor,sigma,fdscale)
     total=torch.sum(per*weights_ref)/float(len(weights_ref))
    total.backward()
    flat_grad=torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])
    grad_copy=flat_grad.cpu().numpy().copy()
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    pre=float(grad_norm.detach().cpu())
    lr_used=float(opt.param_groups[0]['lr'])
    opt.step()
    grads.append(grad_copy);norms.append(pre);losses.append(float(total.detach().cpu()));lrs.append(lr_used);hashes.append(state_hash(model))
    positions.append(position.detach().cpu().numpy().copy());fds.append(fd.detach().cpu().numpy().copy());per_rows.append(per.detach().cpu().numpy().copy())
    rng_after.append(torch.cuda.get_rng_state(device).clone())
   # Confirm that both paths consumed the same dropout RNG draws, then advance once for the next update.
   if not torch.equal(rng_after[0],rng_after[1]): raise RuntimeError(f'{name} dropout RNG consumption differs at step {step}')
   rng=rng_after[0]
   # The loss branches receive identical initialization, inputs, target, dropout stream, optimizer and scheduler.
   ga,gb=grads
   gd=np.abs(ga-gb); denominator=np.maximum(np.abs(ga),np.abs(gb)); rel=np.divide(gd,denominator+1e-12)
   max_loss=abs(losses[0]-losses[1]); max_grad=float(gd.max()); max_rel=float(rel.max())
   max_norm=abs(norms[0]-norms[1]); lr_delta=abs(lrs[0]-lrs[1])
   pos_diff=np.abs(positions[0]-positions[1]); fd_diff=np.abs(fds[0]-fds[1]); per_diff=np.abs(per_rows[0]-per_rows[1])
   batch_ref=float(np.sum(per_rows[1]*weights_ref.detach().cpu().numpy())/len(weights_ref))
   batch_runner=float(np.sum(per_rows[0]*weights.detach().cpu().numpy())/len(weights))
   max_param_delta=float(np.max(np.abs(ga-gb)))
   # Compare actual parameters after each update, not only their hashes.
   # Recreate snapshots from the live models; they are synchronized after each paired step.
   pa=torch.cat([p.detach().reshape(-1) for p in model_a.parameters()]).cpu().numpy()
   pb=torch.cat([p.detach().reshape(-1) for p in model_b.parameters()]).cpu().numpy()
   param_delta=np.abs(pa-pb)
   pred_native_last=anchor[:,None,:]+sigma[None,None,:]*out.action_residual.detach()
   target_residual=(target-anchor[:,None,:])/sigma[None,None,:]
   zero_pos=int((out.action_residual.detach()==target_residual).sum().item())
   zero_fd=int((torch.diff(pred_native_last,dim=1)/fdscale[None,None,:]==torch.diff(target,dim=1)/fdscale[None,None,:]).sum().item())
   step_comparisons.append({'target_kind':name,'step':step,'loss_runner':losses[0],'loss_independent_reference':losses[1],
     'loss_abs_diff':max_loss,'position_per_sample_max_abs_diff':float(pos_diff.max()),'future_29diff_per_sample_max_abs_diff':float(fd_diff.max()),
     'unweighted_per_sample_objective_max_abs_diff':float(per_diff.max()),'sample_weight_formula_max_abs_diff_full_population':weight_formula_max_abs,
     'sample_weight_sum_in_selected_batch':float(weights.detach().cpu().sum()),'sample_weight_min':float(weights.detach().cpu().min()),'sample_weight_max':float(weights.detach().cpu().max()),
     'batch_reduction_denominator_actual_rows':int(len(weights)),'batch_reduction_runner':batch_runner,'batch_reduction_independent_reference':batch_ref,
     'batch_reduction_abs_diff':abs(batch_runner-batch_ref),'position_zero_residual_count':zero_pos,'future_fd_zero_residual_count':zero_fd,
     'safe_fd_uses_29_future_differences_excludes_seam':True,'grad_max_abs_diff':max_grad,'grad_max_rel_diff':max_rel,'dropout_rng_state_after_pair_equal':True,
     'preclip_grad_norm_runner':norms[0],'preclip_grad_norm_reference':norms[1],'preclip_grad_norm_abs_diff':max_norm,
     'clip_applied_runner':norms[0]>1.0,'clip_applied_reference':norms[1]>1.0,'lr_used_runner':lrs[0],'lr_used_reference':lrs[1],'lr_abs_diff':lr_delta,
     'parameter_max_abs_diff_after_step':float(param_delta.max()),'parameter_max_rel_diff_after_step':float(np.max(param_delta/(np.maximum(np.abs(pa),np.abs(pb))+1e-12))),
     'runner_state_hash':hashes[0],'reference_state_hash':hashes[1]})
   if not np.isclose(losses[0],losses[1],atol=LOSS_ATOL,rtol=LOSS_RTOL): raise RuntimeError(f'{name} loss reference mismatch step {step}: {losses}')
   if float(pos_diff.max())>LOSS_ATOL+LOSS_RTOL*max(float(np.max(np.abs(positions[0]))),float(np.max(np.abs(positions[1])))): raise RuntimeError(f'{name} per-sample position mismatch step {step}: {float(pos_diff.max())}')
   if float(fd_diff.max())>LOSS_ATOL+LOSS_RTOL*max(float(np.max(np.abs(fds[0]))),float(np.max(np.abs(fds[1])))): raise RuntimeError(f'{name} per-sample 29-difference mismatch step {step}: {float(fd_diff.max())}')
   if float(per_diff.max())>LOSS_ATOL+LOSS_RTOL*max(float(np.max(np.abs(per_rows[0]))),float(np.max(np.abs(per_rows[1])))): raise RuntimeError(f'{name} per-sample objective mismatch step {step}: {float(per_diff.max())}')
   if abs(batch_runner-batch_ref)>LOSS_ATOL+LOSS_RTOL*max(abs(batch_runner),abs(batch_ref)): raise RuntimeError(f'{name} batch reduction mismatch step {step}: {batch_runner}/{batch_ref}')
   if max_grad>LOSS_ATOL+LOSS_RTOL*max(float(np.max(np.abs(ga))),float(np.max(np.abs(gb)))): raise RuntimeError(f'{name} gradient reference mismatch step {step}: {max_grad}/{max_rel}')
   if max_norm>2e-5 or lr_delta>1e-12: raise RuntimeError(f'{name} clip/scheduler mismatch step {step}')
   if float(param_delta.max())>STEP_ATOL+STEP_RTOL*float(max(np.max(np.abs(pa)),np.max(np.abs(pb)))): raise RuntimeError(f'{name} AdamW parameter step mismatch {float(param_delta.max())}')
  sch_a.close();sch_b.close()
  records.extend(step_comparisons)
 return records

def main():
 start=time.perf_counter()
 if not torch.cuda.is_available(): raise RuntimeError('watcher child has no CUDA device')
 torch.cuda.init()
 logical_index=torch.cuda.current_device()
 device=torch.device(f'cuda:{logical_index}')
 if logical_index != 0: raise RuntimeError(f'watcher child logical CUDA device must be cuda:0, got cuda:{logical_index}')
 torch.manual_seed(202610131);torch.cuda.manual_seed_all(202610131);np.random.seed(202610131)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 torch.cuda.reset_peak_memory_stats()
 data=tr.load_npz(V2/'data/TRAIN_SELECTED_H30.npz'); norm=read_json(V7/'normalizers/NORMALIZER_FORMAL.json')
 resolved=read_json(ROOT/'inventory/RESOLVED_PATHS.json');cache_path=Path(resolved['feature_cache'])
 if sha(cache_path)!=resolved['feature_cache_manifest_sha256']: raise RuntimeError('Q0 feature cache manifest checksum mismatch')
 grid=Q0ShardedGrid(read_json(cache_path)); role_map=read_json(V7/'protocol/TEACHER_ROLE_MAP.json')
 sigma_np=np.asarray(norm['target_sigma'],dtype=np.float64);fd_np=np.asarray(norm['safe_fd_scale'],dtype=np.float64); y=np.asarray(data['target_native_h30'],dtype=np.float64)
 device_meta={'hostname':os.uname().nodename,'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),
  'logical_device':str(device),'device_name':torch.cuda.get_device_name(logical_index),'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
  'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
  'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'dtype':'float32','autocast':False}
 teacher_records=[]
 for seed in SEEDS:
  source=role_map['sources'][str(seed)]['V_MATCHED']; teacher_path=Path(source['model_path']); teacher_manifest=Path(source['model_manifest_path'])
  replay_t,th_before,th_after=replay(teacher_path,teacher_manifest,seed,data,norm,grid,device,'teacher')
  stored_t,trc=load_pred(V7/'teacher_predictions/source/V_MATCHED'/f'seed{seed}','V_MATCHED',seed,data)
  diff=np.abs(replay_t-stored_t); rel=diff/(np.abs(stored_t)+1e-12)
  if not np.allclose(replay_t,stored_t,atol=1e-6,rtol=1e-6): raise RuntimeError(f'accepted V_MATCHED clone replay mismatch seed={seed}: max_abs={diff.max()}')
  z=np.load(ROOT/'targets/SELF_SOFT'/f'seed{seed}'/'Z_TRAIN_NATIVE.npy',mmap_mode='r',allow_pickle=False)
  pos_y,fd_y=loss_np(replay_t,y,sigma_np,fd_np);pos_z,fd_z=loss_np(replay_t,z,sigma_np,fd_np)
  # V7 student model and its saved train predictions are also replayed through the same fresh-process adapter.
  student_slot=V7/'model_only/VD_SELF'/f'seed{seed}'/'attempt_01'; student_path=student_slot/'model.safetensors'; student_manifest=student_slot/'MODEL_MANIFEST.json'
  replay_s,sh_before,sh_after=replay(student_path,student_manifest,seed,data,norm,grid,device,'student')
  stored_s,sr=load_pred(V7/'predictions/train/VD_SELF'/f'seed{seed}'/'attempt_01','VD_SELF',seed,data)
  sdiff=np.abs(replay_s-stored_s)
  if not np.allclose(replay_s,stored_s,atol=1e-6,rtol=1e-6): raise RuntimeError(f'accepted V7 SELF student replay mismatch seed={seed}: max_abs={sdiff.max()}')
  student_pos,student_fd=loss_np(replay_s,z,sigma_np,fd_np)
  teacher_records.append({'seed':seed,'teacher_role':'V_MATCHED','teacher_model_sha256':sha(teacher_path),'teacher_model_manifest_sha256':sha(teacher_manifest),
    'accepted_teacher_prediction_sha256':trc['prediction_sha256'],'clone_replay_max_abs_diff':float(diff.max()),'clone_replay_max_rel_diff':float(rel.max()),
    'clone_replay_atol_rtol': '1e-6/1e-6','teacher_state_hash_before':th_before,'teacher_state_hash_after':th_after,
    'student_model_sha256':sha(student_path),'accepted_student_prediction_sha256':sr['prediction_sha256'],'student_replay_max_abs_diff':float(sdiff.max()),'student_replay_max_rel_diff':float((sdiff/(np.abs(stored_s)+1e-12)).max()),
    'student_state_hash_before':sh_before,'student_state_hash_after':sh_after,'teacher_loss_Y_position':float(pos_y.mean()),'teacher_loss_Y_29diff':float(fd_y.mean()),
    'teacher_loss_Z_position':float(pos_z.mean()),'teacher_loss_Z_29diff':float(fd_z.mean()),'teacher_loss_Z_position_plus_0.1_fd':float((pos_z+0.1*fd_z).mean()),
    'student_loss_Z_position':float(student_pos.mean()),'student_loss_Z_29diff':float(student_fd.mean()),'student_loss_Z_position_plus_0.1_fd':float((student_pos+0.1*student_fd).mean()),
    'teacher_train_predictions_are_oof':False,'optimizer_updates_during_full_replay':0,'dev_target_rows':0})

 # Full-row stored-native algebra has already passed in the CPU audit; this replay checks the model-adapter path.
 weights_all=tr.episode_task_weights(data,np.arange(1280,dtype=np.int64),'shared')
 weights_reference=independent_episode_task_weights(data)
 weight_formula_max_abs=float(np.max(np.abs(weights_all-weights_reference)))
 if not np.allclose(weights_all,weights_reference,atol=1e-12,rtol=1e-12) or not np.isclose(weights_all.sum(),1280.0,atol=1e-10,rtol=0.0): raise RuntimeError(f'episode/task weight formula mismatch: {weight_formula_max_abs}')
 batch=choose_batch(data,weights_all); batch_weights=torch.as_tensor(weights_all[batch],device=device,dtype=torch.float32)
 batch_weights_reference=torch.as_tensor(weights_reference[batch],device=device,dtype=torch.float32)
 init_path=Path(role_map['sources']['202610131']['V_MATCHED']['init_path'])
 prefix,tasks,anchor,y_batch,sigma,fdscale=tr.input_batch(data,batch,norm,grid,'TS11-D',device)
 z_bank=np.load(ROOT/'targets/SELF_SOFT/seed202610131/Z_TRAIN_NATIVE.npy',mmap_mode='r',allow_pickle=False)[batch]
 y_bank=np.asarray(y[batch],dtype=np.float32); sigma=sigma.detach();fdscale=fdscale.detach();
 batch_record={'schema_version':1,'status':'PASS','selected_by':'metadata: deterministic episode order, first accepted anchor per episode, five task quotas totaling 64',
  'rows':64,'seed_for_step_audit':202610131,'task_quotas':{'119':12,'94':13,'90':13,'141':13,'230':13},
  'task_ids':[int(data['task_id'][i]) for i in batch],'anchor_ids':[str(data['anchor_id'][i]) for i in batch],
  'episode_ids':[int(data['episode_id'][i]) for i in batch],'anchor_frames':[int(data['anchor_frame'][i]) for i in batch],
  'full_population_sample_weights':[float(weights_all[i]) for i in batch],'independent_formula_max_abs_diff_full_population':weight_formula_max_abs,
  'weight_formula':'W/(5*E_task*n_episode_windows)','weights_sum_full_population':float(weights_all.sum()),
  'all_five_tasks_present':True,'multiple_episodes_per_task':True,'unequal_weights_present':True,
  'cross_episode':True,'source_feature_cache_manifest_sha256':resolved['feature_cache_manifest_sha256'],'final_heldout_payload_access':0}
 write_json(ROOT/'audit/AUDIT_BATCH.json',batch_record)
 step_records=compare_two_updates(202610131,init_path,device,prefix,tasks,anchor,y_bank,z_bank,sigma,fdscale,batch_weights,batch_weights_reference,weight_formula_max_abs)
 with (ROOT/'audit/TEACHER_CLONE_REPLAY.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(teacher_records[0]));w.writeheader();w.writerows(teacher_records)
 with (ROOT/'audit/NATIVE_LOSS_GRAD_STEP_DIFF.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(step_records[0]));w.writeheader();w.writerows(step_records)
 peak_alloc=float(torch.cuda.max_memory_allocated()/(1024*1024));peak_reserved=float(torch.cuda.max_memory_reserved()/(1024*1024))
 audit={'schema_version':1,'status':'PASS','created_utc':utc(),'device':device_meta,'full_train_rows':1280,'full_train_program_seed_replays':6,
  'teacher_clone_replay_rows':3840,'student_replay_rows':3840,'teacher_model_native_replay_tolerance':{'atol':1e-6,'rtol':1e-6},
  'optimizer_updates_during_teacher_student_full_replay':0,'temporary_clone_optimizer_updates':8,'training_candidate_fits_started':0,
  'dev_target_archive_opened':False,'dev_target_rows':0,'final_heldout_payload_access':0,'cpu_float64_algebra_identity_source':'audit/FULL_ROW_TARGET_ROUNDTRIP.json',
  'batch_manifest_path':str(ROOT/'audit/AUDIT_BATCH.json'),'loss_grad_step_rows':len(step_records),'loss_reference_tolerance':{'atol':LOSS_ATOL,'rtol':LOSS_RTOL},
  'adamw_step_tolerance':{'atol':STEP_ATOL,'rtol':STEP_RTOL},'physical_gpu_memory_peak_allocated_mib':peak_alloc,'physical_gpu_memory_peak_reserved_mib':peak_reserved,
  'wall_seconds':time.perf_counter()-start,'source_roles_same_seed_only':True,'sampling_order_changed_by_target':False}
 write_json(ROOT/'audit/GPU_ZERO_OPTIMIZER_AUDIT.json',audit)
 # Stage0 watcher receipt uses the measured reserved peak for later formal jobs.
 write_json(ROOT/'coordination/STAGE0_RESOURCE_RECEIPT.json',{'schema_version':1,'status':'PASS','created_utc':utc(),'job_id':'v8_zero_optimizer_gpu_audit',
   'physical_gpu':device_meta['physical_gpu'],'cuda_visible_devices':device_meta['cuda_visible_devices'],'logical_device':'cuda:0',
   'peak_allocated_mib':peak_alloc,'peak_reserved_mib':peak_reserved,'registered_measured_peak_mib':max(1,int(np.ceil(peak_reserved))),
   'source_v7_train_peak_allocated_mib':138.14013671875,'source_v7_train_peak_reserved_mib':178.0,'final_heldout_payload_access':0})
 print(json.dumps({'status':'PASS','physical_gpu':device_meta['physical_gpu'],'teacher_replay_max_abs':max(r['clone_replay_max_abs_diff'] for r in teacher_records),
   'student_replay_max_abs':max(r['student_replay_max_abs_diff'] for r in teacher_records),'peak_reserved_mib':peak_reserved,'loss_grad_step_rows':len(step_records)},sort_keys=True))
if __name__=='__main__': main()
