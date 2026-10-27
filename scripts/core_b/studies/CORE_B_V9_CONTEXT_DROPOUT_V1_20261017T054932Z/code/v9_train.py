#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, random, socket, sys, time, traceback
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file, save_file

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code'
LOCKED=CODE/'locked_source'
sys.path.insert(0,str(CODE))
sys.path.insert(0,str(LOCKED))
from d3_eq_contract import episode_equal_weights, per_sample_loss, shuffled_epoch_order
from prepare_v2_data import Q0ShardedGrid
from v2_model import V2TaskConditionedPredictor,parameter_count,state_hash
from v2_scheduler import UpdateDrivenCosineAnnealingLR,write_lr_trace
from v9_common import append_jsonl,atomic_json,load_npz,metric_components,model_input,read_json,rng_state_equal_numpy,sha_file,stable_drop_bit,utc,write_csv

TASKS=(119,94,90,141,230)
SEEDS=(202610131,202610132,202610133)
EFFECTIVE_BATCH=64

def sha_bytes(a:np.ndarray)->str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order='C')).hexdigest()
def atomic_safe(path:Path,tensors:dict[str,torch.Tensor])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.part')
    save_file({k:v.detach().to(device='cpu').contiguous() for k,v in tensors.items()},str(tmp))
    os.replace(tmp,path)
def npstate_equal(a,b):
    return rng_state_equal_numpy(a,b)
def weights_for(data,indices):
    if len(indices)!=1280: raise RuntimeError('V9 fit requires all 1280 accepted outer-train rows')
    tasks=data['task_id'][indices].astype(np.int64);episodes=data['episode_id'][indices].astype(np.int64)
    weights=np.empty(len(indices),dtype=np.float64);unique=sorted(int(x) for x in np.unique(tasks))
    if tuple(unique)!=tuple(sorted(TASKS)): raise RuntimeError('five-task set differs from study lock')
    for tid in unique:
        local=np.flatnonzero(tasks==tid);ep=episodes[local]
        _,inverse,counts=np.unique(ep,return_inverse=True,return_counts=True)
        weights[local]=len(indices)/(5*len(counts)*counts[inverse])
    if not np.isfinite(weights).all() or np.any(weights<=0) or not np.isclose(weights.sum(),len(indices),atol=1e-4,rtol=0):
        raise RuntimeError('episode/task weights invalid')
    return weights

def read_mask_bank(path:Path,seed:int,ordered_ids:np.ndarray,expected_file_sha:str|None=None):
    if expected_file_sha and sha_file(path)!=expected_file_sha: raise RuntimeError('mask bank file SHA mismatch')
    with np.load(path,allow_pickle=False) as z:
        bits=z['drop_bits'].copy();ids=z['anchor_ids'].astype(str).copy()
    if bits.shape!=(160,1280) or bits.dtype!=np.bool_: raise RuntimeError('mask bank schema mismatch')
    if not np.array_equal(ids,ordered_ids.astype(str)): raise RuntimeError('mask bank ordered receiver IDs differ from train data')
    return bits

def seed_everything(seed:int):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
def rng_receipt(device):
    ns=np.random.get_state();h=hashlib.sha256();h.update(ns[0].encode('utf-8'));h.update(np.ascontiguousarray(ns[1]).tobytes());h.update(repr(ns[2:]).encode('utf-8'))
    return {'torch_cpu_sha256':hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        'torch_cuda_logical0_sha256':hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest(),
        'numpy_sha256':h.hexdigest(),'python_random_sha256':hashlib.sha256(repr(random.getstate()).encode('utf-8')).hexdigest()}

def load_inputs(job):
    train_path=Path(job['train_npz_path'])
    data=load_npz(train_path)
    if len(data['anchor_id'])!=1280 or data['history_native'].shape!=(1280,32,71) or data['target_native_h30'].shape!=(1280,30,28):
        raise RuntimeError('locked train archive shape mismatch')
    if len(set(data['anchor_id'].astype(str).tolist()))!=1280: raise RuntimeError('train anchor IDs are not unique')
    if set(int(x) for x in np.unique(data['task_id']))!=set(TASKS): raise RuntimeError('train task IDs differ from lock')
    if not np.isfinite(data['history_native']).all() or not np.isfinite(data['target_native_h30']).all(): raise RuntimeError('train X/Y has nonfinite values')
    return data

def init_model(job,device):
    mode=job['video_mode'];seed=int(job['seed'])
    model=V2TaskConditionedPredictor(seed=seed,video_mode=mode,waypoint_mode=None)
    init_path=Path(job['init_path'])
    if sha_file(init_path)!=job['init_file_sha256']: raise RuntimeError('fresh INIT file hash mismatch')
    state=load_file(str(init_path),device='cpu')
    missing,unexpected=model.load_state_dict(state,strict=True)
    if missing or unexpected: raise RuntimeError('fresh INIT strict load mismatch')
    init_hash=state_hash(model)
    if init_hash!=job['expected_init_state_hash']: raise RuntimeError('fresh INIT state hash mismatch')
    return model.to(device),init_hash

def run_sentinel_qa(job,data,normalizer,grid,model,device,bits):
    seed=int(job['seed']);indices=np.asarray([int(np.flatnonzero(data['task_id']==tid)[0]) for tid in TASKS],dtype=np.int64)
    # Cross-task real batch: fixed first receivers for each task plus deterministic fill.
    chosen=[]
    for tid in TASKS:
        chosen.extend(np.flatnonzero(data['task_id']==tid)[:12].tolist())
    if len(chosen)<64:
        chosen.extend(i for i in range(len(data['anchor_id'])) if i not in set(chosen))
    idx=np.asarray(chosen[:64],dtype=np.int64)
    if set(int(x) for x in data['task_id'][idx])!=set(TASKS): raise RuntimeError('real QA batch is not cross-task')
    full_mask=np.zeros(len(idx),dtype=bool);force=np.ones(len(idx),dtype=bool)
    mode=str(job['video_mode'])
    before_history=sha_bytes(data['history_native'])
    before_anchor=sha_bytes(data['anchor_native'])
    np_state=np.random.get_state();torch_cpu=torch.get_rng_state().clone();torch_cuda=torch.cuda.get_rng_state(device).clone()
    probe_keys=[]
    for i in idx[:10].tolist():
        key=[str(data['source_cluster'][i]),str(data['episode_id'][i]),int(data['task_id'][i]),str(data['anchor_id'][i])]
        probe_keys.append(stable_drop_bit(seed,0,key))
    rng_unchanged=npstate_equal(np_state,np.random.get_state()) and torch.equal(torch_cpu,torch.get_rng_state()) and torch.equal(torch_cuda,torch.cuda.get_rng_state(device))
    if not rng_unchanged: raise RuntimeError('SHA mask generation consumed numpy/torch CPU/CUDA RNG')
    reference_prefix,reference_task,reference_anchor=model_input(data,idx,normalizer,mode,device,grid,drop_mask=None)
    keep_prefix,keep_task,keep_anchor=model_input(data,idx,normalizer,mode,device,grid,drop_mask=full_mask)
    prefix_equal=reference_prefix.keys()==keep_prefix.keys() and all(torch.equal(reference_prefix[k],keep_prefix[k]) for k in reference_prefix)
    task_equal=torch.equal(reference_task,keep_task);anchor_equal=torch.equal(reference_anchor,keep_anchor)
    if not (prefix_equal and task_equal and anchor_equal): raise RuntimeError('p0 inputs differ from accepted V8 hard path')
    y=torch.as_tensor(data['target_native_h30'][idx],device=device,dtype=torch.float32)
    norm_sigma=torch.as_tensor(np.asarray(normalizer['target_sigma'],dtype=np.float32),device=device)
    norm_fd=torch.as_tensor(np.asarray(normalizer['safe_fd_scale'],dtype=np.float32),device=device)
    model.eval();start_hash=state_hash(model)
    model.zero_grad(set_to_none=True)
    ref_out=model(reference_prefix,reference_task).action_residual
    ref_loss=per_sample_loss(ref_out,y,reference_anchor,norm_sigma,norm_fd).mean()
    ref_loss.backward()
    ref_grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    keep_out=model(keep_prefix,keep_task).action_residual
    ref_native=reference_anchor[:,None]+norm_sigma[None,None]*ref_out
    keep_native=keep_anchor[:,None]+norm_sigma[None,None]*keep_out
    native_output_equal=torch.equal(ref_native,keep_native)
    keep_loss=per_sample_loss(keep_out,y,keep_anchor,norm_sigma,norm_fd).mean()
    keep_loss.backward()
    keep_grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    output_equal=torch.equal(ref_out,keep_out) and native_output_equal;loss_equal=torch.equal(ref_loss,keep_loss)
    grad_names_equal=ref_grads.keys()==keep_grads.keys()
    grad_equal=grad_names_equal and all(torch.equal(ref_grads[n],keep_grads[n]) for n in ref_grads)
    if not (output_equal and loss_equal and grad_equal): raise RuntimeError('p0 output/loss/gradient differs from V8 accepted input path')
    # Force-drop is whole-block, out-of-place, and invariant to any changed numeric values.
    dropped_prefix,dropped_task,dropped_anchor=model_input(data,idx,normalizer,mode,device,grid,drop_mask=force)
    altered=dict(data);altered['history_native']=data['history_native'].copy();altered['history_native'][idx]+=np.float32(0.314159)
    changed_prefix,changed_task,changed_anchor=model_input(altered,idx,normalizer,mode,device,grid,drop_mask=force)
    state_zero=bool(torch.equal(dropped_prefix['state'],torch.zeros_like(dropped_prefix['state'])))
    masked_value_invariance=torch.equal(dropped_prefix['state'],changed_prefix['state'])
    preserved=[k for k in reference_prefix if k!='state']
    other_inputs_equal=preserved==[k for k in dropped_prefix if k!='state'] and all(torch.equal(reference_prefix[k],dropped_prefix[k]) for k in preserved)
    independent_anchor=bool(torch.equal(dropped_prefix['anchor_command'],reference_prefix['anchor_command']) and torch.equal(dropped_anchor,reference_anchor))
    task_preserved=torch.equal(dropped_task,reference_task) and torch.equal(changed_task,dropped_task)
    video_preserved=all(torch.equal(dropped_prefix[k],changed_prefix[k]) for k in ('dynamic','static')) if mode!='STATE-CTRL' else True
    drop_out=model(dropped_prefix,dropped_task).action_residual
    changed_out=model(changed_prefix,changed_task).action_residual
    force_output_invariance=torch.equal(drop_out,changed_out)
    # The label is absent from the forward signature. Changing Y leaves inputs and SHA mask bits unchanged.
    changed_y=dict(data);changed_y['target_native_h30']=data['target_native_h30'].copy();changed_y['target_native_h30'][idx]+=np.float32(7.0)
    y_changed_prefix,y_changed_task,y_changed_anchor=model_input(changed_y,idx,normalizer,mode,device,grid,drop_mask=force)
    y_independent=all(torch.equal(dropped_prefix[k],y_changed_prefix[k]) for k in dropped_prefix) and torch.equal(dropped_task,y_changed_task) and torch.equal(dropped_anchor,y_changed_anchor)
    y_mask_unchanged=probe_keys==[stable_drop_bit(seed,0,[str(data['source_cluster'][i]),str(data['episode_id'][i]),int(data['task_id'][i]),str(data['anchor_id'][i])]) for i in idx[:10].tolist()]
    # STATE-CTRL ignores RGB keys and never obtains cache rows.
    state_model=V2TaskConditionedPredictor(seed=seed,video_mode='STATE-CTRL',waypoint_mode=None)
    state_tensor=load_file(str(job['init_path']),device='cpu')
    state_model.load_state_dict(state_tensor,strict=True);state_model=state_model.to(device).eval()
    state_prefix,state_task,_=model_input(data,idx,normalizer,'STATE-CTRL',device,None,drop_mask=full_mask)
    state_rgb_a=state_model(state_prefix,state_task).action_residual
    with_rgb=dict(state_prefix);sentinel=torch.ones((len(idx),16,4,768),device=device,dtype=torch.float32)
    with_rgb['dynamic']=sentinel;with_rgb['static']=-sentinel
    state_rgb_b=state_model(with_rgb,state_task).action_residual
    rgb_invariant=torch.equal(state_rgb_a,state_rgb_b)
    if not all((state_zero,masked_value_invariance,other_inputs_equal,independent_anchor,task_preserved,video_preserved,force_output_invariance,y_independent,y_mask_unchanged,rgb_invariant)):
        raise RuntimeError('numeric-context mask sentinel failed')
    if sha_bytes(data['history_native'])!=before_history or sha_bytes(data['anchor_native'])!=before_anchor: raise RuntimeError('mask modified source arrays/shared storage')
    if state_hash(model)!=start_hash: raise RuntimeError('sentinel QA modified model state')
    model.zero_grad(set_to_none=True);model.train()
    return {'schema_version':1,'status':'PASS','created_utc':utc(),'dataset':'accepted 64-row real cross-task outer-train batch',
        'task_ids_covered':sorted(set(int(x) for x in data['task_id'][idx])),'seed':seed,'video_mode':mode,
        'full_block_field':'history_native -> normalized prefix_inputs.state [B,32,71]',
        'p0_matches_V8_input_path':prefix_equal,'p0_native_model_output_exact':native_output_equal,'p0_true_Y_loss_exact':loss_equal,
        'p0_gradient_exact':grad_equal,'p0_eval_mode':'model.eval(); ordinary NN dropout disabled for exact path comparison',
        'force_drop_state_all_zero':state_zero,'changing_raw_numeric_context_does_not_change_masked_tensor':masked_value_invariance,
        'non_state_inputs_unchanged':other_inputs_equal,'independent_command_anchor_preserved':independent_anchor,'task_preserved':task_preserved,
        'video_preserved':video_preserved,'force_drop_forward_invariant_to_masked_values':force_output_invariance,
        'changing_Y_does_not_change_forward_inputs':y_independent,'changing_Y_does_not_change_schedule_bit':y_mask_unchanged,
        'state_only_RGB_input_invariant':rgb_invariant,'source_arrays_unchanged':True,'mask_out_of_place':True,
        'SHA_mask_generation_preserved_numpy_torch_cpu_cuda_rng_states':rng_unchanged,'AD_VD_same_seed_bank_sha256':read_json(ROOT/'protocol/MASK_SCHEDULE.json')['bank_hashes'][str(seed)]['bits_sha256'],
        'final_heldout_payload_access':0}

def run(job_path:Path)->int:
    job=read_json(job_path);job_root=Path(job['job_root']);fallback=job_root/'OOM_FALLBACK.json'
    stage0=bool(job.get('stage0',False));updates_limit=job.get('updates_limit')
    attempt_dir=Path(job['attempt_dir']);micro=int(job.get('microbatch_size',64));accum=int(job.get('accumulation_steps',1))
    if fallback.exists() and not stage0:
        frec=read_json(fallback)
        if frec.get('job_id')!=job.get('job_id') or int(frec.get('effective_batch',-1))!=EFFECTIVE_BATCH: raise RuntimeError('OOM fallback receipt mismatch')
        micro=int(frec['microbatch_size']);accum=int(frec['accumulation_steps']);attempt_dir=job_root/'attempt_02'
    if micro*accum!=EFFECTIVE_BATCH: raise RuntimeError('effective batch changed from 64')
    attempt_dir.mkdir(parents=True,exist_ok=True)
    status_path=attempt_dir/'RUN_STATUS.json'
    started=utc();timer=time.perf_counter();fit_key=''
    try:
        if not torch.cuda.is_available(): raise RuntimeError('V9 fit requires watcher-authorized CUDA')
        if job.get('code_lock_sha256')!=sha_file(ROOT/'protocol/CODE_LOCK.json'): raise RuntimeError('V9 code lock SHA differs from job config')
        if job.get('training_runner_sha256') and sha_file(Path(__file__))!=job['training_runner_sha256']: raise RuntimeError('V9 training runner hash differs from job lock')
        if job.get('v9_common_sha256') and sha_file(CODE/'v9_common.py')!=job['v9_common_sha256']: raise RuntimeError('V9 shared adapter hash differs from job lock')
        seed=int(job['seed']);program=str(job['program_id']);mode=str(job['video_mode']);p=float(job['numeric_context_drop_probability'])
        stage0=bool(job.get('stage0',False))
        if program not in {'A0','AD','V0','VD'} or (mode,p)!={'A0':('STATE-CTRL',0.0),'AD':('STATE-CTRL',0.5),'V0':('TS11-D',0.0),'VD':('TS11-D',0.5)}[program]:
            raise RuntimeError('program/input mode/drop probability differs from locked V9 matrix')
        if int(job['seed']) not in SEEDS: raise RuntimeError('seed outside V9 lock')
        study=read_json(ROOT/'protocol/STUDY_LOCK.json');recipe=read_json(ROOT/'protocol/RECIPE_LOCK.json')
        if study.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV' or recipe.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV': raise RuntimeError('V9 study/recipe lock missing')
        if job.get('teacher_target_path') or job.get('dev_target_path') or job.get('target_mode') not in {'true_y'}: raise RuntimeError('V9 fit cannot receive teacher/dev target paths')
        seed_everything(seed)
        device=torch.device('cuda:0')
        rng_at_seed=rng_receipt(device)
        data=load_inputs(job);train_idx=np.arange(1280,dtype=np.int64)
        normalizer=read_json(Path(job['normalizer_path']))
        if sha_file(job['normalizer_path'])!=job['normalizer_sha256']: raise RuntimeError('normalizer hash differs from lock')
        bank_path=Path(job['mask_bank_path']);bank=read_mask_bank(bank_path,seed,data['anchor_id'],job['mask_bank_file_sha256'])
        expected_bits=read_json(ROOT/'protocol/MASK_SCHEDULE.json')['bank_hashes'][str(seed)]['bits_sha256']
        if hashlib.sha256(np.ascontiguousarray(bank).tobytes(order='C')).hexdigest()!=expected_bits: raise RuntimeError('mask schedule bits hash differs from lock')
        sampler_path=Path(job['sampler_orders_path'])
        if sha_file(sampler_path)!=job['sampler_orders_sha256']: raise RuntimeError('sampler archive hash differs from lock')
        with np.load(sampler_path,allow_pickle=False) as z:
            sampler=z[f'seed_{seed}'].copy()
        if sampler.shape!=(160,1280) or sampler.dtype!=np.int64: raise RuntimeError('sampler sequence schema mismatch')
        sampler_sequence_sha=hashlib.sha256(np.ascontiguousarray(sampler).tobytes(order='C')).hexdigest()
        if sampler_sequence_sha!=job['sampler_sequence_sha256']: raise RuntimeError('locked seed sampler sequence SHA mismatch')
        weights=weights_for(data,train_idx);weight_by_global=weights
        grid=None
        if mode!='STATE-CTRL':
            cache=read_json(Path(job['feature_cache_manifest_path']))
            if sha_file(job['feature_cache_manifest_path'])!=job['feature_cache_manifest_sha256']: raise RuntimeError('feature cache manifest hash mismatch')
            grid=Q0ShardedGrid(cache)
        model,init_hash=init_model(job,device)
        torch.cuda.reset_peak_memory_stats(device)
        initial_state=state_hash(model)
        if int(job.get('epochs',160))!=160 or int(job.get('effective_batch',64))!=64: raise RuntimeError('formal recipe differs from locked 160 epochs/batch64')
        epochs=int(job.get('epochs',160));updates_per_epoch=int(np.ceil(len(train_idx)/EFFECTIVE_BATCH));tmax=epochs*updates_per_epoch
        if tmax!=3200: raise RuntimeError('scheduler must use 3200 total updates')
        run_qa=None
        if stage0:
            run_qa=run_sentinel_qa(job,data,normalizer,grid,model,device,bank)
            atomic_json(ROOT/'qa/MASK_SENTINEL_AND_RNG.json',run_qa)
            atomic_json(ROOT/'qa/BASELINE_NO_MASK_DIFF.json',{'schema_version':1,'status':'PASS','p0_matches_V8_input_path':run_qa['p0_matches_V8_input_path'],
                'native_output_equal':run_qa['p0_native_model_output_exact'],'true_Y_loss_equal':run_qa['p0_true_Y_loss_exact'],
                'gradient_equal':run_qa['p0_gradient_exact'],'stage0_fit_restarted_from_fresh_INIT_for_all_formal_slots':True})
        if stage0:
            max_updates=int(updates_limit or 2)
            out=attempt_dir
        else:
            max_updates=None;out=attempt_dir
        lr=float(job['lr']);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4)
        scheduler=UpdateDrivenCosineAnnealingLR(optimizer,T_max=tmax,eta_min=1e-5)
        rng_before_first_optimizer_step=rng_receipt(device)
        model.train();optimizer.zero_grad(set_to_none=True);optimizer_steps=0;presentations=0
        epoch_rows=[];mask_epoch_rows=[];grad_norms=[];stopped_stage0=False
        for epoch in range(1,epochs+1):
            scheduler.set_epoch(epoch)
            order=sampler[epoch-1]
            generated=shuffled_epoch_order(len(train_idx),seed,epoch)
            if not np.array_equal(generated,order) or not np.array_equal(np.sort(order),np.arange(1280,dtype=np.int64)):
                raise RuntimeError('sampler order differs from accepted seed+epoch sequence at epoch '+str(epoch))
            pos_sum=fd_sum=seam_sum=obj_sum=weight_sum=0.0;drop_count=0;drop_total=0;step_count=0
            task_drop={t:[0,0] for t in TASKS};clip_count=0
            for group_start in range(0,len(order),EFFECTIVE_BATCH):
                local_group=order[group_start:group_start+EFFECTIVE_BATCH]
                group_idx=train_idx[local_group];group_n=len(group_idx)
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0,group_n,micro):
                    batch_idx=group_idx[micro_start:micro_start+micro]
                    if p>0:
                        drop=bank[epoch-1,batch_idx].copy()
                    else:
                        drop=np.zeros(len(batch_idx),dtype=np.bool_)
                    prefix,task_rank,anchor=model_input(data,batch_idx,normalizer,mode,device,grid,drop_mask=drop)
                    target=torch.as_tensor(data['target_native_h30'][batch_idx],device=device,dtype=torch.float32)
                    sigma=torch.as_tensor(np.asarray(normalizer['target_sigma'],dtype=np.float32),device=device)
                    fdscale=torch.as_tensor(np.asarray(normalizer['safe_fd_scale'],dtype=np.float32),device=device)
                    out_model=model(prefix,task_rank)
                    if out_model.goal_prediction is not None: raise RuntimeError('unexpected auxiliary goal output')
                    losses=per_sample_loss(out_model.action_residual,target,anchor,sigma,fdscale)
                    w_np=weight_by_global[batch_idx]
                    w=torch.as_tensor(w_np,device=device,dtype=torch.float32)
                    contribution=torch.sum(losses*w)/float(group_n)
                    if not torch.isfinite(contribution): raise FloatingPointError('nonfinite true-Y training loss')
                    contribution.backward()
                    normp,fdp,seamp,objp=metric_components(out_model.action_residual,target,anchor,sigma,fdscale,
                        torch.as_tensor(np.asarray(job['metric_sigma'],dtype=np.float32),device=device),
                        torch.as_tensor(np.asarray(job['metric_safe_fd_scale'],dtype=np.float32),device=device))
                    pos_sum+=float(torch.sum(normp*w).detach().cpu());fd_sum+=float(torch.sum(fdp*w).detach().cpu())
                    seam_sum+=float(torch.sum(seamp*w).detach().cpu());obj_sum+=float(torch.sum(objp*w).detach().cpu());weight_sum+=float(w.sum().detach().cpu())
                    presentations+=len(batch_idx);drop_count+=int(drop.sum());drop_total+=len(batch_idx)
                    for tid in TASKS:
                        tcount=int(np.sum(data['task_id'][batch_idx]==tid));task_drop[tid][1]+=tcount;task_drop[tid][0]+=int(drop[data['task_id'][batch_idx]==tid].sum())
                    for name,param in model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all(): raise FloatingPointError('nonfinite gradient '+name)
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0,error_if_nonfinite=True)
                grad_value=float(grad_norm.detach().cpu());grad_norms.append(grad_value)
                if grad_value>1.0:clip_count+=1
                optimizer.step();optimizer_steps+=1;step_count+=1
                if scheduler.step_calls!=optimizer_steps: raise RuntimeError('optimizer/scheduler count diverged')
                if max_updates is not None and optimizer_steps>=max_updates:
                    stopped_stage0=True;break
            if weight_sum and not np.isclose(weight_sum,1280.0,rtol=0,atol=1e-3) and not stage0:
                raise RuntimeError('epoch weights do not sum to population W')
            if stage0:
                epoch_rows.append({'epoch':epoch,'optimizer_steps':optimizer_steps,'lr_after_epoch':scheduler.get_last_lr()[0],
                    'true_y_masked_training_objective_partial':obj_sum/max(weight_sum,1.0),'numeric_context_drop_rate_observed':drop_count/max(drop_total,1),
                    'rows_presented':drop_total,'elapsed_seconds':time.perf_counter()-timer})
                if stopped_stage0: break
                continue
            metrics={'position_mae':pos_sum/1280,'safe_fd_l1':fd_sum/1280,'seam_safe_fd_l1':seam_sum/1280,'true_y_objective':obj_sum/1280}
            actual_rate=drop_count/max(drop_total,1)
            epoch_rows.append({'epoch':epoch,'optimizer_steps':optimizer_steps,'lr_after_epoch':scheduler.get_last_lr()[0],
                'masked_input_true_y_position_mae':metrics['position_mae'],'masked_input_true_y_safe_fd_l1':metrics['safe_fd_l1'],
                'masked_input_true_y_seam_safe_fd_l1':metrics['seam_safe_fd_l1'],'masked_input_true_y_objective':metrics['true_y_objective'],
                'numeric_context_drop_probability_locked':p,'numeric_context_drop_rate_observed':actual_rate,
                'mask_receiver_rows_dropped':drop_count,'mask_receiver_rows_seen':drop_total,'clip_updates':clip_count,
                'preclip_grad_norm_mean':float(np.mean(grad_norms[-step_count:])),'preclip_grad_norm_max':float(np.max(grad_norms[-step_count:])),
                'elapsed_seconds':time.perf_counter()-timer})
            for tid in TASKS:
                dn,tn=task_drop[tid]
                mask_epoch_rows.append({'program_id':program,'seed':seed,'epoch':epoch,'task_id':tid,'receiver_n':tn,
                    'dropped_n':dn,'drop_rate_observed':dn/tn if tn else 0.0,'configured_probability':p})
            if epoch%10==0 or epoch==epochs:
                atomic_json(status_path,{'status':'running','updated_utc':utc(),'job_id':job['job_id'],'program_id':program,'seed':seed,
                    'epoch':epoch,'epochs_target':epochs,'optimizer_steps':optimizer_steps,'steps_per_epoch':updates_per_epoch,
                    'lr':scheduler.get_last_lr()[0],'mask_rate_last_epoch':actual_rate,'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),
                    'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':str(device)})
                write_csv(out/'EPOCH_LOG.csv',epoch_rows)
                write_csv(out/'MASK_EPOCH_LOG.csv',mask_epoch_rows)
        torch.cuda.synchronize(device)
        peak_alloc=float(torch.cuda.max_memory_allocated(device)/(1024*1024));peak_reserved=float(torch.cuda.max_memory_reserved(device)/(1024*1024))
        if stage0:
            result={'schema_version':1,'status':'PASS','study':'V9 real two-update Stage0 QA only','job_id':job['job_id'],'program_id':program,'seed':seed,
                'updates':optimizer_steps,'rows_presented':presentations,'true_Y_only':True,'stage0_mask_qa':run_qa,
                'peak_memory_allocated_mib':peak_alloc,'peak_memory_reserved_mib':peak_reserved,
                'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),
                'logical_device':str(device),'elapsed_seconds':time.perf_counter()-timer,'fresh_INIT_for_formal_restart':init_hash,
                'formal_optimizer_steps':0,'final_heldout_payload_access':0}
            atomic_json(out/'STAGE0_COMPLETE.json',result);atomic_json(status_path,{'status':'complete','stage0':True,'updates':optimizer_steps,'completed_utc':utc()})
            print(json.dumps(result,sort_keys=True));return 0
        if optimizer_steps!=3200 or presentations!=160*1280: raise RuntimeError('formal fit did not complete exact 160x1280 presentations / 3200 steps')
        final_path=out/'model.safetensors';atomic_safe(final_path,model.state_dict())
        lr_trace=out/'LR_TRACE_ACTUAL.csv';write_lr_trace(lr_trace,scheduler.update_records)
        write_csv(out/'EPOCH_LOG.csv',epoch_rows);write_csv(out/'MASK_EPOCH_LOG.csv',mask_epoch_rows)
        model_manifest={'schema_version':1,'status':'PASS','created_utc':utc(),'job_id':job['job_id'],'program_id':program,'seed':seed,
            'video_mode':mode,'numeric_context_drop_probability':p,'target_mode':'true_y','target_role':'TRUE_Y',
            'target_expression':'ground_truth_future_Y','teacher_target_used':False,'teacher_parameters_updated':0,
            'epochs':epochs,'steps_per_epoch':updates_per_epoch,'optimizer_steps':optimizer_steps,'effective_batch':64,
            'microbatch_size':micro,'accumulation_steps':accum,'optimizer':{'name':'AdamW','lr':lr,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':1e-4},
            'scheduler':{'name':'UpdateDrivenCosineAnnealingLR','T_max':tmax,'eta_min':1e-5,'step_location':'optimizer post-step'},
            'gradient_clip_norm':1.0,'dtype':'float32','amp':False,'tf32':False,'deterministic_algorithms':True,
            'init_path':job['init_path'],'init_sha256':job['init_file_sha256'],'init_state_hash':init_hash,'final_state_hash':state_hash(model),
            'model_file':str(final_path),'model_file_sha256':sha_file(final_path),'parameter_count':parameter_count(model),
            'normalizer_path':job['normalizer_path'],'normalizer_sha256':job['normalizer_sha256'],
            'train_npz_path':job['train_npz_path'],'train_npz_sha256':job['train_npz_sha256'],
            'sampler_orders_path':str(sampler_path),'sampler_orders_sha256':sha_file(sampler_path),
            'sampler_sequence_sha256':hashlib.sha256(np.ascontiguousarray(sampler).tobytes(order='C')).hexdigest(),
            'mask_bank_path':str(bank_path),'mask_bank_file_sha256':sha_file(bank_path),'mask_bank_bits_sha256':expected_bits,
            'mask_bank_applied':p>0,'mask_rng_consumes_training_rng':False,
            'recipe_lock_sha256':sha_file(ROOT/'protocol/RECIPE_LOCK.json'),'study_lock_sha256':sha_file(ROOT/'protocol/STUDY_LOCK.json'),
            'mask_field_map_sha256':sha_file(ROOT/'protocol/MASK_FIELD_MAP.json'),'job_config_path':str(job_path),'job_config_sha256':sha_file(job_path),
            'training_runner_path':str(Path(__file__).resolve()),'training_runner_sha256':sha_file(Path(__file__).resolve()),
            'target_presentations':presentations,'train_rows':1280,'dev_or_final_target_rows':0,
            'rng_at_seed':rng_at_seed,'rng_before_first_optimizer_step':rng_before_first_optimizer_step,
            'runtime':{'hostname':socket.gethostname(),'python_version':sys.version.split()[0],'torch_version':torch.__version__,
                'torch_cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,
                'cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32},
            'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),
            'logical_device':str(device),'peak_memory_allocated_mib':peak_alloc,'peak_memory_reserved_mib':peak_reserved,
            'final_heldout_payload_access':0}
        atomic_json(out/'MODEL_MANIFEST.json',model_manifest)
        complete={'schema_version':1,'status':'PASS','completed_utc':utc(),'job_id':job['job_id'],'program_id':program,'seed':seed,
            'epochs_completed':epochs,'optimizer_steps':optimizer_steps,'target_mode':'true_y','target_role':'TRUE_Y',
            'model_sha256':model_manifest['model_file_sha256'],'model_state_hash':model_manifest['final_state_hash'],
            'training_sample_presentations':presentations,'teacher_target_used':False,'dev_or_final_target_rows':0,
            'rng_at_seed':rng_at_seed,'rng_before_first_optimizer_step':rng_before_first_optimizer_step,
            'elapsed_seconds':time.perf_counter()-timer,'final_heldout_payload_access':0}
        atomic_json(out/'RUN_COMPLETE.json',complete);atomic_json(status_path,{'status':'complete','completed_utc':utc(),'job_id':job['job_id'],'epochs':epochs,'optimizer_steps':optimizer_steps})
        print(json.dumps({'status':'PASS','program_id':program,'seed':seed,'epochs':epochs,'optimizer_steps':optimizer_steps,
            'model_sha256':model_manifest['model_file_sha256'],'peak_reserved_mib':peak_reserved,'elapsed_seconds':complete['elapsed_seconds']},sort_keys=True),flush=True)
        return 0
    except Exception as exc:
        text=str(exc).lower();is_oom=('out of memory' in text) or type(exc).__name__ in {'OutOfMemoryError','CudaOutOfMemoryError'}
        if is_oom and not stage0 and micro==64 and accum==1 and not fallback.exists():
            atomic_json(fallback,{'schema_version':1,'status':'RETRY_WITH_EFFECTIVE_BATCH_PRESERVED','job_id':job.get('job_id'),
                'trigger':type(exc).__name__+': '+str(exc),'microbatch_size':32,'accumulation_steps':2,'effective_batch':64,
                'restart_policy':'fresh INIT in attempt_02; attempt_01 retained','created_utc':utc()})
        err={'status':'failed','failed_utc':utc(),'job_id':job.get('job_id'),'program_id':job.get('program_id'),'seed':job.get('seed'),
            'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
            'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'train_rows_presented':int(locals().get('presentations',0)),
            'final_heldout_payload_access':0}
        atomic_json(status_path,err)
        if not stage0: append_jsonl(ROOT/'coordination/FAILED_ATTEMPTS.jsonl',err)
        print(json.dumps({'status':'FAILED','job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc)},sort_keys=True),flush=True)
        return 1

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);args=ap.parse_args()
    raise SystemExit(run(args.job_json))
if __name__=='__main__': main()

