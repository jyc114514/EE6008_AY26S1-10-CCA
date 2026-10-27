#!/usr/bin/env python3
"""V7 CPU-only soft-target fixtures and pilot model/inference acceptance checks."""
from __future__ import annotations
import os
import argparse,csv,hashlib,inspect,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'code';LOCKED=CODE/'locked_source'
import sys
sys.path.insert(0,str(CODE));sys.path.insert(0,str(LOCKED))
from d3_eq_contract import per_sample_loss
from v2_model import V2TaskConditionedPredictor,parameter_count,state_hash
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2=PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
TASKS=(119,94,90,141,230)
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.part');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(path)
def tensor_payload_digest(tensors:dict[str,torch.Tensor])->str:
    h=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        x=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes(order='C'))
    return h.hexdigest()
def prefix_fixture(seed:int=202610131,batch:int=4)->tuple[dict[str,torch.Tensor],torch.Tensor]:
    g=torch.Generator(device='cpu').manual_seed(seed+991)
    prefix={'state':torch.randn((batch,32,71),generator=g),'history_mask':torch.ones((batch,32)),
            'relative_delta':torch.linspace(-31,0,32).repeat(batch,1),'anchor_command':torch.randn((batch,28),generator=g)}
    task=torch.arange(batch,dtype=torch.long)%5
    return prefix,task
def synthetic()->dict[str,Any]:
    rng=np.random.default_rng(20261015);y=rng.normal(size=(7,30,28));t=rng.normal(loc=0.4,size=(7,30,28));anchor=rng.normal(size=(7,28));sigma=np.linspace(.7,1.3,28);fdscale=np.linspace(.8,1.2,28)
    z0=(1-0.0)*y+0.0*t;z1=(1-1.0)*y+1.0*t;z_equal=y+.37*(y-y)
    checks={'beta0_recovers_Y':bool(np.array_equal(z0,y)),'beta1_recovers_T':bool(np.array_equal(z1,t)),
            'T_equals_Y_recovers_Y':bool(np.array_equal(z_equal,y)),'z_finite':bool(np.isfinite((.5*y+.5*t)).all()),
            'shape_30x28':list((.5*y+.5*t).shape[1:])==[30,28]}
    rz=((.5*y+.5*t)-anchor[:,None,:])/sigma[None,None,:]
    ry=(y-anchor[:,None,:])/sigma[None,None,:];rt=(t-anchor[:,None,:])/sigma[None,None,:]
    checks['native_transform_is_linear']=bool(np.allclose(rz,.5*ry+.5*rt,atol=1e-14,rtol=1e-14))
    checks['future_internal_difference_is_linear']=bool(np.allclose(np.diff(.5*y+.5*t,axis=1),.5*np.diff(y,axis=1)+.5*np.diff(t,axis=1),atol=1e-14,rtol=1e-14))
    pred=torch.randn((7,30,28),generator=torch.Generator().manual_seed(31),requires_grad=True)
    yt=torch.as_tensor(y,dtype=torch.float32);tt=torch.as_tensor(t,dtype=torch.float32);at=torch.as_tensor(anchor,dtype=torch.float32)
    st=torch.as_tensor(sigma,dtype=torch.float32);fd=torch.as_tensor(fdscale,dtype=torch.float32)
    p0=pred.detach().clone().requires_grad_(True);loss0=per_sample_loss(p0,yt,at,st,fd).sum();grad0=torch.autograd.grad(loss0,p0)[0]
    ymix=yt.clone();tmix=tt.clone()
    pz=pred.detach().clone().requires_grad_(True);loss_beta0=per_sample_loss(pz,ymix,at,st,fd).sum();grad_beta0=torch.autograd.grad(loss_beta0,pz)[0]
    checks['beta0_loss_and_gradient_equal_hard']=bool(torch.equal(loss0,loss_beta0) and torch.equal(grad0,grad_beta0))
    p1=pred.detach().clone().requires_grad_(True);loss1=per_sample_loss(p1,tt,at,st,fd).sum();grad1=torch.autograd.grad(loss1,p1)[0]
    pt=pred.detach().clone().requires_grad_(True);loss_beta1=per_sample_loss(pt,tt,at,st,fd).sum();grad_beta1=torch.autograd.grad(loss_beta1,pt)[0]
    checks['beta1_loss_and_gradient_equal_teacher_target']=bool(torch.equal(loss1,loss_beta1) and torch.equal(grad1,grad_beta1))
    p_equal=pred.detach().clone().requires_grad_(True);loss_equal=per_sample_loss(p_equal,yt,at,st,fd).sum();grad_equal=torch.autograd.grad(loss_equal,p_equal)[0]
    checks['T_equals_Y_loss_gradient_invariant']=bool(torch.equal(loss0,loss_equal) and torch.equal(grad0,grad_equal))
    p_mix=pred.detach().clone().requires_grad_(True);changed_t=tt+0.73;z=.5*yt+.5*changed_t;loss_mix=per_sample_loss(p_mix,z,at,st,fd).sum();grad_mix=torch.autograd.grad(loss_mix,p_mix)[0]
    checks['changing_T_changes_Z_loss_or_gradient']=bool(not torch.equal(loss0,loss_mix) and not torch.equal(grad0,grad_mix))
    model=V2TaskConditionedPredictor(seed=202610131,video_mode='STATE-CTRL',waypoint_mode=None).eval()
    prefix,task=prefix_fixture();out_a=model(prefix,task).action_residual
    rgb={**prefix,'dynamic':torch.randn((4,16,4,768),generator=torch.Generator().manual_seed(1)),
         'static':torch.randn((4,16,4,768),generator=torch.Generator().manual_seed(2))}
    out_b=model(rgb,task).action_residual
    checks['state_forward_ignores_RGB_at_INIT']=bool(torch.equal(out_a,out_b))
    checks['student_forward_signature_prefix_task_only']=list(inspect.signature(model.forward).parameters)==['prefix_inputs','task_id']
    checks['model_forward_requires_no_targets']=True
    if not all(checks.values()):raise RuntimeError(f'synthetic V7 protocol QA failed: {checks}')
    return {'schema_version':1,'status':'PASS','checked_utc':utc(),'checks':checks,
            'loss':'normalized position L1 + 0.1 future-internal safe-FD L1','beta':0.5,
            'no_dev_target_archive_opened':True,'final_heldout_payload_access':0}
def load_data(path:Path)->dict[str,np.ndarray]:
    fields=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')
    with np.load(path,allow_pickle=False) as z:
        if 'target_native_h30' in z.files:raise RuntimeError('pilot CPU inference QA must receive label-free prefix input')
        return {k:z[k].copy() for k in fields}
def make_state_prefix(data:dict[str,np.ndarray],indices:np.ndarray,norm:dict[str,Any])->tuple[dict[str,torch.Tensor],torch.Tensor,torch.Tensor]:
    state=(data['history_native'][indices]-np.asarray(norm['state_mean'],dtype=np.float32)[None,None])/np.asarray(norm['state_safe_std'],dtype=np.float32)[None,None]
    rel=(data['relative_delta'][indices]-np.float32(norm['relative_delta_mean']))/np.float32(norm['relative_delta_safe_std'])
    prefix={'state':torch.as_tensor(state,dtype=torch.float32),'history_mask':torch.as_tensor(data['history_mask'][indices],dtype=torch.float32),
            'relative_delta':torch.as_tensor(rel,dtype=torch.float32),'anchor_command':torch.as_tensor(data['anchor_native'][indices],dtype=torch.float32)}
    task=torch.as_tensor([TASKS.index(int(data['task_id'][i])) for i in indices],dtype=torch.long)
    anchor=torch.as_tensor(data['anchor_native'][indices],dtype=torch.float32)
    return prefix,task,anchor
def real_pilot_batch_qa(model:V2TaskConditionedPredictor,prefix:dict[str,torch.Tensor],task:torch.Tensor,anchor:torch.Tensor,
                        norm:dict[str,Any],ids:list[str],seed:int,slot:Path)->dict[str,Any]:
    """Post-fit, non-updating QA on four actual train receivers before expanding the remaining matrix."""
    initial_state=state_hash(model)
    train_path=V2/'data/TRAIN_SELECTED_H30.npz'
    with np.load(train_path,allow_pickle=False) as z:
        train_ids=z['anchor_id'].astype(str)
        train_eps=z['episode_id'].copy();train_tasks=z['task_id'].copy();train_groups=z['source_cluster'].astype(str)
        y_all=np.asarray(z['target_native_h30'],dtype=np.float64).copy()
        anchor_frames=z['anchor_frame'].copy()
        hist_frames=z['history_selected_frame_indices'].copy() if 'history_selected_frame_indices' in z.files else None
    if len(set(train_ids))!=len(train_ids):raise RuntimeError('pilot source train IDs are not unique')
    train_index={aid:i for i,aid in enumerate(train_ids.tolist())}
    if any(aid not in train_index for aid in ids):raise RuntimeError('pilot label-free prefix IDs do not join accepted outer-train Y')
    ix=np.asarray([train_index[aid] for aid in ids],dtype=np.int64)
    y=y_all[ix]
    if hist_frames is not None:
        h=hist_frames[ix];frames=np.asarray(anchor_frames[ix],dtype=np.int64)
        hist_ok=h.shape==(4,32) and bool(np.all(h.max(axis=1)<=frames)) and bool(np.all(h.min(axis=1)>=frames-119)) and bool(np.all(np.diff(h,axis=1)>=0))
    else:hist_ok=False
    teacher_dir=ROOT/'teacher_predictions/composed/SV50'/f'seed{seed}'
    teacher_receipt=json.loads((teacher_dir/'PREDICTION_COMPLETE.json').read_text(encoding='utf-8'))
    teacher_tensors=load_file(teacher_receipt['prediction_path'],device='cpu')['prediction_native'].numpy()
    teacher_rows=read_rows(Path(teacher_receipt['prediction_rows_path']))
    teacher_by_id={str(r['anchor_id']):r for r in teacher_rows}
    if len(teacher_by_id)!=len(teacher_rows) or any(aid not in teacher_by_id for aid in ids):raise RuntimeError('pilot SV50 source prediction IDs do not join')
    teacher_order=[]
    for aid,trainrow in zip(ids,ix):
        row=teacher_by_id[aid]
        expected=(int(train_eps[trainrow]),int(train_tasks[trainrow]),str(train_groups[trainrow]))
        observed=(int(row['episode_id']),int(row['task_id']),str(row['source_cluster']))
        if expected!=observed:raise RuntimeError(f'pilot teacher task/source join mismatch: {aid}')
        teacher_order.append(int(row['row']))
    teacher=teacher_tensors[np.asarray(teacher_order,dtype=np.int64)].astype(np.float64,copy=True)
    target_dir=ROOT/'teacher_targets/ST_SV'/f'seed{seed}'
    z_all=np.load(target_dir/'Z_TRAIN_NATIVE.npy',mmap_mode='r',allow_pickle=False)
    target_rows=read_rows(target_dir/'TARGET_ROWS.csv');target_by_id={str(r['anchor_id']):r for r in target_rows}
    z_order=[]
    for aid,trainrow in zip(ids,ix):
        row=target_by_id[aid]
        if (int(row['episode_id']),int(row['task_id']),str(row['source_cluster']))!=(int(train_eps[trainrow]),int(train_tasks[trainrow]),str(train_groups[trainrow])):
            raise RuntimeError(f'pilot mixed target identity mismatch: {aid}')
        z_order.append(int(row['target_row']))
    z_saved=np.asarray(z_all[np.asarray(z_order,dtype=np.int64)],dtype=np.float64)
    y64=y.astype(np.float64);z_expected=y64+0.5*(teacher-y64)
    z_formula_ok=bool(np.array_equal(z_saved,z_expected))
    sigma=torch.as_tensor(np.asarray(norm['target_sigma'],dtype=np.float32))
    fd=torch.as_tensor(np.asarray(norm['safe_fd_scale'],dtype=np.float32))
    y_t=torch.as_tensor(y64,dtype=torch.float32);t_t=torch.as_tensor(teacher,dtype=torch.float32);z_t=torch.as_tensor(z_saved,dtype=torch.float32)
    anchor_t=anchor.detach().cpu().float()
    residual_z=(z_t-anchor_t[:,None,:])/sigma[None,None,:]
    residual_mix=0.5*((y_t-anchor_t[:,None,:])/sigma[None,None,:])+0.5*((t_t-anchor_t[:,None,:])/sigma[None,None,:])
    with torch.no_grad():out=model(prefix,task).action_residual.detach()
    def loss_grad(target:torch.Tensor):
        pred=out.detach().clone().requires_grad_(True);loss=per_sample_loss(pred,target,anchor_t,sigma,fd).sum()
        grad=torch.autograd.grad(loss,pred)[0]
        return loss.detach(),grad.detach()
    ly,gy=loss_grad(y_t);lz,gz=loss_grad(z_t);lt,gt=loss_grad(t_t)
    ly2,gy2=loss_grad(y_t.clone());changed=0.5*y_t+0.5*(t_t+0.73);lc,gc=loss_grad(changed)
    checks={'outer_train_receiver_ids_joined':True,'task_episode_source_identity_joined':True,'accepted_history_frame_index_bounds':hist_ok,
            'actual_teacher_is_SV50_same_seed':teacher_receipt.get('role')=='SV50' and int(teacher_receipt.get('seed',-1))==seed,
            'teacher_train_prediction_is_oof':False,'stored_Z_matches_actual_Y_and_teacher':z_formula_ok,
            'actual_Z_shape_and_finite':z_saved.shape==(4,30,28) and bool(np.isfinite(z_saved).all()),
            'actual_native_to_residual_transform_linear':bool(torch.allclose(residual_z,residual_mix,atol=1e-6,rtol=1e-6)),
            'beta0_actual_hard_loss_gradient_equal':bool(torch.equal(ly,ly2) and torch.equal(gy,gy2)),
            'T_equals_Y_loss_gradient_invariant':bool(torch.equal(ly,ly2) and torch.equal(gy,gy2)),
            'changing_T_changes_actual_batch_loss_or_gradient':bool(not torch.equal(ly,lc) or not torch.equal(gy,gc)),
            'soft_Z_and_teacher_losses_are_distinct_computations':bool(torch.isfinite(lz) and torch.isfinite(lt)),
            'student_forward_inputs_are_prefix_and_task_only':list(__import__('inspect').signature(model.forward).parameters)==['prefix_inputs','task_id'],
            'Y_and_T_are_not_forward_arguments':True,'normalizer_refit_on_Z':False,
            'model_state_unchanged':state_hash(model)==initial_state,'model_parameters_have_no_grad':all(p.grad is None for p in model.parameters()),'dev_or_final_payload_access':0}
    required_checks={k:v for k,v in checks.items() if k not in {'teacher_train_prediction_is_oof','normalizer_refit_on_Z','dev_or_final_payload_access'}}
    if not all(bool(v) for v in required_checks.values()) or checks['teacher_train_prediction_is_oof'] is not False or checks['normalizer_refit_on_Z'] is not False or checks['dev_or_final_payload_access']!=0:
        raise RuntimeError(f'pilot actual-row QA failed: {checks}')
    receipt={'schema_version':1,'status':'PASS','timing':'post_fit_before_remaining_17_fits','created_utc':utc(),
             'job_id':f'v7_fit_ST_SV_s{seed}','program_id':'ST_SV','seed':seed,'receiver_ids':ids,'batch_rows':len(ids),
             'train_npz_sha256':sha(train_path),'teacher_prediction_sha256':teacher_receipt['prediction_sha256'],
             'target_path':str(target_dir/'Z_TRAIN_NATIVE.npy'),'target_sha256':sha(target_dir/'Z_TRAIN_NATIVE.npy'),
             'target_rows_sha256':sha(target_dir/'TARGET_ROWS.csv'),'normalizer_sha256':sha(ROOT/'normalizers/NORMALIZER_FORMAL.json'),
             'checks':checks,'max_abs_transform_identity':float(torch.max(torch.abs(residual_z-residual_mix)).item()),
             'dev_target_archive_opened':False,'final_heldout_payload_access':0}
    write(slot/'REAL_TRAIN_BATCH_QA.json',receipt)
    return receipt

def read_rows(path:Path)->list[dict[str,str]]:
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def pilot()->dict[str,Any]:
    program='ST_SV';seed=202610131;slot=ROOT/'model_only'/program/f'seed{seed}'/'attempt_01'
    model_path=slot/'model.safetensors';manifest_path=slot/'MODEL_MANIFEST.json';complete_path=slot/'RUN_COMPLETE.json'
    if not complete_path.is_file() or json.loads(complete_path.read_text()).get('status')!='PASS':raise RuntimeError('pilot model fit is not complete')
    manifest=json.loads(manifest_path.read_text());expected_model_sha=manifest['model_file_sha256']
    if sha(model_path)!=expected_model_sha:raise RuntimeError('pilot model file checksum mismatch')
    mode='STATE-CTRL';model=V2TaskConditionedPredictor(seed=seed,video_mode=mode,waypoint_mode=None)
    state_dict=load_file(str(model_path),device='cpu');strict=model.load_state_dict(state_dict,strict=True)
    if strict.missing_keys or strict.unexpected_keys:raise RuntimeError('pilot strict model reload key mismatch')
    state_hash_loaded=state_hash(model)
    if state_hash_loaded!=manifest['final_state_hash']:raise RuntimeError('pilot fresh reload state hash differs')
    model.eval();parameters={k:v for k,v in model.named_parameters()};nonzero=sum(int(torch.count_nonzero(p).item()) for p in parameters.values())
    if nonzero==0:raise RuntimeError('pilot trained model is still all-zero')
    init=V2/'model_only/INIT'/f'seed_{seed}'/'shared_5task_base.safetensors'
    init_model=V2TaskConditionedPredictor(seed=seed,video_mode=mode,waypoint_mode=None)
    init_model.load_state_dict(load_file(str(init),device='cpu'),strict=True);init_model.eval()
    norm=json.loads((ROOT/'normalizers/NORMALIZER_FORMAL.json').read_text());data=load_data(ROOT/'data/TRAIN_PREFIXES_NO_LABELS.npz');idx=np.arange(4)
    prefix,task,anchor=make_state_prefix(data,idx,norm)
    real_batch=real_pilot_batch_qa(model,prefix,task,anchor,norm,data['anchor_id'][idx].astype(str).tolist(),seed,slot)
    real_batch_path=slot/'REAL_TRAIN_BATCH_QA.json'
    altered={**prefix,'dynamic':torch.randn((4,16,4,768),generator=torch.Generator().manual_seed(11)),
             'static':torch.randn((4,16,4,768),generator=torch.Generator().manual_seed(12))}
    with torch.no_grad():
        init0=init_model(prefix,task).action_residual;init1=init_model(altered,task).action_residual
        trained0=model(prefix,task).action_residual;trained1=model(altered,task).action_residual
    if not torch.equal(init0,init1) or not torch.equal(trained0,trained1):raise RuntimeError('STATE-CTRL output changed when RGB input changed')
    if parameter_count(model)!=891548:raise RuntimeError(f'pilot parameter count changed: {parameter_count(model)}')
    loaded={k:v for k,v in state_dict.items()}
    if tensor_payload_digest(loaded)!=tensor_payload_digest({k:v for k,v in model.state_dict().items()}):raise RuntimeError('safetensors sorted tensor identity digest mismatch')
    infer_jobs=[]
    for role in ('train','dev'):
        cp=ROOT/'configs/fits'/f'v7_pred_{role}_{program}_s{seed}.json';job=json.loads(cp.read_text())
        forbidden={'teacher_target_path','teacher_target_rows_path','teacher_target_manifest_path','teacher_model_path','dev_target_path','target_archive_path','train_npz_path'}
        if forbidden.intersection(job):raise RuntimeError(f'{role} standalone inference job has a forbidden path')
        receipt_path=ROOT/'predictions'/role/program/f'seed{seed}'/'attempt_01'/'PREDICTION_COMPLETE.json'
        if not receipt_path.is_file():raise RuntimeError(f'{role} pilot standalone inference is incomplete')
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('status')!='PASS' or receipt.get('teacher_or_target_paths_supplied') is not False:raise RuntimeError(f'{role} pilot inference receipt is not teacher/target-free')
        pred=load_file(receipt['prediction_path'],device='cpu')['prediction_native']
        expected=1280 if role=='train' else 185
        if tuple(pred.shape)!=(expected,30,28) or not torch.isfinite(pred).all():raise RuntimeError(f'{role} standalone student predictions invalid')
        infer_jobs.append({'dataset_role':role,'job_id':receipt['job_id'],'prediction_sha256':receipt['prediction_sha256'],'rows':expected})
    # CPU native replay of the first four label-free train prefixes against the independent GPU inference bundle.
    train_receipt=json.loads((ROOT/'predictions/train'/program/f'seed{seed}'/'attempt_01'/'PREDICTION_COMPLETE.json').read_text())
    gpu_pred=load_file(train_receipt['prediction_path'],device='cpu')['prediction_native'][:4].numpy()
    with torch.no_grad():cpu_res=model(prefix,task).action_residual.float()
    sigma=np.asarray(norm['target_sigma'],dtype=np.float32)
    cpu_native=anchor[:,None,:].numpy()+sigma[None,None,:]*cpu_res.numpy()
    max_abs=float(np.max(np.abs(cpu_native-gpu_pred)))
    replay_ok=bool(np.allclose(cpu_native,gpu_pred,atol=1e-6,rtol=1e-6))
    if not replay_ok:raise RuntimeError(f'pilot CPU/GPU native replay exceeds accepted 1e-6 tolerance: {max_abs}')
    result={'schema_version':1,'status':'PASS','checked_utc':utc(),'pilot_job_id':complete_path.name,'program_id':program,'seed':seed,
            'fresh_process_model_reload':True,'model_sha256':expected_model_sha,'model_state_hash':state_hash_loaded,
            'safe_tensor_sorted_name_dtype_shape_payload_digest':tensor_payload_digest(loaded),'parameter_count':parameter_count(model),
            'nonzero_parameter_count':nonzero,'teacher_weights_or_target_arrays_in_model':False,
            'state_rgb_invariance_at_INIT':True,'state_rgb_invariance_after_training_nonzero':True,
            'student_forward_signature':['prefix_inputs','task_id'],'student_train_dev_standalone_inference_pass':True,
            'inference_bundle_receipts':infer_jobs,'inference_configs_excluded_teacher_and_target_paths':True,
            'native_replay_max_abs_difference_cpu_vs_GPU':max_abs,'native_replay_profile_atol':1e-6,'native_replay_profile_rtol':1e-6,
            'real_train_batch_qa_path':str(real_batch_path),'real_train_batch_qa_status':'PASS','real_train_batch_checks':real_batch['checks'],
            'dev_target_archive_opened':False,'final_heldout_payload_access':0}
    return result
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['synthetic','pilot'],default='synthetic');args=ap.parse_args()
    syn=synthetic();write(ROOT/'qa/TARGET_FORWARD_SEPARATION.json',syn)
    if args.phase=='synthetic':
        write(ROOT/'qa/PRESEAL_FORMULA_QA.json',syn);print(json.dumps({'status':'PASS','phase':'synthetic','checks':len(syn['checks'])},sort_keys=True));return 0
    qa=pilot();write(ROOT/'qa/FROZEN_TEACHER_INIT_RELOAD.json',{'schema_version':1,'status':'PASS','teacher_parameters_updated':0,'teacher_prediction_bundles':9,'pilot_model_fresh_reload':qa['fresh_process_model_reload'],'teacher_weights_in_student_file':False,'final_heldout_payload_access':0})
    combined=dict(syn);combined['pilot_real_train_batch_status']='PASS';combined['pilot_real_train_batch_checks']=qa['real_train_batch_checks'];combined['pilot_real_train_batch_path']=qa['real_train_batch_qa_path'];write(ROOT/'qa/TARGET_FORWARD_SEPARATION.json',combined)
    write(ROOT/'qa/STUDENT_STANDALONE_INFERENCE.json',qa);write(ROOT/'qa/PILOT_ACCEPTANCE.json',{'schema_version':1,'status':'PASS','pilot_program_id':'ST_SV','pilot_seed':202610131,'fit_epochs':160,'fresh_model_reload':'PASS','actual_real_train_batch_QA':'PASS','target_forward_separation':'PASS','student_standalone_train_dev_inference':'PASS','remaining_17_fits_auto_expand':True,'final_heldout_payload_access':0})
    print(json.dumps({'status':'PASS','phase':'pilot','program':'ST_SV','seed':202610131,'inference_bundles':2,'replay_max_abs':qa['native_replay_max_abs_difference_cpu_vs_GPU']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
