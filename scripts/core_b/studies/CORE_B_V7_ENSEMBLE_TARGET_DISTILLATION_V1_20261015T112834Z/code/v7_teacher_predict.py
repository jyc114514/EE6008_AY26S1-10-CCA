#!/usr/bin/env python3
"""Replay one frozen V2/V4 teacher on accepted outer-train prefix inputs only."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file, save_file
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code'; LOCKED=CODE/'locked_source'
sys.path.insert(0,str(CODE)); sys.path.insert(0,str(LOCKED))
from prepare_v2_data import Q0ShardedGrid
from v2_model import V2TaskConditionedPredictor, parameter_count, state_hash
TASK_IDS=(119,94,90,141,230)
PREFIX_FIELDS=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')
def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def atomic_json(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def append_jsonl(path:Path,row:dict[str,Any])->None:
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
def load_prefix(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:
        missing=set(PREFIX_FIELDS)-set(z.files)
        if missing: raise RuntimeError(f'prefix archive missing fields: {sorted(missing)}')
        if path.name=='DEV_PREFIXES_NO_LABELS.npz' and 'target_native_h30' in z.files:
            raise RuntimeError('dev prefix archive contains a future target key')
        return {k:z[k].copy() for k in PREFIX_FIELDS}
def row_csv(path:Path,data:dict[str,np.ndarray])->None:
    fields=['row','anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(len(data['anchor_id'])):
            task=int(data['task_id'][i]); rank=TASK_IDS.index(task)
            if int(data['task_rank'][i])!=rank: raise RuntimeError(f'task rank differs at {data["anchor_id"][i]}')
            w.writerow({'row':i,'anchor_id':str(data['anchor_id'][i]),'episode_id':int(data['episode_id'][i]),'task_id':task,'task_rank':rank,'source_cluster':str(data['source_cluster'][i]),'anchor_frame':int(data['anchor_frame'][i])})
def build_prefix(data:dict[str,np.ndarray],ids:np.ndarray,norm:dict[str,Any],device:torch.device,mode:str,grid:Q0ShardedGrid|None)->tuple[dict[str,torch.Tensor],torch.Tensor,torch.Tensor]:
    state_mean=np.asarray(norm['state_mean'],dtype=np.float32); state_std=np.asarray(norm['state_safe_std'],dtype=np.float32)
    rel_mean=np.float32(norm['relative_delta_mean']); rel_std=np.float32(norm['relative_delta_safe_std'])
    state=(data['history_native'][ids]-state_mean[None,None])/state_std[None,None]
    rel=(data['relative_delta'][ids]-rel_mean)/rel_std
    prefix={'state':torch.as_tensor(state,device=device,dtype=torch.float32),
            'history_mask':torch.as_tensor(data['history_mask'][ids],device=device,dtype=torch.float32),
            'relative_delta':torch.as_tensor(rel,device=device,dtype=torch.float32),
            'anchor_command':torch.as_tensor(data['anchor_native'][ids],device=device,dtype=torch.float32)}
    task_rank=np.asarray([TASK_IDS.index(int(data['task_id'][i])) for i in ids],dtype=np.int64)
    task=torch.as_tensor(task_rank,device=device,dtype=torch.long)
    anchor=torch.as_tensor(data['anchor_native'][ids],device=device,dtype=torch.float32)
    if mode!='STATE-CTRL':
        if grid is None: raise RuntimeError('video teacher has no accepted Q0 feature reader')
        raw=np.stack([grid.row(int(data['cache_index'][i]),str(data['anchor_id'][i]),str(data['episode_id'][i])) for i in ids]).astype(np.float32,copy=False)
        vm=np.asarray(norm['video_mean'],dtype=np.float32)[None,None,None,:]
        vs=np.asarray(norm['video_safe_std'],dtype=np.float32)[None,None,None,:]
        prefix['dynamic']=torch.as_tensor((raw[:,0]-vm)/vs,device=device,dtype=torch.float32)
        prefix['static']=torch.as_tensor((raw[:,1]-vm)/vs,device=device,dtype=torch.float32)
    return prefix,task,anchor
def run(job_path:Path)->int:
    job=json.loads(job_path.read_text(encoding='utf-8')); out=Path(job['output_root']); out.mkdir(parents=True,exist_ok=True)
    status_path=out/'RUN_STATUS.json'; complete=out/'PREDICTION_COMPLETE.json'; started=utc(); timer=time.perf_counter()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('teacher replay requires the watcher-authorized CUDA device')
        mode=str(job['video_mode']); role=str(job['teacher_role']); seed=int(job['seed'])
        if role not in {'S_OPT','A_COMMON','V_MATCHED'} or mode != ('TS11-D' if role=='V_MATCHED' else 'STATE-CTRL'):
            raise RuntimeError('teacher role and accepted source input mode differ')
        device=torch.device('cuda:0'); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.use_deterministic_algorithms(True)
        model_path=Path(job['model_path']); manifest_path=Path(job['model_manifest_path'])
        model_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if model_manifest.get('status')!='PASS' or int(model_manifest.get('seed',-1))!=seed:
            raise RuntimeError('frozen teacher manifest is not accepted for this seed')
        if sha(model_path)!=job['model_sha256'] or sha(manifest_path)!=job['model_manifest_sha256']:
            raise RuntimeError('teacher weight or manifest hash mismatch')
        normalizer_path=Path(job['normalizer_path']); norm_sha=sha(normalizer_path)
        if norm_sha!=job['normalizer_sha256'] or model_manifest.get('normalizer_sha256')!=norm_sha:
            raise RuntimeError('teacher normalizer hash differs from accepted model provenance')
        normalizer=json.loads(normalizer_path.read_text(encoding='utf-8'))
        model=V2TaskConditionedPredictor(seed=seed,video_mode=mode,waypoint_mode=None)
        model.load_state_dict(load_file(str(model_path),device='cpu'),strict=True)
        loaded_hash=state_hash(model)
        expected_state=model_manifest.get('final_state_hash',model_manifest.get('model_state_hash'))
        if loaded_hash!=expected_state: raise RuntimeError('teacher strict reload state hash mismatch')
        for p in model.parameters(): p.requires_grad_(False)
        model=model.to(device).eval(); torch.cuda.reset_peak_memory_stats(device)
        data=load_prefix(Path(job['train_prefix_path']))
        n=len(data['anchor_id'])
        if n!=1280 or len(set(data['anchor_id'].astype(str)))!=n: raise RuntimeError('outer-train prefix rows/IDs differ from the 1280-row lock')
        if sha(Path(job['train_prefix_path']))!=job['train_prefix_sha256']: raise RuntimeError('outer-train prefix archive SHA mismatch')
        resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text(encoding='utf-8'))
        grid=None
        if mode!='STATE-CTRL':
            cache_path=Path(resolved['feature_cache'])
            if sha(cache_path)!=resolved['feature_cache_manifest_sha256']: raise RuntimeError('Q0 cache manifest hash mismatch')
            grid=Q0ShardedGrid(json.loads(cache_path.read_text(encoding='utf-8')))
        batch=int(job.get('batch_size',64)); sigma=torch.as_tensor(np.asarray(normalizer['target_sigma'],dtype=np.float32),device=device)
        anchor_all=torch.as_tensor(data['anchor_native'],device=device,dtype=torch.float32); pred=[]
        before_hash=state_hash(model); start=time.perf_counter()
        forward_rows=0
        with torch.no_grad():
            for begin in range(0,n,batch):
                ids=np.arange(begin,min(begin+batch,n),dtype=np.int64)
                prefix,task,anchor=build_prefix(data,ids,normalizer,device,mode,grid)
                output=model(prefix,task)
                native=anchor[:,None,:]+sigma[None,None,:]*output.action_residual.float()
                pred.append(native.detach().to(dtype=torch.float64).cpu())
                forward_rows+=len(ids)
                append_jsonl(out/'EXPOSURE.jsonl',{'event':'frozen_teacher_train_forward','job_id':job['job_id'],'teacher_role':role,'seed':seed,'rows':len(ids),'cumulative_rows':forward_rows,'future_GT_rows':0,'utc':utc()})
        native=torch.cat(pred,dim=0).contiguous()
        if tuple(native.shape)!=(1280,30,28) or not torch.isfinite(native).all(): raise RuntimeError(f'teacher prediction shape/finite failure: {tuple(native.shape)}')
        if any(p.grad is not None for p in model.parameters()): raise RuntimeError('frozen teacher acquired gradients')
        after_hash=state_hash(model)
        if before_hash!=after_hash: raise RuntimeError('frozen teacher state changed during inference')
        pred_path=out/'PREDICTIONS.safetensors'; tmp=pred_path.with_name(f'.{pred_path.name}.{os.getpid()}.part')
        save_file({'prediction_native':native},str(tmp)); os.replace(tmp,pred_path)
        rows_path=out/'PREDICTION_ROWS.csv'; row_csv(rows_path,data)
        reloaded=load_file(str(pred_path),device='cpu')['prediction_native']
        if tuple(reloaded.shape)!=(1280,30,28) or reloaded.dtype!=torch.float64 or not torch.isfinite(reloaded).all(): raise RuntimeError('teacher safe-tensor reload failed')
        if hashlib.sha256(reloaded.numpy().tobytes(order='C')).hexdigest()!=hashlib.sha256(native.numpy().tobytes(order='C')).hexdigest(): raise RuntimeError('teacher prediction payload changed on reload')
        receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'teacher_role':role,'seed':seed,'video_mode':mode,
                 'model_path':str(model_path),'model_sha256':sha(model_path),'model_manifest_path':str(manifest_path),
                 'model_manifest_sha256':sha(manifest_path),'model_state_hash_before':before_hash,'model_state_hash_after':after_hash,
                 'normalizer_path':str(normalizer_path),'normalizer_sha256':norm_sha,'train_prefix_path':str(job['train_prefix_path']),
                 'train_prefix_sha256':job['train_prefix_sha256'],'prediction_path':str(pred_path),'prediction_sha256':sha(pred_path),
                 'prediction_rows_path':str(rows_path),'prediction_rows_sha256':sha(rows_path),'row_count':n,'prediction_shape':list(native.shape),
                 'prediction_native_dtype':'float64','source_compute_dtype':'float32','teacher_parameters_frozen':True,'teacher_gradients':False,
                 'future_GT_requested_by_inference_loader':False,'forward_arguments':['prefix_inputs','task_id'],'forward_rows':forward_rows,'final_heldout_payload_access':0,
                 'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),
                 'logical_device':str(device),'peak_memory_allocated_mib':float(torch.cuda.max_memory_allocated(device)/(1024*1024)),
                 'peak_memory_reserved_mib':float(torch.cuda.max_memory_reserved(device)/(1024*1024)),
                 'elapsed_seconds':time.perf_counter()-timer,'forward_elapsed_seconds':time.perf_counter()-start,'completed_utc':utc()}
        atomic_json(complete,receipt); atomic_json(status_path,{'status':'complete','completed_utc':utc(),'job_id':job['job_id']})
        print(json.dumps({'status':'PASS','job_id':job['job_id'],'rows':n,'prediction_sha256':receipt['prediction_sha256'],'peak_reserved_mib':receipt['peak_memory_reserved_mib']},sort_keys=True)); return 0
    except Exception as exc:
        atomic_json(status_path,{'status':'failed','failed_utc':utc(),'job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'partial_forward_rows':int(locals().get('forward_rows',0)),'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown')})
        print(json.dumps({'status':'FAILED','job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc)},sort_keys=True),flush=True); return 1
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--job-json',type=Path,required=True); return run(ap.parse_args().job_json)
if __name__=='__main__': raise SystemExit(main())
