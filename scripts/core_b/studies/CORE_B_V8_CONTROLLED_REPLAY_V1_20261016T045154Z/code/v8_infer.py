#!/usr/bin/env python3
"""Standalone student inference with model, label-free prefix/cache, and normalizer only."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1]; CODE=ROOT/'code'; LOCKED=CODE/'locked_source'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(LOCKED))
from prepare_v2_data import Q0ShardedGrid
from v2_model import V2TaskConditionedPredictor,parameter_count,state_hash
TASK_IDS=(119,94,90,141,230)
PREFIX_FIELDS=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic_json(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.{os.getpid()}.part')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
def append_jsonl(path:Path,row:dict[str,Any])->None:
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
def load_prefix(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:
        missing=set(PREFIX_FIELDS)-set(z.files)
        if missing:raise RuntimeError(f'prefix archive missing fields: {sorted(missing)}')
        if 'target_native_h30' in z.files:raise RuntimeError('standalone inference archive contains a future-target field')
        return {k:z[k].copy() for k in PREFIX_FIELDS}
def row_csv(path:Path,data:dict[str,np.ndarray],dataset_role:str)->None:
    fields=['row','anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame','dataset_role']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(len(data['anchor_id'])):
            task=int(data['task_id'][i]);rank=TASK_IDS.index(task)
            if int(data['task_rank'][i])!=rank:raise RuntimeError(f'task rank differs at {data["anchor_id"][i]}')
            w.writerow({'row':i,'anchor_id':str(data['anchor_id'][i]),'episode_id':int(data['episode_id'][i]),'task_id':task,'task_rank':rank,'source_cluster':str(data['source_cluster'][i]),'anchor_frame':int(data['anchor_frame'][i]),'dataset_role':dataset_role})
def make_prefix(data:dict[str,np.ndarray],ids:np.ndarray,norm:dict[str,Any],device:torch.device,mode:str,grid:Q0ShardedGrid|None)->tuple[dict[str,torch.Tensor],torch.Tensor,torch.Tensor]:
    state=(data['history_native'][ids]-np.asarray(norm['state_mean'],dtype=np.float32)[None,None])/np.asarray(norm['state_safe_std'],dtype=np.float32)[None,None]
    rel=(data['relative_delta'][ids]-np.float32(norm['relative_delta_mean']))/np.float32(norm['relative_delta_safe_std'])
    prefix={'state':torch.as_tensor(state,device=device,dtype=torch.float32),'history_mask':torch.as_tensor(data['history_mask'][ids],device=device,dtype=torch.float32),'relative_delta':torch.as_tensor(rel,device=device,dtype=torch.float32),'anchor_command':torch.as_tensor(data['anchor_native'][ids],device=device,dtype=torch.float32)}
    task=torch.as_tensor([TASK_IDS.index(int(data['task_id'][i])) for i in ids],device=device,dtype=torch.long)
    anchor=torch.as_tensor(data['anchor_native'][ids],device=device,dtype=torch.float32)
    if mode!='STATE-CTRL':
        if grid is None:raise RuntimeError('video student lacks the accepted feature cache reader')
        raw=np.stack([grid.row(int(data['cache_index'][i]),str(data['anchor_id'][i]),str(data['episode_id'][i])) for i in ids]).astype(np.float32,copy=False)
        vm=np.asarray(norm['video_mean'],dtype=np.float32)[None,None,None,:];vs=np.asarray(norm['video_safe_std'],dtype=np.float32)[None,None,None,:]
        prefix['dynamic']=torch.as_tensor((raw[:,0]-vm)/vs,device=device,dtype=torch.float32)
        prefix['static']=torch.as_tensor((raw[:,1]-vm)/vs,device=device,dtype=torch.float32)
    return prefix,task,anchor
def run(job_path:Path)->int:
    job=json.loads(job_path.read_text(encoding='utf-8'));out=Path(job['output_root']);out.mkdir(parents=True,exist_ok=True)
    status_path=out/'RUN_STATUS.json';complete=out/'PREDICTION_COMPLETE.json';started=utc();timer=time.perf_counter()
    forbidden={'teacher_target_path','teacher_target_rows_path','teacher_target_manifest_path','teacher_model_path','dev_target_path','target_archive_path','train_npz_path'}
    if forbidden.intersection(job):raise RuntimeError(f'standalone inference config contains forbidden GT/teacher paths: {sorted(forbidden.intersection(job))}')
    try:
        if not torch.cuda.is_available():raise RuntimeError('student inference requires the watcher-authorized CUDA device')
        seed=int(job['seed']);mode=str(job['video_mode']);dataset_role=str(job['dataset_role'])
        if dataset_role not in {'train','dev'} or mode not in {'STATE-CTRL','TS11-D'}:raise RuntimeError('invalid student input role/mode')
        model_path=Path(job['model_path']);manifest_path=Path(job['model_manifest_path']);norm_path=Path(job['normalizer_path']);prefix_path=Path(job['prefix_path'])
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status')!='PASS' or not manifest.get('student_inference_requires_teacher') is False:raise RuntimeError('student model manifest is not accepted teacher-free')
        if int(manifest.get('seed',-1))!=seed or manifest.get('video_mode')!=mode:raise RuntimeError('student seed/input mode differs from model manifest')
        if sha(model_path)!=job['model_sha256'] or sha(manifest_path)!=job['model_manifest_sha256']:raise RuntimeError('student model or manifest hash mismatch')
        norm_sha=sha(norm_path)
        if norm_sha!=job['normalizer_sha256'] or manifest.get('normalizer_sha256')!=norm_sha:raise RuntimeError('student normalizer hash mismatch')
        prefix_sha=sha(prefix_path)
        if prefix_sha!=job['prefix_sha256']:raise RuntimeError('label-free prefix archive hash mismatch')
        device=torch.device('cuda:0');torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        model=V2TaskConditionedPredictor(seed=seed,video_mode=mode,waypoint_mode=None)
        model.load_state_dict(load_file(str(model_path),device='cpu'),strict=True)
        loaded_hash=state_hash(model)
        if loaded_hash!=manifest.get('final_state_hash'):raise RuntimeError('student fresh-process state hash mismatch')
        model=model.to(device).eval();torch.cuda.reset_peak_memory_stats()
        data=load_prefix(prefix_path);n=len(data['anchor_id'])
        expected=1280 if dataset_role=='train' else 185
        if n!=expected or n!=int(job['expected_rows']) or len(set(data['anchor_id'].astype(str)))!=n:raise RuntimeError(f'{dataset_role} prefix row/ID support mismatch: {n}')
        norm=json.loads(norm_path.read_text(encoding='utf-8'));resolved=json.loads((ROOT/'inventory/RESOLVED_PATHS.json').read_text(encoding='utf-8'))
        grid=None
        if mode!='STATE-CTRL':
            cache_path=Path(resolved['feature_cache'])
            if sha(cache_path)!=resolved['feature_cache_manifest_sha256']:raise RuntimeError('Q0 cache manifest hash mismatch')
            grid=Q0ShardedGrid(json.loads(cache_path.read_text(encoding='utf-8')))
        sigma=torch.as_tensor(np.asarray(norm['target_sigma'],dtype=np.float32),device=device);rows=[];before_hash=state_hash(model)
        forward_rows=0
        with torch.no_grad():
            for start in range(0,n,int(job.get('batch_size',64))):
                ids=np.arange(start,min(start+int(job.get('batch_size',64)),n),dtype=np.int64)
                prefix,task,anchor=make_prefix(data,ids,norm,device,mode,grid)
                out_model=model(prefix,task)
                native=anchor[:,None,:]+sigma[None,None,:]*out_model.action_residual.float()
                rows.append(native.detach().to(dtype=torch.float64).cpu())
                forward_rows+=len(ids)
                append_jsonl(out/'EXPOSURE.jsonl',{'event':f'student_{dataset_role}_prefix_forward','job_id':job['job_id'],'program_id':job['program_id'],'seed':seed,'rows':len(ids),'cumulative_rows':forward_rows,'GT_rows':0,'utc':utc()})
        native=torch.cat(rows,dim=0).contiguous()
        if tuple(native.shape)!=(n,30,28) or not torch.isfinite(native).all():raise RuntimeError(f'student prediction shape/finite failure: {tuple(native.shape)}')
        if state_hash(model)!=before_hash:raise RuntimeError('student model state changed during inference')
        pred_path=out/'PREDICTIONS.safetensors';tmp=pred_path.with_name(f'.{pred_path.name}.{os.getpid()}.part')
        save_file({'prediction_native':native},str(tmp));os.replace(tmp,pred_path)
        rows_path=out/'PREDICTION_ROWS.csv';row_csv(rows_path,data,dataset_role)
        reloaded=load_file(str(pred_path),device='cpu')['prediction_native']
        if reloaded.dtype!=torch.float64 or tuple(reloaded.shape)!=(n,30,28) or not torch.isfinite(reloaded).all():raise RuntimeError('student safe-tensor reload failed')
        if hashlib.sha256(reloaded.numpy().tobytes(order='C')).digest()!=hashlib.sha256(native.numpy().tobytes(order='C')).digest():raise RuntimeError('student prediction payload changed on reload')
        receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'program_id':job['program_id'],'seed':seed,'student_input':mode,'dataset_role':dataset_role,
                 'model_path':str(model_path),'model_sha256':sha(model_path),'model_manifest_path':str(manifest_path),'model_manifest_sha256':sha(manifest_path),
                 'model_state_hash':loaded_hash,'normalizer_path':str(norm_path),'normalizer_sha256':norm_sha,'prefix_path':str(prefix_path),'prefix_sha256':prefix_sha,
                 'prediction_path':str(pred_path),'prediction_sha256':sha(pred_path),'prediction_rows_path':str(rows_path),'prediction_rows_sha256':sha(rows_path),
                 'prediction_shape':list(native.shape),'prediction_native_dtype':'float64','forward_arguments':['prefix_inputs','task_id'],
                 'teacher_or_target_paths_supplied':False,'future_GT_requested_by_inference_loader':False,'teacher_dependencies_required_at_inference':False,
                 'fresh_model_state_reload':True,'prediction_safetensors_reload':True,'parameter_count':parameter_count(model),'forward_rows':forward_rows,
                 'job_config_path':str(job_path),'job_config_sha256':sha(job_path),'inference_runner_path':str(Path(__file__).resolve()),'inference_runner_sha256':sha(Path(__file__).resolve()),
                 'runtime':{'hostname':os.uname().nodename,'python_version':sys.version.split()[0],'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,
                            'cudnn_version':torch.backends.cudnn.version(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
                            'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast':False},
                 'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':str(device),
                 'peak_memory_allocated_mib':float(torch.cuda.max_memory_allocated()/(1024*1024)),'peak_memory_reserved_mib':float(torch.cuda.max_memory_reserved()/(1024*1024)),
                 'elapsed_seconds':time.perf_counter()-timer,'completed_utc':utc(),'final_heldout_payload_access':0}
        atomic_json(complete,receipt);atomic_json(status_path,{'status':'complete','completed_utc':utc(),'job_id':job['job_id']})
        print(json.dumps({'status':'PASS','job_id':job['job_id'],'rows':n,'prediction_sha256':receipt['prediction_sha256'],'peak_reserved_mib':receipt['peak_memory_reserved_mib']},sort_keys=True));return 0
    except Exception as exc:
        atomic_json(status_path,{'status':'failed','failed_utc':utc(),'job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'partial_forward_rows':int(locals().get('forward_rows',0)),'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown')})
        print(json.dumps({'status':'FAILED','job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc)},sort_keys=True),flush=True);return 1
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);return run(ap.parse_args().job_json)
if __name__=='__main__':raise SystemExit(main())
