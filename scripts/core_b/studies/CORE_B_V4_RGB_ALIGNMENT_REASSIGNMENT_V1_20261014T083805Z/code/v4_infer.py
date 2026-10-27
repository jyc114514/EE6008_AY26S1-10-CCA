#!/usr/bin/env python3
"""Run one sealed V4 dev-prefix prediction job; never loads dev targets."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code';LOCKED=CODE/'locked_source'
import sys
sys.path.insert(0,str(CODE));sys.path.insert(0,str(LOCKED))
from prepare_v2_data import Q0ShardedGrid  # noqa:E402
from v2_model import V2TaskConditionedPredictor,parameter_count,state_hash  # noqa:E402
TASK_IDS=(119,94,90,141,230)
DEV_FIELDS=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')
TRAIN_FIELDS=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame')
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path:Path)->str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.name}.{os.getpid()}.part');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def load_allowed_npz(path:Path,fields:tuple[str,...])->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:
        if 'target_native_h30' in z.files and path.name=='DEV_PREFIXES_NO_LABELS.npz':raise RuntimeError('dev prefix archive unexpectedly contains target key')
        missing=set(fields)-set(z.files)
        if missing:raise RuntimeError(f'{path.name}: missing fields {sorted(missing)}')
        return {k:z[k].copy() for k in fields}
def row_csv(path:Path,data:dict[str,np.ndarray])->None:
    fields=['row','anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(len(data['anchor_id'])):
            task=int(data['task_id'][i]);rank=TASK_IDS.index(task)
            if int(data['task_rank'][i])!=rank:raise RuntimeError(f'accepted task rank order mismatch at {data["anchor_id"][i]}')
            w.writerow({'row':i,'anchor_id':str(data['anchor_id'][i]),'episode_id':int(data['episode_id'][i]),'task_id':task,'task_rank':rank,'source_cluster':str(data['source_cluster'][i]),'anchor_frame':int(data['anchor_frame'][i])})
def map_rows(path:Path,map_id:str,dev:dict[str,np.ndarray],train:dict[str,np.ndarray])->tuple[np.ndarray,list[dict[str,str]]]:
    lookup={str(a):i for i,a in enumerate(train['anchor_id'])}
    expected={str(a):i for i,a in enumerate(dev['anchor_id'])}
    chosen={}
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['map_id']!=map_id:continue
            rid=str(r['receiver_anchor_id'])
            if rid not in expected or rid in chosen:raise RuntimeError(f'{map_id}: unknown/duplicate receiver {rid}')
            i=expected[rid];dr=int(r['donor_row'])
            if dr<0 or dr>=len(train['anchor_id']):raise RuntimeError(f'{map_id}: invalid train donor row')
            if str(train['anchor_id'][dr])!=r['donor_anchor_id'] or int(train['episode_id'][dr])!=int(r['donor_episode_id']):raise RuntimeError(f'{map_id}: donor row ID mismatch')
            if int(dev['task_id'][i])!=int(r['task_id']) or int(train['task_id'][dr])!=int(r['task_id']):raise RuntimeError(f'{map_id}: donor task mismatch')
            if int(train['episode_id'][dr])==int(dev['episode_id'][i]) or str(train['source_cluster'][dr])==str(dev['source_cluster'][i]):raise RuntimeError(f'{map_id}: donor source group overlaps receiver')
            chosen[rid]=r
    if set(chosen)!=set(expected):raise RuntimeError(f'{map_id}: missing {len(set(expected)-set(chosen))} receiver maps')
    donor_rows=np.asarray([int(chosen[str(a)]['donor_row']) for a in dev['anchor_id']],dtype=np.int64)
    return donor_rows,[chosen[str(a)] for a in dev['anchor_id']]
def prefix_batch(receiver:dict[str,np.ndarray],ids:np.ndarray,norm:dict[str,Any],device:torch.device,mode:str,grid:Q0ShardedGrid|None,feature_data:dict[str,np.ndarray]|None,donor_rows:np.ndarray|None)->tuple[dict[str,torch.Tensor],torch.Tensor]:
    state=(receiver['history_native'][ids]-np.asarray(norm['state_mean'],dtype=np.float32)[None,None])/np.asarray(norm['state_safe_std'],dtype=np.float32)[None,None]
    rel=(receiver['relative_delta'][ids]-np.float32(norm['relative_delta_mean']))/np.float32(norm['relative_delta_safe_std'])
    prefix={'state':torch.as_tensor(state,device=device,dtype=torch.float32),'history_mask':torch.as_tensor(receiver['history_mask'][ids],device=device,dtype=torch.float32),'relative_delta':torch.as_tensor(rel,device=device,dtype=torch.float32),'anchor_command':torch.as_tensor(receiver['anchor_native'][ids],device=device,dtype=torch.float32)}
    tasks=np.asarray([TASK_IDS.index(int(receiver['task_id'][i])) for i in ids],dtype=np.int64)
    task_id=torch.as_tensor(tasks,device=device,dtype=torch.long)
    if mode!='STATE-CTRL':
        if grid is None:raise RuntimeError('fusion inference lacks Q0 reader')
        src=receiver if donor_rows is None else feature_data
        src_idx=ids if donor_rows is None else donor_rows[ids]
        if src is None:raise RuntimeError('donor map selected without train feature records')
        raw=np.stack([grid.row(int(src['cache_index'][i]),str(src['anchor_id'][i]),str(src['episode_id'][i])) for i in src_idx]).astype(np.float32,copy=False)
        vm=np.asarray(norm['video_mean'],dtype=np.float32)[None,None,None,:];vs=np.asarray(norm['video_safe_std'],dtype=np.float32)[None,None,None,:]
        prefix['dynamic']=torch.as_tensor((raw[:,0]-vm)/vs,device=device,dtype=torch.float32)
        prefix['static']=torch.as_tensor((raw[:,1]-vm)/vs,device=device,dtype=torch.float32)
    return prefix,task_id
def run(job_path:Path)->int:
    job=json.loads(job_path.read_text());out=Path(job['output_root']);out.mkdir(parents=True,exist_ok=True)
    started=utc();status_path=out/'RUN_STATUS.json';complete_path=out/'PREDICTION_COMPLETE.json'
    if complete_path.exists():
        done=json.loads(complete_path.read_text())
        if done.get('status')=='PASS':print(json.dumps({'status':'already_complete','job_id':job['job_id']}));return 0
    try:
        if not torch.cuda.is_available():raise RuntimeError('V4 inference requires the watcher-authorized CUDA device')
        device=torch.device('cuda:0');torch.manual_seed(int(job['seed']));torch.cuda.manual_seed_all(int(job['seed']))
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        model_path=Path(job['model_path']);manifest_path=Path(job['model_manifest_path']);model_manifest=json.loads(manifest_path.read_text())
        if model_manifest.get('status')!='PASS':raise RuntimeError('model completion manifest is not PASS')
        model_sha=sha(model_path)
        expected=job.get('model_sha256_expected') or model_manifest.get('model_file_sha256')
        if model_sha!=expected:raise RuntimeError(f'model weight hash mismatch: {model_sha} != {expected}')
        if model_manifest.get('normalizer_sha256')!=sha(Path(job['normalizer_path'])):raise RuntimeError('model and inference normalizer hashes differ')
        model=V2TaskConditionedPredictor(seed=int(job['seed']),video_mode=str(job['video_mode']),waypoint_mode=None)
        model.load_state_dict(load_file(str(model_path),device='cpu'),strict=True)
        loaded_hash=state_hash(model)
        if loaded_hash!=model_manifest.get('final_state_hash'):raise RuntimeError('strictly reloaded model state hash differs from fit manifest')
        model=model.to(device).eval();torch.cuda.reset_peak_memory_stats(device)
        dev=load_allowed_npz(Path(job['data_path']),DEV_FIELDS)
        if len(dev['anchor_id'])!=int(job['expected_rows']) or len(dev['anchor_id'])!=185:raise RuntimeError('dev prefix row support mismatch')
        if len(set(dev['anchor_id'].astype(str)))!=185:raise RuntimeError('duplicate dev anchor IDs')
        train=load_allowed_npz(Path(job['train_data_path']),TRAIN_FIELDS)
        norm=json.loads(Path(job['normalizer_path']).read_text())
        resolved=json.loads(Path(job['resolved_paths_path']).read_text())
        grid=None;feature_data=None;donor_rows=None;map_records=[]
        condition=str(job['condition'])
        if str(job['video_mode'])!='STATE-CTRL':
            cache_manifest=json.loads(Path(resolved['feature_cache']).read_text());grid=Q0ShardedGrid(cache_manifest)
            if condition=='MATCHED':feature_data=None
            elif condition in {'D1','D2','D3'}:
                feature_data=train;donor_rows,map_records=map_rows(Path(job['eval_donor_maps_path']),condition,dev,train)
            else:raise RuntimeError(f'invalid V4 inference condition {condition}')
        elif condition!='MATCHED':raise RuntimeError('A_STATE_COMMON must be scored only on its masked matched view')
        rows=[]
        anchors=torch.as_tensor(dev['anchor_native'],device=device,dtype=torch.float32)
        sigma=torch.as_tensor(np.asarray(norm['target_sigma'],dtype=np.float32),device=device,dtype=torch.float32)
        with torch.no_grad():
            for start in range(0,len(dev['anchor_id']),int(job.get('batch_size',32))):
                ids=np.arange(start,min(start+int(job.get('batch_size',32)),len(dev['anchor_id'])),dtype=np.int64)
                prefix,task_id=prefix_batch(dev,ids,norm,device,str(job['video_mode']),grid,feature_data,donor_rows)
                model_output=model(prefix,task_id)
                rows.append((anchors[ids,None,:]+sigma[None,None,:]*model_output.action_residual.float()).detach().float().cpu())
        native=torch.cat(rows,dim=0).contiguous()
        if tuple(native.shape)!=(185,30,28) or not torch.isfinite(native).all():raise RuntimeError(f'prediction shape/finite failure {tuple(native.shape)}')
        residual=(native.to(device)-anchors[:,None,:])/sigma[None,None,:]
        prediction_path=outdir=out_path=out/'PREDICTIONS.safetensors'
        tmp=prediction_path.with_name(f'.{prediction_path.name}.{os.getpid()}.part');save_file({'action_residual':residual.detach().float().cpu().contiguous(),'prediction_native':native},str(tmp));os.replace(tmp,prediction_path)
        ids_path=out/'PREDICTION_ROWS.csv';row_csv(ids_path,dev)
        reload_tensors=load_file(str(prediction_path),device='cpu')
        if tuple(reload_tensors['prediction_native'].shape)!=(185,30,28) or not torch.isfinite(reload_tensors['prediction_native']).all():raise RuntimeError('prediction safe-tensor reload failed')
        donor_counts={}
        for r in map_records:donor_counts[r['donor_anchor_id']]=donor_counts.get(r['donor_anchor_id'],0)+1
        receipt={'schema_version':1,'status':'PASS','job_id':job['job_id'],'program_id':job['program_id'],'model_id':job['model_id'],'condition':condition,'seed':int(job['seed']),'task_ids':list(TASK_IDS),'model_path':str(model_path),'model_sha256':model_sha,'model_manifest_path':str(manifest_path),'model_manifest_sha256':sha(manifest_path),'model_state_hash':loaded_hash,'normalizer_path':str(job['normalizer_path']),'normalizer_sha256':sha(Path(job['normalizer_path'])),'dev_prefix_path':str(job['data_path']),'dev_prefix_sha256':sha(Path(job['data_path'])),'train_prefix_metadata_sha256':sha(Path(job['train_data_path'])),'feature_cache_manifest_sha256':resolved['feature_cache_manifest_sha256'],'donor_map_path':job.get('eval_donor_maps_path') if condition in {'D1','D2','D3'} else None,'donor_map_sha256':sha(Path(job['eval_donor_maps_path'])) if condition in {'D1','D2','D3'} else None,'prediction_path':str(prediction_path),'prediction_sha256':sha(prediction_path),'prediction_rows_path':str(ids_path),'prediction_rows_sha256':sha(ids_path),'row_count':len(dev['anchor_id']),'prediction_shape':list(native.shape),'all_finite':True,'fresh_model_state_reload':True,'prediction_safetensors_reload':True,'donor_unique_anchors':len(donor_counts),'donor_anchors_reused':sum(c>1 for c in donor_counts.values()),'donor_reuse_fraction':float(sum(c-1 for c in donor_counts.values())/len(map_records)) if map_records else 'NA','dev_future_target_arrays_requested':False,'final_heldout_payload_access':0,'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unknown'),'logical_device':str(device),'peak_memory_allocated_mib':float(torch.cuda.max_memory_allocated(device)/(1024*1024)),'peak_memory_reserved_mib':float(torch.cuda.max_memory_reserved(device)/(1024*1024)),'elapsed_seconds':(datetime.now(timezone.utc)-datetime.fromisoformat(started.replace('Z','+00:00'))).total_seconds()}
        atomic_json(complete_path,receipt);atomic_json(status_path,{'status':'complete','completed_utc':utc(),'job_id':job['job_id']})
        print(json.dumps({'status':'PASS','job_id':job['job_id'],'rows':185,'prediction_sha256':receipt['prediction_sha256']}));return 0
    except Exception as exc:
        atomic_json(status_path,{'status':'failed','failed_utc':utc(),'job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'physical_gpu':os.environ.get('GPU_WATCHER_PHYSICAL_GPU','unknown')})
        print(json.dumps({'status':'FAILED','job_id':job.get('job_id'),'error_type':type(exc).__name__,'error':str(exc)}),flush=True);return 1
def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--job-json',type=Path,required=True);return run(ap.parse_args().job_json)
if __name__=='__main__':raise SystemExit(main())
