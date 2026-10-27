from __future__ import annotations
import os
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
TASK_IDS=(119,94,90,141,230)
PREFIX_FIELDS=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')

def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha_file(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read_json(path:Path)->dict[str,Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic_json(path:Path,obj:dict[str,Any])->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.part')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)
def append_jsonl(path:Path,row:dict[str,Any])->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    line=(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n').encode('utf-8')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
    try: os.write(fd,line);os.fsync(fd)
    finally: os.close(fd)
def write_csv(path:Path,rows:list[dict[str,Any]],fields:list[str]|None=None)->None:
    import csv
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise RuntimeError('refusing empty CSV: '+str(path))
    if fields is None:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.part')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def load_npz(path:Path,fields:tuple[str,...]|None=None)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:
        if fields is not None:
            missing=set(fields)-set(z.files)
            if missing: raise RuntimeError('npz missing fields: '+str(sorted(missing)))
            return {k:z[k].copy() for k in fields}
        return {k:z[k].copy() for k in z.files}
def stable_drop_bit(seed:int,epoch0:int,receiver_key:list[Any])->bool:
    payload=json.dumps(['v9','numctx',int(seed),int(epoch0),receiver_key],ensure_ascii=False,separators=(',',':')).encode('utf-8')
    u64=int(hashlib.sha256(payload).hexdigest()[:16],16)
    return u64<2**63
def model_input(
    data:dict[str,np.ndarray],indices:np.ndarray,normalizer:dict[str,Any],video_mode:str,
    device:torch.device,grid:Any=None,drop_mask:np.ndarray|torch.Tensor|None=None,
)->tuple[dict[str,torch.Tensor],torch.Tensor,torch.Tensor]:
    state_mean=np.asarray(normalizer['state_mean'],dtype=np.float32)
    state_std=np.asarray(normalizer['state_safe_std'],dtype=np.float32)
    relative_mean=np.float32(normalizer['relative_delta_mean'])
    relative_std=np.float32(normalizer['relative_delta_safe_std'])
    state_np=(data['history_native'][indices]-state_mean[None,None])/state_std[None,None]
    state=torch.as_tensor(state_np,device=device,dtype=torch.float32)
    if drop_mask is not None:
        mask=torch.as_tensor(drop_mask,device=device,dtype=torch.bool)
        if tuple(mask.shape)!=(len(indices),): raise RuntimeError('numeric-context mask must be [B]')
        # Out-of-place whole-block masking after normalization, before state_projection.
        state=torch.where(mask[:,None,None],torch.zeros_like(state),state)
    relative=(data['relative_delta'][indices]-relative_mean)/relative_std
    prefix={
        'state':state,
        'history_mask':torch.as_tensor(data['history_mask'][indices],device=device,dtype=torch.float32),
        'relative_delta':torch.as_tensor(relative,device=device,dtype=torch.float32),
        'anchor_command':torch.as_tensor(data['anchor_native'][indices],device=device,dtype=torch.float32),
    }
    task=torch.as_tensor([TASK_IDS.index(int(data['task_id'][i])) for i in indices],device=device,dtype=torch.long)
    anchor=torch.as_tensor(data['anchor_native'][indices],device=device,dtype=torch.float32)
    if video_mode!='STATE-CTRL':
        if grid is None: raise RuntimeError('video model requires the accepted feature cache reader')
        features=np.empty((len(indices),2,16,4,768),dtype=np.float32)
        for pos,i in enumerate(indices.tolist()):
            features[pos]=grid.row(int(data['cache_index'][i]),str(data['anchor_id'][i]),str(data['episode_id'][i]))
        vm=np.asarray(normalizer['video_mean'],dtype=np.float32)[None,None,None,:]
        vs=np.asarray(normalizer['video_safe_std'],dtype=np.float32)[None,None,None,:]
        prefix['dynamic']=torch.as_tensor((features[:,0]-vm)/vs,device=device,dtype=torch.float32)
        prefix['static']=torch.as_tensor((features[:,1]-vm)/vs,device=device,dtype=torch.float32)
    return prefix,task,anchor
def prefix_row_csv(path:Path,data:dict[str,np.ndarray],dataset_role:str)->None:
    import csv
    fields=['row','anchor_id','episode_id','task_id','task_rank','source_cluster','anchor_frame','dataset_role']
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(len(data['anchor_id'])):
            task=int(data['task_id'][i]);rank=TASK_IDS.index(task)
            if int(data['task_rank'][i])!=rank: raise RuntimeError('task rank mismatch at '+str(data['anchor_id'][i]))
            w.writerow({'row':i,'anchor_id':str(data['anchor_id'][i]),'episode_id':int(data['episode_id'][i]),'task_id':task,'task_rank':rank,
                'source_cluster':str(data['source_cluster'][i]),'anchor_frame':int(data['anchor_frame'][i]),'dataset_role':dataset_role})
def metric_components(pred_residual:torch.Tensor,target_native:torch.Tensor,anchor_native:torch.Tensor,
                      target_sigma:torch.Tensor,safe_fd_scale:torch.Tensor,
                      metric_sigma:torch.Tensor|None=None,metric_safe_fd_scale:torch.Tensor|None=None):
    pred_native=anchor_native[:,None]+target_sigma[None,None]*pred_residual
    eval_sigma=target_sigma if metric_sigma is None else metric_sigma
    eval_fd_scale=safe_fd_scale if metric_safe_fd_scale is None else metric_safe_fd_scale
    pos_err=torch.abs(pred_native-target_native)/eval_sigma[None,None]
    position=0.5*(pos_err[...,:14].mean(dim=(1,2))+pos_err[...,14:].mean(dim=(1,2)))
    target_fd=torch.diff(target_native,dim=1)/eval_fd_scale[None,None]
    pred_fd=torch.diff(pred_native,dim=1)/eval_fd_scale[None,None]
    fd_err=torch.abs(pred_fd-target_fd)
    safe_fd=0.5*(fd_err[...,:14].mean(dim=(1,2))+fd_err[...,14:].mean(dim=(1,2)))
    seam_err=torch.abs((pred_native[:,0]-anchor_native)/eval_fd_scale-(target_native[:,0]-anchor_native)/eval_fd_scale)
    seam=0.5*(seam_err[:,:14].mean(dim=1)+seam_err[:,14:].mean(dim=1))
    objective=position+0.1*safe_fd
    return position,safe_fd,seam,objective
def rng_state_equal_numpy(a,b)->bool:
    if a[0]!=b[0] or not np.array_equal(a[1],b[1]): return False
    return a[2:]==b[2:]

