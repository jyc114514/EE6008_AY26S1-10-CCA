from __future__ import annotations
import os
import argparse,csv,hashlib,json,math,os,random
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader,Dataset
from model import V3Full28Predictor,parameter_count,state_hash
from experiment_contracts import make_matched_alt_predictor,validate_physical_gpu

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/"repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
DATA_PATH=V3/"data/V3_MAIN_INPUTS.npz"
FEATURE_PATH=EXP/"cache/R3D_FEATURES_UNION.npy"
PRIMARY_INDEX_PATH=EXP/"cache/R3D_PRIMARY_INDEX.npy"
NORMALIZER_PATH=EXP/"cache/R3D_NORMALIZER.json"
EPOCHS=40;BATCH_SIZE=64;ACTION_DIM=28;HALF=14;VIDEO_DIM=512;VIDEO_STEPS=16

def digest(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def atomic_json(path:Path,value:object)->None:
 part=path.with_name(f'.{path.name}.{os.getpid()}.part')
 part.write_text(json.dumps(value,indent=2,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x)))+'\n',encoding='utf-8')
 os.replace(part,path)

def seed_all(seed:int,use_cuda:bool)->None:
 random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
 if use_cuda:
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

class AltDataset(Dataset):
 def __init__(self,data:dict[str,np.ndarray],features:np.ndarray,rows:np.ndarray,mean:Tensor,std:Tensor,horizon:int):
  self.data=data;self.features=features;self.rows=rows.astype(np.int64);self.mean=mean;self.std=std;self.horizon=horizon
 def __len__(self):return len(self.rows)
 def __getitem__(self,i:int):
  row=int(self.rows[i]);feature=np.asarray(self.features[row],dtype=np.float32)
  video=(feature-self.mean.numpy())/self.std.numpy()
  video=np.broadcast_to(video[None,:],(VIDEO_STEPS,VIDEO_DIM)).copy()
  return (torch.from_numpy(self.data['history_standardized'][row]),torch.from_numpy(self.data['history_mask'][row]),torch.from_numpy(self.data['history_relative_delta_standardized'][row]),torch.from_numpy(video),torch.from_numpy(self.data['anchor_command_native'][row]),torch.from_numpy(self.data['targets_native'][row,:self.horizon]),torch.tensor(int(self.data['episode_index'][row]),dtype=torch.long))

def load_training_data():
 allowed=('history_standardized','history_mask','history_relative_delta_standardized','anchor_command_native','targets_native','episode_index','split_code','anchor_id')
 with np.load(DATA_PATH,allow_pickle=False) as z:data={k:z[k].copy() for k in allowed}
 if set(np.unique(data['split_code']).tolist())-{0,1}:raise RuntimeError('input contains a split outside train/dev')
 idx=np.load(PRIMARY_INDEX_PATH,allow_pickle=False)
 union=np.load(FEATURE_PATH,mmap_mode='r')
 features=np.asarray(union[idx],dtype=np.float32)
 if features.shape!=(len(data['split_code']),VIDEO_DIM) or not np.isfinite(features).all():raise RuntimeError('primary R3D cache shape/finite check failed')
 norm=json.loads(NORMALIZER_PATH.read_text())
 target=json.loads((V3/'data/V3_NORMALIZERS.json').read_text())
 stats={'target_sigma':torch.tensor(target['target_sigma'],dtype=torch.float32),'safe_fd':torch.tensor(target['safe_fd_scale'],dtype=torch.float32),'video_mean':torch.tensor(norm['mean'],dtype=torch.float32),'video_std':torch.tensor(norm['safe_std'],dtype=torch.float32)}
 return data,features,stats

def legacy_weighted_batch_objective(per_sample:Tensor,weights:Tensor)->Tensor:
 if per_sample.ndim!=1 or weights.shape!=per_sample.shape:raise ValueError('per-sample loss/weight shape mismatch')
 denominator=weights.sum()
 if not torch.isfinite(denominator) or denominator<=0:raise ValueError('batch weight sum must be finite and positive')
 return (per_sample*weights).sum()/denominator

def batch_loss(model:V3Full28Predictor,batch,stats:dict[str,Tensor],horizon:int,device:torch.device):
 state,mask,rel,video,anchor,target,episode=[v.to(device) for v in batch]
 if video.shape[1:]!=(VIDEO_STEPS,VIDEO_DIM):raise ValueError(f'R3D video tensor shape {tuple(video.shape)}')
 video=video.mean(dim=1,keepdim=True).expand(-1,VIDEO_STEPS,-1)
 residual=model(state,mask,rel,video,anchor,use_state=True,use_video=True)
 sigma=stats['target_sigma'].to(device).view(1,1,ACTION_DIM);fdscale=stats['safe_fd'].to(device).view(1,1,ACTION_DIM)
 pred=anchor.unsqueeze(1)+residual*sigma
 pos=0.5*(torch.abs(pred[:,:,:HALF]-target[:,:,:HALF])/sigma[:,:,:HALF]).mean((1,2))+0.5*(torch.abs(pred[:,:,HALF:]-target[:,:,HALF:])/sigma[:,:,HALF:]).mean((1,2))
 fd_err=torch.abs(torch.diff(pred,dim=1)-torch.diff(target,dim=1))/fdscale
 fd=0.5*fd_err[:,:,:HALF].mean((1,2))+0.5*fd_err[:,:,HALF:].mean((1,2))
 return pos+0.1*fd,pos,fd,pred,episode

def atomic_torch(path:Path,value:object)->None:
 part=path.with_name(f'.{path.name}.{os.getpid()}.part')
 with part.open('wb') as f:torch.save(value,f)
 os.replace(part,path)

def main()->int:
 ap=argparse.ArgumentParser()
 ap.add_argument('--seed',type=int,required=True);ap.add_argument('--horizon',type=int,choices=(30,60),default=30);ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--stage0',action='store_true');ap.add_argument('--max-optimizer-steps',type=int);ap.add_argument('--device',choices=('cuda','cpu'),default='cuda')
 args=ap.parse_args();physical=validate_physical_gpu(os.environ.get('GPU_WATCHER_PHYSICAL_GPU')) if args.device=='cuda' else None
 run_root=args.run_root.resolve()
 if not run_root.is_relative_to(EXP.resolve()):raise RuntimeError('run output must stay inside EXP_ROOT')
 if run_root.exists() and any(run_root.iterdir()):raise FileExistsError(run_root)
 if args.stage0 and args.max_optimizer_steps!=4:raise ValueError('predictor Stage0 requires exactly 4 optimizer steps')
 if not args.stage0 and args.max_optimizer_steps is not None:raise ValueError('max_optimizer_steps is reserved for Stage0')
 if args.device=='cuda':
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable after approved host launch')
  torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
 else:device=torch.device('cpu')
 run_root.mkdir(parents=True,exist_ok=False)
 atomic_json(run_root/'RUN_STATUS.json',{'status':'running','optimizer_steps':0,'final_test_access':0})
 try:
  seed_all(args.seed,args.device=='cuda')
  data,features,stats=load_training_data();all_train=np.flatnonzero(data['split_code']==0)
  train_rows=all_train[:BATCH_SIZE*args.max_optimizer_steps] if args.stage0 else all_train
  if args.stage0 and len(train_rows)<BATCH_SIZE*args.max_optimizer_steps:raise RuntimeError('too few train rows for four complete Stage0 steps')
  episode_ids=data['episode_index'][train_rows];counts={int(e):int(np.sum(episode_ids==e)) for e in np.unique(episode_ids)}
  episode_count=len(counts);total_windows=len(train_rows)
  model,init_report=make_matched_alt_predictor(args.horizon,args.seed)
  model=model.to(device);initial_hash=state_hash(model)
  optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4)
  steps_per_epoch=math.ceil(total_windows/BATCH_SIZE);expected=args.max_optimizer_steps if args.stage0 else EPOCHS*steps_per_epoch
  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(expected,1),eta_min=1e-5)
  cfg={'status':'LOCKED','protocol_id':'CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1','condition':'ALT-POOL-R3D18','horizon_steps':args.horizon,'seed':args.seed,'epochs':0 if args.stage0 else EPOCHS,'batch_size':BATCH_SIZE,'effective_batch':BATCH_SIZE,'steps_per_epoch':steps_per_epoch,'expected_optimizer_steps':expected,'train_windows':total_windows,'train_episodes':episode_count,'parameter_count':parameter_count(model),'initial_model_hash':initial_hash,'shared_initialization':init_report,'input_sha256':digest(DATA_PATH),'primary_feature_index_sha256':digest(PRIMARY_INDEX_PATH),'feature_cache_sha256':digest(FEATURE_PATH),'normalizer_sha256':digest(NORMALIZER_PATH),'model_code_sha256':digest(EXP/'code/model.py'),'train_code_sha256':digest(Path(__file__)),'physical_gpu':physical,'logical_device':'cuda:0' if args.device=='cuda' else 'cpu','visual_encoder_updated':False,'final_test_access':0}
  atomic_json(run_root/'RUN_CONFIG_LOCKED.json',cfg)
  dataset=AltDataset(data,features,train_rows,stats['video_mean'],stats['video_std'],args.horizon)
  fixture=next(iter(DataLoader(dataset,batch_size=min(BATCH_SIZE,8),shuffle=False,num_workers=0)))
  model.eval()
  with torch.inference_mode():_,_,_,fixture_pred,_=batch_loss(model,fixture,stats,args.horizon,device)
  persist=fixture[4].to(device).unsqueeze(1)
  initial_persistence=float(torch.max(torch.abs(fixture_pred-persist)).detach().cpu())
  if initial_persistence>1e-6:raise RuntimeError(f'zero output does not equal Persistence: {initial_persistence}')
  steps=0;curves=[];target_steps=expected
  try:
   for epoch in range(1,(EPOCHS+1 if not args.stage0 else 100000)):
    if steps>=target_steps:break
    model.train();gen=torch.Generator().manual_seed(args.seed+epoch)
    loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,generator=gen,num_workers=0)
    optimizer.zero_grad(set_to_none=True);sum_loss=sum_pos=sum_fd=sum_weight=0.0
    for batch in loader:
     if steps>=target_steps:break
     total,pos,fd,_pred,episode=batch_loss(model,batch,stats,args.horizon,device)
     ep=episode.detach().cpu().numpy();per_sample=np.asarray([total_windows/(episode_count*counts[int(e)]) for e in ep],dtype=np.float32)
     weight=torch.from_numpy(per_sample).to(device)
     # This intentionally matches the accepted historical V3 batch-local denominator.
     loss=legacy_weighted_batch_objective(total,weight);loss.backward()
     sum_loss+=float((total.detach()*weight).sum().cpu());sum_pos+=float((pos.detach()*weight).sum().cpu());sum_fd+=float((fd.detach()*weight).sum().cpu());sum_weight+=float(weight.sum().detach().cpu())
     torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step();steps+=1
    curves.append({'epoch':epoch,'optimizer_steps':steps,'loss_weighted':sum_loss/max(sum_weight,1e-12),'position_weighted':sum_pos/max(sum_weight,1e-12),'fd_weighted':sum_fd/max(sum_weight,1e-12),'learning_rate':scheduler.get_last_lr()[0]})
    atomic_json(run_root/'RUN_STATUS.json',{'status':'running','last_completed_epoch':epoch,'optimizer_steps':steps,'final_test_access':0})
   if steps!=expected:raise RuntimeError(f'optimizer update count mismatch: {steps} != {expected}')
   name='checkpoint_stage0.pt' if args.stage0 else 'checkpoint_last.pt';model.eval();atomic_torch(run_root/name,{'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'config':cfg,'epoch':epoch,'optimizer_steps':steps,'seed':args.seed,'rng_state_torch':torch.get_rng_state(),'rng_state_numpy':np.random.get_state(),'rng_state_python':random.getstate()})
   with torch.inference_mode():_,_,_,pred,_=batch_loss(model,fixture,stats,args.horizon,device)
   weights_only_path=run_root/'checkpoint_model_weights.pt'
   atomic_torch(weights_only_path,{'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'config':cfg,'seed':args.seed,'final_test_access':0})
   reload_model,_=make_matched_alt_predictor(args.horizon,args.seed);payload=torch.load(weights_only_path,map_location='cpu',weights_only=True);reload_model.load_state_dict(payload['model_state_dict'],strict=True);reload_model=reload_model.to(device).eval()
   with torch.inference_mode():_,_,_,reloaded,_=batch_loss(reload_model,fixture,stats,args.horizon,device)
   reload_diff=float(torch.max(torch.abs(pred-reloaded)).detach().cpu())
   if reload_diff>1e-6:raise RuntimeError(f'checkpoint reload mismatch: {reload_diff}')
   peak_alloc=int(torch.cuda.max_memory_allocated()//(1024*1024)) if args.device=='cuda' else 0
   peak_reserved=int(torch.cuda.max_memory_reserved()//(1024*1024)) if args.device=='cuda' else 0
   atomic_json(run_root/'EPOCH_CURVES.json',{'status':'PASS','rows':curves,'final_test_access':0})
   complete={'status':'PASS','condition':'ALT-POOL-R3D18','horizon_steps':args.horizon,'seed':args.seed,'epoch':epoch,'optimizer_steps':steps,'checkpoint':name,'inference_checkpoint':str(weights_only_path),'inference_checkpoint_sha256':digest(weights_only_path),'parameter_count':parameter_count(reload_model),'initial_persistence_max_abs':initial_persistence,'checkpoint_reload_prediction_max_abs':reload_diff,'peak_memory_allocated_mib':peak_alloc,'peak_memory_reserved_mib':peak_reserved,'train_windows':total_windows,'physical_gpu':physical,'final_test_access':0}
   atomic_json(run_root/'RUN_COMPLETE.json',complete);atomic_json(run_root/'RUN_STATUS.json',{'status':'complete','optimizer_steps':steps,'checkpoint':name,'final_test_access':0})
   print(json.dumps(complete,sort_keys=True));return 0
  except Exception:raise
 except Exception as exc:
  atomic_json(run_root/'RUN_STATUS.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}','final_test_access':0});raise

if __name__=='__main__':raise SystemExit(main())
