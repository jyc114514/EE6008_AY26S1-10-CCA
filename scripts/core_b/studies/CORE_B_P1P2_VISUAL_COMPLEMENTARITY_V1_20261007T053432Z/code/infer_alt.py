from __future__ import annotations
import os
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from model import parameter_count
from experiment_contracts import make_matched_alt_predictor,validate_physical_gpu

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/"repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
DENSE_INPUT=PROJECT/"reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d2/D2_DENSE_INPUTS.npz"
HORIZON=30;VIDEO_DIM=512;VIDEO_STEPS=16

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

class InferenceDataset(Dataset):
 def __init__(self,data,features,rows,mean,std):self.data=data;self.features=features;self.rows=np.asarray(rows,dtype=np.int64);self.mean=mean;self.std=std
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=int(self.rows[i]);video=(np.asarray(self.features[i],dtype=np.float32)-self.mean)/self.std
  video=np.broadcast_to(video[None,:],(VIDEO_STEPS,VIDEO_DIM)).copy()
  return (torch.from_numpy(self.data['history_standardized'][r]),torch.from_numpy(self.data['history_mask'][r]),torch.from_numpy(self.data['history_relative_delta_standardized'][r]),torch.from_numpy(video),torch.from_numpy(self.data['anchor_command_native'][r]))

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--kind',choices=('primary','dense'),required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=('cuda','cpu'),default='cuda');args=ap.parse_args()
 physical=validate_physical_gpu(os.environ.get('GPU_WATCHER_PHYSICAL_GPU')) if args.device=='cuda' else None
 if not args.output.resolve().is_relative_to(EXP.resolve()):raise RuntimeError('prediction output must remain under EXP_ROOT')
 if args.output.exists():raise FileExistsError(args.output)
 if args.device=='cuda':
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable after approved host launch')
  torch.cuda.set_device(0);device=torch.device('cuda:0')
 else:device=torch.device('cpu')
 # Target arrays are intentionally not opened by inference.
 if args.kind=='primary':
  with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in ('anchor_id','episode_index','split_code','anchor_frame','anchor_timestamp','history_standardized','history_mask','history_relative_delta_standardized','anchor_command_native')}
  if set(np.unique(data['split_code']).tolist())-{0,1}:raise RuntimeError('primary input includes non-train/dev split')
  rows=np.flatnonzero(data['split_code']==1)
  indices=np.load(EXP/'cache/R3D_PRIMARY_INDEX.npy',allow_pickle=False)[rows]
  source_ids=data['anchor_id'][rows]
  frames=data['anchor_frame'][rows];times=data['anchor_timestamp'][rows]
 else:
  with np.load(DENSE_INPUT,allow_pickle=False) as z:data={k:z[k].copy() for k in ('dense_id','episode_index','anchor_frame','anchor_timestamp','history_standardized','history_mask','history_relative_delta_standardized','anchor_command_native')}
  rows=np.arange(len(data['dense_id']),dtype=np.int64);indices=np.load(EXP/'cache/R3D_DENSE_INDEX.npy',allow_pickle=False)
  source_ids=data['dense_id'];frames=data['anchor_frame'];times=data['anchor_timestamp']
 union=np.load(EXP/'cache/R3D_FEATURES_UNION.npy',mmap_mode='r');features=np.asarray(union[indices],dtype=np.float32)
 if not np.isfinite(features).all():raise RuntimeError('non-finite feature in inference input')
 norm=json.loads((EXP/'cache/R3D_NORMALIZER.json').read_text());target=json.loads((V3/'data/V3_NORMALIZERS.json').read_text())
 mean=np.asarray(norm['mean'],dtype=np.float32);std=np.asarray(norm['safe_std'],dtype=np.float32);sigma=np.asarray(target['target_sigma'],dtype=np.float32)
 if len(mean)!=VIDEO_DIM or len(std)!=VIDEO_DIM or len(sigma)!=28:raise RuntimeError('normalizer dimension mismatch')
 sigma_tensor=torch.as_tensor(sigma,dtype=torch.float32,device=device).reshape(1,1,28)
 groups={}
 with (EXP/'inventory/SOURCE_GROUP_MAP.csv').open(newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):groups[int(row['episode_index'])]=row['source_group_id']
 ds=InferenceDataset(data,features,rows,mean,std);loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0)
 model,init_report=make_matched_alt_predictor(HORIZON,args.seed)
 checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
 model.load_state_dict(checkpoint['model_state_dict'],strict=True);model=model.to(device).eval()
 predictions=[]
 with torch.inference_mode():
  for batch in loader:
   state,mask,rel,video,anchor=[v.to(device) for v in batch]
   video=video.mean(dim=1,keepdim=True).expand(-1,VIDEO_STEPS,-1)
   residual=model(state,mask,rel,video,anchor,use_state=True,use_video=True)
   predictions.append((anchor.unsqueeze(1)+residual*sigma_tensor).float().cpu().numpy())
 pred=np.concatenate(predictions,axis=0)
 if pred.shape!=(len(rows),HORIZON,28) or not np.isfinite(pred).all():raise RuntimeError(f'prediction schema failure {pred.shape}')
 episode=data['episode_index'][rows]
 group=np.asarray([groups[int(e)] for e in episode],dtype='U64')
 args.output.parent.mkdir(parents=True,exist_ok=True)
 payload={'source_id':np.asarray(source_ids),'episode_index':episode,'source_group_id':group,'anchor_frame':np.asarray(frames),'anchor_timestamp':np.asarray(times),'predictions_native':pred}
 if args.kind=='primary':payload['anchor_id']=np.asarray(source_ids)
 else:payload['dense_id']=np.asarray(source_ids)
 np.savez_compressed(args.output,**payload)
 receipt={'status':'PREDICTIONS_WRITTEN_UNSCORED','kind':args.kind,'seed':args.seed,'horizon_steps':HORIZON,'rows':len(rows),'prediction_shape':list(pred.shape),'prediction_sha256':sha256(args.output),'checkpoint_path':str(args.checkpoint.resolve()),'checkpoint_sha256':sha256(args.checkpoint),'feature_cache_sha256':sha256(EXP/'cache/R3D_FEATURES_UNION.npy'),'normalizer_sha256':sha256(EXP/'cache/R3D_NORMALIZER.json'),'prediction_code_sha256':sha256(Path(__file__)),'predictor_parameters':parameter_count(model),'physical_gpu':physical,'logical_device':'cuda:0' if args.device=='cuda' else 'cpu','target_arrays_opened':False,'final_test_access':0}
 sidecar=args.output.with_suffix('.json');sidecar.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
