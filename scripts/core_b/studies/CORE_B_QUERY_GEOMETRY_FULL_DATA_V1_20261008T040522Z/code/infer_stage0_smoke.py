#!/usr/bin/env python3
"""Measure fresh-process H30 query-wrapper inference on 128 train-only monitor rows."""
from __future__ import annotations
import os
import argparse,json,os,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from d3_model import D3TimeSpacePredictor
from fixed_future_query_model import FixedFutureQueryOverride
from train_full_query import runtime_metadata
from infer_full_query import read_source_groups,query_table_and_hash,load_split_data,GridStore,predict_rows,sha_file

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--batch-size',type=int,default=64);args=ap.parse_args()
 physical=int(os.environ.get('GPU_WATCHER_PHYSICAL_GPU','-1'));visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if physical<0 or visible!=str(physical) or not torch.cuda.is_available():raise RuntimeError('Stage0 inference smoke requires a watcher-assigned visible CUDA device')
 torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(0)
 run_root=ROOT/'runs/stage0'/args.run_id
 stage= json.loads((run_root/'STAGE0_REPORT.json').read_text())
 cp=run_root/'checkpoint_stage0.pt';cp_sha=sha_file(cp)
 if stage.get('status')!='PASS' or stage.get('optimizer_steps')!=4 or stage.get('query_table_sha256') is None:raise RuntimeError('Stage0 training checkpoint not accepted')
 query,qsha=query_table_and_hash()
 payload=torch.load(cp,map_location='cpu',weights_only=False)
 base=D3TimeSpacePredictor(horizon=30,seed=int(stage['seed']),video_mode=stage['config'])
 model=FixedFutureQueryOverride(base,torch.from_numpy(query));model.load_state_dict(payload['model'],strict=True);model.to(device).eval()
 if hashlib.sha256(model.future_query_table.detach().cpu().numpy().dtype.str.encode()+b'\0'+np.asarray(model.future_query_table.shape,dtype=np.int64).tobytes()+model.future_query_table.detach().cpu().numpy().tobytes()).hexdigest()!=qsha:raise RuntimeError('fresh process did not retain H30 query buffer')
 panel=list(__import__('csv').DictReader((ROOT/'protocol/TRAIN_MONITOR_PANEL.csv').open(newline='',encoding='utf-8')))
 ids=np.asarray([x['anchor_id'] for x in panel],dtype='U96')
 group_map=read_source_groups();archive=np.load(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz'),allow_pickle=False)
 data=load_split_data('train',group_map,archive);byid={v:i for i,v in enumerate(data['anchor_id'].astype('U96').tolist())};rows=np.asarray([byid[x] for x in ids],dtype=np.int64)
 cm=json.loads(Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json')).read_text())
 vn=json.loads(Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_VIDEO_NORMALIZER.json')).read_text())
 norm=json.loads(Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_NORMALIZERS.json')).read_text())
 grid=GridStore(cm);sigma=torch.as_tensor(np.asarray(norm['target_sigma'],dtype=np.float32),device=device).view(1,1,28)
 torch.cuda.reset_peak_memory_stats(device)
 pred=predict_rows(model,data,rows,grid,np.asarray(vn['mean'],dtype=np.float32),np.asarray(vn['safe_std'],dtype=np.float32),sigma,device,args.batch_size)
 if pred.shape!=(128,30,28) or not np.isfinite(pred).all():raise RuntimeError('inference smoke output failed shape/finite')
 peak_alloc=int(torch.cuda.max_memory_allocated(device));peak_reserved=int(torch.cuda.max_memory_reserved(device))
 result={'schema_version':1,'status':'PASS','run_id':args.run_id,'stage0_checkpoint_sha256':cp_sha,'query_table_sha256':qsha,
  'train_monitor_rows':128,'prediction_shape':list(pred.shape),'prediction_sha256':hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest(),
  'fresh_process_reload':True,'target_arrays_loaded':False,'dev_targets_opened':False,'optimizer_created':False,'backward_called':False,
  'device':str(device),'physical_gpu':physical,'logical_device':'cuda:0','cuda_visible_devices':visible,'runtime':runtime_metadata(device),
  'peak_allocated_bytes':peak_alloc,'peak_reserved_bytes':peak_reserved,'measured_peak_mib':int(np.ceil(peak_reserved/1024**2)),
  'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'final_test_access':0}
 out=run_root/'INFER_CANARY.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
