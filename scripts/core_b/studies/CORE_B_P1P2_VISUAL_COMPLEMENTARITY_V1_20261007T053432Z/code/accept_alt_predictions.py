from __future__ import annotations
import os
import argparse,csv,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
DENSE=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d2/D2_DENSE_INPUTS.npz'
SEEDS=(20260950,20260951,20260952)

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def utc_tag():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def backup_then_write(path:Path,text:str):
 if path.exists():
  b=path.with_name(path.name+'.bak_'+utc_tag())
  shutil.copyfile(path,b)
 path.write_text(text,encoding='utf-8')
def read_manifest(path):
 if not path.exists():return []
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_manifest(path,rows):
 fields=['condition','seed','horizon_steps','bundle_path','bundle_sha256','sidecar_path','rows','prediction_shape','status','target_arrays_opened','final_test_access']
 merged={}
 for r in read_manifest(path):merged[(r['condition'],r['seed'],r['horizon_steps'])]=r
 for r in rows:merged[(r['condition'],r['seed'],r['horizon_steps'])]=r
 path.parent.mkdir(parents=True,exist_ok=True)
 text=''
 import io
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=fields);w.writeheader();w.writerows(merged[k] for k in sorted(merged,key=lambda x:(x[0],int(x[1]),int(x[2]))));text=buf.getvalue()
 if path.exists():
  b=path.with_name(path.name+'.bak_'+utc_tag());shutil.copyfile(path,b)
 path.write_text(text,encoding='utf-8')

def expected(kind):
 if kind=='primary':
  with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
   split=z['split_code'].copy();ids=z['anchor_id'].astype('U64').copy();eps=z['episode_index'].copy();frames=z['anchor_frame'].copy();times=z['anchor_timestamp'].copy()
  if set(np.unique(split).tolist())-{0,1}:raise RuntimeError('non train/dev split code in V3 input')
  rows=np.flatnonzero(split==1);return ids[rows],eps[rows],frames[rows],times[rows],30
 with np.load(DENSE,allow_pickle=False) as z:
  ids=z['dense_id'].astype('U96').copy();eps=z['episode_index'].copy();frames=z['anchor_frame'].copy();times=z['anchor_timestamp'].copy()
 return ids,eps,frames,times,30

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('primary','dense'),required=True);ap.add_argument('--seed',type=int,choices=SEEDS,required=True);args=ap.parse_args()
 condition='ALT-POOL-R3D18';h=30
 bundle=EXP/'predictions'/args.kind/f'ALT-POOL-R3D18-H30-S{args.seed}.npz';sidecar=bundle.with_suffix('.json')
 status='rejected';problems=[];shape=None;count=0
 try:
  if not bundle.is_file() or not sidecar.is_file():raise FileNotFoundError('prediction bundle/sidecar missing')
  receipt=json.loads(sidecar.read_text())
  with np.load(bundle,allow_pickle=False) as z:
   forbidden=[k for k in z.files if 'target' in k.lower() or 'future' in k.lower()]
   if forbidden:raise RuntimeError(f'target/future keys found in prediction bundle: {forbidden}')
   ids=z['source_id'].astype('U96');eps=z['episode_index'].copy();frames=z['anchor_frame'].copy();times=z['anchor_timestamp'].copy();groups=z['source_group_id'].astype('U64');pred=z['predictions_native'].copy()
  exp_ids,exp_eps,exp_frames,exp_times,h=expected(args.kind)
  if len(set(ids.tolist()))!=len(ids) or set(ids.tolist())!=set(exp_ids.tolist()):problems.append('exact ID set/uniqueness mismatch')
  lookup={str(x):i for i,x in enumerate(ids.tolist())}
  aligned=np.asarray([lookup.get(str(x),-1) for x in exp_ids],dtype=np.int64)
  if np.any(aligned<0):raise RuntimeError('prediction IDs missing expected input IDs')
  if pred.shape!=(len(ids),h,28) or not np.isfinite(pred).all():problems.append('prediction shape/finiteness mismatch')
  if not np.array_equal(eps[aligned],exp_eps) or not np.array_equal(frames[aligned],exp_frames) or np.max(np.abs(times[aligned]-exp_times))>1e-6:problems.append('episode/frame/timestamp mismatch')
  if receipt.get('target_arrays_opened') is not False or int(receipt.get('final_test_access',-1))!=0:problems.append('target-blind/final-test receipt failure')
  if receipt.get('seed')!=args.seed or receipt.get('kind')!=args.kind or receipt.get('horizon_steps')!=h:problems.append('receipt identity mismatch')
  if not problems:status='accepted'
  shape=list(pred.shape);count=len(ids)
 except Exception as exc:problems.append(f'{type(exc).__name__}: {exc}')
 row={'condition':condition,'seed':args.seed,'horizon_steps':h,'bundle_path':str(bundle.relative_to(EXP)),'bundle_sha256':sha256(bundle) if bundle.exists() else '','sidecar_path':str(sidecar.relative_to(EXP)),'rows':count,'prediction_shape':json.dumps(shape),'status':status,'target_arrays_opened':False,'final_test_access':0}
 manifest=EXP/'predictions'/('PRIMARY_MANIFEST.csv' if args.kind=='primary' else 'DENSE_MANIFEST.csv')
 write_manifest(manifest,[row])
 qa_path=EXP/'qa'/('PRIMARY_PREDICTION_ACCEPTANCE.json' if args.kind=='primary' else 'DENSE_PREDICTION_ACCEPTANCE.json')
 acceptance={'status':status,'condition':condition,'seed':args.seed,'kind':args.kind,'horizon_steps':h,'rows':count,'shape':shape,'bundle_sha256':row['bundle_sha256'],'problems':problems,'target_arrays_opened':False,'final_test_access':0}
 backup_then_write(qa_path,json.dumps(acceptance,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps(acceptance,sort_keys=True));return 0 if status=='accepted' else 2
if __name__=='__main__':raise SystemExit(main())
