#!/usr/bin/env python3
from __future__ import annotations
import os
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
ROOT=Path(__file__).resolve().parents[1]
TASKS=[119,94,90,141,230];SEEDS=[202610131,202610132,202610133]
V4=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def main():
 seal=json.loads((ROOT/'qa/PREDICTION_SEAL.json').read_text());formula=json.loads((ROOT/'protocol/FORMULA_LOCK.json').read_text());fh=sha(ROOT/'protocol/FORMULA_LOCK.json')
 if seal.get('status')!='PASS' or len(seal.get('entries',[]))!=18 or seal.get('formula_lock_sha256')!=fh:raise RuntimeError('prediction seal gate failed')
 if json.loads((ROOT/'protocol/CONTRAST_LOCK.json').read_text()).get('formula_lock_sha256')!=fh:raise RuntimeError('contrast lock mismatch')
 ids_rows=read(ROOT/'protocol/ORDERED_RECEIVER_IDS.csv');ids=[x['anchor_id'] for x in sorted(ids_rows,key=lambda x:int(x['canonical_row_index']))]
 idhash=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
 if len(ids)!=185 or idhash!=formula['ordered_receiver_ids_sha256']:raise RuntimeError('ordered ID lock mismatch')
 sm=json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text())
 prefix_rec=next(x for x in sm['source_artifacts'] if Path(x['path']).name=='DEV_PREFIXES_NO_LABELS.npz')
 if sha(prefix_rec['path'])!=prefix_rec['sha256']:raise RuntimeError('prefix hash mismatch')
 with np.load(prefix_rec['path'],allow_pickle=False) as z:
  pids=[str(x) for x in z['anchor_id'].tolist()];meta={str(z['anchor_id'][i]):(str(z['episode_id'][i]),str(z['task_id'][i]),str(z['source_cluster'][i])) for i in range(len(pids))}
 if set(pids)!=set(ids):raise RuntimeError('prefix membership mismatch')
 # Every sealed derived array, sidecar, and completion is checked with no target archive path opened.
 for e in seal['entries']:
  for k,hk in [('prediction_path','prediction_sha256'),('prediction_rows_path','prediction_rows_sha256'),('completion_path','completion_sha256')]:
   if sha(e[k])!=e[hk]:raise RuntimeError(f'sealed file hash mismatch {e[k]}')
  a=np.asarray(load_file(e['prediction_path'])['prediction_native']);rr=read(e['prediction_rows_path'])
  if a.shape!=(185,30,28) or a.dtype!=np.float64 or not np.isfinite(a).all() or [r['anchor_id'] for r in rr]!=ids:raise RuntimeError(f'derived array schema invalid {e["bundle_id"]}')
 # Core source arrays are validated by hash and explicit metadata join.
 for s in sm['source_prediction_bundles']:
  if s['role'] not in {'S_OPT','V_MATCHED','A_COMMON','C_REASSIGNED'}:continue
  if sha(s['prediction_path'])!=s['prediction_sha256'] or sha(s['prediction_rows_path'])!=s['prediction_rows_sha256']:raise RuntimeError('core source hash mismatch')
  a=np.asarray(load_file(s['prediction_path'])['prediction_native']);rr=read(s['prediction_rows_path'])
  if a.shape!=(185,30,28) or not np.isfinite(a).all() or {r['anchor_id'] for r in rr}!=set(ids):raise RuntimeError('core source schema/IDs invalid')
  for r in rr:
   if (r['episode_id'],r['task_id'],r['source_cluster'])!=meta[r['anchor_id']]:raise RuntimeError('core source metadata join invalid')
 # Filter only the accepted V4 matched/reference methods. Donor-map conditions are excluded.
 mapping={'BASE_STATE':'S','B_M':'V','A_M':'A','C_M':'C','BIAS_STATE':'BIAS_STATE','BIAS_VIDEO':'BIAS_VIDEO','PERSISTENCE':'PERSISTENCE'}
 counts={k:0 for k in mapping}
 for r in read(V4/'results/PER_WINDOW_LOSSES.csv'):
  if r['method'] not in mapping:continue
  counts[r['method']]+=1
  if r['anchor_id'] not in meta or (str(r['episode_id']),str(r['task_id']),str(r['source_cluster']))!=meta[r['anchor_id']]:raise RuntimeError('accepted V4 source reference ID/meta mismatch')
 if any(v!=555 for v in counts.values()):raise RuntimeError(f'accepted V4 reference support invalid {counts}')
 primary=read(V4/'results/PRIMARY_TASK_SEED.csv')
 if len(primary)!=15 or {(int(x['task_id']),int(x['seed'])) for x in primary}!={(t,s) for t in TASKS for s in SEEDS}:raise RuntimeError('V4 primary task-seed source table invalid')
 # Metric lock and synthetic scorer path check; the source development archive is not opened here.
 metric=json.loads(Path(sm['metric_scale_lock']['path']).read_text())
 if len(metric['target_sigma'])!=28 or len(metric['safe_fd_scale'])!=28 or any(float(v)<=0 for v in metric['target_sigma']+metric['safe_fd_scale']):raise RuntimeError('train-only scorer scales invalid')
 sys.path.insert(0,str(ROOT/'code'))
 from score_v5 import target_score
 rng=np.random.default_rng(20261015);p=rng.normal(size=(2,30,28));y=rng.normal(size=(2,30,28));a=rng.normal(size=(2,28))
 m=target_score(p,y,a,np.ones(28),np.ones(28))
 if set(m)!={'position_mae','safe_fd_l1','seam_safe_fd_l1'} or any(v.shape!=(2,) or not np.isfinite(v).all() for v in m.values()):raise RuntimeError('synthetic accepted scorer path invalid')
 # Confirm source path exists by metadata only; this does not read the archive payload.
 target_path=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/DEV_TARGETS_SCORER_ONLY.npz'))
 if not target_path.is_file():raise RuntimeError('scorer-only target path absent')
 out={'schema_version':1,'status':'PASS','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),
  'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),'formula_lock_sha256':fh,'derived_bundles_revalidated':18,'core_source_bundles_revalidated':12,
  'label_free_prefix_path':str(prefix_rec['path']),'label_free_prefix_hash_verified':True,'ordered_receiver_ids_sha256':idhash,
  'accepted_v4_per_window_reference_counts':counts,'v4_donor_map_rows_used':0,'v4_primary_task_seed_rows':len(primary),
  'metric_scale_lock_sha256':sm['metric_scale_lock']['sha256'],'synthetic_scorer':'PASS','target_archive_payload_opened_by_preflight':False,
  'target_archive_path_checked_by_stat_only':True,'target_archive_load_operations_added':0,'final_heldout_payload_access':0}
 write(ROOT/'qa/SCORER_PREFLIGHT.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
