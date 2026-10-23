#!/usr/bin/env python3
"""Seal all V4 prediction files before any scorer can open the dev target archive."""
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
V2=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'))
V3=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'))
TASKS=(119,94,90,141,230);SEEDS=(202610131,202610132,202610133)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_meta(path):
    with safe_open(str(path),framework='np') as f:
        if 'prediction_native' not in f.keys():raise RuntimeError(f'missing native prediction tensor: {path}')
        x=f.get_tensor('prediction_native')
        if x.shape!=(185,30,28) or not np.isfinite(x).all():raise RuntimeError(f'prediction shape/finite failure: {path} {x.shape}')
        return {'shape':list(x.shape),'dtype':str(x.dtype)}
def read_ids(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    with np.load(ROOT/'data/DEV_PREFIXES_NO_LABELS.npz',allow_pickle=False) as z:
        ids=[str(x) for x in z['anchor_id'].tolist()]
        meta=[(str(z['episode_id'][i]),str(z['task_id'][i]),str(z['source_cluster'][i])) for i in range(len(ids))]
        meta_by_id={ids[i]:meta[i] for i in range(len(ids))}
    if len(ids)!=185 or len(set(ids))!=185:raise RuntimeError('locked dev support is not 185 unique receivers')
    queue=json.loads((ROOT/'coordination/queues/INFERENCE.json').read_text())
    if queue.get('phase')!='INFERENCE' or len(queue.get('jobs',[]))!=24:raise RuntimeError('V4 inference plan must contain exactly 24 nonempty jobs')
    new_entries=[]
    for watcher_cfg in queue['jobs']:
        w=json.loads(Path(watcher_cfg).read_text());cfgpath=Path(w['command'][-1]);cfg=json.loads(cfgpath.read_text());out=Path(cfg['output_root'])
        complete_path=out/'PREDICTION_COMPLETE.json';pred=out/'PREDICTIONS.safetensors';rows=out/'PREDICTION_ROWS.csv'
        if not all(p.is_file() for p in (complete_path,pred,rows)):raise RuntimeError(f"missing prediction artifacts for {cfg['job_id']}")
        complete=json.loads(complete_path.read_text())
        if complete.get('status')!='PASS' or complete.get('dev_future_target_arrays_requested') is not False:raise RuntimeError(f"invalid completion marker {cfg['job_id']}")
        pred_sha=sha(pred);rows_sha=sha(rows)
        if complete.get('prediction_sha256')!=pred_sha or complete.get('prediction_rows_sha256')!=rows_sha:raise RuntimeError(f"completion hashes disagree for {cfg['job_id']}")
        meta_tensor=tensor_meta(pred);rid=read_ids(rows)
        if len(rid)!=185 or [r['anchor_id'] for r in rid]!=ids:raise RuntimeError(f"prediction IDs/order differ from fixed dev receiver set: {cfg['job_id']}")
        for i,r in enumerate(rid):
            if (str(r['episode_id']),str(r['task_id']),str(r['source_cluster']))!=meta[i]:raise RuntimeError(f"prediction row metadata join mismatch: {cfg['job_id']} row {i}")
        if complete.get('model_sha256')!=sha(Path(cfg['model_path'])):raise RuntimeError(f"model weight changed after prediction: {cfg['job_id']}")
        new_entries.append({'job_id':cfg['job_id'],'program_id':cfg['program_id'],'model_id':cfg['model_id'],'seed':int(cfg['seed']),'condition':cfg['condition'],'prediction_path':str(pred),'prediction_sha256':pred_sha,'prediction_rows_path':str(rows),'prediction_rows_sha256':rows_sha,'completion_path':str(complete_path),'completion_sha256':sha(complete_path),'completion_status':'PASS','rows':185,'prediction_shape':meta_tensor['shape'],'dtype':meta_tensor['dtype'],'model_path':cfg['model_path'],'model_sha256':complete['model_sha256'],'normalizer_sha256':complete['normalizer_sha256'],'donor_map_sha256':complete.get('donor_map_sha256'),'dev_targets_requested':False})
    ref_seal=json.loads((V3/'protocol/PREDICTION_BATCH_SEAL.json').read_text())
    refs=ref_seal['reused_v2_prediction_references'];b_entries=[]
    for seed in SEEDS:
        matches=[x for x in refs if x.get('model_id')=='G-V' and int(x.get('seed',-1))==seed]
        if len(matches)!=1:raise RuntimeError(f'expected one accepted V2 G-V matched reference at seed {seed}')
        x=matches[0];pred=Path(x['prediction_path']);rows=Path(x['prediction_rows_path'])
        if not pred.is_file() or not rows.is_file() or sha(pred)!=x['prediction_sha256'] or sha(rows)!=x['prediction_rows_sha256']:raise RuntimeError(f'accepted V2 B_MATCHED reference invalid at seed {seed}')
        tm=tensor_meta(pred);rid=read_ids(rows)
        if len(rid)!=185 or {r['anchor_id'] for r in rid}!=set(ids):raise RuntimeError(f'accepted B_MATCHED ID set differs at seed {seed}')
        rowmap={r['anchor_id']:r for r in rid}
        for i,a in enumerate(ids):
            r0=rowmap[a]
            if (str(r0['episode_id']),str(r0['task_id']),str(r0['source_cluster']))!=meta_by_id[a]:raise RuntimeError(f'accepted B_MATCHED metadata join differs at seed {seed}: {a}')
        b_entries.append({'model_id':'B_MATCHED','source_model_id':'G-V','condition':'MATCHED','seed':seed,'prediction_path':str(pred),'prediction_sha256':sha(pred),'prediction_rows_path':str(rows),'prediction_rows_sha256':sha(rows),'rows':185,'prediction_shape':tm['shape'],'source':'accepted V2 current corrected split; reused read-only','source_prediction_seal_sha256':sha(V2/'protocol/PREDICTION_BATCH_SEAL.json'),'dev_target_read_in_v4_preseal':False})
    expected_ids={f'v4_pred_A_STATE_COMMON_s{s}_M' for s in SEEDS}|{f'v4_pred_C_STATE_NEIGHBOR_REASSIGNED_s{s}_M' for s in SEEDS}
    expected_ids|={f'v4_pred_{program}_s{s}_{m}' for s in SEEDS for program in ('B_MATCHED','C_STATE_NEIGHBOR_REASSIGNED') for m in ('D1','D2','D3')}
    if {x['job_id'] for x in new_entries}!=expected_ids:raise RuntimeError('sealed V4 job IDs do not equal predeclared 24-job matrix')
    if len(new_entries)!=24 or len(b_entries)!=3:raise RuntimeError('nonempty seal count mismatch')
    seal={'schema_version':1,'status':'SEALED_BEFORE_TARGET_ACCESS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'study':'post_V3_RGB_alignment_training_strategy_development','tasks':list(TASKS),'seeds':list(SEEDS),'planned_new_jobs':24,'sealed_new_prediction_jobs':len(new_entries),'missing_new_prediction_jobs':0,'new_jobs_primary_matched':6,'new_jobs_cross_evaluation':18,'reused_B_MATCHED_jobs':len(b_entries),'new_task_condition_seed_views':120,'reused_B_MATCHED_task_seed_views':15,'total_primary_A_B_C_task_seed_views':45,'total_cross_evaluation_task_seed_views':90,'all_jobs_cover_same_185_dev_receiver_ids':True,'all_prediction_hashes_IDs_shapes_and_finiteness_revalidated':True,'empty_prediction_batch_refused':True,'dev_target_reader_called':False,'dev_target_rows_read':0,'future_labels_used_in_forward':False,'final_heldout_payload_access':0,'new_prediction_entries':new_entries,'reused_B_MATCHED_entries':b_entries}
    p=ROOT/'protocol/PREDICTION_BATCH_SEAL.json';p.write_text(json.dumps(seal,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    (ROOT/'qa/PREDICTION_SEAL.json').write_text(json.dumps({'schema_version':1,'status':'PASS','seal_path':str(p),'seal_sha256':sha(p),'new_jobs':24,'reused_B_MATCHED_jobs':3,'missing_jobs':0,'prediction_task_method_seed_views':135,'dev_target_rows_read':0,'nonempty_seal':True,'final_heldout_payload_access':0},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    exp_path=ROOT/'inventory/OUTCOME_EXPOSURE.json';exp=json.loads(exp_path.read_text());exp.update({'v4_prediction_seal_sha256':sha(p),'v4_prediction_jobs_sealed':24,'v4_reused_B_MATCHED_jobs':3,'v4_preseal_dev_target_rows_read':0,'status':'SEALED_NONEMPTY_NO_V4_TARGET_ACCESS'});exp_path.write_text(json.dumps(exp,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS','seal_sha256':sha(p),'new_jobs':24,'B_matched_reused':3,'targets_read':0,'final_access':0},indent=2))
if __name__=='__main__':main()
