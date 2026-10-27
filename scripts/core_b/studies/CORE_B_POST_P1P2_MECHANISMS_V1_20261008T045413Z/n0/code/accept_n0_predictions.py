#!/usr/bin/env python3
"""Branchwise target-blind N0 acceptance. Failed branches do not suppress valid branches."""
from __future__ import annotations
import os
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

N0_ROOT=Path(__file__).resolve().parents[1]
NEXT_ROOT=N0_ROOT.parent
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
P1_ROOT=PROJECT/'reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z'
V3_ROOT=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'

def sha(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_bundle(path:Path,expected_sha:str,ids:np.ndarray)->np.ndarray:
    if not path.is_file() or sha(path)!=expected_sha:raise RuntimeError('bundle missing/hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        forbidden=[k for k in z.files if 'target' in k.lower() or 'future' in k.lower()]
        if forbidden:raise RuntimeError(f'forbidden target/future keys: {forbidden}')
        if not {'anchor_id','predictions_native'}.issubset(z.files):raise RuntimeError('prediction bundle lacks anchor_id/predictions_native')
        got=z['anchor_id'].astype(str);pred=z['predictions_native'].astype(np.float32,copy=True)
    lookup={x:i for i,x in enumerate(got.tolist())};want=ids.astype(str).tolist()
    if len(lookup)!=len(got) or set(lookup)!=set(want):raise RuntimeError('prediction ID set mismatch')
    pred=pred[np.asarray([lookup[x] for x in want],dtype=np.int64)]
    if pred.shape!=(2322,30,28) or not np.isfinite(pred).all():raise RuntimeError(f'prediction shape/finite failure {pred.shape}')
    return pred

def main()->int:
    out=NEXT_ROOT/'qa/PREDICTION_ACCEPTANCE.json'
    if out.exists():raise FileExistsError(out)
    matrix=list(csv.DictReader((NEXT_ROOT/'protocol/N0_MATRIX.csv').open(newline='',encoding='utf-8')))
    if len(matrix)!=12:raise RuntimeError('locked N0 intervention denominator is not 12')
    with np.load(V3_ROOT/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
        ids=z['anchor_id'][z['split_code']==1].astype(str)
    rows=[];real_records=[];probe_records=[];branches=[];node_records=[]
    for model,seed in sorted({(r['model'],int(r['seed'])) for r in matrix}):
        real_path=N0_ROOT/f'predictions/REAL_REPLAY/{model}-H30-S{seed}-REAL.npz'
        real_side_path=real_path.with_suffix('.json')
        branch={'model':model,'seed':seed,'real_replay':'missing','probes':{},'blocked_reason':''}
        try:
            side=json.loads(real_side_path.read_text(encoding='utf-8'))
            row=next(r for r in matrix if r['model']==model and int(r['seed'])==seed)
            if side.get('status')!='PASS_STRICT_REPLAY' or side.get('strict_replay_exact') is not True:raise RuntimeError('strict accepted-P1 replay failed')
            if side.get('target_arrays_opened') is not False or int(side.get('final_test_access',-1))!=0 or int(side.get('optimizer_steps',-1))!=0:raise RuntimeError('REAL replay crossed N0 boundary')
            if side.get('checkpoint_sha256')!=row['checkpoint_sha256']:raise RuntimeError('REAL checkpoint does not match the N0 matrix')
            real=load_bundle(real_path,side['bundle_sha256'],ids)
            ref_path=Path(row['accepted_real_prediction_path'])
            if not ref_path.is_absolute():ref_path=P1_ROOT/ref_path
            if sha(ref_path)!=row['accepted_real_prediction_sha256']:raise RuntimeError('accepted P1 reference hash changed')
            ref=load_bundle(ref_path,row['accepted_real_prediction_sha256'],ids)
            if not np.array_equal(real,ref):raise RuntimeError('fresh REAL output is not exact against accepted P1 bundle')
            branch['real_replay']='accepted_strict_exact'
            real_records.append({'model':model,'seed':seed,'bundle_path':str(real_path.relative_to(NEXT_ROOT)),'bundle_sha256':side['bundle_sha256'],
                'p1_reference_sha256':row['accepted_real_prediction_sha256'],'checkpoint_sha256':side['checkpoint_sha256'],'physical_gpu':side.get('physical_gpu'),
                'max_abs_difference':0.0,'final_test_access':0})
        except Exception as exc:
            branch['real_replay']='blocked_or_failed'
            branch['blocked_reason']=f'{type(exc).__name__}: {exc}'
            for cond in ('ZERO','PERM'):
                m=next(r for r in matrix if r['model']==model and int(r['seed'])==seed and r['condition']==cond)
                rows.append({'model':model,'seed':seed,'condition':cond,'accepted':False,'reason':'REAL replay did not pass; branch blocked','bundle_path':m['new_prediction_bundle'],'bundle_sha256':''})
            branches.append(branch)
            continue
        for condition in ('ZERO','PERM'):
            m=next(r for r in matrix if r['model']==model and int(r['seed'])==seed and r['condition']==condition)
            path=NEXT_ROOT/m['new_prediction_bundle'];sidepath=path.with_suffix('.json')
            try:
                ps=json.loads(sidepath.read_text(encoding='utf-8'))
                if ps.get('status')!='PREDICTIONS_WRITTEN_UNSCORED':raise RuntimeError('probe bundle was not left unscored')
                if (ps.get('model'),int(ps.get('seed',-1)),ps.get('condition'))!=(model,seed,condition):raise RuntimeError('probe sidecar cell key mismatch')
                if ps.get('target_arrays_opened') is not False or ps.get('encoder_updated') is not False or int(ps.get('optimizer_steps',-1))!=0 or int(ps.get('final_test_access',-1))!=0:raise RuntimeError('probe crossed N0 boundary')
                if ps.get('checkpoint_sha256')!=m['checkpoint_sha256'] or ps.get('accepted_reference_bundle_sha256')!=m['accepted_real_prediction_sha256']:raise RuntimeError('probe source binding mismatch')
                pred=load_bundle(path,ps['bundle_sha256'],ids)
                if ps.get('non_video_input_sha256')!=side.get('non_video_input_sha256'):raise RuntimeError('non-video inputs changed')
                if ps.get('parameter_state_sha256_before')!=ps.get('parameter_state_sha256_after') or ps.get('buffer_state_sha256_before')!=ps.get('buffer_state_sha256_after'):raise RuntimeError('parameters or buffers changed')
                if ps.get('p1_video_projection_input_sha256')==side.get('p1_video_projection_input_sha256'):raise RuntimeError(f'{condition} did not alter projection input')
                branch['probes'][condition]='accepted'
                probe_records.append({'model':model,'seed':seed,'condition':condition,'bundle_path':str(path.relative_to(NEXT_ROOT)),
                    'bundle_sha256':ps['bundle_sha256'],'checkpoint_sha256':ps['checkpoint_sha256'],'rows':len(pred),'shape':list(pred.shape),
                    'physical_gpu':ps.get('physical_gpu'),'target_arrays_opened':False,'final_test_access':0})
                node_records.append({'model':model,'seed':seed,'condition':condition,
                    'real_node_sha256':side.get('p1_video_projection_input_sha256'),'probe_node_sha256':ps.get('p1_video_projection_input_sha256'),
                    'node_changed':ps.get('p1_video_projection_input_sha256')!=side.get('p1_video_projection_input_sha256'),
                    'non_video_input_sha256':ps.get('non_video_input_sha256'),
                    'parameters_unchanged':ps.get('parameter_state_sha256_before')==ps.get('parameter_state_sha256_after'),
                    'buffers_unchanged':ps.get('buffer_state_sha256_before')==ps.get('buffer_state_sha256_after'),
                    'target_arrays_opened':False,'final_test_access':0})
                rows.append({'model':model,'seed':seed,'condition':condition,'accepted':True,'reason':'pass','bundle_path':str(path.relative_to(NEXT_ROOT)),'bundle_sha256':ps['bundle_sha256']})
            except Exception as exc:
                branch['probes'][condition]='blocked_or_failed'
                rows.append({'model':model,'seed':seed,'condition':condition,'accepted':False,'reason':f'{type(exc).__name__}: {exc}','bundle_path':str(path.relative_to(NEXT_ROOT)),'bundle_sha256':''})
        branches.append(branch)
    accepted_probe=len(probe_records);accepted_real=len(real_records)
    status='PASS_ALL_6_REAL_REPLAYS_AND_12_INTERVENTIONS' if accepted_real==6 and accepted_probe==12 else 'PARTIAL_N0_ACCEPTANCE'
    result={'status':status,'real_replays_expected':6,'real_replays_accepted':accepted_real,'perturbed_bundles_expected':12,'perturbed_bundles_accepted':accepted_probe,
        'condition_counts':{'ZERO':sum(r['condition']=='ZERO' for r in probe_records),'PERM':sum(r['condition']=='PERM' for r in probe_records)},
        'rows_per_bundle':2322,'prediction_shape':[2322,30,28],'target_arrays_opened':False,'optimizer_steps':0,'new_encoder_extraction':0,
        'final_heldout_access':0,'real_replays':real_records,'interventions':probe_records,'branch_status':branches,'expected_but_not_accepted':rows}
    node_path=N0_ROOT/'results/N0_NODE_INPUT_DIFF.csv'
    if node_path.exists():raise FileExistsError(node_path)
    node_path.parent.mkdir(parents=True,exist_ok=True)
    node_fields=['model','seed','condition','real_node_sha256','probe_node_sha256','node_changed','non_video_input_sha256','parameters_unchanged','buffers_unchanged','target_arrays_opened','final_test_access']
    with node_path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=node_fields);writer.writeheader();writer.writerows(node_records)
    result['node_input_diff_csv']=str(node_path.relative_to(NEXT_ROOT))
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':status,'real_replays':accepted_real,'interventions':accepted_probe,'final_heldout_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
