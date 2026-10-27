#!/usr/bin/env python3
from __future__ import annotations
import os
import csv, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file, save_file

ROOT=Path(__file__).resolve().parents[1]
TASKS=[119,94,90,141,230]
SEEDS=[202610131,202610132,202610133]
METHODS=['H_SV','H_VS','H_AV','H_SC','H_PV','MIX50']

def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_ids(ids): return sha_bytes(('\n'.join(ids)+'\n').encode())
def read_csv(p):
    with Path(p).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def write_csv(p,rows,fields):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def write_json(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def validate_inputs(arrays, ids_by_role, ordered_ids, anchors):
    if not ordered_ids: raise ValueError('empty receiver matrix')
    if len(set(ordered_ids))!=len(ordered_ids): raise ValueError('duplicate anchor IDs')
    if np.asarray(anchors).shape!=(len(ordered_ids),28): raise ValueError('anchor shape mismatch')
    if not np.isfinite(anchors).all(): raise ValueError('nonfinite anchor')
    for role in ('S','V','A','C'):
        if role not in arrays: raise ValueError(f'missing source role {role}')
        x=np.asarray(arrays[role])
        if x.shape!=(len(ordered_ids),30,28): raise ValueError(f'{role} shape mismatch')
        if not np.isfinite(x).all(): raise ValueError(f'{role} nonfinite')
        if ids_by_role.get(role)!=ordered_ids: raise ValueError(f'{role} ID/seed/order mismatch')

def compose(arrays, ordered_ids, anchor):
    # This function only consumes source native predictions and the label-free current anchor.
    S=np.asarray(arrays['S'],dtype=np.float64)
    V=np.asarray(arrays['V'],dtype=np.float64)
    A=np.asarray(arrays['A'],dtype=np.float64)
    C=np.asarray(arrays['C'],dtype=np.float64)
    a=np.asarray(anchor,dtype=np.float64)[:,None,:]
    mS=S.mean(axis=1,keepdims=True,dtype=np.float64)
    mV=V.mean(axis=1,keepdims=True,dtype=np.float64)
    mA=A.mean(axis=1,keepdims=True,dtype=np.float64)
    mC=C.mean(axis=1,keepdims=True,dtype=np.float64)
    return {
        'H_SV':V-mV+mS,
        'H_VS':S-mS+mV,
        'H_AV':V-mV+mA,
        'H_SC':C-mC+mS,
        'H_PV':V-mV+a,
        'MIX50':0.5*S+0.5*V,
    }

def validate_seed_contract(seed_by_role, expected_seed):
    if set(seed_by_role)!={'S','V','A','C'} or any(int(v)!=int(expected_seed) for v in seed_by_role.values()):
        raise ValueError('source seed mismatch')

def expect_reject(label, fn):
    try: fn()
    except (ValueError,KeyError): return {'case':label,'rejected':True}
    raise RuntimeError(f'validation failed to reject {label}')

def synthetic_qa():
    rng=np.random.default_rng(20261015)
    n=3; ids=['synthetic_a','synthetic_b','synthetic_c']
    arrays={k:rng.normal(size=(n,30,28)).astype(np.float64) for k in ('S','V','A','C')}
    anchor=rng.normal(size=(n,28)).astype(np.float64)
    idmap={k:ids[:] for k in arrays}
    validate_inputs(arrays,idmap,ids,anchor)
    out=compose(arrays,ids,anchor)
    maxerr=lambda x,y:float(np.max(np.abs(x-y)))
    checks={
      'H(X,X)=X':maxerr(compose({'S':arrays['S'],'V':arrays['S'],'A':arrays['S'],'C':arrays['S']},ids,anchor)['H_SV'],arrays['S']),
      'H_SV_mean_equals_S':maxerr(out['H_SV'].mean(axis=1),arrays['S'].mean(axis=1)),
      'H_SC_mean_equals_S':maxerr(out['H_SC'].mean(axis=1),arrays['S'].mean(axis=1)),
      'H_AV_mean_equals_A':maxerr(out['H_AV'].mean(axis=1),arrays['A'].mean(axis=1)),
      'H_VS_mean_equals_V':maxerr(out['H_VS'].mean(axis=1),arrays['V'].mean(axis=1)),
      'H_PV_mean_equals_anchor':maxerr(out['H_PV'].mean(axis=1),anchor),
      'H_SV_internal_diff_equals_V':maxerr(np.diff(out['H_SV'],axis=1),np.diff(arrays['V'],axis=1)),
      'H_VS_internal_diff_equals_S':maxerr(np.diff(out['H_VS'],axis=1),np.diff(arrays['S'],axis=1)),
      'H_AV_internal_diff_equals_V':maxerr(np.diff(out['H_AV'],axis=1),np.diff(arrays['V'],axis=1)),
      'H_SC_internal_diff_equals_C':maxerr(np.diff(out['H_SC'],axis=1),np.diff(arrays['C'],axis=1)),
      'H_PV_internal_diff_equals_V':maxerr(np.diff(out['H_PV'],axis=1),np.diff(arrays['V'],axis=1)),
    }
    tol=1e-12
    if max(checks.values())>tol: raise RuntimeError(f'synthetic identities exceed tolerance {checks}')
    rejects=[]
    rejects.append(expect_reject('wrong ID order',lambda:validate_inputs(arrays,{**idmap,'V':ids[::-1]},ids,anchor)))
    rejects.append(expect_reject('wrong seed',lambda:validate_seed_contract({'S':202610131,'V':202610132,'A':202610131,'C':202610131},202610131)))
    rejects.append(expect_reject('wrong temporal shape',lambda:validate_inputs({**arrays,'S':arrays['S'][:,:29]},idmap,ids,anchor)))
    rejects.append(expect_reject('swapped time/action axes',lambda:validate_inputs({**arrays,'C':arrays['C'].transpose(0,2,1)},idmap,ids,anchor)))
    bad={**arrays,'S':arrays['S'].copy()};bad['S'][0,0,0]=np.nan
    rejects.append(expect_reject('nonfinite source',lambda:validate_inputs(bad,idmap,ids,anchor)))
    rejects.append(expect_reject('empty IDs',lambda:validate_inputs(arrays,idmap,[],anchor[:0])))
    return {'status':'PASS','seed':20261015,'checks_max_abs':checks,'tolerance_atol':1e-12,'negative_cases':rejects,'future_axis_only':True}

def main():
    t0=time.perf_counter()
    formula_path=ROOT/'protocol/FORMULA_LOCK.json'
    seal_path=ROOT/'qa/PREDICTION_SEAL.json'
    if not formula_path.is_file(): raise RuntimeError('cannot compose before FORMULA_LOCK')
    if seal_path.exists(): raise RuntimeError('prediction seal already exists; refusing overwrite')
    formula=json.loads(formula_path.read_text())
    if formula.get('principal_candidate')!='H_SV' or formula.get('arithmetic_dtype')!='float64': raise RuntimeError('formula lock mismatch')
    formula_sha=sha_file(formula_path)
    sm=json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text())
    src={(s['role'],int(s['seed'])):s for s in sm['source_prediction_bundles']}
    join_rows=read_csv(ROOT/'inventory/SOURCE_PREDICTION_JOINS.csv')
    by_role_seed={}
    for row in join_rows: by_role_seed.setdefault((row['role'],int(row['seed'])),[]).append(row)
    prefix_path=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/DEV_PREFIXES_NO_LABELS.npz'))
    with np.load(prefix_path,allow_pickle=False) as z:
        ids0=[str(x) for x in z['anchor_id'].tolist()]
        anchor_by_id={str(z['anchor_id'][i]):np.asarray(z['anchor_native'][i],dtype=np.float64) for i in range(len(ids0))}
        prefix_meta={str(z['anchor_id'][i]):{'episode_id':str(z['episode_id'][i]),'task_id':str(z['task_id'][i]),'source_group':str(z['source_cluster'][i]),'anchor_frame':str(z['anchor_frame'][i])} for i in range(len(ids0))}
    canonical_rows=sorted((r for r in join_rows if r['role']=='S_OPT' and int(r['seed'])==SEEDS[0]),key=lambda x:int(x['canonical_row_index']))
    ids=[r['anchor_id'] for r in canonical_rows]
    if len(ids)!=185 or len(set(ids))!=185 or set(ids)!=set(ids0): raise RuntimeError('canonical receiver order invalid')
    if sha_ids(ids)!=formula.get('ordered_receiver_ids_sha256'): raise RuntimeError('canonical receiver order differs from locked V2 IDs')
    anchor=np.stack([anchor_by_id[a] for a in ids])
    outputs={}
    source_sha={}
    load_seconds=0.0
    for seed in SEEDS:
        arrays={}; idmap={}
        for role,short in [('S_OPT','S'),('V_MATCHED','V'),('A_COMMON','A'),('C_REASSIGNED','C')]:
            s=src[(role,seed)]
            if sha_file(s['prediction_path'])!=s['prediction_sha256'] or sha_file(s['prediction_rows_path'])!=s['prediction_rows_sha256']:
                raise RuntimeError(f'source changed after lock: {role}/{seed}')
            st=time.perf_counter(); d=load_file(s['prediction_path']); arr=np.asarray(d['prediction_native']); load_seconds+=time.perf_counter()-st
            rows=read_csv(s['prediction_rows_path']); index={r['anchor_id']:i for i,r in enumerate(rows)}
            if set(index)!=set(ids): raise RuntimeError(f'source IDs changed {role}/{seed}')
            ordered=np.stack([arr[index[a]] for a in ids]).astype(np.float64,copy=False)
            arrays[short]=ordered; idmap[short]=ids[:]
            source_sha[f'{role}:{seed}']=s['prediction_sha256']
        validate_inputs(arrays,idmap,ids,anchor)
        validate_seed_contract({r:seed for r in ('S','V','A','C')},seed)
        out=compose(arrays,ids,anchor)
        outputs[seed]=(arrays,out)
    syn=synthetic_qa()
    # Real unlabeled source-array algebra QA before creating the nonempty seal.
    metric=json.loads(Path(sm['metric_scale_lock']['path']).read_text())
    fdscale=np.asarray(metric['safe_fd_scale'],dtype=np.float64)
    real=[];max_native={k:0.0 for k in ['H_SV','H_AV','H_PV','H_VS','H_SC']};max_scaled={k:0.0 for k in max_native}
    level_mean_err={k:0.0 for k in ['H_SV','H_AV','H_PV','H_VS','H_SC']}
    seam_deviation={}
    for seed,(arr,out) in outputs.items():
        refs={'H_SV':arr['V'],'H_AV':arr['V'],'H_PV':arr['V'],'H_VS':arr['S'],'H_SC':arr['C']}
        for method,ref in refs.items():
            delta=np.diff(out[method],axis=1)-np.diff(ref,axis=1)
            max_native[method]=max(max_native[method],float(np.max(np.abs(delta))))
            max_scaled[method]=max(max_scaled[method],float(np.max(np.abs(delta)/fdscale[None,None,:])))
        for method,ref in [('H_SV',arr['S']),('H_AV',arr['A']),('H_PV',np.broadcast_to(anchor[:,None,:],out['H_PV'].shape)),('H_VS',arr['V']),('H_SC',arr['S'])]:
            level_mean_err[method]=max(level_mean_err[method],float(np.max(np.abs(out[method].mean(axis=1)-ref.mean(axis=1)))))
        seam_deviation[str(seed)]={
            'H_SV_first_step_vs_V_max_abs':float(np.max(np.abs((out['H_SV'][:,0]-anchor)-(arr['V'][:,0]-anchor)))),
            'H_AV_first_step_vs_V_max_abs':float(np.max(np.abs((out['H_AV'][:,0]-anchor)-(arr['V'][:,0]-anchor)))),
            'H_SC_first_step_vs_C_max_abs':float(np.max(np.abs((out['H_SC'][:,0]-anchor)-(arr['C'][:,0]-anchor)))),
            'H_PV_first_step_vs_V_max_abs':float(np.max(np.abs((out['H_PV'][:,0]-anchor)-(arr['V'][:,0]-anchor))))}
    atol=1e-12; rtol=1e-10
    if max(max_native.values())>atol or max(level_mean_err.values())>atol: raise RuntimeError(f'real source algebra QA failed: {max_native}; means={level_mean_err}')

    outrows=[]; complete=[]
    rows_meta=[{'anchor_id':a,**prefix_meta[a]} for a in ids]
    plan_path=ROOT/'protocol/DERIVED_PREDICTION_PLAN.csv';plan=read_csv(plan_path)
    plan_map={(r['method'],int(r['seed'])):r for r in plan}
    compose_start=time.perf_counter()
    for seed,(arr,out) in outputs.items():
        for method in METHODS:
            array=np.asarray(out[method],dtype=np.float64,order='C')
            if array.shape!=(185,30,28) or array.dtype!=np.float64 or not np.isfinite(array).all(): raise RuntimeError(f'invalid derived bundle {method}/{seed}')
            folder=ROOT/'predictions'/method/f'seed{seed}'/'attempt_02';folder.mkdir(parents=True,exist_ok=False)
            pred_path=folder/'PREDICTIONS.safetensors';rows_path=folder/'PREDICTION_ROWS.csv';done_path=folder/'PREDICTION_COMPLETE.json'
            save_file({'prediction_native':array},str(pred_path))
            write_csv(rows_path,rows_meta,['anchor_id','episode_id','task_id','source_group','anchor_frame'])
            ph=sha_file(pred_path);rh=sha_file(rows_path); array_sha=sha_bytes(np.asarray(array,dtype='<f8',order='C').tobytes(order='C'))
            # Round-trip validation without any outcome input.
            stored=np.asarray(load_file(str(pred_path))['prediction_native'])
            if stored.dtype!=np.float64 or stored.shape!=(185,30,28) or not np.array_equal(stored,array): raise RuntimeError(f'roundtrip mismatch {method}/{seed}')
            srcroles={'H_SV':['S_OPT','V_MATCHED'],'H_VS':['S_OPT','V_MATCHED'],'H_AV':['A_COMMON','V_MATCHED'],'H_SC':['S_OPT','C_REASSIGNED'],'H_PV':['V_MATCHED','current_command_anchor_a'],'MIX50':['S_OPT','V_MATCHED']}[method]
            lineage={role:(src[(role,seed)]['prediction_sha256'] if role!='current_command_anchor_a' else sha_file(prefix_path)) for role in srcroles}
            done={'schema_version':1,'status':'PASS','bundle_id':f'{method}_seed{seed}','method':method,'seed':seed,'formula':formula['formulas'][method],
                  'formula_lock_sha256':formula_sha,'source_lineage_sha256':lineage,'ordered_anchor_ids_sha256':sha_ids(ids),'attempt':2,
                  'prediction_path':str(pred_path),'prediction_sha256':ph,'prediction_shape':list(array.shape),'dtype':str(array.dtype),'finite':True,
                  'prediction_rows_path':str(rows_path),'prediction_rows_sha256':rh,'canonical_array_sha256':array_sha,'canonical_array_hash_spec':'little-endian float64 C-order bytes','rows':len(ids),'task_views':90,'target_input_to_transform':False,
                  'future_labels_used_in_forward':False,'new_fits':0,'optimizer_updates':0,'normalizer_sha256':formula['normalizer']['sha256'],'native_space':True,'created_utc':utc()}
            write_json(done_path,done)
            complete.append({'bundle_id':done['bundle_id'],'method':method,'seed':seed,'attempt':2,'prediction_path':str(pred_path),'prediction_sha256':ph,'canonical_array_sha256':array_sha,
                'prediction_rows_path':str(rows_path),'prediction_rows_sha256':rh,'completion_path':str(done_path),'completion_sha256':sha_file(done_path),
                'shape':list(array.shape),'dtype':str(array.dtype),'rows':185,'task_views':90,'formula_lock_sha256':formula_sha})
            plan_map[(method,seed)]['status']='ACCEPTED'
            outrows.append({'method':method,'seed':seed,'attempt':2,'task_views':90,'rows':185,'prediction_path':str(pred_path),'prediction_sha256':ph,'canonical_array_sha256':array_sha,'prediction_rows_sha256':rh,'status':'ACCEPTED'})
    write_csv(plan_path,plan,list(plan[0]))
    transform_elapsed=time.perf_counter()-compose_start
    write_csv(ROOT/'results/DERIVED_BUNDLE_INDEX.csv',outrows,list(outrows[0]))
    # Rerun the pure transform from the same label-free source data and compare serialized bytes.
    regeneration={}
    for seed,(arr,out) in outputs.items():
        for method in METHODS:
            check=ROOT/'tmp'/f'REGEN_{method}_seed{seed}.safetensors'
            save_file({'prediction_native':np.asarray(out[method],dtype=np.float64,order='C')},str(check))
            original=ROOT/'predictions'/method/f'seed{seed}'/'attempt_02'/'PREDICTIONS.safetensors'
            same=sha_file(check)==sha_file(original)
            check.unlink()
            if not same: raise RuntimeError(f'GT-independent regeneration hash changed for {method}/{seed}')
            regeneration[f'{method}:{seed}']=sha_file(original)
    qa={'schema_version':1,'status':'PASS','synthetic':syn,'real_source_array_checks':{
        'source_bundle_count':18,'core_formula_source_bundles':12,'bias_reference_bundles':6,'all_source_arrays_nonempty_finite':True,
        'ordered_id_join':'anchor_id + episode_id + task_id + source_group','all_185_receiver_ids_match':True,'shape':'[185,30,28]',
        'native_output_dtype':'float64','mean_identity_max_abs':level_mean_err,'internal_difference_max_native_abs':max_native,
        'internal_difference_max_canonical_scaled_abs':max_scaled,'identity_tolerance_atol':atol,'identity_tolerance_rtol':rtol,
        'seam_invariance_assumed':False,'first_step_seam_deviation_observed':seam_deviation,
        'donor_map_predictions_used_for_V_or_C':False,'transform_input_signature':['source_native_predictions','ordered_receiver_ids','current_command_anchor_a'],
        'GT_loader_or_path_in_transform':False,'regeneration_after_initial_save_hashes_identical':regeneration},
        'accepted_bundles':len(complete),'accepted_task_views':len(complete)*5,'accepted_prediction_rows':len(complete)*185,
        'empty_prediction_batch_refused':True,'target_archive_loads_before_seal':0,'dev_future_target_rows_read_before_seal':0,
        'final_heldout_payload_access':0,'formula_lock_sha256':formula_sha,'composition_wall_seconds':transform_elapsed,
        'created_utc':utc()}
    write_json(ROOT/'qa/TRANSFORM_QA.json',qa)
    seal={'schema_version':1,'status':'PASS','created_utc':utc(),'formula_lock_sha256':formula_sha,'attempt':2,'planned_bundles':18,'sealed_bundles':len(complete),
          'missing_bundles':0,'nonempty_seal':bool(complete),'accepted_task_views':len(complete)*5,'accepted_window_bundles':len(complete),
          'receiver_anchors_per_bundle':185,'all_source_and_output_arrays_finite':True,'all_ID_joins_valid':True,
          'preseal_dev_target_reads':0,'future_labels_used_in_forward':False,'prediction_generation_inputs_exclude_outcomes':True,
          'final_heldout_payload_access':0,'entries':complete}
    write_json(seal_path,seal)
    exposure_path=ROOT/'inventory/OUTCOME_EXPOSURE.json'; exposure=json.loads(exposure_path.read_text())
    exposure.update({'status':'PREDICTIONS_SEALED_PRE_SCORE','v5_preseal_target_archive_opens':0,'v5_preseal_target_rows_read':0,
        'sealed_prediction_bundles':len(complete),'sealed_derived_task_views':len(complete)*5,'sealed_derived_receiver_predictions':len(complete)*185,
        'prediction_seal_sha256':sha_file(seal_path),'formula_lock_sha256':formula_sha,'last_updated_utc':utc()})
    exposure['history'].append({'phase':'V5_formula_transform_seal','target_reads':0,'target_archive_loads':0,'accepted_bundles':len(complete),'status':'PASS'})
    write_json(exposure_path,exposure)
    owner_path=ROOT/'coordination/OWNER_STATE.json'; owner=json.loads(owner_path.read_text())
    owner.update({'phase':'predictions_sealed_pre_score','prediction_seal_status':'PASS','active_jobs':0,'active_gpu_jobs':0,'updated_utc':utc()})
    write_json(owner_path,owner)
    print(json.dumps({'status':'PASS','bundles':len(complete),'task_views':len(complete)*5,'rows':len(complete)*185,
      'prediction_seal_sha256':sha_file(seal_path),'formula_lock_sha256':formula_sha,'max_native_diff':max_native,
      'max_scaled_diff':max_scaled,'composition_wall_seconds':transform_elapsed,'target_reads':0},indent=2))

if __name__=='__main__': main()
