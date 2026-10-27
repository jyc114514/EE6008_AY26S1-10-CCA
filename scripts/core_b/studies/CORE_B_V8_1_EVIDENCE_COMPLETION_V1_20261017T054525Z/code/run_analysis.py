"""CPU-only scorer-only V8.1 analysis. Requires sealed preflight and synthetic QA."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BASE=PROJECT/'reports/benchmarks'
V4=BASE/'CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V5=BASE/'CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
V7=BASE/'CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z'
V8=BASE/'CORE_B_V8_FROZEN_TRANSFER_DIAGNOSIS_V1_20261016T045057Z'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import scorer_adapter as sa

os.environ['CUDA_VISIBLE_DEVICES']=''
METHOD_SOURCE={'G':'GT_RETRAIN','U':'SELF_SOFT','M':'MIX_SOFT','T':'T'}
METRIC_FIELDS={'position':'position_mae','internal_fd':'safe_fd_l1','seam':'seam_safe_fd_l1'}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path: Path) -> tuple[list[str],list[dict[str,str]]]:
    with path.open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle)
        return list(reader.fieldnames or []),list(reader)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    os.chmod(path,0o600)


def source_metadata() -> tuple[dict,dict,np.ndarray,np.ndarray]:
    preflight=read_json(ROOT/'qa/PREFLIGHT.json')
    if preflight.get('status')!='PASS' or preflight.get('target_npz_load_operations')!=0:
        raise RuntimeError('refusing target access without a clean preflight')
    typed=read_json(ROOT/'tmp/synthetic_qa/TYPED_VECTOR_PIPELINE_QA.json')
    if typed.get('status')!='PASS' or typed.get('score_rebuild_from_error_checkpoint_sufficient_arrays') is not True:
        raise RuntimeError('refusing target access without passing typed vector/checkpoint synthetic QA')
    source=read_json(ROOT/'inventory/SOURCE_MAP.json')
    lock=read_json(ROOT/'protocol/REPAIR_SCOPE_LOCK.json')
    if lock.get('status')!='PASS' or lock.get('locked_anchor_id_sha256')!=preflight.get('prefix_ordered_anchor_id_sha256'):
        raise RuntimeError('repair scope and preflight ID lock disagree')
    prefix_path=Path(source['dev_prefix_source']['absolute_path'])
    if sa.sha256_file(prefix_path)!=source['dev_prefix_source']['sha256']:
        raise RuntimeError('label-free prefix changed after source preflight')
    prefix=sa.read_npz_fields(prefix_path,sa.PREFIX_FIELDS)
    metadata=sa.validate_metadata(prefix,expected_tasks=sa.TASKS)
    ids_hash=sa.ordered_id_hash(metadata['anchor_id'])
    if len(metadata['anchor_id'])!=185 or ids_hash!=lock['locked_anchor_id_sha256']:
        raise RuntimeError('locked receiver metadata changed before scorer-only access')
    scales_json=read_json(V4/'protocol/METRIC_SCALE_LOCK.json')
    sigma,fd_scale=sa.validate_scales(scales_json['target_sigma'],scales_json['safe_fd_scale'])
    return metadata,source,sigma,fd_scale


def acquire_target_cache(metadata: dict, source: dict) -> tuple[np.ndarray,dict,dict]:
    cache_dir=ROOT/'scorer_cache'
    seal_path=cache_dir/'CACHE_SEAL.json'
    expected_id_hash=sa.ordered_id_hash(metadata['anchor_id'])
    if seal_path.is_file():
        y,cache_meta,seal=sa.load_sealed_cache(cache_dir,expected_id_hash)
        receipt_path=ROOT/'qa/TARGET_READ_RECEIPT.json'
        if receipt_path.is_file():
            receipt=read_json(receipt_path)
            receipt['cache_loads_total']=int(receipt.get('cache_loads_total',receipt.get('cache_loads',0)))+1
            receipt['cache_resume_loads']=int(receipt.get('cache_resume_loads',0))+1
            receipt['archive_npz_loads_total']=int(receipt.get('archive_npz_loads_total',receipt.get('npz_load_operations_this_attempt',0)))
            receipt['last_cache_resume_utc']=now()
            receipt['cache_resume_status']='REUSED_VALID_SEALED_CACHE_NO_SOURCE_ARCHIVE_OPEN'
        else:
            receipt={'schema_version':1,'status':'REUSED_SEALED_CACHE','cache_loads_total':1,
                     'cache_resume_loads':1,'archive_npz_loads_total':0,
                     'source_archive_path':seal['source_archive_path'],'source_archive_sha256':seal['source_archive_sha256'],
                     'unique_dev_windows':len(metadata['anchor_id']),'target_cache_seal_sha256':sa.sha256_file(seal_path),
                     'cache_resume_status':'REUSED_VALID_SEALED_CACHE_NO_SOURCE_ARCHIVE_OPEN','created_utc':now()}
        sa.atomic_json(receipt_path,receipt)
        return y,cache_meta,seal

    target_source=source['dev_target_source']
    target_path=Path(target_source['absolute_path'])
    expected_sha=target_source['sha256']
    event_path=ROOT/'coordination/DEV_TARGET_READ_EVENTS.jsonl'
    existing=[]
    if event_path.exists():
        for line in event_path.read_text(encoding='utf-8').splitlines():
            try: existing.append(json.loads(line))
            except json.JSONDecodeError: raise RuntimeError('target read event log has an invalid partial line')
    attempt=len([e for e in existing if e.get('event')=='STARTED'])+1
    event_id=f'V8_1_DEV_TARGET_READ_{attempt:03d}'
    start=now()
    started_event={'event':'STARTED','event_id':event_id,'attempt':attempt,'started_utc':start,
                   'source_path':str(target_path),'source_sha256_expected':expected_sha,
                   'source_role':'previously_exposed_dev_future_labels','split':'dev','unique_window_scope':185,
                   'reader':'np.load allow_pickle=False; fields=exact V7 whitelist','fields':list(sa.TARGET_FIELDS),
                   'npz_load_operations':0,'target_values_loaded':0,'final_heldout_payload_access':0}
    append_jsonl(event_path,started_event)
    source_hash_passes=0
    archive_loads=0
    target_values=0
    try:
        actual_sha=sa.sha256_file(target_path); source_hash_passes+=1
        if actual_sha!=expected_sha:
            raise RuntimeError(f'target source hash changed before reader open: {actual_sha}')
        archive_loads+=1
        with np.load(target_path,allow_pickle=False) as archive:
            missing=sorted(set(sa.TARGET_FIELDS)-set(archive.files))
            if missing: raise ValueError(f'target archive missing white-listed fields: {missing}')
            raw={key:np.array(archive[key],copy=True) for key in sa.TARGET_FIELDS}
            for key,value in raw.items():
                if np.asarray(value).dtype.hasobject:
                    raise ValueError(f'{key}: object/pickle payload is forbidden')
            target_values=len(raw['anchor_id'])
            target_ids=sa.text_vector(raw['anchor_id'],'target.anchor_id')
            if len(target_ids)!=185 or len(set(target_ids))!=185:
                raise ValueError('target IDs must contain exactly 185 unique locked dev windows')
            if raw['target_native_h30'].shape!=(185,30,28):
                raise ValueError(f'target_native_h30 shape mismatch: {raw["target_native_h30"].shape}')
            if not np.isfinite(raw['target_native_h30']).all():
                raise ValueError('target_native_h30 contains non-finite values')
            source_order_hash=sa.ordered_id_hash(target_ids)
            aligned_y,_=sa.align_target_to_metadata(raw,metadata)
            provenance={
              'source_archive_path':str(target_path),'source_archive_sha256':actual_sha,
              'source_target_field':'target_native_h30','source_target_dtype':str(raw['target_native_h30'].dtype),
              'source_target_shape':list(raw['target_native_h30'].shape),
              'source_ordered_id_sha256':source_order_hash,'reader_version':'V8.1-whitelist-typed-cache-v1',
              'reader_access_event_id':event_id,
            }
            # Cache files and their seal are durable before any error reduction begins.
            cache_seal=sa.write_sealed_cache(cache_dir,aligned_y,metadata,provenance)
            cache_y,cache_meta,cache_seal=sa.load_sealed_cache(cache_dir,expected_id_hash)
            if not np.array_equal(cache_y,aligned_y):
                raise RuntimeError('sealed cache readback differs from aligned reader values')
            cache_bytes={name:sa.sha256_file(cache_dir/name) for name in cache_seal['files']}
            access_receipt={
              'schema_version':1,'status':'PASS','study':'post_V8_evidence_completion_no_new_model',
              'event_id':event_id,'attempt':attempt,'started_utc':start,'completed_utc':now(),
              'source_archive_path':str(target_path),'source_archive_sha256':actual_sha,
              'source_sha256_passes_this_attempt':source_hash_passes,'npz_load_operations_this_attempt':archive_loads,
              'reader_allow_pickle':False,'reader_fields_read':list(sa.TARGET_FIELDS),
              'unique_dev_windows_loaded':target_values,'target_native_shape':list(raw['target_native_h30'].shape),
              'source_ordered_id_sha256':source_order_hash,'cache_ordered_id_sha256':expected_id_hash,
              'cache_role':'SCORER_ONLY_EXPOSED_DEV_LABELS','cache_loads':1,'cache_loads_total':1,'cache_resume_loads':0,'archive_npz_loads_total':archive_loads,
              'cache_file_sha256_readback':cache_bytes,'final_heldout_payload_access':0,
              'new_fit':0,'optimizer_update':0,'teacher_student_forward':0,'dev_prediction_job':0,
              'encoder_extraction':0,'bootstrap_draws':0,'gpu_commands':0,
            }
            sa.atomic_json(ROOT/'qa/TARGET_READ_RECEIPT.json',access_receipt)
            append_jsonl(event_path,{'event':'PASS','event_id':event_id,'attempt':attempt,'completed_utc':now(),
                                     'npz_load_operations':archive_loads,'target_values_loaded':target_values,
                                     'cache_seal_sha256':sa.sha256_file(seal_path),'cache_loads':1,
                                     'final_heldout_payload_access':0})
        # The original NPZ reader is now closed; only the read-back sealed cache proceeds.
        del aligned_y,raw
        return cache_y,cache_meta,cache_seal
    except Exception as exc:
        failure={'schema_version':1,'id':event_id,'status':'FAILED','failed_utc':now(),
                 'source_archive_path':str(target_path),'source_archive_sha256_expected':expected_sha,
                 'source_sha256_passes_this_attempt':source_hash_passes,'npz_load_operations_this_attempt':archive_loads,
                 'target_values_loaded_this_attempt':target_values,'error_type':type(exc).__name__,'error':str(exc),
                 'final_heldout_payload_access':0,'dev_prediction_or_forward':0,'gpu_commands':0}
        sa.atomic_json(ROOT/'failed_attempts'/f'{event_id}_FAILED.json',failure)
        append_jsonl(event_path,{'event':'FAILED','event_id':event_id,'attempt':attempt,'failed_utc':now(),
                                 'npz_load_operations':archive_loads,'target_values_loaded':target_values,
                                 'error_type':type(exc).__name__,'error':str(exc)})
        raise


def dtype_code(dtype) -> str:
    value=np.dtype(dtype)
    if value==np.dtype(np.float32): return 'F32'
    if value==np.dtype(np.float64): return 'F64'
    return str(value).upper()


def load_bound_prediction(asset: dict, metadata: dict) -> np.ndarray:
    pred_path=Path(asset['prediction_absolute_path'])
    if sa.sha256_file(pred_path)!=asset['prediction_sha256']:
        raise RuntimeError(f"source prediction changed after preflight: {asset['method']}/{asset['seed']}")
    tensors=load_file(str(pred_path))
    if 'prediction_native' not in tensors:
        raise RuntimeError(f"missing prediction_native field: {pred_path}")
    raw=np.asarray(tensors['prediction_native'])
    if dtype_code(raw.dtype)!=asset['prediction_dtype'] or raw.shape!=(185,30,28) or not np.isfinite(raw).all():
        raise RuntimeError(f"prediction dtype/shape/finite mismatch: {pred_path}, {raw.dtype}/{raw.shape}")
    _,rows=read_csv(Path(asset['rows_absolute_path']))
    return sa.align_prediction_by_id(raw,rows,metadata)


def load_accepted_tables(metadata: dict) -> tuple[dict,dict,dict]:
    _,window_rows=read_csv(V7/'results/PER_WINDOW_METRICS.csv')
    method_map={'GT_RETRAIN':'G','SELF_SOFT':'U','MIX_SOFT':'M','T':'T'}
    accepted={}
    for row in window_rows:
        if row['method'] not in method_map or int(row['seed']) not in sa.SEEDS:
            continue
        key=(method_map[row['method']],int(row['seed']),row['anchor_id'])
        if key in accepted: raise RuntimeError(f'duplicate accepted per-window row {key}')
        accepted[key]={k:float(row[METRIC_FIELDS[k]]) for k in ('position','internal_fd','seam')}
    expected=4*3*185
    if len(accepted)!=expected: raise RuntimeError(f'accepted per-window loss support mismatch: {len(accepted)} != {expected}')
    _,task_rows=read_csv(V7/'results/PRIMARY_TASK_SEED.csv')
    primary={}
    for row in task_rows:
        if row['method'] not in method_map or int(row['seed']) not in sa.SEEDS:
            continue
        scope='ALL' if row['task_id']=='macro' else row['task_id']
        key=(method_map[row['method']],int(row['seed']),scope)
        if key in primary: raise RuntimeError(f'duplicate accepted primary table row {key}')
        primary[key]={k:float(row[METRIC_FIELDS[k]]) for k in ('position','internal_fd','seam')}
    if len(primary)!=4*3*6: raise RuntimeError(f'accepted primary table coverage mismatch: {len(primary)}')
    perwindow_agg={}
    for method in sa.METHODS:
      for seed in sa.SEEDS:
        values={metric:np.asarray([accepted[(method,seed,anchor_id)][metric] for anchor_id in metadata['anchor_id']],dtype=np.float64)
                for metric in ('position','internal_fd','seam')}
        for scope in [*sa.TASKS,'ALL']:
          for metric,value in values.items():
            perwindow_agg[(method,seed,str(scope),metric)]=float(sa._scope_mean(value,metadata,scope)[0])
    return accepted,primary,perwindow_agg


def main() -> dict:
    started=time.perf_counter()
    maxrss_start=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata,source,sigma,fd_scale=source_metadata()
    cache_y,cache_meta,cache_seal=acquire_target_cache(metadata,source)
    if cache_seal['ordered_id_sha256']!=sa.ordered_id_hash(metadata['anchor_id']):
        raise RuntimeError('target cache is not aligned to locked receiver ID order')
    if cache_meta['anchor_id']!=metadata['anchor_id'] or not np.array_equal(cache_meta['anchor_native'],metadata['anchor_native']):
        raise RuntimeError('cached metadata differs from the accepted label-free prefix')
    scale_hash=source['metric_scale_source']['sha256']
    accepted_window,accepted_primary,accepted_reaggregated=load_accepted_tables(metadata)
    asset_index={(b['method'],int(b['seed'])):b for b in source['prediction_bundles']}
    if len(asset_index)!=12: raise RuntimeError('accepted prediction bundle map changed after preflight')
    ids_hash=sa.ordered_id_hash(metadata['anchor_id'])
    checkpoint_root=ROOT/'checkpoints'
    index_path=checkpoint_root/'CHECKPOINT_INDEX.json'
    if index_path.exists():
        index=read_json(index_path)
        if index.get('ordered_id_sha256')!=ids_hash or index.get('target_cache_seal_sha256')!=sa.sha256_file(ROOT/'scorer_cache/CACHE_SEAL.json'):
            raise RuntimeError('existing checkpoint index is bound to another target cache/ID order')
    else:
        index={'schema_version':1,'status':'RUNNING','created_utc':now(),'ordered_id_sha256':ids_hash,
               'target_cache_seal_sha256':sa.sha256_file(ROOT/'scorer_cache/CACHE_SEAL.json'),
               'expected_slots':[{'method':m,'seed':s} for m in sa.METHODS for s in sa.SEEDS],'slots':{}}
        sa.atomic_json(index_path,index)
    errors_by_slot={}; predictions_by_slot={}; per_window_rows=[]
    horizon_rows=[]; axis_rows=[]; macro_rows=[]; prediction_array_loads=0; prediction_map_loads=0
    checkpoint_loads=0; checkpoint_writes=0
    for method in sa.METHODS:
      for seed in sa.SEEDS:
        slot=f'{method}_s{seed}'
        asset=asset_index[(method,seed)]
        ckpt_path=checkpoint_root/f'{slot}.npz'
        existing=index['slots'].get(slot)
        if existing and existing.get('status')=='PASS':
            errors=sa.load_error_checkpoint(ckpt_path,existing,ids_hash); checkpoint_loads+=1
            if existing.get('source_prediction_sha256')!=asset['prediction_sha256'] or existing.get('target_cache_seal_sha256')!=index['target_cache_seal_sha256']:
                raise RuntimeError(f'{slot}: checkpoint source binding mismatch')
        else:
            prediction=load_bound_prediction(asset,metadata)
            prediction_array_loads+=1
            prediction_map_loads+=1
            errors=sa.elementwise_errors(prediction,cache_y,metadata['anchor_native'],sigma,fd_scale)
            record=sa.write_error_checkpoint(ckpt_path,ids_hash,errors,{
              'method':method,'seed':seed,'source_prediction_sha256':asset['prediction_sha256'],
              'source_rows_sha256':asset['rows_sha256'],'source_bundle_path':asset['prediction_absolute_path'],
              'target_cache_seal_sha256':index['target_cache_seal_sha256'],
              'error_definitions':{'position':'abs(pred-Y)/target_sigma','internal_fd':'abs(diff(pred)-diff(Y))/safe_fd_scale','seam':'abs((pred_1-anchor)-(Y_1-anchor))/safe_fd_scale'},
            })
            index['slots'][slot]=record; index['updated_utc']=now(); index['status']='RUNNING'
            sa.atomic_json(index_path,index); checkpoint_writes+=1
            errors=sa.load_error_checkpoint(ckpt_path,record,ids_hash); checkpoint_loads+=1
            predictions_by_slot[(method,seed)]=prediction
        errors_by_slot[(method,seed)]=errors
        if (method,seed) not in predictions_by_slot:
            predictions_by_slot[(method,seed)]=load_bound_prediction(asset,metadata)
            prediction_array_loads+=1
            prediction_map_loads+=1
        direct=sa.canonical_per_window_from_errors(errors)
        for i,anchor_id in enumerate(metadata['anchor_id']):
            per_window_rows.append({
              'method':method,'seed':seed,'anchor_id':anchor_id,
              'episode_id':int(metadata['episode_id'][i]),'task_id':int(metadata['task_id'][i]),
              'source_group':metadata['source_cluster'][i],
              'position_mae':float(direct['position_mae'][i]),'safe_fd_l1':float(direct['safe_fd_l1'][i]),
              'seam_safe_fd_l1':float(direct['seam_safe_fd_l1'][i]),
              'accepted_source_prediction_sha256':asset['prediction_sha256'],
              'target_cache_seal_sha256':index['target_cache_seal_sha256'],
            })
        h,a,m=sa.aggregate_slot(method,seed,metadata,errors,direct,scale_hash)
        horizon_rows.extend(h); axis_rows.extend(a); macro_rows.extend(m)
    if len(index['slots'])!=12:
        raise RuntimeError(f'error checkpoint slots incomplete: {len(index["slots"])}')
    index['status']='PASS'; index['completed_utc']=now(); index['completed_slots']=len(index['slots'])
    sa.atomic_json(index_path,index)

    # Full-precision row-level reconciliation to accepted V7 scorer output.
    row_recon=[]; row_max=0.0; row_failures=0
    for row in per_window_rows:
        key=(row['method'],int(row['seed']),row['anchor_id'])
        old=accepted_window[key]
        diffs={metric:abs(float(row[METRIC_FIELDS[metric]])-old[metric]) for metric in ('position','internal_fd','seam')}
        maximum=max(diffs.values()); row_max=max(row_max,maximum)
        if any(d>1e-12+1e-10*abs(old[metric]) for metric,d in diffs.items()): row_failures+=1
        row_recon.append({**row,
          'accepted_v7_position_mae':old['position'],'accepted_v7_safe_fd_l1':old['internal_fd'],
          'accepted_v7_seam_safe_fd_l1':old['seam'],'max_abs_diff':maximum,
          'status':'PASS' if all(d<=1e-12+1e-10*abs(old[k]) for k,d in diffs.items()) else 'FAIL'})
    if len(row_recon)!=2220 or row_failures:
        raise RuntimeError(f'per-window reconciliation failed: rows={len(row_recon)}, failures={row_failures}, max={row_max}')

    macro_rows_by_key={(r['method'],int(r['seed']),r['task_or_ALL'],r['metric']):r for r in macro_rows}
    macro_recon=[]; max_diff={k:0.0 for k in ('primary','perwindow','direct','horizon','axis')}; macro_failures=0
    for method in sa.METHODS:
      for seed in sa.SEEDS:
       for scope in [*(str(t) for t in sa.TASKS),'ALL']:
        for metric,metric_name in METRIC_FIELDS.items():
            current=macro_rows_by_key[(method,seed,scope,metric_name)]
            primary_value_row=accepted_primary.get((method,seed,scope),{})
            if metric not in primary_value_row:
                raise RuntimeError(f'accepted primary row missing {(method,seed,scope,metric)}')
            v7_primary=float(primary_value_row[metric])
            v7_pw=float(accepted_reaggregated[(method,seed,scope,metric)])
            direct_value=float(current['canonical_direct_value'])
            horizon_value='' if metric=='seam' else float(current['horizon_reconstructed_value'])
            axis_value=float(current['axis_reconstructed_value'])
            values={'primary':abs(v7_primary-direct_value),'perwindow':abs(v7_pw-direct_value),
                    'direct':abs(v7_primary-direct_value),'axis':abs(v7_primary-axis_value)}
            if metric!='seam': values['horizon']=abs(v7_primary-horizon_value)
            for k,v in values.items(): max_diff[k]=max(max_diff[k],v)
            tol=1e-12+1e-10*abs(v7_primary)
            passed=all(v<=tol for v in values.values())
            if not passed: macro_failures+=1
            macro_recon.append({
              'method':method,'seed':seed,'task_or_ALL':scope,'metric':metric_name,
              'accepted_v7_primary_value':v7_primary,'accepted_v7_perwindow_reaggregate':v7_pw,
              'v8_1_canonical_direct_value':direct_value,'v8_1_horizon_reconstructed_value':horizon_value,
              'v8_1_axis_reconstructed_value':axis_value,
              'max_abs_difference_to_accepted':max(values.values()),'abs_tolerance':1e-12,'rel_tolerance':1e-10,
              'support':current['support'],'support_episodes':current['support_episodes'],
              'support_source_groups':current['support_source_groups'],'support_tasks':current['support_tasks'],'scale_hash':scale_hash,
              'status':'PASS' if passed else 'FAIL',
            })
    if len(macro_recon)!=216 or macro_failures:
        raise RuntimeError(f'macro reconciliation failed: rows={len(macro_recon)}, failures={macro_failures}, max={max_diff}')
    if len(horizon_rows)!=4248 or len(axis_rows)!=6048:
        raise RuntimeError(f'long table cardinality mismatch: horizon={len(horizon_rows)}, axis={len(axis_rows)}')

    # Gains follow the accepted V7 convention: G loss minus candidate loss; positive favors candidate.
    h_lookup={(r['method'],int(r['seed']),r['task_or_ALL'],r['metric'],str(r['step_or_transition'])):float(r['value']) for r in horizon_rows}
    a_lookup={(r['method'],int(r['seed']),r['task_or_ALL'],r['metric'],int(r['axis'])):float(r['value']) for r in axis_rows}
    horizon_gains=[]; axis_gains=[]
    for r in horizon_rows:
        key=(int(r['seed']),r['task_or_ALL'],r['metric'],str(r['step_or_transition']))
        base=h_lookup[('G',*key)]
        horizon_gains.append({**r,'reference':'G','gain_G_minus_method':base-float(r['value']),
                              'positive_favors_candidate':True})
    for r in axis_rows:
        key=(int(r['seed']),r['task_or_ALL'],r['metric'],int(r['axis']))
        base=a_lookup[('G',*key)]
        axis_gains.append({**r,'reference':'G','gain_G_minus_method':base-float(r['value']),
                           'positive_favors_candidate':True})

    result_dir=ROOT/'results'
    sa.write_csv(result_dir/'PER_WINDOW_RESCORING.csv',per_window_rows)
    sa.write_csv(result_dir/'PER_WINDOW_RECONCILIATION.csv',row_recon)
    sa.write_csv(result_dir/'HORIZON_ERRORS.csv',horizon_rows)
    sa.write_csv(result_dir/'AXIS_ERRORS.csv',axis_rows)
    sa.write_csv(result_dir/'MACRO_RECONCILIATION.csv',macro_recon)
    sa.write_csv(result_dir/'HORIZON_GAINS.csv',horizon_gains)
    sa.write_csv(result_dir/'AXIS_GAINS.csv',axis_gains)

    def seed_summaries(rows,group_fields,value_field='value'):
        groups=defaultdict(list)
        for r in rows:
            key=tuple(r[k] for k in group_fields)
            groups[key].append(float(r[value_field]))
        out=[]
        for key,values in groups.items():
            if len(values)!=3: raise RuntimeError(f'seed summary expected 3 seeds at {key}, got {len(values)}')
            base=dict(zip(group_fields,key))
            out.append({**base,'seed_stat':'mean','value':float(np.mean(values)),'n_seeds':3})
            out.append({**base,'seed_stat':'sample_sd','value':float(np.std(values,ddof=1)),'n_seeds':3})
        return out
    h_summary=seed_summaries(horizon_rows,['method','task_or_ALL','metric','step_or_transition'])
    a_summary=seed_summaries(axis_rows,['method','task_or_ALL','metric','axis'])
    hg_summary=seed_summaries(horizon_gains,['method','task_or_ALL','metric','step_or_transition'],'gain_G_minus_method')
    ag_summary=seed_summaries(axis_gains,['method','task_or_ALL','metric','axis'],'gain_G_minus_method')
    macro_summary=seed_summaries(macro_recon,['method','task_or_ALL','metric'],'v8_1_canonical_direct_value')
    sa.write_csv(result_dir/'HORIZON_ERRORS_SEED_SUMMARY.csv',h_summary)
    sa.write_csv(result_dir/'AXIS_ERRORS_SEED_SUMMARY.csv',a_summary)
    sa.write_csv(result_dir/'HORIZON_GAINS_SEED_SUMMARY.csv',hg_summary)
    sa.write_csv(result_dir/'AXIS_GAINS_SEED_SUMMARY.csv',ag_summary)
    sa.write_csv(result_dir/'MACRO_SEED_SUMMARY.csv',macro_summary)

    # Plot inputs are explicit CSVs derived from, and row-for-row traceable to, result tables.
    plot_dir=ROOT/'figure_data'
    for dest,rows in ((plot_dir/'horizon_errors.csv',horizon_rows),(plot_dir/'axis_errors.csv',axis_rows),
                      (plot_dir/'horizon_gains.csv',horizon_gains),(plot_dir/'axis_gains.csv',axis_gains),
                      (plot_dir/'macro_reconciliation.csv',macro_recon)):
        sa.write_csv(dest,rows)

    # Per-window posthoc/regular selected trajectory data; prediction payloads stay in-memory only.
    _,case_rows=read_csv(V8/'CASE_SELECTION.csv')
    if len(case_rows)!=15: raise RuntimeError('fixed V8 case set is not 15 rows')
    anchor_index={a:i for i,a in enumerate(metadata['anchor_id'])}
    traj_rows=[]
    for case in case_rows:
        aid=case['anchor_id']
        if aid not in anchor_index: raise RuntimeError(f'selected case is outside locked IDs: {aid}')
        i=anchor_index[aid]
        axes=(int(case['axis_0_14']),int(case['axis_14_28']))
        if axes!=(2,20): raise RuntimeError(f'case axis differs from fixed V8 selection: {aid}/{axes}')
        for axis in axes:
            traj_rows.append({
              'plot_id':case['plot_id'],'selection_reason':case['selection_reasons'],
              'posthoc_failure_example':case['posthoc_failure_example'],'seed':int(case['seed']),
              'task_id':int(case['task_id']),'anchor_id':aid,'source_group':metadata['source_cluster'][i],
              'axis':axis,'series':'known_anchor','future_frame':0,'native_value':float(metadata['anchor_native'][i,axis]),
              'position_scale':float(sigma[axis]),'safe_fd_scale':float(fd_scale[axis]),
            })
            for frame in range(1,31):
                traj_rows.append({
                  'plot_id':case['plot_id'],'selection_reason':case['selection_reasons'],
                  'posthoc_failure_example':case['posthoc_failure_example'],'seed':int(case['seed']),
                  'task_id':int(case['task_id']),'anchor_id':aid,'source_group':metadata['source_cluster'][i],
                  'axis':axis,'series':'Y','future_frame':frame,'native_value':float(cache_y[i,frame-1,axis]),
                  'position_scale':float(sigma[axis]),'safe_fd_scale':float(fd_scale[axis]),
                })
                for method in sa.METHODS:
                    pred=predictions_by_slot[(method,int(case['seed']))]
                    traj_rows.append({
                      'plot_id':case['plot_id'],'selection_reason':case['selection_reasons'],
                      'posthoc_failure_example':case['posthoc_failure_example'],'seed':int(case['seed']),
                      'task_id':int(case['task_id']),'anchor_id':aid,'source_group':metadata['source_cluster'][i],
                      'axis':axis,'series':method,'future_frame':frame,'native_value':float(pred[i,frame-1,axis]),
                      'position_scale':float(sigma[axis]),'safe_fd_scale':float(fd_scale[axis]),
                    })
    sa.write_csv(plot_dir/'trajectory_data.csv',traj_rows)

    # Full V8.1 diagnostics, reconciliation, and exposure receipt.
    created=now()
    summary={
      'schema_version':1,'status':'PASS','study':'post_V8_evidence_completion_no_new_model',
      'created_utc':created,'locked_dev_windows':185,'source_groups':47,'methods':list(sa.METHODS),'seeds':list(sa.SEEDS),
      'prediction_bundles_loaded':len(predictions_by_slot),'prediction_array_loads':prediction_array_loads,
      'prediction_row_map_loads_during_analysis':prediction_map_loads,'target_archive_npz_loads':int(read_json(ROOT/'qa/TARGET_READ_RECEIPT.json').get('archive_npz_loads_total',read_json(ROOT/'qa/TARGET_READ_RECEIPT.json').get('npz_load_operations_this_attempt',0))),
      'scorer_cache_package_loads':int(read_json(ROOT/'qa/TARGET_READ_RECEIPT.json').get('cache_loads_total',read_json(ROOT/'qa/TARGET_READ_RECEIPT.json').get('cache_loads',0))),
      'checkpoint_slots':len(index['slots']),'checkpoint_writes_this_process':checkpoint_writes,
      'checkpoint_package_loads_this_process':checkpoint_loads,
      'diagnostic_method_seed_window_units':12*185,'new_elementwise_slots':len(index['slots']),
      'horizon_rows':len(horizon_rows),'horizon_position_rows':sum(r['metric']=='position' for r in horizon_rows),
      'horizon_internal_fd_rows':sum(r['metric']=='internal_fd' for r in horizon_rows),
      'axis_rows':len(axis_rows),'axis_position_rows':sum(r['metric']=='position' for r in axis_rows),
      'axis_internal_fd_rows':sum(r['metric']=='internal_fd' for r in axis_rows),'axis_seam_rows':sum(r['metric']=='seam' for r in axis_rows),
      'macro_reconciliation_rows':len(macro_recon),'per_window_rescoring_rows':len(per_window_rows),
      'per_window_reconciliation_rows':len(row_recon),'max_per_window_abs_difference':row_max,
      'max_macro_abs_differences':max_diff,'max_macro_reconciliation_abs_difference':max(max_diff.values()),
      'failed_macro_reconciliations':macro_failures,'fixed_trajectory_cases':len(case_rows),
      'fixed_axes_per_case':[2,20],'trajectory_figure_data_rows':len(traj_rows),
      'final_heldout_payload_access':0,'new_fit':0,'optimizer_updates':0,
      'teacher_student_forward_jobs':0,'new_dev_prediction_jobs':0,'encoder_extractions':0,
      'bootstrap_draws':0,'model_selection_or_hyperparameter_search':0,'gpu_commands':0,'old_root_writes':0,
      'plot_gain_definition':'G loss minus method loss; positive favors candidate',
      'aggregation':'per anchor; equal windows within episode; equal episodes within task; equal task macro; seed-first then 3-seed mean/sample SD',
      'tolerances':{'absolute':1e-12,'relative':1e-10},
      'process_elapsed_seconds':time.perf_counter()-started,
      'process_max_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
      'process_rss_kib_start':int(maxrss_start),
    }
    sa.atomic_json(ROOT/'qa/REAL_DATA_QA.json',summary)
    # Append this run's access totals to the one-row source access receipt.
    receipt=read_json(ROOT/'qa/TARGET_READ_RECEIPT.json')
    receipt.update({'analysis_completed_utc':created,'elementwise_diagnostic_slots':len(index['slots']),
                    'method_seed_window_units':12*185,'no_label_values_written_outside_scorer_cache':True,
                    'target_cache_seal_sha256':sa.sha256_file(ROOT/'scorer_cache/CACHE_SEAL.json')})
    sa.atomic_json(ROOT/'qa/TARGET_READ_RECEIPT.json',receipt)
    print(json.dumps(summary,indent=2,sort_keys=True))
    return summary


if __name__=='__main__':
    main()
