"""Read-only source and prediction binding gate; never reads dev target values."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BASE = PROJECT/'reports/benchmarks'
V2 = BASE/'CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4 = BASE/'CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V5 = BASE/'CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
V7 = BASE/'CORE_B_V7_SINGLE_HEAD_SOFT_TARGET_V1_20261015T112823Z'
V8 = BASE/'CORE_B_V8_FROZEN_TRANSFER_DIAGNOSIS_V1_20261016T045057Z'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import scorer_adapter as sa

EXPECTED = {
  'V7_manifest':'58c50a9928106c2cf95fe78c5d3a0bdaa05c050a271803d01a6ef8e011810a29',
  'V8_manifest':'7778803b295f0bd5c3563fa69b940e740ced7b77a7e77c5650f4a0790b88d1c0',
  'V5_manifest':'5866688b48bef3ed940920affc15081dea052c51eca6eb86e7bcfd3a09c5acfa',
  'V8_source_map':'0f35f976f3702df68273350427116f341dea02da2e14c9331fe803b681e8b475',
  'V7_source_map':'9dec27b5f36dcfbcd6ee2c17603b3f13958a76a38cb5c02c1850e1c98387170a',
  'V7_prediction_seal':'c28ee3038e4a41e3a8dbae1a85e2ca7ed5afd9d545dcd24b5b85e6ffb4fc91d7',
  'V7_per_window':'271ce67b3d36fdb298ea55cb540540022581e645d0ba5dd69cedac07b80cbcae',
  'V7_primary_task':'c739cdf2c29a4a32db97e557cc6ac2b71026a3d855e4d0289c7b9e4de9a55f4e',
  'V4_prefix':'8cc74fef6d794320b33bbb6d43c0a93230b21373af1c2b910cb74d1057d09090',
  'V4_scales':'dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32',
  'V2_dev_target':'233a0cdb31d1ddd7cea6ecfaa5e9d02e9c42e9822af7fe488c7fe5a40febee27',
  'dev_ids':'5c6e5ef759ce864e69a186b7d2fe6063b3c9e1bc8d5c98640d7a2873d1ba3025',
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def manifest_entries(root: Path) -> dict[str,str]:
    entries={}
    for line in (root/'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest, rel=line.split('  ',1)
        entries[rel]=digest
    return entries


def verify_root(root: Path, expected: str, label: str) -> dict:
    actual=sa.sha256_file(root/'CHECKSUMS.sha256')
    if actual!=expected:
        raise RuntimeError(f'{label} manifest hash changed: {actual} != {expected}')
    receipt=read_json(root/'qa/FINAL_CHECKSUM_VERIFICATION.json')
    if receipt.get('status')!='PASS' or receipt.get('manifest_sha256')!=expected or receipt.get('mismatches') or receipt.get('missing'):
        raise RuntimeError(f'{label} final checksum receipt failed')
    return {'root':str(root),'manifest_sha256':actual,'manifest_entries':receipt['manifest_entry_count'],
            'receipt_path':str(root/'qa/FINAL_CHECKSUM_VERIFICATION.json'),
            'receipt_sha256':sa.sha256_file(root/'qa/FINAL_CHECKSUM_VERIFICATION.json'),
            'receipt_status':receipt['status']}


def verify_selected_file(root: Path, path: Path, expected: str | None=None, *, check_manifest: bool=True) -> dict:
    rel=path.relative_to(root).as_posix()
    actual=sa.sha256_file(path)
    if expected and actual!=expected:
        raise RuntimeError(f'selected dependency hash mismatch: {rel}: {actual} != {expected}')
    if check_manifest:
        entries=manifest_entries(root)
        if rel not in entries:
            raise RuntimeError(f'selected dependency absent from its source manifest: {rel}')
        if entries[rel]!=actual:
            raise RuntimeError(f'selected dependency manifest mismatch: {rel}')
    return {'source_root':str(root),'relative_path':rel,'absolute_path':str(path),
            'sha256':actual,'manifest_verified':check_manifest}


def csv_rows(path: Path) -> tuple[list[str],list[dict[str,str]]]:
    with path.open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle)
        return list(reader.fieldnames or []),list(reader)


def main() -> dict:
    start=time.perf_counter()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name,root in (('V7',V7),('V8',V8),('V5',V5)):
        if not root.is_dir(): raise FileNotFoundError(root)
    root_records={
      'V7':verify_root(V7,EXPECTED['V7_manifest'],'V7'),
      'V8':verify_root(V8,EXPECTED['V8_manifest'],'V8'),
      'V5':verify_root(V5,EXPECTED['V5_manifest'],'V5'),
    }
    v8_map_path=V8/'inventory/SOURCE_MAP.json'
    v7_map_path=V7/'protocol/SOURCE_MAP.json'
    v8_map=read_json(v8_map_path); v7_map=read_json(v7_map_path)
    deps=[]
    deps.append(verify_selected_file(V8,v8_map_path,EXPECTED['V8_source_map']))
    deps.append(verify_selected_file(V7,v7_map_path,EXPECTED['V7_source_map']))

    # Verify prior V7 access receipt and binding, while keeping this run's label load at zero.
    target_src=v8_map['original_dev_target_source']
    target_path=Path(target_src['path'])
    if target_path != V2/'data/DEV_TARGETS_SCORER_ONLY.npz':
        raise RuntimeError('target path does not equal accepted V2 scorer-only artifact')
    target_hash=sa.sha256_file(target_path)  # V7's accepted reader hashes the archive before np.load.
    if target_hash!=EXPECTED['V2_dev_target'] or target_src['sha256']!=target_hash:
        raise RuntimeError('dev target source archive hash mismatch')
    v7_access=read_json(V7/'qa/DEV_TARGET_ACCESS_RECEIPT.json')
    if (v7_access.get('target_archive_path')!=str(target_path)
        or v7_access.get('target_archive_sha256')!=target_hash
        or int(v7_access.get('unique_target_windows_loaded',-1))!=185
        or int(v7_access.get('final_heldout_payload_access',-1))!=0):
        raise RuntimeError('accepted V7 dev target access receipt does not bind the selected source')
    deps.append(verify_selected_file(V7,V7/'qa/DEV_TARGET_ACCESS_RECEIPT.json',
                                     v8_map['original_dev_target_source']['V7_access_receipt_sha256']))
    v4_data=v7_map['data']
    prefix_path=Path(v4_data['dev_prefix_npz'])
    scale_path=Path(v4_data['metric_scale_lock'])
    prefix_dep=verify_selected_file(V4,prefix_path,EXPECTED['V4_prefix'])
    scale_dep=verify_selected_file(V4,scale_path,EXPECTED['V4_scales'])
    deps.extend((prefix_dep,scale_dep))
    scales=read_json(scale_path)
    sigma,fd_scale=sa.validate_scales(scales['target_sigma'],scales['safe_fd_scale'])

    # Only label-free, explicitly listed receiver/anchor fields are read here.
    prefix_pack=sa.read_npz_fields(prefix_path,sa.PREFIX_FIELDS)
    metadata=sa.validate_metadata(prefix_pack,expected_tasks=sa.TASKS)
    ids=metadata['anchor_id']
    id_sha=sa.ordered_id_hash(ids)
    if len(ids)!=185 or id_sha!=EXPECTED['dev_ids'] or id_sha!=v8_map['dev_ordered_anchor_id_sha256']:
        raise RuntimeError('accepted 185 dev receiver ID order does not match V7/V8 source maps')
    task_support={str(t):int(np.sum(metadata['task_id']==t)) for t in sa.TASKS}
    if task_support!={'119':48,'94':48,'90':41,'141':20,'230':28}:
        raise RuntimeError(f'locked task support changed: {task_support}')
    if len(set(metadata['source_cluster']))!=47 or len(set(metadata['episode_id'].tolist()))!=47:
        raise RuntimeError('locked dev group/episode support differs from V7 acceptance')

    v7_seal_path=V7/'qa/PREDICTION_SEAL.json'
    v7_seal=read_json(v7_seal_path)
    if v7_seal.get('status')!='PASS' or len(v7_seal.get('entries',[]))!=9 or v7_seal.get('dev_target_values_read_before_seal')!=0:
        raise RuntimeError('V7 prediction seal is not a complete pre-label 9-bundle seal')
    deps.append(verify_selected_file(V7,v7_seal_path,EXPECTED['V7_prediction_seal']))
    accepted=v8_map['accepted_prediction_assets']
    bundles=[e for e in accepted if e.get('split')=='dev' and e.get('method') in {'GT_RETRAIN','SELF_SOFT','MIX_SOFT','T'}]
    by_key={}
    method_map={'GT_RETRAIN':'G','SELF_SOFT':'U','MIX_SOFT':'M','T':'T'}
    for e in bundles:
        method=method_map[e['method']]; seed=int(e['seed']); key=(method,seed)
        if key in by_key: raise RuntimeError(f'duplicate accepted prediction slot {key}')
        by_key[key]=e
    required={(m,s) for m in sa.METHODS for s in sa.SEEDS}
    if set(by_key)!=required or len(by_key)!=12:
        raise RuntimeError(f'accepted prediction slots incomplete: {sorted(set(by_key))}')

    asset_records=[]
    source_roots={'V7':V7,'V5':V5,'V2':V2}
    for method in sa.METHODS:
      for seed in sa.SEEDS:
        e=by_key[(method,seed)]
        asset_root=source_roots[e['source_root']]
        prediction_path=Path(e['prediction_path']); rows_path=Path(e['rows_path']); seal_path=Path(e['seal_path'])
        if list(e['prediction_shape'])!=[185,30,28] or int(e['receiver_rows'])!=185:
            raise RuntimeError(f'{method}/{seed}: prediction source shape/count mismatch')
        pred_dep=verify_selected_file(asset_root,prediction_path,e['prediction_sha256'])
        row_dep=verify_selected_file(asset_root,rows_path,e['rows_sha256'])
        seal_dep=verify_selected_file(asset_root,seal_path,e['seal_sha256'])
        fields,rows=csv_rows(rows_path)
        if len(rows)!=185 or len({r.get('anchor_id','') for r in rows})!=185:
            raise RuntimeError(f'{method}/{seed}: row map is not 185 unique IDs')
        row_ids=[r['anchor_id'] for r in rows]
        row_id_hash=sa.ordered_id_hash(row_ids)
        if row_id_hash!=e['ordered_anchor_id_sha256']:
            raise RuntimeError(f'{method}/{seed}: ordered row-map hash mismatch')
        # This validates IDs and metadata against the accepted prefix, without decoding any prediction values.
        sa.align_prediction_by_id(np.zeros((185,30,28),dtype=np.float64),rows,metadata)
        expected_source_method={'G':'GT_RETRAIN','U':'SELF_SOFT','M':'MIX_SOFT','T':'T'}[method]
        if method!='T':
            matching=[x for x in v7_seal['entries'] if x.get('program_id')==expected_source_method and int(x.get('seed',-1))==seed]
            if len(matching)!=1 or matching[0].get('prediction_sha256')!=e['prediction_sha256'] or matching[0].get('prediction_path')!=e['prediction_path']:
                raise RuntimeError(f'{method}/{seed}: prediction asset disagrees with accepted V7 seal')
        asset_records.append({
          'method':method,'source_method':e['method'],'seed':seed,'source_root':e['source_root'],
          'source_root_manifest_sha256':root_records[e['source_root']]['manifest_sha256'],
          'prediction_relative_path':pred_dep['relative_path'],'prediction_absolute_path':str(prediction_path),
          'prediction_sha256':pred_dep['sha256'],'prediction_dtype':e['prediction_dtype'],
          'prediction_shape':list(e['prediction_shape']),'prediction_field':'prediction_native',
          'rows_relative_path':row_dep['relative_path'],'rows_absolute_path':str(rows_path),
          'rows_sha256':row_dep['sha256'],'ordered_anchor_id_sha256':row_id_hash,
          'rows':len(rows),'seal_relative_path':seal_dep['relative_path'],'seal_sha256':seal_dep['sha256'],
          'metadata_join':'PASS','prediction_payload_decoded':False,
        })
        deps.extend((pred_dep,row_dep,seal_dep))

    # Accepted full-precision loss rows provide both row-level reconciliation and fixed case selection.
    accepted_loss_paths=(
      (V7/'results/PER_WINDOW_METRICS.csv',EXPECTED['V7_per_window']),
      (V7/'results/PRIMARY_TASK_SEED.csv',EXPECTED['V7_primary_task']),
    )
    for path,digest in accepted_loss_paths:
        deps.append(verify_selected_file(V7,path,digest))
    _,pw=csv_rows(accepted_loss_paths[0][0])
    selected_pw=[r for r in pw if r['method'] in {'GT_RETRAIN','SELF_SOFT','MIX_SOFT','T'} and int(r['seed']) in sa.SEEDS]
    _,pt=csv_rows(accepted_loss_paths[1][0])
    selected_pt=[r for r in pt if r['method'] in {'GT_RETRAIN','SELF_SOFT','MIX_SOFT','T'} and int(r['seed']) in sa.SEEDS]
    if len(selected_pw)!=2220 or len(selected_pt)!=72:
        raise RuntimeError(f'accepted loss coverage changed: window={len(selected_pw)}, task_seed={len(selected_pt)}')
    if any(r['task_id'] not in {str(t) for t in sa.TASKS} for r in selected_pw):
        raise RuntimeError('selected accepted per-window loss has an unlocked task')
    if any(r['task_id'] not in {*[str(t) for t in sa.TASKS],'macro'} for r in selected_pt):
        raise RuntimeError('selected accepted task/seed loss has an unlocked scope')

    case_path=V8/'CASE_SELECTION.csv'
    case_dep=verify_selected_file(V8,case_path)
    axis_score_path=V8/'figure_data/train_axis_scores.csv'
    axis_dep=verify_selected_file(V8,axis_score_path)
    deps.extend((case_dep,axis_dep))
    _,cases=csv_rows(case_path)
    if len(cases)!=15 or any(r.get('trajectory_plot_status')!='UNAVAILABLE_DEV_Y_CLOSED' for r in cases):
        raise RuntimeError('V8 fixed case set/status differs from the sealed selection record')

    status={
      'schema_version':1,'status':'PASS','study':'post_V8_evidence_completion_no_new_model',
      'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cpu_only':True,
      'old_roots_read_only':True,'target_npz_load_operations':0,'target_label_values_read':0,
      'target_archive_hash_passes':1,'target_source_path':str(target_path),'target_source_sha256':target_hash,
      'prefix_archive_loads':1,'prefix_fields_read':list(sa.PREFIX_FIELDS),'prefix_rows':len(ids),
      'prefix_ordered_anchor_id_sha256':id_sha,'task_support':task_support,
      'source_groups':len(set(metadata['source_cluster'])),'episodes':len(set(metadata['episode_id'].tolist())),
      'locked_scale_sha256':scale_dep['sha256'],'sigma_count':len(sigma),'safe_fd_scale_count':len(fd_scale),
      'prediction_payloads_decoded':0,'prediction_asset_file_hashes_verified':12,
      'prediction_row_maps_read_and_join_checked':12,'accepted_prediction_slots':len(asset_records),
      'accepted_per_window_rows_reused':len(selected_pw),'accepted_task_seed_rows_reused':len(selected_pt),
      'fixed_case_count':len(cases),'fixed_axis_pairs':sorted({(int(r['axis_0_14']),int(r['axis_14_28'])) for r in cases}),
      'roots':root_records,'elapsed_seconds':time.perf_counter()-start,
      'max_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
      'numpy_version':np.__version__,'python_version':sys.version.split()[0],
    }
    source_map={
      'schema_version':1,'study':'post_V8_evidence_completion_no_new_model','status':'PREFLIGHT_PASS',
      'V2_root':str(V2),'V4_root':str(V4),'V5_root':str(V5),'V7_root':str(V7),'V8_root':str(V8),
      'roots':root_records,'dev_target_source':{
        'exact_relative_path':'data/DEV_TARGETS_SCORER_ONLY.npz','absolute_path':str(target_path),
        'sha256':target_hash,'source_role':'already_exposed_dev_future_labels','field':'target_native_h30',
        'shape':[185,30,28],'V7_access_receipt_sha256':v7_access.get('target_archive_sha256'),
        'npz_load_operations_this_run':0,'hash_only_this_preflight':True,
      },
      'dev_prefix_source':prefix_dep,'accepted_id_order_sha256':id_sha,'tasks':list(sa.TASKS),
      'seeds':list(sa.SEEDS),'methods':list(sa.METHODS),'task_support':task_support,
      'metric_scale_source':{**scale_dep,'target_sigma_values':sigma.tolist(),'safe_fd_scale_values':fd_scale.tolist(),
                             'position_scale_definition':'locked V7 target_sigma; two equal coordinate blocks 0:14 and 14:28',
                             'internal_fd_scale_definition':'locked V7 safe_fd_scale on 29 first differences within future frames',
                             'seam_definition':'separate anchor-to-first-future transition under safe_fd_scale'},
      'prediction_bundles':asset_records,
      'accepted_loss_sources':[verify_selected_file(V7,p,d) for p,d in accepted_loss_paths],
      'case_selection_source':case_dep,'axis_selection_source':axis_dep,
      'unavailable_v8_target_payload_replaced_by_authorized_v8_1_read':True,
      'no_target_values_read_yet':True,
    }
    sa.atomic_json(ROOT/'inventory/SOURCE_MAP.json',source_map)
    sa.atomic_json(ROOT/'qa/PREFLIGHT.json',status)
    sa.atomic_json(ROOT/'protocol/REPAIR_SCOPE_LOCK.json',{
      'schema_version':1,'status':'PASS','study':'post_V8_evidence_completion_no_new_model',
      'locked_tasks':list(sa.TASKS),'locked_seeds':list(sa.SEEDS),'locked_methods':list(sa.METHODS),
      'locked_dev_windows':185,'locked_source_groups':47,'locked_anchor_id_sha256':id_sha,
      'metrics':['position_mae','safe_fd_l1','seam_safe_fd_l1'],
      'prediction_sources_sha256':[{'method':r['method'],'seed':r['seed'],'prediction_sha256':r['prediction_sha256'],'rows_sha256':r['rows_sha256']} for r in asset_records],
      'scale_lock_sha256':scale_dep['sha256'],'target_source_path':str(target_path),'target_source_sha256':target_hash,
      'final_heldout_payload_access':0,'new_fits':0,'optimizer_updates':0,'teacher_student_forwards':0,
      'new_dev_prediction_jobs':0,'encoder_extractions':0,'bootstrap_draws':0,'model_selection_or_search':0,
      'gpu_commands':0,'old_root_writes':0,'target_open_gate':'synthetic_typed_vector_pipeline_QA_must_pass',
    })

    ledger=[]
    for d in deps:
      ledger.append({'source_path':d['absolute_path'],'source_sha256':d['sha256'],'purpose':'locked source/prediction/row-map binding',
                     'prior_status':'accepted and sealed','V8_1_action':'verified read-only preflight','output_status':'PASS'})
    ledger.extend([
      {'source_path':str(target_path),'source_sha256':target_hash,'purpose':'authorized scorer-only dev Y for exact 185 locked anchors',
       'prior_status':'V8 exposed once but raw values not persisted','V8_1_action':'hash-verified only; await synthetic QA then cache before reduction','output_status':'NOT_OPENED'},
      {'source_path':'V8/V7 scorer-cache search in selected roots','source_sha256':'','purpose':'reuse eligible same-ID/same-role/same-source cache if present',
       'prior_status':'none found in V2/V7/V8 selected roots','V8_1_action':'new scorer-only cache required','output_status':'MISSING_REUSABLE_CACHE'},
      {'source_path':str(V7/'results/PER_WINDOW_METRICS.csv'),'source_sha256':EXPECTED['V7_per_window'],'purpose':'accepted per-window reconciliation and fixed posthoc case selection',
       'prior_status':'2220 selected method-seed-window rows exist','V8_1_action':'reuse; no re-bootstrap','output_status':'READY'},
      {'source_path':str(V7/'results/PRIMARY_TASK_SEED.csv'),'source_sha256':EXPECTED['V7_primary_task'],'purpose':'accepted task/seed/macro reconciliation',
       'prior_status':'72 selected method-seed-scope rows exist','V8_1_action':'reuse; no re-bootstrap','output_status':'READY'},
      {'source_path':str(V8/'results/HORIZON_AXIS_ERRORS.csv'),'source_sha256':sa.sha256_file(V8/'results/HORIZON_AXIS_ERRORS.csv'),
       'purpose':'V8 known gap inventory','prior_status':'4 explicit UNAVAILABLE rows; no Y-bearing trajectory','V8_1_action':'replace only in V8.1 appendix','output_status':'GAP_TO_FILL'},
    ])
    sa.write_csv(ROOT/'inventory/GAP_AND_REUSE_LEDGER.csv',ledger)
    result={**status,'output_root':str(ROOT),'source_map_sha256':sa.sha256_file(ROOT/'inventory/SOURCE_MAP.json'),
            'scope_lock_sha256':sa.sha256_file(ROOT/'protocol/REPAIR_SCOPE_LOCK.json'),
            'gap_ledger_rows':len(ledger)}
    print(json.dumps(result,indent=2,sort_keys=True))
    return result


if __name__=='__main__':
    main()
