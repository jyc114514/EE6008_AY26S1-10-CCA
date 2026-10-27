#!/usr/bin/env python3
from __future__ import annotations
import os
import csv, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / 'reports/benchmarks'
V2 = BENCH / 'CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V3 = BENCH / 'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'
V4 = BENCH / 'CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
ROOT = Path(__file__).resolve().parents[1]
TASKS = [119, 94, 90, 141, 230]
SEEDS = [202610131, 202610132, 202610133]
METHODS = {
    'H_SV': 'V - mean_t(V) + mean_t(S)',
    'H_VS': 'S - mean_t(S) + mean_t(V)',
    'H_AV': 'V - mean_t(V) + mean_t(A)',
    'H_SC': 'C - mean_t(C) + mean_t(S)',
    'H_PV': 'V - mean_t(V) + a',
    'MIX50': '0.5*S + 0.5*V',
}

def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
def sha_file(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def sha_text(s: str) -> str: return hashlib.sha256(s.encode('utf-8')).hexdigest()
def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
def read_csv(p: Path):
    with p.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))
def write_csv(p: Path, rows, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def require_hash(path: Path, expected: str, role: str):
    actual = sha_file(path)
    if actual != expected: raise RuntimeError(f'{role} hash mismatch: {path} expected={expected} actual={actual}')
    return actual
def read_rows(path: Path): return read_csv(path)
def safetensors_array(path: Path, key='prediction_native'):
    d = load_file(str(path))
    if key not in d: raise RuntimeError(f'missing {key} in {path}')
    return np.asarray(d[key])
def rows_digest(ids): return sha_text('\n'.join(ids) + '\n')

def main():
    if not ROOT.is_dir(): raise RuntimeError('V5 root must exist and be unique')
    if (ROOT / 'protocol/FORMULA_LOCK.json').exists(): raise RuntimeError('formula lock already exists; do not overwrite')
    v4seal_path = V4 / 'protocol/PREDICTION_BATCH_SEAL.json'
    v3seal_path = V3 / 'protocol/PREDICTION_BATCH_SEAL.json'
    v2seal_path = V2 / 'protocol/PREDICTION_BATCH_SEAL.json'
    v4seal = json.loads(v4seal_path.read_text()); v3seal = json.loads(v3seal_path.read_text()); v2seal = json.loads(v2seal_path.read_text())
    v4status = json.loads((V4/'qa/FINAL_ACCEPTANCE.json').read_text())
    v4owner = json.loads((V4/'coordination/OWNER_STATE.json').read_text())
    if v4status.get('status') != 'complete' or int(v4status.get('final_heldout_payload_access', -1)) != 0: raise RuntimeError('V4 terminal acceptance not complete/locked')
    if v4owner.get('active_jobs') != 0 or v4owner.get('owner_lock_acquired') is True or v4owner.get('all_gpu_leases_released') is not True: raise RuntimeError('V4 owner/writer is not idle')
    if v4seal.get('status') != 'SEALED_BEFORE_TARGET_ACCESS' or v4seal.get('dev_target_reader_called') is not False: raise RuntimeError('V4 prediction seal invalid')
    if v3seal.get('status') != 'SEALED_BEFORE_TARGET_ACCESS' or v3seal.get('dev_target_reader_called') is not False: raise RuntimeError('V3 prediction seal invalid')
    if v2seal.get('status') != 'SEALED_BEFORE_TARGET_ACCESS' or int(v2seal.get('dev_target_labels_read_before_seal', -1)) != 0: raise RuntimeError('V2 prediction seal invalid')

    norm_path = V2/'normalizers/NORMALIZER_FORMAL.json'
    metric_path = V2/'protocol/METRIC_SCALE_LOCK.json'
    split_path = V2/'protocol/INNER_SPLIT.csv'
    support_path = V2/'inventory/DEV_SUPPORT_V2.csv'
    prefix_path = V2/'data/DEV_PREFIXES_NO_LABELS.npz'
    source_scores_path = V4/'results/PER_WINDOW_LOSSES.csv'
    norm = json.loads(norm_path.read_text()); metric = json.loads(metric_path.read_text())
    if norm.get('status') != 'LOCKED_FORMAL' or norm.get('final_heldout_payload_access') != 0: raise RuntimeError('formal normalizer not locked')
    if metric.get('status') != 'LOCKED' or metric.get('final_heldout_payload_access') != 0: raise RuntimeError('metric lock not locked')
    norm_sha = sha_file(norm_path); metric_sha = sha_file(metric_path)
    if norm_sha != '22f66a54f6106de68c1a0bf0aa3a46ca3340c3077b4121c6a0fdf9d83528d65d': raise RuntimeError('formal normalizer hash differs from accepted source map')
    if metric_sha != 'dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32': raise RuntimeError('metric lock hash differs from accepted source map')
    if sha_file(split_path) != 'ba31002f0a17b826f57cec93c7dfaa484d76b01a07fec5b9cda10de03fe041a8': raise RuntimeError('corrected V2 literal split hash mismatch')
    if source_scores_path.exists() is False: raise RuntimeError('accepted V4 per-window reference scores missing')

    # Label-free receiver metadata and current command anchors only.
    with np.load(prefix_path, allow_pickle=False) as z:
        forbidden = {'target_native_h30', 'future_target', 'target'} & set(z.files)
        if forbidden: raise RuntimeError(f'label-free prefix contains forbidden outcome keys: {sorted(forbidden)}')
        prefix = {k: z[k].copy() for k in ('anchor_id','episode_id','task_id','source_cluster','anchor_frame','anchor_native')}
    ids_prefix = [str(x) for x in prefix['anchor_id'].tolist()]
    if len(ids_prefix) != 185 or len(set(ids_prefix)) != 185: raise RuntimeError('label-free receiver support invalid')
    prefix_meta = {ids_prefix[i]: (str(prefix['episode_id'][i]),str(prefix['task_id'][i]),str(prefix['source_cluster'][i])) for i in range(len(ids_prefix))}

    # V4 case is retained for task 119; other cases use a fixed metadata-only rule.
    v4_case = 'e001712_a00_f000119'
    if v4_case not in prefix_meta or prefix_meta[v4_case][1] != '119': raise RuntimeError('accepted V4 case anchor is not in this receiver set')
    selected_cases = {119: v4_case}
    for task in TASKS:
        if task == 119: continue
        cand = [i for i,a in enumerate(ids_prefix) if int(prefix['task_id'][i]) == task]
        if not cand: raise RuntimeError(f'no receiver for task {task}')
        cand.sort(key=lambda i: (int(prefix['episode_id'][i]), int(prefix['anchor_frame'][i]), ids_prefix[i]))
        selected_cases[task] = ids_prefix[cand[0]]

    v2_entries = v2seal['prediction_entries']
    v4_entries = v4seal['new_prediction_entries']
    v3_entries = v3seal['new_prediction_entries']
    v4_reuse = v4seal['reused_B_MATCHED_entries']
    sources = []
    def add_source(role, seed, entry, version, condition='MATCHED'):
        if version == 'V2':
            p = Path(entry['prediction_path']); rp = Path(entry['rows_path']); predhash=entry['prediction_sha256']; rowhash=entry['rows_sha256']; sealpath=v2seal_path
            model=entry['model_id']; profile='accepted V2 formal G-S/G-V native command prediction; same corrected outer train, H30 Full-28';
        elif version == 'V4':
            p = Path(entry['prediction_path']); rp = Path(entry['prediction_rows_path']); predhash=entry['prediction_sha256']; rowhash=entry['prediction_rows_sha256']; sealpath=v4seal_path
            model=entry['model_id']; profile='accepted V4 matched RGB program native command prediction; condition=MATCHED';
        else:
            p = Path(entry['prediction_path']); rp = Path(entry['prediction_rows_path']); predhash=entry['prediction_sha256']; rowhash=entry['prediction_rows_sha256']; sealpath=v3seal_path
            model=entry['model_id']; profile='accepted V3 fixed readout bias reference native command prediction';
        require_hash(p, predhash, f'{role} seed {seed} array')
        require_hash(rp, rowhash, f'{role} seed {seed} rows')
        arr = safetensors_array(p)
        rows = read_rows(rp)
        if arr.shape != (185,30,28) or not np.isfinite(arr).all(): raise RuntimeError(f'{role}/{seed} array shape/finite failure: {arr.shape}')
        if len(rows) != 185 or len({r['anchor_id'] for r in rows}) != 185: raise RuntimeError(f'{role}/{seed} row IDs invalid')
        row_by_id = {r['anchor_id']: i for i,r in enumerate(rows)}
        if set(row_by_id) != set(ids_prefix): raise RuntimeError(f'{role}/{seed} receiver IDs differ')
        for r in rows:
            if (str(r['episode_id']),str(r['task_id']),str(r['source_cluster'])) != prefix_meta[r['anchor_id']]: raise RuntimeError(f'{role}/{seed} row metadata mismatch {r["anchor_id"]}')
        if version in ('V2','V4') and entry.get('normalizer_sha256') not in (None, norm_sha): raise RuntimeError(f'{role}/{seed} normalizer hash mismatch')
        if version == 'V4' and condition != 'MATCHED': raise RuntimeError(f'{role} donor-map condition prohibited')
        complete_path = Path(entry.get('completion_path', p.parent/'PREDICTION_COMPLETE.json'))
        complete_sha = sha_file(complete_path) if complete_path.exists() else None
        complete = json.loads(complete_path.read_text()) if complete_path.exists() else {}
        if complete and complete.get('status') != 'PASS': raise RuntimeError(f'{role}/{seed} completion not PASS')
        sources.append({
            'role':role,'seed':int(seed),'source_version':version,'model_id':model,'condition':condition,
            'prediction_path':str(p),'prediction_sha256':predhash,'prediction_rows_path':str(rp),'prediction_rows_sha256':rowhash,
            'completion_path':str(complete_path) if complete_path.exists() else None,'completion_sha256':complete_sha,
            'seal_path':str(sealpath),'seal_sha256':sha_file(sealpath),'normalizer_path':str(norm_path),'normalizer_sha256':norm_sha,
            'native_field':'prediction_native','dtype':str(arr.dtype),'shape':list(arr.shape),'ordered_source_ids_sha256':rows_digest([r['anchor_id'] for r in rows]),
            'source_profile':profile,'elapsed_seconds':complete.get('elapsed_seconds'),'physical_gpu_historical':complete.get('physical_gpu'),
            'source_rows':rows,'source_index_by_id':row_by_id
        })

    for seed in SEEDS:
        for role, model in [('S_OPT','G-S'),('V_MATCHED','G-V')]:
            matches = [e for e in v2_entries if e['model_id']==model and int(e['seed'])==seed]
            if len(matches)!=1: raise RuntimeError(f'V2 source not unique {model}/{seed}')
            add_source(role,seed,matches[0],'V2')
        for role, model in [('A_COMMON','A_STATE_COMMON'),('C_REASSIGNED','C_STATE_NEIGHBOR_REASSIGNED')]:
            matches=[e for e in v4_entries if e.get('model_id')==model and int(e['seed'])==seed and e.get('condition')=='MATCHED']
            if len(matches)!=1: raise RuntimeError(f'V4 source not unique matched {model}/{seed}')
            add_source(role,seed,matches[0],'V4','MATCHED')
        for role, method in [('BIAS_STATE','BIAS_STATE'),('BIAS_VIDEO','BIAS_VIDEO')]:
            matches=[e for e in v3_entries if e.get('method_id')==method and int(e['seed'])==seed]
            if len(matches)!=1: raise RuntimeError(f'V3 bias source not unique {method}/{seed}')
            add_source(role,seed,matches[0],'V3')

    # V4 explicitly accepted B_M as a reuse of V2 G-V; verify each link exactly.
    for seed in SEEDS:
        v2v = next(s for s in sources if s['role']=='V_MATCHED' and s['seed']==seed)
        links=[e for e in v4_reuse if int(e['seed'])==seed]
        if len(links)!=1 or links[0]['prediction_sha256'] != v2v['prediction_sha256'] or links[0]['prediction_path'] != v2v['prediction_path']:
            raise RuntimeError(f'V4 B_M reuse link does not match accepted V2 G-V {seed}')

    # Split, normalizer, metric-scale, and accepted reference-score provenance.
    v4_source_map_path=V4/'inventory/SOURCE_MAP.json'
    v4_primary_path=V4/'results/PRIMARY_TASK_SEED.csv'
    source_reference_files=[
        {'path':str(p),'sha256':sha_file(p),'role':role}
        for p,role in [(split_path,'corrected literal split'),(support_path,'accepted dev support'),(norm_path,'formal normalizer'),
                       (metric_path,'canonical metric scale lock'),(prefix_path,'label-free dev prefixes/current anchor'),
                       (source_scores_path,'accepted V4 per-window reference scores'),(v4_primary_path,'accepted V4 primary task-seed table'),
                       (v4_source_map_path,'accepted V4 source map'),(v2seal_path,'accepted V2 prediction seal'),
                       (v3seal_path,'accepted V3 prediction seal'),(v4seal_path,'accepted V4 prediction seal')]
    ]
    # Seed and task support from accepted ID-level metadata.
    support=[]
    for t in TASKS:
        ix=[i for i in range(185) if int(prefix['task_id'][i])==t]
        support.append({'task_id':t,'anchors':len(ix),'episodes':len(set(int(prefix['episode_id'][i]) for i in ix)),
                        'source_groups':len(set(str(prefix['source_cluster'][i]) for i in ix))})
    if sum(x['anchors'] for x in support)!=185 or sum(x['episodes'] for x in support)!=47: raise RuntimeError('V5 support counts differ from accepted contract')

    # All source joins are keyed by anchor ID; the V4 row order is intentionally not assumed.
    join_rows=[]
    canonical_ids=[r['anchor_id'] for r in next(x for x in sources if x['role']=='S_OPT' and x['seed']==SEEDS[0])['source_rows']]
    if len(canonical_ids)!=185 or set(canonical_ids)!=set(ids_prefix): raise RuntimeError('V2 ordered receiver IDs differ from prefix membership')
    for s in sources:
        for ci,aid in enumerate(canonical_ids):
            si=s['source_index_by_id'][aid]; row=s['source_rows'][si]
            join_rows.append({'role':s['role'],'seed':s['seed'],'source_version':s['source_version'],'condition':s['condition'],
                'anchor_id':aid,'episode_id':row['episode_id'],'task_id':row['task_id'],'source_group':row['source_cluster'],
                'canonical_row_index':ci,'source_row_index':si,'prediction_path':s['prediction_path'],'prediction_sha256':s['prediction_sha256'],
                'rows_path':s['prediction_rows_path'],'rows_sha256':s['prediction_rows_sha256'],'native_field':s['native_field'],
                'dtype':s['dtype'],'shape':'x'.join(map(str,s['shape'])),'ordered_source_ids_sha256':s['ordered_source_ids_sha256'],
                'normalizer_sha256':s['normalizer_sha256']})
    join_fields=list(join_rows[0])
    write_csv(ROOT/'protocol/ORDERED_RECEIVER_IDS.csv',[{'canonical_row_index':i,'anchor_id':a} for i,a in enumerate(canonical_ids)],['canonical_row_index','anchor_id'])
    write_csv(ROOT/'inventory/SOURCE_PREDICTION_JOINS.csv',join_rows,join_fields)

    source_lock=[]
    for s in sources:
        source_lock.append({k:s[k] for k in ['role','seed','source_version','model_id','condition','prediction_path','prediction_sha256','prediction_rows_path','prediction_rows_sha256','seal_path','seal_sha256','normalizer_path','normalizer_sha256','native_field','dtype','shape','ordered_source_ids_sha256','source_profile']})
    formula_lock={
        'schema_version':1,'protocol_id':'CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1','study':'post_V4_frozen_level_shape_composition_development',
        'created_utc':utc(),'principal_candidate':'H_SV','tasks':TASKS,'formal_seeds':SEEDS,
        'accepted_support':{'train_episodes':220,'train_anchors':1280,'dev_episodes':47,'dev_anchors':185,'task_support':support,'minimum_support_gate':False},
        'split':{'path':str(split_path),'sha256':sha_file(split_path),'reuse_literal_membership_ordered_ids':True,'repartition':False},
        'source_predictions':source_lock,'source_reference_files':source_reference_files,
        'normalizer':{'path':str(norm_path),'sha256':norm_sha,'output_space':'native command values','source_arrays_native_field':'prediction_native',
                      'native_reconstruction_formula':'pred_native=anchor_native+target_sigma*pred_residual'},
        'metric_scale_lock':{'path':str(metric_path),'sha256':metric_sha,'target_sigma_from_primary_train':True,'safe_fd_scale_from_accepted_lock':True},
        'ordered_receiver_ids_sha256':rows_digest(canonical_ids),'ordered_receiver_ids_source':'S_OPT accepted V2 G-S row order',
        'ordered_receiver_ids_csv_path':str(ROOT/'protocol/ORDERED_RECEIVER_IDS.csv'),'ordered_receiver_ids_csv_sha256':sha_file(ROOT/'protocol/ORDERED_RECEIVER_IDS.csv'),'ordered_receiver_ids_count':len(canonical_ids),
        'formula_axis':'mean over future_step axis=1 only, shape [N,30,28]; a broadcast to [N,1,28]',
        'formulas':METHODS,'arithmetic_dtype':'float64','saved_dtype':'float64','clipping':False,'smoothing':False,'learned_gates_or_weights':False,
        'prediction_generation_inputs':['accepted native source prediction arrays S,V,A,C','label-free prefix current command anchor a','ordered receiver IDs'],
        'target_or_future_observation_input_to_generation':False,'new_neural_fits':0,'optimizer_updates':0,'parameter_calibration_fits':0,
        'encoder_training':0,'encoder_extraction':0,'frozen_source_replays':0,'final_heldout_payload_access':0,
        'metadata_selected_cases':{'selection_rule':'reuse V4 selected anchor for task 119; for each other task choose the lowest episode_id, then lowest anchor_frame, then lexical anchor_id from label-free receiver metadata','selected_anchor_ids':{str(k):v for k,v in selected_cases.items()},'selection_uses_outcomes':False}
    }
    write_json(ROOT/'protocol/FORMULA_LOCK.json',formula_lock)
    formula_sha=sha_file(ROOT/'protocol/FORMULA_LOCK.json')
    plan=[]
    for method,formula in METHODS.items():
        inputs={'H_SV':['S_OPT','V_MATCHED'],'H_VS':['S_OPT','V_MATCHED'],'H_AV':['A_COMMON','V_MATCHED'],'H_SC':['S_OPT','C_REASSIGNED'],'H_PV':['V_MATCHED','current_command_anchor_a'],'MIX50':['S_OPT','V_MATCHED']}[method]
        for seed in SEEDS:
            plan.append({'bundle_id':f'{method}_seed{seed}','method':method,'seed':seed,'formula':formula,'source_roles':'+'.join(inputs),
                'tasks':','.join(map(str,TASKS)),'receiver_anchors':185,'task_views':90,'expected_shape':'185x30x28','dtype':'float64',
                'formula_lock_sha256':formula_sha,'dev_future_target_input':False,'optimizer_updates':0,'status':'PLANNED'})
    write_csv(ROOT/'protocol/DERIVED_PREDICTION_PLAN.csv',plan,list(plan[0]))
    contrast_lock={
        'schema_version':1,'created_utc':utc(),'formula_lock_sha256':formula_sha,'gain_sign':'gain=L(reference)-L(candidate); positive favors candidate',
        'contrasts':[
            {'name':'LEVEL_TRANSFER_GAIN','reference':'V','candidate':'H_SV'},
            {'name':'HYBRID_VS_OPT_STATE','reference':'S','candidate':'H_SV'},
            {'name':'HYBRID_VS_BIAS_STATE','reference':'BIAS_STATE','candidate':'H_SV'},
            {'name':'HYBRID_VS_BIAS_VIDEO','reference':'BIAS_VIDEO','candidate':'H_SV'},
            {'name':'MATCHED_SHAPE_GAIN','reference':'H_SC','candidate':'H_SV'},
            {'name':'OPT_VS_COMMON_LEVEL_GAIN','reference':'H_AV','candidate':'H_SV'},
            {'name':'HYBRID_VS_ANCHOR_LEVEL','reference':'H_PV','candidate':'H_SV'},
            {'name':'HYBRID_VS_MIX50','reference':'MIX50','candidate':'H_SV'}],
        'metrics':['position_mae','safe_fd_l1','seam_safe_fd_l1'],'aggregation':'anchor to episode mean, episode equal within task, five task equal; seed-specific then three-seed summaries',
        'source_group_bootstrap':{'draws':5000,'seed':20261015,'sampling':'within each task, resample source groups with replacement, retaining all episode anchors; same draw indices shared across methods and seeds','interval':'percentile 2.5%,97.5%','p_values':False},
        'seed_loo':'leave one formal seed out from five-task macro; use remaining two seeds',
        'task_loo':'leave one task out within each seed; macro over remaining four tasks',
        'ci_scope':'conditional on fixed models, formulas, and repeated exposure to development outcomes; not cross-training-seed population inference',
        'source_groups_are_inference_units':True,'timesteps_dimensions_methods_seeds_are_not_independent_replicates':True
    }
    write_json(ROOT/'protocol/CONTRAST_LOCK.json',contrast_lock)
    write_json(ROOT/'inventory/SOURCE_MAP.json',{
        'schema_version':1,'study':'post_V4_frozen_level_shape_composition_development','created_utc':utc(),
        'v2_root':str(V2),'v3_root':str(V3),'v4_root':str(V4),'v5_root':str(ROOT),
        'source_prediction_bundles':source_lock,'source_artifacts':source_reference_files,
        'fold_membership':{'sha256':sha_file(split_path),'train_episodes':220,'train_anchors':1280,'dev_episodes':47,'dev_anchors':185,'task_support':support,
                           'v2_v3_v4_byte_equal_source_asserted_by_v4_source_map':True},
        'v4_B_M_reuse':{'accepted':True,'map':'V4 B_M uses accepted V2 G-V matched outputs','per_seed_prediction_hashes':{str(s):next(x['prediction_sha256'] for x in source_lock if x['role']=='V_MATCHED' and x['seed']==s) for s in SEEDS}},
        'v4_matched_only_sources':True,'donor_map_outputs_used_for_V_or_C':False,
        'normalizer':{'path':str(norm_path),'sha256':norm_sha},'metric_scale_lock':{'path':str(metric_path),'sha256':metric_sha},
        'canonical_receiver_order_hash':rows_digest(canonical_ids),'canonical_receiver_order_path':str(ROOT/'protocol/ORDERED_RECEIVER_IDS.csv'),'canonical_receiver_order_csv_sha256':sha_file(ROOT/'protocol/ORDERED_RECEIVER_IDS.csv'),'canonical_receiver_order_source':'V2 accepted G-S/G-V prediction rows','final_heldout_payload_access':0
    })
    write_json(ROOT/'inventory/OUTCOME_EXPOSURE.json',{
        'schema_version':1,'status':'PRESEAL_LOCKED','created_utc':utc(),'prior_cumulative_dev_future_window_logical_reads':35523,
        'unique_dev_receiver_windows':185,'v5_preseal_target_archive_opens':0,'v5_preseal_target_rows_read':0,
        'v5_preseal_formula_prediction_target_reads':0,'final_heldout_payload_access':0,
        'expected_new_derived_prediction_bundles':18,'expected_derived_window_loss_evaluations':3330,
        'history':[{'phase':'V5_preseal_sources_formula_plan','target_reads':0,'target_archive_loads':0,'status':'LOCKED'}]
    })
    write_json(ROOT/'coordination/OWNER_STATE.json',{
        'schema_version':1,'study':'post_V4_frozen_level_shape_composition_development','out_root':str(ROOT),
        'status':'running','phase':'formula_and_source_lock','active_jobs':0,'active_gpu_jobs':0,'owner_lock_acquired':False,
        'same_contract_v5_existing_root_found':False,'v4_writer_idle':True,'v4_owner_state_status':v4owner.get('status'),
        'source_roots_read_only':True,'old_root_writes':0,'final_heldout_payload_access':0,'updated_utc':utc()
    })
    print(json.dumps({'status':'FORMULA_LOCKED','formula_lock_sha256':formula_sha,'sources':len(sources),'joins':len(join_rows),
                      'support':support,'cases':selected_cases,'target_accesses':0,'root':str(ROOT)},indent=2))

if __name__=='__main__': main()
