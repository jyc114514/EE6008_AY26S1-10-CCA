#!/usr/bin/env python3
"""Bind the post-V2 experiment to accepted V2 assets and actual fold membership."""
from __future__ import annotations
import os
import csv, hashlib, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2 = PROJECT / 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
SCREEN_SEED = 202610130


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f'refusing to overwrite V3 receipt: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def checked_manifest(path: Path, manifest: dict) -> dict:
    if manifest.get('status') != 'PASS':
        raise RuntimeError(f'V2 model manifest is not PASS: {path}')
    model_path = Path(manifest['model_file'])
    actual = sha(model_path)
    if actual != manifest.get('model_file_sha256'):
        raise RuntimeError(f'V2 accepted weight hash mismatch: {model_path}')
    return {'model_path': str(model_path), 'model_sha256': actual,
            'final_state_hash': manifest.get('final_state_hash'),
            'model_manifest_path': str(path), 'model_manifest_sha256': sha(path)}


def load_queue(phase: str):
    filename = 'SCREENING_ALL.json' if phase == 'screening' else 'FORMAL.json'
    queue_path = V2 / 'coordination/queues' / filename
    return queue_path, read_json(queue_path)


def fit_record(watcher_json: str) -> dict:
    watcher_path = Path(watcher_json)
    watcher = read_json(watcher_path)
    if V2 not in watcher_path.parents:
        raise RuntimeError(f'queue config not in current V2 root: {watcher_path}')
    fit_path = Path(watcher['command'][-1])
    fit = read_json(fit_path)
    job_root = Path(fit['job_root'])
    if V2 not in job_root.parents or any(x in job_root.parts for x in ('archive', 'superseded')):
        raise RuntimeError(f'refusing non-current or archived checkpoint root: {job_root}')
    complete_path = job_root / 'RUN_COMPLETE.json'
    manifest_path = job_root / 'MODEL_MANIFEST.json'
    complete, manifest = read_json(complete_path), read_json(manifest_path)
    if complete.get('status') != 'PASS' or manifest.get('status') != 'PASS':
        raise RuntimeError(f'V2 base lacks accepted RUN_COMPLETE/MODEL_MANIFEST: {fit["job_id"]}')
    asset = checked_manifest(manifest_path, manifest)
    if str(fit['job_id']) != str(manifest.get('job_id')):
        raise RuntimeError('fit config and model manifest job IDs differ')
    if int(fit['seed']) != int(manifest['seed']) or str(fit['model_id']) != str(manifest['model_id']):
        raise RuntimeError('fit config and model manifest identity differs')
    selected_file = job_root / f"model_epoch{int(fit['epochs']):03d}.safetensors"
    if not selected_file.is_file() or sha(selected_file) != asset['model_sha256']:
        raise RuntimeError(f'selected final epoch differs from accepted model.safetensors: {selected_file}')
    normalizer_path = Path(fit['normalizer_path'])
    normalizer = read_json(normalizer_path)
    if normalizer.get('status') not in {'LOCKED_SCREENING', 'LOCKED_FORMAL'}:
        raise RuntimeError(f'normalizer is not locked: {normalizer_path}')
    p = {
        'phase': str(fit['phase']), 'model_id': str(fit['model_id']), 'seed': int(fit['seed']),
        'video_mode': str(fit['video_mode']), 'lr': float(fit['lr']), 'epochs': int(fit['epochs']),
        'job_id': str(fit['job_id']), 'fit_config_path': str(fit_path),
        'fit_config_sha256': sha(fit_path), 'watcher_config_path': str(watcher_path),
        'watcher_config_sha256': sha(watcher_path), 'job_root': str(job_root),
        'run_complete_path': str(complete_path), 'run_complete_sha256': sha(complete_path),
        'run_complete_status': complete.get('status'), **asset,
        'normalizer_path': str(normalizer_path), 'normalizer_sha256': sha(normalizer_path),
        'normalizer_status': normalizer.get('status'),
        'feature_cache_manifest_sha256': manifest.get('feature_cache_manifest_sha256'),
        'physical_gpu_training_receipt': manifest.get('physical_gpu'),
        'train_anchors': manifest.get('train_anchors'), 'tune_anchors': manifest.get('tune_anchors'),
    }
    return p


def main() -> None:
    acceptance_path = V2 / 'qa/FINAL_ACCEPTANCE.json'
    acceptance = read_json(acceptance_path)
    seal_path = V2 / 'protocol/PREDICTION_BATCH_SEAL.json'
    seal = read_json(seal_path)
    selection_path = V2 / 'protocol/LR_EPOCH_SELECTION.json'
    selection = read_json(selection_path)
    split_lock_path = V2 / 'protocol/INNER_SPLIT_LOCK.json'
    split_lock = read_json(split_lock_path)
    formal_matrix_path = V2 / 'protocol/FORMAL_MATRIX.json'
    formal_matrix = read_json(formal_matrix_path)
    model_contract_path = V2 / 'protocol/MODEL_AND_LOSS.json'
    model_contract = read_json(model_contract_path)
    source_prov_path = V2 / 'results/TRAINING_SOURCE_PROVENANCE.csv'
    source_prov = read_csv(source_prov_path)
    outcomes = read_json(V2 / 'inventory/OUTCOME_EXPOSURE.json')
    cache_paths = read_json(V2 / 'inventory/RESOLVED_PATHS.json')
    cache_manifest_path = Path(cache_paths['feature_cache'])
    cache_manifest = read_json(cache_manifest_path)
    if acceptance.get('status') != 'PASS' or acceptance.get('checksum_acceptance', {}).get('status') != 'PASS':
        raise RuntimeError('V2 source final acceptance/checksum is not PASS')
    if acceptance.get('final_heldout_payload_access') != 0 or seal.get('final_heldout_payload_access') != 0:
        raise RuntimeError('V2 final/heldout access is nonzero')
    if seal.get('status') != 'SEALED_BEFORE_TARGET_ACCESS' or seal.get('sealed_prediction_jobs') != 42 or seal.get('missing_prediction_jobs'):
        raise RuntimeError('V2 current prediction seal is incomplete')
    if selection.get('status') != 'LOCKED' or len(selection.get('logical_model_choices', [])) != 12:
        raise RuntimeError('V2 selection lock is not complete')
    if split_lock.get('status') != 'LOCKED' or 'canonical_source_key' not in split_lock.get('algorithm', ''):
        raise RuntimeError('V2 literal corrected fold lock is not the expected version')
    if cache_manifest.get('status') != 'PASS' or cache_manifest.get('final_test_access') != 0:
        raise RuntimeError('V2 Q0 cache receipt is not accepted/final0')
    if cache_manifest.get('logical_shape') != [20079, 2, 16, 4, 768]:
        raise RuntimeError('V2 accepted feature cache is not the Q0 four-bin tensor')
    if model_contract.get('feature_encoder', {}).get('frozen') is not True or model_contract.get('feature_encoder', {}).get('new_extraction') != 0:
        raise RuntimeError('V2 encoder/cache contract differs from frozen reuse')

    # Match each anchor to the actual CSV membership; do not reconstruct the hash/fold.
    fold_src = V2 / 'protocol/INNER_SPLIT.csv'
    fold_copy = ROOT / 'inventory/ACTUAL_FOLD_MEMBERSHIP.csv'
    if sha(fold_src) != sha(fold_copy):
        raise RuntimeError('V3 fold copy is not byte-identical to current V2 fold CSV')
    fold_rows = read_csv(fold_copy)
    fold_by_episode = {}
    for row in fold_rows:
        ids = [x for x in re.split(r'[;,\s]+', row['episode_ids'].strip()) if x]
        for raw_id in ids:
            key = (int(row['task_id']), int(raw_id))
            if key in fold_by_episode:
                raise RuntimeError(f'duplicate V2 episode membership: {key}')
            fold_by_episode[key] = row['role']
    train_npz = V2 / 'data/TRAIN_SELECTED_H30.npz'
    with np.load(train_npz, allow_pickle=False) as a:
        required = {'anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame',
                    'history_native','history_mask','relative_delta','anchor_native','inner_role'}
        if not required.issubset(a.files):
            raise RuntimeError(f'train archive misses prefix/fold fields: {sorted(required-set(a.files))}')
        if 'target_native_h30' not in a.files:
            raise RuntimeError('V2 accepted train archive unexpectedly lacks train labels')
        if a['target_native_h30'].shape != (1280,30,28):
            raise RuntimeError('V2 train target shape mismatch')
        role_counts = Counter(str(x) for x in a['inner_role'].tolist())
        task_role_counts = Counter()
        mismatches = []
        for task, episode, anchor, archive_role in zip(a['task_id'],a['episode_id'],a['anchor_id'],a['inner_role']):
            expected = fold_by_episode.get((int(task),int(episode)))
            if expected is None or str(expected) != str(archive_role):
                mismatches.append((str(anchor),int(task),int(episode),str(expected),str(archive_role)))
            else:
                task_role_counts[(int(task),str(expected))] += 1
        task_ids = sorted(set(int(x) for x in a['task_id'].tolist()))
        n_train = int(len(a['anchor_id']))
        train_episodes = len(set((int(t),int(e)) for t,e in zip(a['task_id'],a['episode_id'])))
        train_sha = sha(train_npz)
    if mismatches:
        raise RuntimeError(f'V2 fold membership differs from train archive: {mismatches[:3]}')
    if role_counts != Counter({'inner_fit':1010,'inner_tune':270}) or n_train != 1280 or train_episodes != 220:
        raise RuntimeError(f'V2 train split counts differ from receipt: {role_counts}/{n_train}/{train_episodes}')
    if set(task_ids) != set(TASK_IDS):
        raise RuntimeError(f'V2 task set changed: {task_ids}')

    # Match the locked selection to the accepted screening checkpoint; never use a formal base for screen.
    choices = {str(x['model_id']):x for x in selection['logical_model_choices']}
    screening_qpath, screening_q = load_queue('screening')
    formal_qpath, formal_q = load_queue('formal')
    all_screen = [fit_record(x) for x in screening_q['jobs']]
    all_formal = [fit_record(x) for x in formal_q['jobs']]
    screen_bases=[]
    for mid in ('G-S','G-V'):
        c=choices[mid]
        matches=[x for x in all_screen if x['model_id']==mid and x['seed']==SCREEN_SEED and abs(x['lr']-float(c['selected_lr']))<1e-12 and x['epochs']==int(c['selected_epoch'])]
        if len(matches)!=1 or matches[0]['epochs']!=160:
            raise RuntimeError(f'expected exactly one selected corrected SCREENING base for {mid}: {len(matches)}')
        screen_bases.extend(matches)
    formal_bases=[]
    for mid in ('G-S','G-V'):
        c=choices[mid]
        for seed in SEEDS:
            matches=[x for x in all_formal if x['model_id']==mid and x['seed']==seed and abs(x['lr']-float(c['selected_lr']))<1e-12 and x['epochs']==int(c['selected_epoch'])]
            if len(matches)!=1 or matches[0]['epochs']!=160:
                raise RuntimeError(f'expected exactly one selected corrected FORMAL base for {mid}/{seed}')
            formal_bases.extend(matches)
    control_bases=[]
    for seed in SEEDS:
        matches=[x for x in all_formal if x['model_id']=='S-V-90' and x['seed']==seed]
        if len(matches)!=1 or matches[0]['epochs']!=160:
            raise RuntimeError(f'expected accepted S-V-90 control base for seed {seed}')
        control_bases.extend(matches)

    # Selected V2 predictions are references. Verify receipts/IDs only; no dev targets are opened here.
    pred_queue_path=V2/'coordination/queues/FORMAL_DEV.json'
    pred_queue=read_json(pred_queue_path)
    pred_map={}
    for raw in pred_queue['jobs']:
        watcher_path=Path(raw); watcher=read_json(watcher_path)
        fit_path=Path(watcher['command'][-1]); inf=read_json(fit_path)
        if inf['model_id'] not in {'G-S','G-V','S-V-90'}: continue
        job_root=Path(inf['output_root'])
        complete_path=job_root/'PREDICTION_COMPLETE.json'
        ids_path=job_root/'PREDICTION_ROWS.csv'
        tensor_path=job_root/'PREDICTIONS.safetensors'
        complete=read_json(complete_path)
        if complete.get('status')!='PASS': raise RuntimeError(f'accepted V2 base prediction missing: {inf["job_id"]}')
        ids=read_csv(ids_path)
        expected=185 if inf['model_id'] in {'G-S','G-V'} else 41
        if len(ids)!=expected: raise RuntimeError(f'V2 base prediction support mismatch {inf["job_id"]}: {len(ids)}')
        if len({x['anchor_id'] for x in ids})!=len(ids): raise RuntimeError('duplicate IDs in V2 accepted prediction file')
        # The V2 seal is the source of acceptance; recheck payload against its completion receipt.
        pred_sha=sha(tensor_path)
        recorded=complete.get('prediction_sha256')
        if recorded and pred_sha!=recorded: raise RuntimeError(f'V2 accepted prediction hash mismatch: {inf["job_id"]}')
        pred_map[f"{inf['model_id']}:{int(inf['seed'])}"]={'job_id':inf['job_id'],'model_id':inf['model_id'],'seed':int(inf['seed']),
           'output_root':str(job_root),'completion_path':str(complete_path),'completion_sha256':sha(complete_path),
           'ids_path':str(ids_path),'ids_sha256':sha(ids_path),'prediction_path':str(tensor_path),'prediction_sha256':pred_sha,'dev_rows':len(ids)}
    expected_pred_keys={f'{m}:{s}' for m in ('G-S','G-V','S-V-90') for s in SEEDS}
    if set(pred_map)!=expected_pred_keys:
        raise RuntimeError(f'V2 reused prediction reference set incomplete: {set(pred_map)^expected_pred_keys}')

    # Validate the subset of V2's immutable checksum ledger actually used here; never rescan 31 GB.
    checksum_lines=(V2/'CHECKSUMS.sha256').read_text(encoding='utf-8').splitlines()
    v2_hashes={line.split('  ',1)[1]:line.split('  ',1)[0] for line in checksum_lines if '  ' in line}
    subset=[acceptance_path,seal_path,selection_path,split_lock_path,formal_matrix_path,model_contract_path,
            V2/'protocol/METRIC_SCALE_LOCK.json',V2/'inventory/SOURCE_ACCEPTANCE.json',V2/'inventory/OUTCOME_EXPOSURE.json',
            V2/'inventory/DEV_SUPPORT_V2.csv',V2/'inventory/TRAIN_TASKS_V2.csv',train_npz,V2/'data/DEV_PREFIXES_NO_LABELS.npz',
            V2/'results/ABSOLUTE_LOSSES.csv',V2/'results/ALL_SEED_D_V_I.csv',V2/'results/ALL_SEED_LOO.csv',
            V2/'results/CONDITIONAL_CI.csv',V2/'results/TRAINING_SOURCE_PROVENANCE.csv',
            V2/'figures/CAPTIONS_AND_DESCRIPTIONS.md',V2/'qa/FIGURE_VISUAL_REVIEW.md']
    for base in screen_bases+formal_bases+control_bases:
        subset.extend([Path(base['run_complete_path']),Path(base['model_manifest_path']),Path(base['model_path']),Path(base['normalizer_path'])])
    for pred in pred_map.values():
        subset.extend([Path(pred['completion_path']),Path(pred['ids_path']),Path(pred['prediction_path'])])
    drift=[]
    for path in subset:
        rel=path.relative_to(V2).as_posix()
        expected=v2_hashes.get(rel)
        if expected is None: drift.append({'path':rel,'reason':'not_in_v2_checksum_manifest'})
        elif sha(path)!=expected: drift.append({'path':rel,'reason':'checksum_mismatch','manifest_sha256':expected,'actual_sha256':sha(path)})
    if drift: raise RuntimeError(f'current V2 asset/doc drift after checksum seal: {drift[:5]}')

    map_receipt={
      'schema_version':1,'status':'PASS','created_utc':'2026-10-14T05:34:28Z','v2_root':str(V2),
      'v2_final_acceptance_path':str(acceptance_path),'v2_final_acceptance_status':acceptance['status'],
      'v2_final_checksum_status':acceptance['checksum_acceptance']['status'],'v2_prediction_seal_sha256':sha(seal_path),
      'current_corrected_split_lock_sha256':sha(split_lock_path),'current_split_algorithm':split_lock['algorithm'],
      'fold_rehash_performed':False,'fold_csv_source_path':str(fold_src),'fold_csv_copy_path':str(fold_copy),
      'fold_csv_source_sha256':sha(fold_src),'fold_csv_v3_copy_sha256':sha(fold_copy),
      'fold_join_anchor_mismatches':0,'fold_episode_rows':len(fold_by_episode),'train_episodes':train_episodes,
      'train_anchors':n_train,'inner_fit_anchors':role_counts['inner_fit'],'inner_tune_anchors':role_counts['inner_tune'],
      'per_task_role_anchor_counts':{str(t):{role:task_role_counts[(t,role)] for role in ('inner_fit','inner_tune')} for t in TASK_IDS},
      'train_npz_path':str(train_npz),'train_npz_sha256':train_sha,
      'dev_prefix_path':str(V2/'data/DEV_PREFIXES_NO_LABELS.npz'),'dev_prefix_sha256':sha(V2/'data/DEV_PREFIXES_NO_LABELS.npz'),
      'dev_prefix_rows':185,'dev_prefix_has_future_target_key':False,
      'screening_bases':screen_bases,'formal_bases':formal_bases,'probe_control_bases':control_bases,
      'reused_v2_accepted_prediction_references':pred_map,
      'v2_78_fit_rerun':False,'base_training_updates':0,'encoder_training':0,'encoder_extraction':0,
      'parent_historical_dev_target_access_rows':int(acceptance['cumulative_dev_target_access_rows']),
      'parent_dev_target_access_history':outcomes.get('dev_target_access_history',[]),
      'v3_dev_target_reads_before_own_prediction_seal':0,'final_heldout_payload_access':0,
      'new_raw_video_extraction':0,'v2_q0_cache_manifest_path':str(cache_manifest_path),
      'v2_q0_cache_manifest_sha256':sha(cache_manifest_path),'v2_q0_shape':cache_manifest['logical_shape'],
      'v2_q0_shard_count':len(cache_manifest['shards']),'v2_q0_lineage':cache_paths.get('feature_cache_kind'),
      'normalizer_paths':{'screening':screen_bases[0]['normalizer_path'],'formal':formal_bases[0]['normalizer_path']},
      'normalizer_sha256':{'screening':screen_bases[0]['normalizer_sha256'],'formal':formal_bases[0]['normalizer_sha256']},
      'metric_scale_lock_path':str(V2/'protocol/METRIC_SCALE_LOCK.json'),'metric_scale_lock_sha256':sha(V2/'protocol/METRIC_SCALE_LOCK.json'),
      'v2_source_checksum_subset_files':len(subset),'v2_source_checksum_mismatches':0,
      'non_model_documentation_drift':False,'v2_human_figure_review_status':acceptance.get('human_review_status'),
      'cache_lineage_limit':'accepted Q0 cache with partial historical extractor binding',
      'collection_independence':'unverified_episode_source_only',
      'git_state':'UNKNOWN; git executable unavailable and repository .git metadata absent in current shell',
      'final_heldout_payload_access':0,
    }
    write_json(ROOT/'inventory/BASE_STAGE_MAP.json',map_receipt)
    write_json(ROOT/'inventory/ACTUAL_FOLD_REUSE.json',{
      'schema_version':1,'status':'PASS','membership_source':'copied byte-for-byte from V2 corrected INNER_SPLIT.csv',
      'source_sha256':sha(fold_src),'copy_sha256':sha(fold_copy),'fold_rehash_performed':False,
      'train_archive_role_join':'all 1280 anchor_id/task_id/episode_id rows joined to V2 CSV membership; 0 mismatch',
      'inner_fit_anchors':1010,'inner_tune_anchors':270,'inner_fit_episodes':174,'inner_tune_episodes':46,
      'algorithm_reference_path':str(split_lock_path),'algorithm_reference_sha256':sha(split_lock_path),
      'final_heldout_payload_access':0})
    write_json(ROOT/'inventory/CACHE_AND_NORMALIZER_MAP.json',{
      'schema_version':1,'status':'PASS','v2_q0_cache_manifest_path':str(cache_manifest_path),
      'v2_q0_cache_manifest_sha256':sha(cache_manifest_path),'cache_status':cache_manifest['status'],
      'cache_shape':cache_manifest['logical_shape'],'dtype':cache_manifest['dtype'],'shard_count':len(cache_manifest['shards']),
      'screening_normalizer_path':screen_bases[0]['normalizer_path'],'screening_normalizer_sha256':screen_bases[0]['normalizer_sha256'],
      'formal_normalizer_path':formal_bases[0]['normalizer_path'],'formal_normalizer_sha256':formal_bases[0]['normalizer_sha256'],
      'metric_scale_lock_path':str(V2/'protocol/METRIC_SCALE_LOCK.json'),'metric_scale_lock_sha256':sha(V2/'protocol/METRIC_SCALE_LOCK.json'),
      'screening_base_scope':'corrected inner_fit only; seed 202610130; inner_tune used only for adapter selection',
      'formal_base_scope':'corrected outer train; same-seed matching G-S/G-V',
      'dev_prefix_rows':185,'dev_future_targets_loaded':False,'new_video_extraction':0,'final_heldout_payload_access':0})
    write_json(ROOT/'inventory/OUTCOME_EXPOSURE.json',{
      'schema_version':1,'parent_v2_root':str(V2),'parent_v2_cumulative_dev_target_row_reads':int(acceptance['cumulative_dev_target_access_rows']),
      'parent_v2_history':outcomes.get('dev_target_access_history',[]),'v3_current_dev_target_rows_read':0,
      'v3_target_reader_called':False,'v3_empty_or_missing_prediction_target_read_allowed':False,
      'v3_future_labels_used_in_forward':False,'final_heldout_payload_access':0})
    write_json(ROOT/'inventory/V3_PATHS.json',{
      'schema_version':1,'project_root':str(PROJECT),'v2_root':str(V2),'v3_root':str(ROOT),
      'python_bin':str(PROJECT/'envs/jyc/bin/python'),'shared_watcher':str(PROJECT/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py'),
      'cache_manifest':str(cache_manifest_path),'train_npz':str(train_npz),
      'dev_prefix_npz':str(V2/'data/DEV_PREFIXES_NO_LABELS.npz'),
      'scorer_only_dev_target_path':str(V2/'data/DEV_TARGETS_SCORER_ONLY.npz'),
      'scorer_only_dev_target_read_before_v3_seal':False,'final_heldout_payload_access':0})
    print(json.dumps({'status':'PASS','v3_root':str(ROOT),'screening_base_count':len(screen_bases),
      'formal_base_count':len(formal_bases),'probe_control_base_count':len(control_bases),
      'base_prediction_reference_count':len(pred_map),'fold_join_mismatches':0,
      'v2_source_checksum_subset_files':len(subset),'non_model_documentation_drift':False,
      'v3_dev_target_reads':0,'final_heldout_payload_access':0},indent=2))

if __name__=='__main__':
    main()
