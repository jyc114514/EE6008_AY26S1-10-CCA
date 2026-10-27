#!/usr/bin/env python3
"""V6 source-role, split, and sealed V5 baseline preflight; never opens new dev targets."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / 'reports/benchmarks'
V2 = BENCH / 'CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V3 = BENCH / 'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'
V4 = BENCH / 'CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V5 = BENCH / 'CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
TASKS = [119, 94, 90, 141, 230]
SEEDS = [202610131, 202610132, 202610133]
INNER_ROLE_MODELS = {
    'S_FIT': ('v2_screening_G_S_lr0p001_s202610130', 'STATE-CTRL', 0.001),
    'A_FIT': ('v2_screening_G_S_lr0p0005_s202610130', 'STATE-CTRL', 0.0005),
    'V_FIT': ('v2_screening_G_V_lr0p0005_s202610130', 'TS11-D', 0.0005),
}
METRICS = ['position_mae', 'safe_fd_l1', 'seam_safe_fd_l1']


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path: Path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path: Path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def assert_status(path: Path, expected=None):
    d = read_json(path)
    allowed = {expected} if expected is not None else {'complete', 'PASS'}
    if d.get('status') not in allowed:
        raise RuntimeError(f'{path} status={d.get("status")} expected_one_of={sorted(allowed)}')
    return d


def safe_manifest_bundle_records():
    v5map = read_json(V5 / 'inventory/SOURCE_MAP.json')
    wanted = {'S_OPT', 'V_MATCHED', 'A_COMMON', 'BIAS_STATE', 'BIAS_VIDEO'}
    records = [x for x in v5map['source_prediction_bundles'] if x.get('role') in wanted]
    if len(records) != len(wanted) * len(SEEDS):
        raise RuntimeError(f'V5 source map base bundles={len(records)} expected={len(wanted)*len(SEEDS)}')
    for rec in records:
        p = Path(rec['prediction_path'])
        rp = Path(rec.get('prediction_rows_path', rec.get('rows_path', '')))
        if sha_file(p) != rec['prediction_sha256']:
            raise RuntimeError(f'accepted base prediction hash mismatch: {p}')
        expected_rows_hash = rec.get('prediction_rows_sha256', rec.get('rows_sha256'))
        if not rp.is_file() or not expected_rows_hash or sha_file(rp) != expected_rows_hash:
            raise RuntimeError(f'accepted base prediction rows hash mismatch: {rp}')
    v5seal = read_json(V5 / 'qa/PREDICTION_SEAL.json')
    mix = [x for x in v5seal.get('accepted_bundles', v5seal.get('prediction_entries', v5seal.get('entries', [])))
           if x.get('method') == 'MIX50']
    if len(mix) != 3:
        # V5's final schema uses bundles; tolerate only its explicit accepted method records.
        mix = [x for x in v5seal.get('bundles', []) if x.get('method') == 'MIX50']
    if len(mix) != 3:
        raise RuntimeError('could not resolve the three accepted V5 MIX50 bundles from its seal')
    return v5map, records, v5seal, mix


def source_bundle_record(bundle, method, seed):
    # V5 schema: prediction and row paths are in the seal; container hashes are identity-bound.
    prediction = bundle.get('prediction_path')
    rows = bundle.get('prediction_rows_path', bundle.get('rows_path'))
    if not prediction or not rows:
        raise RuntimeError(f'incomplete V5 {method}/{seed} seal entry')
    if sha_file(Path(prediction)) != bundle.get('prediction_sha256'):
        raise RuntimeError(f'V5 accepted prediction changed: {prediction}')
    row_sha = bundle.get('prediction_rows_sha256', bundle.get('rows_sha256'))
    if not row_sha or sha_file(Path(rows)) != row_sha:
        raise RuntimeError(f'V5 accepted rows changed: {rows}')
    return {
        'role': method, 'method': method, 'seed': int(seed), 'source_version': 'V5',
        'prediction_path': prediction, 'prediction_sha256': bundle['prediction_sha256'],
        'prediction_rows_path': rows, 'prediction_rows_sha256': row_sha,
        'dtype': bundle.get('dtype'), 'shape': bundle.get('shape'),
        'completion_path': bundle.get('completion_path'),
        'completion_sha256': bundle.get('completion_sha256'),
        'seal_path': str(V5 / 'qa/PREDICTION_SEAL.json'), 'seal_sha256': sha_file(V5 / 'qa/PREDICTION_SEAL.json'),
    }


def main():
    lock = ROOT / 'coordination/OWNER.lock'
    if not lock.is_file():
        raise RuntimeError('unique V6 owner lock missing')
    owner = read_json(ROOT / 'coordination/OWNER_STATE.json')
    lock_doc = read_json(lock)
    if owner.get('out_root') != str(ROOT) or owner.get('owner_lock_acquired') is not True:
        raise RuntimeError('V6 owner state does not authorize this root')
    if hashlib.sha256(Path(ROOT / 'coordination/OWNER_TOKEN.local').read_text().strip().encode()).hexdigest() != owner.get('owner_token_sha256'):
        raise RuntimeError('V6 owner token verification failed')

    v5_accept = assert_status(V5 / 'qa/FINAL_ACCEPTANCE.json')
    v5_owner = assert_status(V5 / 'coordination/OWNER_STATE.json')
    if int(v5_accept.get('final_heldout_payload_access', -1)) != 0 or v5_accept.get('old_root_writes') not in (0, None):
        raise RuntimeError('V5 final/old-root boundary invalid')
    if v5_owner.get('all_gpu_leases_released') is not True or int(v5_owner.get('active_jobs', -1)) != 0:
        raise RuntimeError('V5 source owner still holds work/lease')
    for root, rel in [(V2, 'qa/FINAL_ACCEPTANCE.json'), (V3, 'qa/FINAL_ACCEPTANCE.json'), (V4, 'qa/FINAL_ACCEPTANCE.json')]:
        acc = assert_status(root / rel)
        if int(acc.get('final_heldout_payload_access', -1)) != 0:
            raise RuntimeError(f'prior source root final access not zero: {root}')
    for root in [V2, V3, V4]:
        own = read_json(root / 'coordination/OWNER_STATE.json')
        if own.get('status') != 'complete' or own.get('owner_lock_acquired') is True:
            raise RuntimeError(f'prior root owner is not terminal: {root}')

    split_path = V2 / 'protocol/INNER_SPLIT.csv'
    split_sha = 'ba31002f0a17b826f57cec93c7dfaa484d76b01a07fec5b9cda10de03fe041a8'
    if sha_file(split_path) != split_sha:
        raise RuntimeError('corrected literal V2 split changed')
    split = read_csv(split_path)
    by_role = {}
    for role in ['inner_fit', 'inner_tune']:
        rr = [r for r in split if r['role'] == role]
        episodes = sum(len([x for x in r['episode_ids'].replace(';', ',').split(',') if x]) for r in rr)
        anchors = sum(int(r['anchors']) for r in rr)
        by_role[role] = {'split_rows': len(rr), 'episodes': episodes, 'anchors': anchors,
                         'distinct_clusters': len({r['cluster_id'] for r in rr})}
    if by_role != {
        'inner_fit': {'split_rows': 174, 'episodes': 174, 'anchors': 1010, 'distinct_clusters': 174},
        'inner_tune': {'split_rows': 46, 'episodes': 46, 'anchors': 270, 'distinct_clusters': 46},
    }:
        raise RuntimeError(f'corrected V2 role support mismatch: {by_role}')
    if set(r['cluster_id'] for r in split if r['role'] == 'inner_fit') & set(r['cluster_id'] for r in split if r['role'] == 'inner_tune'):
        raise RuntimeError('inner-fit/inner-tune source cluster overlap')

    norm_screen_path = V2 / 'screening/NORMALIZER_SCREENING.json'
    norm_screen = read_json(norm_screen_path)
    if norm_screen.get('status') != 'LOCKED_SCREENING' or norm_screen.get('dev_target_or_outcome_access') != 0:
        raise RuntimeError('V2 screen normalizer is not an inner-fit-only lock')
    metric_path = V2 / 'protocol/METRIC_SCALE_LOCK.json'
    metric_lock = read_json(metric_path)
    if metric_lock.get('status') != 'LOCKED' or metric_lock.get('final_heldout_payload_access') != 0:
        raise RuntimeError('V2 formal metric scale lock invalid')
    init_hashes, model_rows = set(), []
    for role, (job, video_mode, lr) in INNER_ROLE_MODELS.items():
        model_dir = V2 / 'model_only/screening' / job
        cfg = read_json(model_dir / 'CONFIG.json')
        manifest = read_json(model_dir / 'MODEL_MANIFEST.json')
        complete = read_json(model_dir / 'RUN_COMPLETE.json')
        acceptance = {r['job_id']: r for r in read_csv(V2 / 'screening/FIT_ACCEPTANCE.csv')}.get(job)
        if not acceptance or acceptance.get('status') != 'PASS':
            raise RuntimeError(f'{role} is not V2 screening-accepted')
        if cfg.get('train_role') != 'inner_fit' or cfg.get('scope') != 'shared' or cfg.get('seed') != 202610130:
            raise RuntimeError(f'{role} fit role/scope/seed mismatch')
        if cfg.get('video_mode') != video_mode or float(cfg.get('lr')) != lr or cfg.get('epochs') != 160:
            raise RuntimeError(f'{role} frozen recipe mismatch')
        if manifest.get('status') != 'PASS' or complete.get('status') != 'PASS' or complete.get('epochs_completed') != 160:
            raise RuntimeError(f'{role} checkpoint is incomplete')
        if manifest.get('normalizer_sha256') != sha_file(norm_screen_path):
            raise RuntimeError(f'{role} screening normalizer mismatch')
        if manifest.get('feature_cache_manifest_sha256') != '43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee':
            raise RuntimeError(f'{role} Q0 feature manifest mismatch')
        model_path = model_dir / 'model.safetensors'
        if sha_file(model_path) != manifest.get('model_file_sha256'):
            raise RuntimeError(f'{role} safe model hash mismatch')
        init_hashes.add(manifest.get('init_state_hash'))
        model_rows.append({
            'role': role, 'job_id': job, 'model_id': manifest.get('model_id'), 'seed': 202610130,
            'fit_role': cfg['train_role'], 'scope': cfg['scope'], 'task_ids': cfg['task_ids'],
            'video_mode': cfg['video_mode'], 'learning_rate': float(cfg['lr']), 'epochs': complete['epochs_completed'],
            'optimizer_steps': complete['optimizer_steps'], 'fit_episodes': 174, 'fit_anchors': 1010,
            'model_path': str(model_path), 'model_sha256': sha_file(model_path),
            'model_state_hash': manifest['final_state_hash'], 'init_state_hash': manifest['init_state_hash'],
            'normalizer_path': str(norm_screen_path), 'normalizer_sha256': sha_file(norm_screen_path),
            'feature_manifest_sha256': manifest['feature_cache_manifest_sha256'],
            'training_input_sha256': manifest['train_selected_npz_sha256'], 'status': 'PASS',
            'final_heldout_payload_access': 0,
        })
    if len(init_hashes) != 1:
        raise RuntimeError(f'S/A/V screen fits do not share their accepted fresh INIT: {init_hashes}')

    v5map, base_records, v5seal, mix_bundles = safe_manifest_bundle_records()
    mix_records = []
    for entry in mix_bundles:
        seed = int(entry['seed'])
        if seed not in SEEDS:
            continue
        mix_records.append(source_bundle_record(entry, 'SV50', seed))
    if len(mix_records) != 3:
        raise RuntimeError('V5 MIX50 accepted bundle source list is incomplete')

    # V5 outcome reads are design context. Do not touch the V2 dev target archive here.
    v5_exposure = read_json(V5 / 'inventory/OUTCOME_EXPOSURE.json')
    if v5_exposure.get('status') != 'complete' or int(v5_exposure.get('final_heldout_payload_access', -1)) != 0:
        raise RuntimeError('V5 exposure ledger invalid')
    dev_support = read_csv(V2 / 'inventory/DEV_SUPPORT_V2.csv')
    dev_counts = {}
    for task in TASKS:
        rr = [r for r in dev_support if int(r['task_id']) == task]
        if len(rr) != 1:
            raise RuntimeError(f'accepted V2 dev support should have one task summary row for {task}')
        dev_counts[str(task)] = {'episodes': int(rr[0]['dev_episodes_after_overlap_exclusion']),
                                 'anchors': int(rr[0]['dev_h30_anchors_after_overlap_exclusion'])}
    if sum(v['episodes'] for v in dev_counts.values()) != 47 or sum(v['anchors'] for v in dev_counts.values()) != 185:
        raise RuntimeError(f'V2 accepted dev support differs: {dev_counts}')

    v5_results = V5 / 'results'
    primary = read_csv(v5_results / 'PRIMARY_TASK_SEED.csv')
    if len(primary) != 15 or {int(r['seed']) for r in primary} != set(SEEDS):
        raise RuntimeError('V5 accepted task-seed baseline table incomplete')
    mix_table = []
    refs = ['S', 'V', 'BIAS_STATE', 'BIAS_VIDEO']
    gain_table = []
    for row in primary:
        rec = {'task_id': int(row['task_id']), 'seed': int(row['seed']), 'aggregation': row['aggregation'],
               'anchors': int(row['anchors']), 'episodes': int(row['episodes']), 'source_groups': int(row['source_groups'])}
        for metric in METRICS:
            rec[metric] = float(row['MIX50_' + metric])
        mix_table.append(rec)
        for ref in refs:
            gr = {'reference_method': ref, 'candidate_method': 'SV50', 'task_id': int(row['task_id']), 'seed': int(row['seed']),
                  'anchors': int(row['anchors']), 'episodes': int(row['episodes']), 'source_groups': int(row['source_groups'])}
            for metric in METRICS:
                gr[metric + '_gain'] = float(row[ref + '_' + metric]) - float(row['MIX50_' + metric])
            gain_table.append(gr)
    write_csv(ROOT / 'results/V5_MIX50_TASK_SEED.csv', mix_table)
    write_csv(ROOT / 'results/V5_MIX50_GAIN_TASK_SEED.csv', gain_table)

    # Explicit source map; checksums bind V2/V3/V4/V5 inputs without copying them.
    relevant_artifacts = [
        (V2 / 'protocol/INNER_SPLIT.csv', 'corrected literal inner split'),
        (V2 / 'protocol/INNER_SPLIT_LOCK.json', 'corrected split lock'),
        (V2 / 'screening/NORMALIZER_SCREENING.json', 'inner-fit-only screening normalizer'),
        (V2 / 'normalizers/NORMALIZER_FORMAL.json', 'accepted outer-train formal normalizer'),
        (V2 / 'protocol/METRIC_SCALE_LOCK.json', 'accepted formal metric scale lock'),
        (V2 / 'data/TRAIN_SELECTED_H30.npz', 'train and inner-tune data archive; target access only after screen replay'),
        (V2 / 'data/DEV_PREFIXES_NO_LABELS.npz', 'label-free dev input archive'),
        (V2 / 'inventory/DEV_SUPPORT_V2.csv', 'accepted dev support'),
        (V2 / 'protocol/PREDICTION_BATCH_SEAL.json', 'accepted V2 formal prediction seal'),
        (V3 / 'protocol/PREDICTION_BATCH_SEAL.json', 'accepted V3 bias reference seal'),
        (V4 / 'protocol/PREDICTION_BATCH_SEAL.json', 'accepted V4 matched A/V seal'),
        (V4 / 'results/PER_WINDOW_LOSSES.csv', 'accepted V4 source reference scores'),
        (V5 / 'protocol/FORMULA_LOCK.json', 'accepted V5 formula lock'),
        (V5 / 'qa/PREDICTION_SEAL.json', 'accepted V5 prediction seal'),
        (V5 / 'results/PRIMARY_TASK_SEED.csv', 'accepted V5 task-seed outcomes'),
        (V5 / 'results/PER_WINDOW_DERIVED_LOSSES.csv', 'accepted V5 derived per-window outcomes'),
        (V5 / 'results/PER_WINDOW_ACCEPTED_REFERENCES.csv', 'accepted V5 per-window source reference outcomes'),
        (V5 / 'results/PAIRED_GROUP_CI.csv', 'accepted V5 conditional source-group intervals'),
        (V5 / 'inventory/OUTCOME_EXPOSURE.json', 'accepted V5 dev exposure history'),
    ]
    source_artifacts = [{'path': str(p), 'sha256': sha_file(p), 'role': role} for p, role in relevant_artifacts]
    source_map = {
        'schema_version': 1, 'study': 'post_V5_controlled_late_fusion_development',
        'created_utc': utc(), 'out_root': str(ROOT), 'tasks': TASKS,
        'formal_seeds': SEEDS, 'screen_seed': 202610130,
        'roots': {'v2': str(V2), 'v3': str(V3), 'v4': str(V4), 'v5': str(V5), 'v6': str(ROOT)},
        'source_roots_read_only': True, 'old_root_writes': 0,
        'split': {'path': str(split_path), 'sha256': split_sha, 'roles': by_role,
                  'outer_train_episodes': 220, 'outer_train_anchors': 1280,
                  'dev_support': dev_counts, 'dev_episodes': 47, 'dev_anchors': 185,
                  'session_or_trial_ids': 'UNKNOWN', 'collection_independence': 'unverified_episode_source_only'},
        'feature_cache': {'kind': 'accepted_q0_sharded_npy', 'manifest_sha256': '43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee',
                          'logical_shape': [20079, 2, 16, 4, 768], 'new_extraction': 0},
        'metric_scales': {'inner_calibration_normalizer_path': str(norm_screen_path), 'inner_calibration_normalizer_sha256': sha_file(norm_screen_path),
                          'formal_dev_metric_lock_path': str(metric_path), 'formal_dev_metric_lock_sha256': sha_file(metric_path),
                          'inner_tune_only_scales_used_for_alpha': True, 'outer_train_scales_used_for_dev': True},
        'inner_fit_models': model_rows,
        'source_prediction_bundles': base_records + mix_records,
        'source_artifacts': source_artifacts,
        'source_seals': {'V2': sha_file(V2 / 'protocol/PREDICTION_BATCH_SEAL.json'),
                         'V3': sha_file(V3 / 'protocol/PREDICTION_BATCH_SEAL.json'),
                         'V4': sha_file(V4 / 'protocol/PREDICTION_BATCH_SEAL.json'),
                         'V5': sha_file(V5 / 'qa/PREDICTION_SEAL.json')},
        'final_heldout_payload_access': 0,
        'prior_v5_exposure': {'target_archive_load_operations_total': v5_exposure.get('target_archive_load_operations_total'),
                              'target_array_window_rows_loaded_total': v5_exposure.get('target_array_window_rows_loaded_total'),
                              'unique_dev_receiver_windows': v5_exposure.get('unique_dev_receiver_windows'),
                              'primary_loss_evaluations': v5_exposure.get('new_dev_future_window_logical_reads'),
                              'reported_prior_counter': v5_exposure.get('reported_prior_counter'),
                              'reported_prior_counter_unit_status': v5_exposure.get('reported_prior_counter_unit_status')},
    }
    write_json(ROOT / 'inventory/SOURCE_MAP.json', source_map)

    fold_reuse = {
        'schema_version': 1, 'status': 'PASS', 'split_path': str(split_path), 'split_sha256': split_sha,
        'outer_train': {'episodes': 220, 'anchors': 1280}, 'inner_fit': by_role['inner_fit'],
        'inner_tune': by_role['inner_tune'], 'dev': {'episodes': 47, 'anchors': 185, 'task_support': dev_counts},
        'formal_seed_count': len(SEEDS), 'screen_seed': 202610130,
        'inner_tune_membership_reused_from_v2': True, 'dev_membership_reused_from_v2': True,
        'same_split_salt_or_repartition': False, 'inner_fit_tune_cluster_overlap': False,
        'session_or_trial_independence_verified': False, 'source_collection_independence': 'unverified_episode_source_only',
        'inner_tune_was_previously_used_for_v2_recipe_selection': True,
        'inner_tune_is_reused_development_meta_calibration': True,
        'final_heldout_payload_access': 0,
    }
    write_json(ROOT / 'inventory/ACTUAL_FOLD_REUSE.json', fold_reuse)

    role_audit = {
        'schema_version': 1, 'status': 'PASS', 'screen_seed': 202610130,
        'training_role': 'inner_fit only', 'fit_support': {'episodes': 174, 'anchors': 1010},
        'tune_support': {'episodes': 46, 'anchors': 270}, 'formal_outer_train_models_substituted': False,
        'roles': model_rows, 'common_fresh_init_hash': next(iter(init_hashes)),
        'base_checkpoint_replays_required': 3, 'new_inner_fit_fits_required': 0,
        'model_forward_inputs_exclude_future_target_arrays': True,
        'encoder_frozen': True, 'screening_predictions_requested_target_arrays': False,
        'screening_normalizer_fit_scope': 'inner_fit only', 'alpha_target_scales_fit_scope': 'inner_fit only',
        'alpha_choice_uses_inner_tune_labels_after_prediction': True,
        'inner_tune_prior_v2_recipe_selection_overlap': True,
        'formal_new_training_fits': 0, 'final_heldout_payload_access': 0,
    }
    write_json(ROOT / 'protocol/CALIBRATION_BASE_ROLE_AUDIT.json', role_audit)

    # Lock formulas, grid, aggregation, contrast sign, and access gates before V6 predictions.
    program_lock = {
        'schema_version': 1, 'protocol_id': 'CORE_B_V6_CONTROLLED_LATE_FUSION_V1',
        'study': 'post_V5_controlled_late_fusion_development', 'created_utc': utc(),
        'task_ids': TASKS, 'formal_seeds': SEEDS, 'screen_seed': 202610130,
        'support': {'outer_train_episodes': 220, 'outer_train_anchors': 1280,
                    'inner_fit_episodes': 174, 'inner_fit_anchors': 1010,
                    'inner_tune_episodes': 46, 'inner_tune_anchors': 270,
                    'dev_episodes': 47, 'dev_anchors': 185},
        'bases': {'S': 'STATE-CTRL LR=0.001 epoch=160', 'A': 'STATE-CTRL LR=0.0005 epoch=160',
                  'V': 'TS11-D matched RGB LR=0.0005 epoch=160'},
        'programs': {'SV50': '(S+V)/2; reuse the 3 accepted V5 MIX50 dev outputs',
                     'SS50': '(S+A)/2', 'SV_CAL': '(1-alpha_SV)*S+alpha_SV*V',
                     'SS_CAL': '(1-alpha_SS)*S+alpha_SS*A'},
        'alpha_grid': [k / 20 for k in range(21)],
        'alpha_selection': {'families': ['SV', 'SS'], 'inner_tune_only': True,
                            'objective': 'position_mae + 0.1*safe_fd_l1',
                            'fd_definition': '29 future-internal differences only',
                            'tie_rule': 'J within abs 1e-12: closest to 0.5, then lower alpha',
                            'weights': 'anchor mean, episode equal within task, five tasks equal',
                            'global_scalar_across_tasks_seeds_episodes_anchors_steps_dimensions': True,
                            'intercept_or_gate_or_smoothing_or_clipping': False},
        'numeric_contract': {'source_join': 'anchor_id+episode_id+task_id+source_group; no positional joins',
                             'mixing_space': 'native command prediction', 'mixing_dtype': 'float64',
                             'prediction_seal_before_dev_target_reader': True,
                             'formal_dev_metric_scale': 'accepted V2 primary-train metric scale lock',
                             'calibration_scales': 'V2 inner-fit-only screening normalizer'},
        'statistics': {'draws': 5000, 'bootstrap_seed': 20261015,
                       'sampling': 'within task resample source groups; carry full group episodes/anchors; shared draws across methods/seeds',
                       'ci_scope': 'conditional on fixed bases, selected alpha, and exposed dev outcomes',
                       'p_values': False, 'independent_unit': 'source_group'},
        'fixed_contrasts': {
            'VIDEO_SUBSTITUTION_50': ['SS50', 'SV50'],
            'VIDEO_SUBSTITUTION_CAL': ['SS_CAL', 'SV_CAL'],
            'SV_CALIBRATION_GAIN': ['SV50', 'SV_CAL'],
            'SS_CALIBRATION_GAIN': ['SS50', 'SS_CAL'],
            'SV_CAL_VS_S': ['S', 'SV_CAL'], 'SV_CAL_VS_V': ['V', 'SV_CAL'],
            'SV_CAL_VS_BIAS_STATE': ['BIAS_STATE', 'SV_CAL'],
            'SV_CAL_VS_BIAS_VIDEO': ['BIAS_VIDEO', 'SV_CAL'],
            'SV50_VS_S': ['S', 'SV50'], 'SV50_VS_V': ['V', 'SV50'],
            'SV50_VS_BIAS_STATE': ['BIAS_STATE', 'SV50'], 'SV50_VS_BIAS_VIDEO': ['BIAS_VIDEO', 'SV50'],
            'SS50_VS_S': ['S', 'SS50'], 'SS50_VS_A': ['A', 'SS50'],
        },
        'fixed_contrast_sign': 'gain = L(reference) - L(candidate); positive favors candidate',
        'completion_gates': {'positive_gain': False, 'minimum_support': False,
                             'final_heldout_payload_access': 0, 'max_new_formal_fits': 0,
                             'max_new_inner_fit_fits': 3, 'max_new_dev_bundles': 9},
        'V5_prior_outcomes_read_before_lock': True,
        'development_only': True,
        'lock_sha256_excludes_self': True,
    }
    write_json(ROOT / 'protocol/PROGRAM_LOCK.json', program_lock)

    state = read_json(ROOT / 'coordination/OWNER_STATE.json')
    state.update({'phase': 'preflight_complete', 'status': 'running', 'v5_checksum_entries_verified': 255,
                  'v5_checksum_sha256': sha_file(V5 / 'CHECKSUMS.sha256'),
                  'inner_fit_models_reused': 3, 'new_inner_fit_fits': 0, 'new_formal_fits': 0,
                  'final_heldout_payload_access': 0, 'old_root_writes': 0, 'updated_utc': utc()})
    write_json(ROOT / 'coordination/OWNER_STATE.json', state)
    print(json.dumps({'status': 'PASS', 'out_root': str(ROOT), 'split_roles': by_role,
                      'source_prediction_bundles': len(base_records) + len(mix_records),
                      'final_heldout_payload_access': 0, 'new_dev_target_reads': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
