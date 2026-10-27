#!/usr/bin/env python3
"""Select one global scalar per family from the sealed screen-seed inner-tune predictions."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2 = PROJECT / 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
TASKS = [119, 94, 90, 141, 230]
ALPHAS = [k / 20 for k in range(21)]


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for key in r:
                if key not in fields:
                    fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def score_native(pred, target, anchor, target_sigma, safe_fd_scale):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (30, 28):
        raise ValueError('score input must align as [N,30,28]')
    if anchor.shape != (len(pred), 28):
        raise ValueError('anchor must be [N,28]')
    pos = (np.abs(pred - target) / target_sigma[None, None, :]).mean(axis=(1, 2))
    fd = (np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)) / safe_fd_scale[None, None, :]).mean(axis=(1, 2))
    seam = (np.abs((pred[:, 0, :] - anchor) - (target[:, 0, :] - anchor)) / safe_fd_scale[None, :]).mean(axis=1)
    return {'position_mae': pos, 'safe_fd_l1': fd, 'seam_safe_fd_l1': seam}


def task_equal_episode_aggregate(values, episodes, tasks):
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes).astype(str)
    tasks = np.asarray(tasks).astype(np.int64)
    by_task = {}
    for task in TASKS:
        ix = np.flatnonzero(tasks == task)
        ep_means = [float(values[ix[episodes[ix] == ep]].mean()) for ep in sorted(set(episodes[ix]))]
        if not ep_means:
            raise RuntimeError(f'inner_tune has no rows for task {task}')
        by_task[task] = {'episode_equal_mean': float(np.mean(ep_means)), 'episodes': len(ep_means), 'anchors': int(len(ix))}
    return by_task, float(np.mean([by_task[t]['episode_equal_mean'] for t in TASKS]))


def synthetic_qa():
    target = np.zeros((3, 30, 28), dtype=np.float64)
    anchor = np.zeros((3, 28), dtype=np.float64)
    sigma = np.ones(28, dtype=np.float64)
    fd = np.ones(28, dtype=np.float64)
    s = np.ones_like(target)
    v = -np.ones_like(target)
    y0 = score_native(s, target, anchor, sigma, fd)
    y1 = score_native(v, target, anchor, sigma, fd)
    ym = score_native((s + v) / 2, target, anchor, sigma, fd)
    if not np.allclose(y0['position_mae'], 1) or not np.allclose(y1['position_mae'], 1):
        raise RuntimeError('synthetic endpoint position loss failed')
    if not np.allclose(ym['position_mae'], 0) or not np.allclose(ym['seam_safe_fd_l1'], 0):
        raise RuntimeError('synthetic midpoint/seam loss failed')
    if not np.allclose(y0['safe_fd_l1'], 0) or not np.allclose(y1['safe_fd_l1'], 0):
        raise RuntimeError('synthetic internal-FD calculation failed')
    return {'status': 'PASS', 'rows': 3, 'shape': [3, 30, 28], 'endpoint_identity': True,
            'convexity_midpoint_position': True, 'seam_path': True, 'internal_fd_29_steps': True,
            'development_targets_read': False, 'created_utc': utc()}


def load_bundle(role, seal):
    rec = next(x for x in seal['bundles'] if x['role'] == role)
    tensors = load_file(rec['prediction_path'])
    pred = np.asarray(tensors['prediction_native'])
    rows = read_csv(Path(rec['rows_path']))
    ids = [r['anchor_id'] for r in rows]
    return pred.astype(np.float64), ids, rec, rows


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    alpha_path = ROOT / 'protocol/ALPHA_LOCK.json'
    if alpha_path.exists():
        raise RuntimeError('ALPHA_LOCK already exists; alpha selection is write-once')
    seal_path = ROOT / 'qa/INNER_TUNE_PREDICTION_SEAL.json'
    if not seal_path.is_file():
        raise RuntimeError('calibration refuses missing screen-seed inner-tune prediction seal')
    seal = json.loads(seal_path.read_text())
    if seal.get('status') != 'SEALED_BEFORE_TUNE_LABEL_READER' or int(seal.get('accepted_bundle_count', 0)) != 3:
        raise RuntimeError('calibration refuses unsealed/empty/incomplete inner-tune predictions')
    if seal.get('target_archive_opened') is not False or seal.get('dev_target_archive_opened') is not False:
        raise RuntimeError('tune prediction seal reports target/dev payload opened before calibration')
    qa = synthetic_qa()
    write_json(ROOT / 'qa/CALIBRATION_SYNTHETIC_QA.json', qa)

    # Target arrays are read only now: predictions and exact role/ID joins are sealed.
    train_archive = V2 / 'data/TRAIN_SELECTED_H30.npz'
    archive_sha = sha_file(train_archive)
    with np.load(train_archive, allow_pickle=False) as z:
        required = {'anchor_id', 'episode_id', 'task_id', 'source_cluster', 'inner_role', 'anchor_native', 'target_native_h30'}
        if not required.issubset(set(z.files)):
            raise RuntimeError(f'V2 train/tune archive missing required calibration keys: {sorted(required-set(z.files))}')
        all_ids = z['anchor_id'].astype(str)
        all_roles = z['inner_role'].astype(str)
        selected = np.flatnonzero(all_roles == 'inner_tune')
        # Compressed NPZ array members are materialized as a whole; only these 270 rows are scored.
        target_all = z['target_native_h30']
        anchor_all = z['anchor_native']
        episode_all = z['episode_id']
        task_all = z['task_id']
        group_all = z['source_cluster'].astype(str)
        tune_ids = all_ids[selected].tolist()
        tune_targets = target_all[selected].astype(np.float64)
        tune_anchors = anchor_all[selected].astype(np.float64)
        tune_episodes = episode_all[selected].astype(np.int64)
        tune_tasks = task_all[selected].astype(np.int64)
        tune_groups = group_all[selected].tolist()
        tune_metadata = {str(all_ids[i]): (str(int(episode_all[i])), str(int(task_all[i])), str(group_all[i])) for i in selected}
    if len(selected) != 270 or len(set(tune_ids)) != 270:
        raise RuntimeError(f'inner_tune target support invalid: rows={len(selected)}')
    if tune_targets.shape != (270, 30, 28) or tune_anchors.shape != (270, 28):
        raise RuntimeError(f'inner_tune target/anchor shape mismatch: {tune_targets.shape}, {tune_anchors.shape}')

    loaded = {}
    for role in ['S_FIT', 'A_FIT', 'V_FIT']:
        arr, ids, rec, rows = load_bundle(role, seal)
        if len(ids) != 270 or len(set(ids)) != 270 or set(ids) != set(tune_ids):
            raise RuntimeError(f'{role} sealed tune rows do not match the explicit target ID set')
        if arr.shape != (270, 30, 28) or not np.isfinite(arr).all():
            raise RuntimeError(f'{role} tune predictions have invalid shape or nonfinite values')
        for row in rows:
            aid = str(row['anchor_id'])
            observed = (str(int(row['episode_id'])), str(int(row['task_id'])), str(row['source_cluster']))
            if aid not in tune_metadata or observed != tune_metadata[aid]:
                raise RuntimeError(f'{role} tune prediction metadata mismatch for anchor_id={aid}')
        loaded[role] = {'prediction': arr, 'ids': ids, 'record': rec, 'rows': rows}
    canonical_ids = loaded['S_FIT']['ids']
    if any(loaded[role]['ids'] != canonical_ids for role in ['A_FIT', 'V_FIT']):
        raise RuntimeError('screen-seed tune prediction ordered IDs differ; explicit join cannot proceed')
    target_index = {aid: i for i, aid in enumerate(tune_ids)}
    reindex = np.asarray([target_index[aid] for aid in canonical_ids], dtype=np.int64)
    tune_targets = tune_targets[reindex]
    tune_anchors = tune_anchors[reindex]
    tune_episodes = tune_episodes[reindex]
    tune_tasks = tune_tasks[reindex]
    tune_groups = [tune_groups[i] for i in reindex.tolist()]
    tune_ids = canonical_ids

    normalizer_path = V2 / 'screening/NORMALIZER_SCREENING.json'
    normalizer = json.loads(normalizer_path.read_text())
    if normalizer.get('status') != 'LOCKED_SCREENING' or int(normalizer.get('inner_tune_anchors_excluded_from_all_fits', -1)) != 270:
        raise RuntimeError('alpha scales are not the accepted inner-fit-only screening scales')
    target_sigma = np.asarray(normalizer['target_sigma'], dtype=np.float64)
    safe_fd_scale = np.asarray(normalizer.get('safe_fd_scale', normalizer.get('fd_scale')), dtype=np.float64)
    if target_sigma.shape != (28,) or safe_fd_scale.shape != (28,) or not np.isfinite(target_sigma).all() or not np.isfinite(safe_fd_scale).all():
        raise RuntimeError('inner-fit target/difference scale vectors invalid')
    if np.any(target_sigma <= 0) or np.any(safe_fd_scale <= 0):
        raise RuntimeError('inner-fit target/difference scales must be positive')

    curves = []
    selected_alphas = {}
    family_members = {'SV': ('S_FIT', 'V_FIT'), 'SS': ('S_FIT', 'A_FIT')}
    for family, (first_role, second_role) in family_members.items():
        first = loaded[first_role]['prediction']
        second = loaded[second_role]['prediction']
        family_scores = []
        for alpha in ALPHAS:
            candidate = (1.0 - alpha) * first + alpha * second
            if not np.isfinite(candidate).all():
                raise FloatingPointError(f'{family} alpha={alpha} is nonfinite')
            per_anchor = score_native(candidate, tune_targets, tune_anchors, target_sigma, safe_fd_scale)
            task_results = {}
            for metric, values in per_anchor.items():
                task_results[metric], _ = task_equal_episode_aggregate(values, tune_episodes, tune_tasks)
            pos_macro = float(np.mean([task_results['position_mae'][t]['episode_equal_mean'] for t in TASKS]))
            fd_macro = float(np.mean([task_results['safe_fd_l1'][t]['episode_equal_mean'] for t in TASKS]))
            seam_macro = float(np.mean([task_results['seam_safe_fd_l1'][t]['episode_equal_mean'] for t in TASKS]))
            j = pos_macro + 0.1 * fd_macro
            rec = {'family': family, 'first_role': first_role, 'second_role': second_role, 'alpha': alpha,
                   'position_mae_macro': pos_macro, 'safe_fd_l1_macro': fd_macro,
                   'seam_safe_fd_l1_macro': seam_macro, 'J': j,
                   'inner_tune_rows': 270, 'episodes': 46, 'tasks': 5}
            for task in TASKS:
                key = str(task)
                for metric in ['position_mae', 'safe_fd_l1', 'seam_safe_fd_l1']:
                    rec[f'task_{task}_{metric}'] = task_results[metric][task]['episode_equal_mean']
                    rec[f'task_{task}_episodes'] = task_results[metric][task]['episodes']
                    rec[f'task_{task}_anchors'] = task_results[metric][task]['anchors']
            curves.append(rec)
            family_scores.append(rec)
        best = min(x['J'] for x in family_scores)
        tied = [x for x in family_scores if x['J'] <= best + 1e-12]
        selected = sorted(tied, key=lambda x: (abs(float(x['alpha']) - 0.5), float(x['alpha'])))[0]
        selected_alphas[family] = {
            'alpha': selected['alpha'], 'selected_J': selected['J'],
            'position_mae_macro': selected['position_mae_macro'],
            'safe_fd_l1_macro': selected['safe_fd_l1_macro'],
            'seam_safe_fd_l1_macro': selected['seam_safe_fd_l1_macro'],
            'tied_candidates_within_abs_1e_12': [float(x['alpha']) for x in tied],
            'tie_break_rule': 'closest to 0.5, then lower alpha',
            'first_role': first_role, 'second_role': second_role,
        }
    if len(curves) != 42:
        raise RuntimeError(f'expected exactly 42 family/alpha risk rows, found {len(curves)}')
    curve_path = ROOT / 'results/INNER_TUNE_ALPHA_CURVES.csv'
    write_csv(curve_path, curves)

    source_map = json.loads((ROOT / 'inventory/SOURCE_MAP.json').read_text())
    program_path = ROOT / 'protocol/PROGRAM_LOCK.json'
    lock = {
        'schema_version': 1, 'status': 'LOCKED_BEFORE_V6_DEV_PREDICTION',
        'study': 'post_V5_controlled_late_fusion_development', 'created_utc': utc(),
        'program_lock_path': str(program_path), 'program_lock_sha256': sha_file(program_path),
        'inner_tune_prediction_seal_path': str(seal_path), 'inner_tune_prediction_seal_sha256': sha_file(seal_path),
        'inner_fit_only_normalizer_path': str(normalizer_path), 'inner_fit_only_normalizer_sha256': sha_file(normalizer_path),
        'train_tune_archive_path': str(train_archive), 'train_tune_archive_sha256': archive_sha,
        'train_tune_archive_logical_rows': 1280, 'target_member_rows_materialized': int(len(target_all)),
        'inner_tune_target_rows_scored': 270, 'inner_fit_targets_used_by_alpha_scoring': 0,
        'target_member_read_after_nonempty_inner_tune_prediction_seal': True,
        'dev_target_archive_opened': False, 'final_heldout_payload_access': 0,
        'inner_tune_was_used_for_v2_recipe_selection': True,
        'interpretation': 'development meta-calibration on previously used inner-tune split; not independent confirmation',
        'scales': {'position_target_sigma_sha256': hashlib.sha256(np.asarray(target_sigma, dtype='<f8').tobytes()).hexdigest(),
                   'safe_fd_scale_sha256': hashlib.sha256(np.asarray(safe_fd_scale, dtype='<f8').tobytes()).hexdigest(),
                   'source_scope': 'inner_fit only', 'outer_train_dev_metric_scales_substituted': False},
        'objective': 'task-equal episode-equal position_mae + 0.1*safe_fd_l1; future-internal 29 differences only',
        'grid': ALPHAS, 'grid_points_per_family': 21,
        'candidate_risk_rows': 42, 'selected_alphas': selected_alphas,
        'selection_target': 'inner_tune only', 'alpha_selected_using_dev': False,
        'alpha_curves_path': str(curve_path), 'alpha_curves_sha256': sha_file(curve_path),
        'native_float64_combination': True, 'one_global_scalar_per_family': True,
        'positive_gain_completion_gate': False, 'minimum_support_completion_gate': False,
    }
    write_json(alpha_path, lock)
    owner_path = ROOT / 'coordination/OWNER_STATE.json'
    owner = json.loads(owner_path.read_text())
    owner.update({'phase': 'alpha_locked', 'status': 'running', 'alpha_selected_using_dev': False,
                  'alpha_sv': selected_alphas['SV']['alpha'], 'alpha_ss': selected_alphas['SS']['alpha'],
                  'inner_tune_loss_evaluations': 42 * 270, 'dev_target_access': 0,
                  'updated_utc': utc()})
    write_json(owner_path, owner)
    print(json.dumps({'status': 'PASS', 'selected_alphas': selected_alphas,
                      'risk_rows': len(curves), 'inner_tune_rows': 270,
                      'dev_target_archive_opened': False}, sort_keys=True))


if __name__ == '__main__':
    main()
