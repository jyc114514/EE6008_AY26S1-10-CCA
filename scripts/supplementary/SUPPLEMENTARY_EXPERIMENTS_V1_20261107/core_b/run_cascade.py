#!/usr/bin/env python3
"""Replay original Core B weights with paired, unrestricted Core A top-1 tasks.

Only task conditioning changes. Original task metadata stays in the prefix and
is used solely by the explicitly named oracle control and later scoring.
Unsupported Core A tasks produce abstentions, never oracle substitutions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

OLD_ROOT = os.environ.get("EE6008_LEGACY_SOURCE_PREFIX", "").rstrip("/")
V9 = 'reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z'
REC = 'reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z'
RUN = os.environ.get("EE6008_LEGACY_REPLAY_REL", "")
MATRIX = 'reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z/protocol/FINAL_MODEL_MATRIX.csv'
METRIC = 'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/protocol/METRIC_SCALE_LOCK.json'
DEV_TARGET = 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/DEV_TARGETS_SCORER_ONLY.npz'
TASKS = (119, 94, 90, 141, 230)
PAIRS = ((20260825, 202610131), (20260826, 202610132), (20260827, 202610133))
PROGRAMS = ('A0', 'AD', 'V0', 'VD', 'S_OPT')
METRICS = ('position_mae', 'safe_fd_l1', 'seam_safe_fd_l1')
PINNED = {
    'v2_model.py': '6e1e967c89afeeb6e676e9265a893fed54f66682f54af3bf04cf3d41766c8c24',
    'locked_source/D3_model.py': '6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589',
    'v9_common.py': 'f023fa7af55f681d1308b0165dbcb4b1fe75091fcf3f9ed1df59d456d1a12615',
}


def require_external_paths():
    """External historical manifests retain their own canonical path prefix."""
    if not OLD_ROOT:
        raise ValueError("Set EE6008_LEGACY_SOURCE_PREFIX to the exact prefix in the authorized source manifests")
    if not RUN or Path(RUN).is_absolute() or ".." in Path(RUN).parts:
        raise ValueError("Set EE6008_LEGACY_REPLAY_REL to the recovery output directory relative to --source-root")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def relocated(root, original):
    if not OLD_ROOT:
        raise ValueError("Set EE6008_LEGACY_SOURCE_PREFIX before relocating external assets")
    p = str(original)
    if p.startswith(OLD_ROOT + '/'):
        p = p[len(OLD_ROOT) + 1:]
    if Path(p).is_absolute() or '..' in Path(p).parts:
        raise ValueError('Path is outside the original asset tree: ' + p)
    return Path(root) / p


def verify(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Hash mismatch for {path}: {actual} != {expected}')
    return actual


def exact_profile_lock(root, expected):
    """Bind a feature manifest to its exact immutable historical lock version."""
    directory = Path(root) / REC / 'core_b'
    current = directory / 'CURRENT_PROFILE_CANDIDATE_LOCK.json'
    candidates = [current] + sorted(p for p in directory.glob('CURRENT_PROFILE_CANDIDATE_LOCK.json*') if p != current)
    for path in candidates:
        if path.is_file() and sha(path) == expected:
            return path
    raise ValueError(f'No current or historical candidate lock has required SHA-256 {expected}')


def prefix_path(root, split):
    role = 'DEV' if split == 'dev' else 'FINAL'
    return root / RUN / f'replay/core_b/{role}_PREFIXES_CURRENT_PROFILE_V2.npz'


def load_prefix(root, split):
    path = prefix_path(root, split)
    with np.load(path, allow_pickle=False) as z:
        if any(any(t in k.lower() for t in ('target', 'future', 'label', 'ground_truth')) for k in z.files):
            raise ValueError('Prefix archive contains a future/target field')
        data = {k: z[k].copy() for k in z.files}
    n = 185 if split == 'dev' else 162
    needed = {'canonical_id', 'anchor_id', 'episode_id', 'task_id', 'cache_index', 'anchor_frame',
              'history_native', 'history_mask', 'relative_delta', 'anchor_native', 'source_cluster'}
    if not needed.issubset(data):
        raise ValueError('Missing prefix fields: ' + str(needed - set(data)))
    if len(data['anchor_id']) != n or len(set(data['anchor_id'].astype(str))) != n:
        raise ValueError('Original fixed anchor population mismatch')
    if len(set(data['episode_id'].tolist())) != (47 if split == 'dev' else 41):
        raise ValueError('Original fixed episode population mismatch')
    shapes = {'history_native': (n, 32, 71), 'history_mask': (n, 32), 'relative_delta': (n, 32), 'anchor_native': (n, 28)}
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid prefix {key} shape/finite values')
    if not set(data['task_id'].astype(int)).issubset(TASKS):
        raise ValueError('Original Core B population includes unregistered task')
    if np.min(data['anchor_frame']) < 119:
        raise ValueError('Core B anchor is earlier than the original 120-frame prefix')
    return data


class Grid:
    """Read the identical current-profile archive, relocating paths only."""
    def __init__(self, root, split, data):
        role = 'DEV' if split == 'dev' else 'FINAL'
        path = root / REC / f'core_b/CURRENT_PROFILE_V2_{role}_FEATURE_MANIFEST.json'
        manifest = read_json(path)
        if manifest.get('status') != 'PASS' or manifest.get('profile_id') != 'CURRENT_PROFILE_V2' or manifest.get('split') != split:
            raise ValueError('Current-profile feature manifest is not accepted for this split')
        prefix_hash_key = 'dev_prefix_sha256' if split == 'dev' else 'final_prefix_sha256'
        verify(prefix_path(root, split), manifest[prefix_hash_key])
        lock_path = exact_profile_lock(root, manifest['profile_lock_sha256'])
        verify(lock_path, manifest['profile_lock_sha256'])
        archive_path = relocated(root, manifest['feature_archive_path'])
        verify(archive_path, manifest['feature_archive_sha256'])
        with np.load(archive_path, allow_pickle=False) as z:
            ids, anchors = z['canonical_ids'].astype(str), z['anchor_ids'].astype(str)
            episodes, tasks, indices = z['episode_ids'].astype(int), z['task_ids'].astype(int), z['cache_indices'].astype(int)
            features = z['grid_features'].copy()
        n = len(ids)
        if features.shape != (n, 2, 16, 4, 768) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError('Current-profile feature shape/dtype/finite mismatch')
        for values, key in ((ids, 'canonical_id'), (anchors, 'anchor_id'), (episodes, 'episode_id'), (tasks, 'task_id'), (indices, 'cache_index')):
            if not np.array_equal(values, data[key]):
                raise ValueError('Current-profile feature identity/order mismatch: ' + key)
        if len(set(indices.tolist())) != n:
            raise ValueError('Duplicate cache indices')
        for values, key in ((ids, 'canonical_id_order_sha256'), (anchors, 'anchor_id_order_sha256')):
            if hashlib.sha256('\n'.join(values.tolist()).encode()).hexdigest() != manifest[key]:
                raise ValueError('Current-profile manifest order hash mismatch')
        self.rows = {int(index): (str(anchor), int(ep), features[i]) for i, (index, anchor, ep) in enumerate(zip(indices, anchors, episodes))}
        self.receipt = {'manifest_path': str(path), 'manifest_sha256': sha(path), 'archive_path': str(archive_path),
                        'archive_sha256': sha(archive_path), 'profile_lock_path': str(lock_path),
                        'profile_lock_sha256': sha(lock_path)}

    def row(self, index, anchor, episode):
        a, e, values = self.rows[int(index)]
        if a != str(anchor) or e != int(episode):
            raise ValueError('Feature row identity mismatch')
        return values


def load_classification(path, data, a_seed):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
    else:
        obj = read_json(path)
        rows = obj if isinstance(obj, list) else obj['predictions']
    if not rows:
        raise ValueError('Empty Core A predictions')
    use_anchor = all(str(r.get('anchor_id', '')).strip() for r in rows)
    key = 'anchor_id' if use_anchor else 'episode_id'
    lookup = {}
    for row in rows:
        value = str(row[key]) if use_anchor else str(int(row[key]))
        if value in lookup:
            raise ValueError('Duplicate Core A prediction identity: ' + value)
        for required in ('predicted_task_id', 'observation_end_frame'):
            if required not in row:
                raise ValueError('Core A output must explicitly provide ' + required)
        if 'seed' in row and str(row['seed']).strip() and int(row['seed']) != a_seed:
            raise ValueError('Core A prediction seed differs from fixed pairing')
        lookup[value] = row
    audit, predicted, valid = [], [], []
    for i in range(len(data['anchor_id'])):
        value = str(data[key][i]) if use_anchor else str(int(data[key][i]))
        row = lookup.get(value)
        task, cutoff = (int(row['predicted_task_id']), int(row['observation_end_frame'])) if row else (-1, -1)
        reason = 'accepted'
        if row is None:
            reason = 'missing_core_a_prediction'
        elif cutoff < 0:
            reason = 'missing_observation_cutoff'
        elif cutoff > int(data['anchor_frame'][i]):
            reason = 'classification_observes_after_core_b_anchor'
        elif task not in TASKS:
            reason = 'unsupported_core_a_task'
        predicted.append(task)
        valid.append(reason == 'accepted')
        audit.append({'anchor_id': str(data['anchor_id'][i]), 'episode_id': int(data['episode_id'][i]),
                      'true_task_id': int(data['task_id'][i]), 'predicted_task_id': task,
                      'observation_end_frame': cutoff, 'anchor_frame': int(data['anchor_frame'][i]),
                      'core_a_seed': a_seed, 'status': reason})
    return np.asarray(predicted, dtype=np.int64), np.asarray(valid, dtype=bool), audit


def save_prediction(path, data, values, valid, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, canonical_ids=data['canonical_id'].astype(str), prediction_native=values,
                        valid=np.asarray(valid, dtype=bool))
    return dict(path=str(path), sha256=sha(path), **metadata)


def run(args):
    require_external_paths()
    import torch
    from safetensors.torch import load_file

    root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(output))
    code = root / V9 / 'code'
    for relative, expected in PINNED.items():
        verify(code / relative, expected)
    sys.path.insert(0, str(code))
    model_mod, common = importlib.import_module('v2_model'), importlib.import_module('v9_common')
    data = load_prefix(root, args.split)
    n = len(data['anchor_id'])
    grid = Grid(root, args.split, data)
    with (root / MATRIX).open(encoding='utf-8-sig', newline='') as f:
        matrix = list(csv.DictReader(f))
    matrix = [r for r in matrix if r['core'] == 'B' and r['program'] in PROGRAMS and r['status'].startswith('accepted_source_bound')]
    if len(matrix) != 15 or len({(r['program'], int(r['seed'])) for r in matrix}) != 15:
        raise ValueError('Expected original 15 Core B models')
    source_predictions = {}
    for spec in args.core_a:
        seed, path = spec.split('=', 1)
        if int(seed) in source_predictions:
            raise ValueError('Repeated Core A seed')
        source_predictions[int(seed)] = Path(path).resolve()
    if set(source_predictions) != {p[0] for p in PAIRS}:
        raise ValueError('Exactly the three original Core A seeds must be supplied')
    classifiers = {seed: load_classification(path, data, seed) for seed, path in source_predictions.items()}
    output.mkdir(parents=True)
    recipe = {'split': args.split, 'source_root': str(root), 'prefix_path': str(prefix_path(root, args.split)),
              'prefix_sha256': sha(prefix_path(root, args.split)), 'source_matrix_sha256': sha(root / MATRIX),
              'core_a_source': args.core_a_source, 'seed_pairs': [list(p) for p in PAIRS],
              'pairing_rule': 'fixed positional pairing before outcomes, no best-seed selection',
              'class_policy': 'unrestricted Core A top-1; reject unsupported tasks; no truth fallback',
              'causality_rule': 'core_a observation_end_frame <= core_b anchor_frame for every accepted row',
              'new_training_runs': 0, 'feature_receipt': grid.receipt, 'batch_size': 64,
              'classifiers': {str(s): {'path': str(p), 'sha256': sha(p)} for s, p in source_predictions.items()},
              'started_utc': datetime.now(timezone.utc).isoformat(), 'predictions': []}
    write_json(output / 'RUN_CONTRACT.json', recipe)
    recipe['classification_audits'] = {}
    for seed, (_, _, audit) in classifiers.items():
        audit_path = output / f'classification_audit_seed{seed}.csv'
        csv_write(audit_path, audit)
        recipe['classification_audits'][str(seed)] = {'path': str(audit_path), 'sha256': sha(audit_path)}
    if args.preflight:
        for row in matrix:
            verify(relocated(root, row['checkpoint_path']), row['checkpoint_sha256'])
            verify(relocated(root, row['model_manifest_path']), row['model_manifest_sha256'])
            verify(relocated(root, row['normalizer_path']), row['normalizer_sha256'])
        recipe['status'] = 'PREFLIGHT_PASS_NO_PREDICTIONS'
        write_json(output / 'PREFLIGHT.json', recipe)
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    records, predictions = [], {}
    for a_seed, b_seed in PAIRS:
        predicted_task, valid, audit = classifiers[a_seed]
        predicted_data = dict(data)
        predicted_data['task_id'] = predicted_task  # isolated model input view; true metadata stays untouched.
        for row in sorted((r for r in matrix if int(r['seed']) == b_seed), key=lambda r: PROGRAMS.index(r['program'])):
            program, mode = row['program'], row['mode']
            model_path = relocated(root, row['checkpoint_path'])
            verify(model_path, row['checkpoint_sha256'])
            verify(relocated(root, row['model_manifest_path']), row['model_manifest_sha256'])
            normalizer_path = relocated(root, row['normalizer_path'])
            verify(normalizer_path, row['normalizer_sha256'])
            normalizer = read_json(normalizer_path)
            torch.manual_seed(b_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(b_seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = program != 'S_OPT'
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            model = model_mod.V2TaskConditionedPredictor(seed=b_seed, video_mode=mode, waypoint_mode=None)
            model.load_state_dict(load_file(str(model_path), device='cpu'), strict=True)
            if model_mod.state_hash(model) != row['model_state_hash']:
                raise ValueError('Original model state hash mismatch')
            model = model.to(device).eval()
            for condition, source, mask in (('oracle_task', data, np.ones(n, bool)), ('core_a_task', predicted_data, valid)):
                values = np.full((n, 30, 28), np.nan, dtype=np.float64)
                indices = np.flatnonzero(mask)
                with torch.no_grad():
                    for start in range(0, len(indices), 64):
                        ix = indices[start:start + 64]
                        prefix, task, anchor = common.model_input(source, ix, normalizer, mode, device, grid if mode != 'STATE-CTRL' else None)
                        residual = model(prefix, task).action_residual
                        sigma = torch.as_tensor(normalizer['target_sigma'], device=device, dtype=torch.float32)
                        native = anchor[:, None, :] + sigma[None, None, :] * residual
                        values[ix] = native.detach().cpu().numpy().astype(np.float64)
                if not np.isfinite(values[mask]).all() or np.isfinite(values[~mask]).any():
                    raise ValueError('Prediction validity/abstention contract failed')
                path = output / 'predictions' / condition / program / f'A{a_seed}_B{b_seed}.npz'
                records.append(save_prediction(path, data, values, mask, condition=condition, program=program,
                                               a_seed=a_seed, b_seed=b_seed, checkpoint_sha256=row['checkpoint_sha256']))
                predictions[(condition, program, b_seed)] = (values, mask)
            del model
        for condition in ('oracle_task', 'core_a_task'):
            for mix, video in (('MIX0', 'V0'), ('MIXD', 'VD')):
                state, mask = predictions[(condition, 'S_OPT', b_seed)]
                visual, mask_v = predictions[(condition, video, b_seed)]
                if not np.array_equal(mask, mask_v):
                    raise ValueError('Ensemble masks mismatch')
                values = 0.5 * state + 0.5 * visual
                records.append(save_prediction(output / 'predictions' / condition / mix / f'A{a_seed}_B{b_seed}.npz',
                    data, values, mask, condition=condition, program=mix, a_seed=a_seed, b_seed=b_seed))
    persistence = np.repeat(data['anchor_native'].astype(np.float64)[:, None, :], 30, axis=1)
    records.append(save_prediction(output / 'predictions' / 'PERSISTENCE.npz', data, persistence, np.ones(n, bool),
                                   condition='persistence', program='PERSISTENCE', a_seed=None, b_seed=None))
    recipe.update(status='PREDICTIONS_SEALED', predictions=records,
                  completed_utc=datetime.now(timezone.utc).isoformat(), future_targets_loaded=False)
    write_json(output / 'PREDICTION_SEAL.json', recipe)
    print(json.dumps({'status': recipe['status'], 'predictions': len(records), 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--split', choices=['dev', 'final_test'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--core-a', action='append', required=True, metavar='SEED=CSV_OR_JSON')
    parser.add_argument('--core-a-source', default='original_dynamic_A')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
