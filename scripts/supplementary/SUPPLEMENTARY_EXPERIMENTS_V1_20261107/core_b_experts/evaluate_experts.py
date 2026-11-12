#!/usr/bin/env python3
"""Freeze, predict and score fresh shared V0 versus five class-specific V0 experts.

Inference never opens future targets. Both dev and final predictions must be
sealed before score can open either target archive. Historical cascade scores
are retained separately and never pooled with this study's fitted models.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

# On the server these two already-reviewed modules live beside this script.
# The sibling fallback supports local synthetic checks without copying files.
if not (Path(__file__).parent / 'run_cascade.py').is_file():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core_b'))
import run_cascade as replay
import score_cascade as scoring

TASKS, PAIRS, METRICS = replay.TASKS, replay.PAIRS, replay.METRICS
SEEDS = tuple(b for _, b in PAIRS)
SPLITS = ('dev', 'final_test')
SYSTEMS = ('SHARED_V0', 'EXPERT_V0')
CONDITIONS = ('oracle_task', 'core_a_task')
BOOTSTRAP_SEED, DRAWS = 2026110702, 5000
MODE = 'TS11-D'
NORMALIZER_SHA = '22f66a54f6106de68c1a0bf0aa3a46ca3340c3077b4121c6a0fdf9d83528d65d'
METRIC_SHA = 'dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32'
sha, verify, read_json = replay.sha, replay.verify, replay.read_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def write_csv(path, rows):
    require(bool(rows), 'Refusing empty result table: ' + str(path))
    replay.csv_write(path, rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def locked_json(path, contract):
    if path.exists():
        require(read_json(path)['contract'] == contract, 'Immutable contract changed: ' + str(path))
    else:
        atomic_json(path, {'created_utc': utc(), 'contract': contract})
    return sha(path)


def classification_path(run_root, split, seed):
    return run_root / 'inputs' / f'core_a_dynamic_{split}_seed{seed}.csv'


def source_asset(source, value):
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            candidate.resolve().relative_to(source.resolve())
            return candidate.resolve()
        except ValueError:
            pass
    return replay.relocated(source, value)


def validate_population(data):
    """The original episode bootstrap is valid only for a 1:1 source mapping."""
    require(set(data['task_id'].astype(int)) == set(TASKS), 'Population lacks a fixed task')
    episode_map, cluster_map = {}, {}
    for episode, task, cluster in zip(data['episode_id'], data['task_id'], data['source_cluster']):
        episode, task, cluster = int(episode), int(task), str(cluster)
        require(episode_map.setdefault(episode, (task, cluster)) == (task, cluster),
                'Episode maps to multiple tasks/source clusters')
        require(cluster_map.setdefault(cluster, (task, episode)) == (task, episode),
                'Source cluster maps to multiple tasks/episodes')
    require(len(set(data['canonical_id'].astype(str))) == len(data['canonical_id']), 'Duplicate canonical IDs')
    return episode_map


def resampling_plan(data, mask, seed=BOOTSTRAP_SEED, draws=DRAWS):
    """One draw per episode/source cluster, sorted by original cluster string."""
    validate_population(data)
    rng, plans = np.random.default_rng(seed), {}
    for task in TASKS:
        ix = mask & (data['task_id'] == task)
        clusters = sorted(set(data['source_cluster'][ix].astype(str)))
        episodes = np.asarray([int(data['episode_id'][ix & (data['source_cluster'].astype(str) == c)][0])
                               for c in clusters], dtype=np.int64)
        indices = (rng.integers(0, len(clusters), size=(draws, len(clusters)), dtype=np.int32)
                   if clusters else np.empty((draws, 0), dtype=np.int32))
        plans[task] = {'cluster_ids': np.asarray(clusters, dtype=str), 'episode_ids': episodes,
                       'indices': indices}
    return plans


def load_inputs(source, run_root, with_grid):
    inputs = {}
    for split in SPLITS:
        data = replay.load_prefix(source, split)
        validate_population(data)
        classifiers = {}
        for a_seed, _ in PAIRS:
            path = classification_path(run_root, split, a_seed)
            classifiers[a_seed] = replay.load_classification(path, data, a_seed)
        inputs[split] = {'data': data, 'classifiers': classifiers,
                         'grid': replay.Grid(source, split, data) if with_grid else None}
    return inputs


def establish_lock(source, run_root, inputs, historical_root):
    evaluation = run_root / 'evaluation'
    evaluation.mkdir(parents=True, exist_ok=True)
    for relative, expected in replay.PINNED.items():
        verify(source / replay.V9 / 'code' / relative, expected)
    verify(source / replay.METRIC, METRIC_SHA)
    contract = {
        'schema_version': 1, 'source_root': str(source), 'run_root': str(run_root),
        'architecture': 'V2TaskConditionedPredictor', 'video_mode': MODE,
        'task_order': list(TASKS), 'seed_pairs': [list(p) for p in PAIRS],
        'systems': list(SYSTEMS), 'conditions': list(CONDITIONS), 'splits': list(SPLITS),
        'model_count': 18, 'prediction_bundles_per_split': 13, 'batch_size': 64,
        'class_source': 'original_dynamic_CoreA_unrestricted_120_class_top1',
        'route_rule': 'oracle: true class; core_a: predicted class and same task token; unsupported/missing/post-anchor abstain; no fallback',
        'model_selection': 'all 18 fresh fixed last-epoch160 fits, no dev/final selection',
        'normalizer_sha256': NORMALIZER_SHA, 'metric_scale_sha256': METRIC_SHA,
        'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_draws': DRAWS,
        'bootstrap_method': 'int32 PCG64; sorted source clusters within fixed task; 1:1 episode-source check; fixed seed-pair average; same draws across all systems/contrasts; percentile95; no seed resampling',
        'aggregation': 'anchor mean within episode, equal episodes within true task, equal five-task macro; missing task makes fixed macro not estimable',
        'capacity_qualification': 'five full-size expert models per seed versus one shared model; 5x total model parameters but one expert active per row; same160epochs with fewer updates per expert',
        'history_qualification': 'historical cascade remains a separate reference; environment/profile differences prevent attributing historical differences solely to specialization',
        'source_code_sha256': replay.PINNED,
        'implementation_sha256': {Path(p).name: sha(p) for p in (__file__, replay.__file__, scoring.__file__, run_root / 'scripts/train_experts.py')},
        'inputs': {}, 'bootstrap_plans': {}, 'historical_reference': {}}
    for split in SPLITS:
        history = historical_root / split
        require(read_json(history / 'PREDICTION_SEAL.json')['status'] == 'PREDICTIONS_SEALED', 'Historical cascade predictions are not sealed')
        require(read_json(history / 'scores/SCORE_RECEIPT.json')['status'] == 'PASS', 'Historical cascade scoring incomplete')
        contract['historical_reference'][split] = {
            name: {'path': str(history / name), 'sha256': sha(history / name)}
            for name in ('PREDICTION_SEAL.json', 'scores/SCORE_RECEIPT.json', 'scores/SUMMARY_MEAN_SD.csv')}
    for split, item in inputs.items():
        data, classifiers = item['data'], item['classifiers']
        full = np.ones(len(data['anchor_id']), bool)
        common = np.logical_and.reduce([classifiers[a][1] for a, _ in PAIRS])
        arrays, metadata = {}, {}
        for name, mask in (('full', full), ('common', common)):
            plan = resampling_plan(data, mask)
            for task, part in plan.items():
                for field, value in part.items():
                    arrays[f'{name}_task{task}_{field}'] = value
            metadata[name] = {'anchors': int(mask.sum()), 'episodes': len(np.unique(data['episode_id'][mask])),
                              'episodes_by_task': {str(t): len(plan[t]['episode_ids']) for t in TASKS},
                              'fixed_five_task_estimable': all(len(plan[t]['episode_ids']) for t in TASKS)}
        plan_path = evaluation / f'BOOTSTRAP_PLAN_{split}.npz'
        if plan_path.exists():
            with np.load(plan_path, allow_pickle=False) as saved:
                require(set(saved.files) == set(arrays), 'Locked bootstrap schema changed')
                require(all(np.array_equal(saved[k], v) and saved[k].dtype == v.dtype for k, v in arrays.items()),
                        'Locked bootstrap draws changed')
        else:
            save_npz(plan_path, arrays)
        contract['bootstrap_plans'][split] = {'path': str(plan_path), 'sha256': sha(plan_path), **metadata}
        contract['inputs'][split] = {
            'prefix_path': str(replay.prefix_path(source, split)), 'prefix_sha256': sha(replay.prefix_path(source, split)),
            'classifiers': {str(a): {'path': str(classification_path(run_root, split, a)),
                                   'sha256': sha(classification_path(run_root, split, a))} for a, _ in PAIRS}}
    locked_json(evaluation / 'EVALUATION_LOCK.json', contract)
    return evaluation, contract


def freeze_models(source, run_root, evaluation):
    rows = []
    training_runner_sha = sha(run_root / 'scripts/train_experts.py')
    for seed in SEEDS:
        for task in (None, *TASKS):
            folder = 'shared' if task is None else f'task_{task}'
            directory = run_root / 'models' / folder / f'seed{seed}'
            manifest_path, done_path = directory / 'MODEL_MANIFEST.json', directory / 'RUN_COMPLETE.json'
            manifest, done = read_json(manifest_path), read_json(done_path)
            require(manifest['status'] == done['status'] == 'PASS', 'Incomplete fit: ' + str(directory))
            verify(manifest_path, done['model_manifest_sha256'])
            require(manifest['mode'] == ('shared' if task is None else 'expert') and manifest['task_id'] == task,
                    'Model role differs from fixed matrix')
            require(int(manifest['seed']) == seed, 'Model seed mismatch')
            require(manifest['video_mode'] == MODE and manifest['architecture'] == 'V2TaskConditionedPredictor', 'Model architecture differs')
            require(int(manifest['epochs']) == int(manifest['epochs_completed']) == int(done['epochs_completed']) == 160, 'Not a complete fixed160epoch fit')
            require(manifest['new_train_from_fresh_init'] is True, 'Model is not a fresh fit')
            require(manifest['training_runner_sha256'] == training_runner_sha, 'Fitted models used a different training implementation')
            for receipt in (manifest, done):
                require(receipt['dev_or_final_target_rows'] == receipt['final_heldout_payload_access'] == 0, 'Heldout targets used by training')
            expected_tasks = list(TASKS) if task is None else [task]
            require(len(manifest['task_ids']) == len(expected_tasks) and set(manifest['task_ids']) == set(expected_tasks), 'Training task subset differs')
            require(tuple(manifest.get('task_order', TASKS)) == TASKS, 'Five-way task-token order differs')
            require(manifest['constant_task_rank'] == (None if task is None else TASKS.index(task)), 'Expert token rank differs')
            path = directory / 'model.safetensors'
            verify(path, manifest['model_file_sha256'])
            require(done['model_sha256'] == manifest['model_file_sha256'] and done['model_state_hash'] == manifest['final_state_hash'], 'Completion checkpoint identity differs')
            normalizer = source_asset(source, manifest['normalizer_path'])
            require(manifest['normalizer_sha256'] == NORMALIZER_SHA, 'Normalizer differs from original global training normalizer')
            verify(normalizer, NORMALIZER_SHA)
            count, updates = int(manifest['train_rows']), int(manifest['optimizer_steps'])
            require(count > 0 and int(manifest['steps_per_epoch']) == math.ceil(count / 64), 'Training update schedule invalid')
            require(updates == 160 * math.ceil(count / 64), 'Training updates do not match160epochs')
            rows.append({'family': 'shared' if task is None else 'expert', 'task_id': task, 'seed': seed,
                         'checkpoint_path': str(path), 'checkpoint_sha256': sha(path), 'model_state_hash': manifest['final_state_hash'],
                         'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
                         'complete_path': str(done_path), 'complete_sha256': sha(done_path),
                         'normalizer_path': str(normalizer), 'normalizer_sha256': NORMALIZER_SHA,
                         'train_rows': count, 'train_episodes': int(manifest['train_episodes']),
                         'epochs': 160, 'optimizer_steps': updates, 'steps_per_epoch': int(manifest['steps_per_epoch']),
                         'parameter_count': int(manifest['parameter_count'])})
    require(len({r['parameter_count'] for r in rows}) == 1 and rows[0]['parameter_count'] > 0, 'Model capacities differ unexpectedly')
    for seed in SEEDS:
        shared = next(r for r in rows if r['seed'] == seed and r['family'] == 'shared')
        experts = [r for r in rows if r['seed'] == seed and r['family'] == 'expert']
        require(shared['train_rows'] == 1280 and sum(r['train_rows'] for r in experts) == shared['train_rows'], 'Expert subsets do not partition original1280 training rows')
        require(sum(r['train_episodes'] for r in experts) == shared['train_episodes'], 'Expert subsets do not partition training episodes')
    locked_json(evaluation / 'MODEL_MATRIX.json', {'status': 'FROZEN', 'training_runner_sha256': training_runner_sha, 'models': rows})
    return rows


def route_partition(route, valid, task=None):
    """A shared model receives all valid rows; an expert only its routed class."""
    route, valid = np.asarray(route, dtype=np.int64), np.asarray(valid, dtype=bool)
    require(route.shape == valid.shape, 'Routing shape mismatch')
    require(set(route[valid].tolist()).issubset(TASKS), 'Accepted route outside five tasks')
    return valid.copy() if task is None else valid & (route == task)


def predict(source, run_root, evaluation, contract, inputs, models, device_name):
    seal_path = evaluation / 'PREDICTION_SEAL.json'
    require(not seal_path.exists(), 'Predictions are already globally sealed; score them without rerunning inference')
    require(not any((evaluation / s / 'scores').exists() for s in SPLITS), 'Scoring already started; predictions cannot change')
    import torch
    from safetensors.torch import load_file
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(source / replay.V9 / 'code'))
    model_mod, common = importlib.import_module('v2_model'), importlib.import_module('v9_common')
    device = torch.device(device_name)
    require(device.type != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    bundles, audits, feature_receipts = {}, {}, {}
    for split, item in inputs.items():
        data, classifiers = item['data'], item['classifiers']
        n = len(data['anchor_id'])
        feature_receipts[split] = item['grid'].receipt
        audits[split] = {}
        for a_seed, b_seed in PAIRS:
            predicted, valid, audit = classifiers[a_seed]
            path = evaluation / split / f'classification_audit_seed{a_seed}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(path, audit)
            audits[split][str(a_seed)] = {'path': str(path), 'sha256': sha(path)}
            for system in SYSTEMS:
                for condition in CONDITIONS:
                    mask = np.ones(n, bool) if condition == 'oracle_task' else valid
                    route = data['task_id'].astype(np.int64) if condition == 'oracle_task' else predicted
                    bundles[split, system, condition, b_seed] = {
                        'values': np.full((n, 30, 28), np.nan, dtype=np.float64), 'valid': mask.copy(),
                        'route': route.copy(), 'written': np.zeros(n, dtype=np.int32)}
    for row in models:
        seed, task = row['seed'], row['task_id']
        system = 'SHARED_V0' if row['family'] == 'shared' else 'EXPERT_V0'
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        model = model_mod.V2TaskConditionedPredictor(seed=seed, video_mode=MODE, waypoint_mode=None)
        model.load_state_dict(load_file(row['checkpoint_path'], device='cpu'), strict=True)
        require(model_mod.state_hash(model) == row['model_state_hash'], 'Loaded checkpoint state hash differs')
        require(sum(p.numel() for p in model.parameters()) == row['parameter_count'], 'Loaded model parameter count differs')
        model.to(device).eval()
        normalizer = read_json(row['normalizer_path'])
        sigma = torch.as_tensor(normalizer['target_sigma'], device=device, dtype=torch.float32)
        for split, item in inputs.items():
            for condition in CONDITIONS:
                bundle = bundles[split, system, condition, seed]
                selected = route_partition(bundle['route'], bundle['valid'], task)
                indices = np.flatnonzero(selected)
                routed = dict(item['data'])
                routed['task_id'] = bundle['route']
                with torch.no_grad():
                    for start in range(0, len(indices), 64):
                        ix = indices[start:start + 64]
                        prefix, token, anchor = common.model_input(routed, ix, normalizer, MODE, device, item['grid'])
                        if task is not None:
                            require(bool(torch.all(token == TASKS.index(task))), 'Expert received wrong constant task token')
                        values = anchor[:, None] + sigma[None, None] * model(prefix, token).action_residual
                        bundle['values'][ix] = values.detach().cpu().numpy().astype(np.float64)
                        bundle['written'][ix] += 1
        del model
        print(json.dumps({'status': 'MODEL_PREDICTIONS_COMPLETE', 'family': row['family'], 'task_id': task, 'seed': seed}), flush=True)
    records = []
    for (split, system, condition, b_seed), bundle in bundles.items():
        data = inputs[split]['data']
        a_seed = next(a for a, b in PAIRS if b == b_seed)
        valid, values, route = bundle['valid'], bundle['values'], bundle['route']
        require(np.all(bundle['written'][valid] == 1) and np.all(bundle['written'][~valid] == 0), 'Rows were omitted or routed twice')
        require(np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all(), 'Prediction finite/abstention contract differs')
        path = evaluation / split / 'predictions' / system / condition / f'A{a_seed}_B{b_seed}.npz'
        save_npz(path, {'canonical_ids': data['canonical_id'].astype(str), 'prediction_native': values,
                        'valid': valid, 'conditioning_task_id': np.where(valid, route, -1),
                        'expert_task_id': np.where(valid, route, -1) if system == 'EXPERT_V0' else np.full(len(valid), -1, dtype=np.int64)})
        records.append({'split': split, 'system': system, 'condition': condition, 'a_seed': a_seed, 'b_seed': b_seed,
                        'path': str(path), 'sha256': sha(path)})
    for split, item in inputs.items():
        data = item['data']
        path = evaluation / split / 'predictions/PERSISTENCE.npz'
        save_npz(path, {'canonical_ids': data['canonical_id'].astype(str),
                        'prediction_native': np.repeat(data['anchor_native'].astype(np.float64)[:, None], 30, axis=1),
                        'valid': np.ones(len(data['anchor_id']), bool)})
        records.append({'split': split, 'system': 'PERSISTENCE', 'condition': 'persistence', 'a_seed': None,
                        'b_seed': None, 'path': str(path), 'sha256': sha(path)})
    require(len(records) == 26, 'Not all dev/final predictions were produced')
    # Re-check all fitted artifacts after inference before releasing the seal.
    freeze_models(source, run_root, evaluation)
    seal = {'status': 'ALL_DEV_FINAL_PREDICTIONS_SEALED', 'created_utc': utc(), 'future_targets_loaded': False,
            'evaluation_lock_sha256': sha(evaluation / 'EVALUATION_LOCK.json'),
            'model_matrix_sha256': sha(evaluation / 'MODEL_MATRIX.json'), 'prediction_count': 26,
            'classification_audits': audits, 'feature_receipts': feature_receipts, 'predictions': records,
            'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'device': str(device), 'batch_size': 64, 'tf32': False, 'amp': False,
                        'deterministic_algorithms': True},
            'target_read_policy': 'both dev and final sealed before either future target archive is opened'}
    atomic_json(seal_path, seal)
    print(json.dumps({'status': seal['status'], 'prediction_count': 26, 'seal': str(seal_path)}))


def verify_seal(evaluation, contract, inputs):
    seal_path = evaluation / 'PREDICTION_SEAL.json'
    seal = read_json(seal_path)
    require(seal['status'] == 'ALL_DEV_FINAL_PREDICTIONS_SEALED' and seal['future_targets_loaded'] is False,
            'Both splits must be sealed before targets are read')
    verify(evaluation / 'EVALUATION_LOCK.json', seal['evaluation_lock_sha256'])
    verify(evaluation / 'MODEL_MATRIX.json', seal['model_matrix_sha256'])
    bundles = {}
    for record in seal['predictions']:
        split, system, condition, seed = (record[k] for k in ('split', 'system', 'condition', 'b_seed'))
        key = split, system, condition, seed
        require(key not in bundles and split in SPLITS, 'Unexpected duplicate/split in global seal')
        data = inputs[split]['data']
        n = len(data['anchor_id'])
        verify(record['path'], record['sha256'])
        with np.load(record['path'], allow_pickle=False) as z:
            expected_fields = {'canonical_ids', 'prediction_native', 'valid'}
            if system != 'PERSISTENCE':
                expected_fields |= {'conditioning_task_id', 'expert_task_id'}
            require(set(z.files) == expected_fields, 'Unexpected fields in prediction-only bundle')
            require(np.array_equal(z['canonical_ids'].astype(str), data['canonical_id'].astype(str)), 'Sealed row identity differs')
            values, valid = z['prediction_native'].copy(), z['valid'].copy()
            require(values.shape == (n, 30, 28) and valid.shape == (n,) and valid.dtype == np.bool_, 'Prediction schema differs')
            require(np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all(), 'Prediction abstention contract differs')
            if system == 'PERSISTENCE':
                require(condition == 'persistence' and seed is None and record['a_seed'] is None and valid.all(), 'Persistence record invalid')
                require(np.array_equal(values, np.repeat(data['anchor_native'].astype(np.float64)[:, None], 30, axis=1)), 'Persistence values differ')
            else:
                require(system in SYSTEMS and condition in CONDITIONS and seed in SEEDS, 'Unexpected prediction cell')
                a_seed = next(a for a, b in PAIRS if b == seed)
                require(record['a_seed'] == a_seed, 'Seed pairing differs')
                predicted, accepted, _ = inputs[split]['classifiers'][a_seed]
                expected = np.ones(n, bool) if condition == 'oracle_task' else accepted
                route = data['task_id'] if condition == 'oracle_task' else predicted
                require(np.array_equal(valid, expected), 'Mask differs from audited classification acceptance')
                require(np.array_equal(z['conditioning_task_id'], np.where(valid, route, -1)), 'Sealed conditioning token differs')
                expert = np.where(valid, route, -1) if system == 'EXPERT_V0' else np.full(n, -1)
                require(np.array_equal(z['expert_task_id'], expert), 'Sealed expert route differs')
        bundles[key] = values, valid
    expected = {(s, p, c, b) for s in SPLITS for p in SYSTEMS for c in CONDITIONS for b in SEEDS}
    expected |= {(s, 'PERSISTENCE', 'persistence', None) for s in SPLITS}
    require(set(bundles) == expected and seal['prediction_count'] == 26, 'Global prediction matrix incomplete')
    for split, item in inputs.items():
        for a_seed, _ in PAIRS:
            record = seal['classification_audits'][split][str(a_seed)]
            verify(record['path'], record['sha256'])
            saved = read_csv(record['path'])
            expected_audit = [{k: str(v) for k, v in row.items()} for row in item['classifiers'][a_seed][2]]
            require(saved == expected_audit, 'Audit differs from original classification CSV')
        feature = seal['feature_receipts'][split]
        for path_key, sha_key in (('manifest_path', 'manifest_sha256'), ('archive_path', 'archive_sha256'), ('profile_lock_path', 'profile_lock_sha256')):
            verify(feature[path_key], feature[sha_key])
    return seal, bundles


def fixed_aggregate(values, mask, data):
    avg, tasks, episodes = scoring.aggregate(values, mask, data)
    estimable = {r['task_id'] for r in tasks} == set(TASKS)
    if not estimable:
        avg = {m: None for m in METRICS}
    by_task = {r['task_id']: r for r in tasks}
    full_tasks = [by_task.get(t, {'task_id': t, 'episode_support': 0, 'anchor_support': 0,
                                'estimand_status': 'not_estimable_no_accepted_episode',
                                **{m: None for m in METRICS}}) for t in TASKS]
    return {'estimand_status': 'estimable' if estimable else 'not_estimable_missing_true_task',
            'anchor_support': int(mask.sum()), 'episode_support': len(np.unique(data['episode_id'][mask])),
            'task_support': len(tasks), **avg}, full_tasks, episodes


def bootstrap_values(values, mask, data, plan):
    """Fixed-model episode means; return per-task and five-task bootstrap arrays."""
    task_draws, task_points = {}, {}
    for task in TASKS:
        episodes, indices = plan[task]['episode_ids'], plan[task]['indices']
        if len(episodes) == 0:
            continue
        means = np.stack([values[mask & (data['task_id'] == task) & (data['episode_id'] == ep)].mean(axis=0) for ep in episodes])
        require(np.isfinite(means).all(), 'Nonfinite bootstrap episode mean')
        task_points[task], task_draws[task] = means.mean(axis=0), means[indices].mean(axis=1)
    point = np.mean(list(task_points.values()), axis=0) if len(task_points) == 5 else None
    draws = np.mean(list(task_draws.values()), axis=0) if len(task_draws) == 5 else None
    return point, draws, task_points, task_draws


def read_plan(path, cohort):
    with np.load(path, allow_pickle=False) as z:
        return {task: {field: z[f'{cohort}_task{task}_{field}'].copy()
                       for field in ('episode_ids', 'cluster_ids', 'indices')} for task in TASKS}


def summarize(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, group in grouped.items():
        row = dict(zip(group_keys, key))
        row.update(seed_pairs=len(group), anchor_support_min=min(r['anchor_support'] for r in group),
                   anchor_support_max=max(r['anchor_support'] for r in group))
        for metric in METRICS:
            values = [r[metric] for r in group]
            complete = all(v is not None for v in values)
            row[metric + '_mean'] = float(np.mean(values)) if complete else None
            row[metric + '_sample_sd'] = float(np.std(values, ddof=1)) if complete and len(values) > 1 else None
        row['estimand_status'] = 'estimable' if all(r.get('estimand_status', 'estimable') == 'estimable' for r in group) else 'not_estimable_missing_true_task'
        out.append(row)
    return out


def interval_rows(meta, values, mask, data, plan):
    point, draws, task_points, task_draws = bootstrap_values(values, mask, data, plan)
    rows, tasks = [], []
    for task in TASKS:
        for j, metric in enumerate(METRICS):
            if task in task_points:
                low, high = np.quantile(task_draws[task][:, j], [.025, .975])
                value = float(task_points[task][j])
            else:
                value, low, high = None, None, None
            tasks.append({**meta, 'task_id': task, 'metric': metric, 'value': value,
                          'conditional_ci95_low': None if low is None else float(low),
                          'conditional_ci95_high': None if high is None else float(high),
                          'episode_support': len(plan[task]['episode_ids'])})
    for j, metric in enumerate(METRICS):
        low, high = np.quantile(draws[:, j], [.025, .975]) if draws is not None else (None, None)
        rows.append({**meta, 'metric': metric, 'value': None if point is None else float(point[j]),
                     'conditional_ci95_low': None if low is None else float(low),
                     'conditional_ci95_high': None if high is None else float(high),
                     'estimand_status': 'estimable' if point is not None else 'not_estimable_missing_true_task',
                     'anchor_support': int(mask.sum()), 'episode_support': len(np.unique(data['episode_id'][mask])),
                     'draws': DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
                     'conditional_on': 'fixed18models_and_fixed_seed_pairing; no_training_seed_resampling'})
    return rows, tasks


def score_split(source, evaluation, split, item, bundles, seal, contract):
    data, classifiers = item['data'], item['classifiers']
    n, full = len(data['anchor_id']), np.ones(len(data['anchor_id']), bool)
    output = evaluation / split / 'scores'
    if (output / 'SCORE_RECEIPT.json').exists():
        receipt = read_json(output / 'SCORE_RECEIPT.json')
        require(receipt['status'] == 'PASS' and receipt['global_prediction_seal_sha256'] == sha(evaluation / 'PREDICTION_SEAL.json'),
                'Existing scoring result belongs to a different seal')
        for name, digest in receipt['tables'].items():
            verify(output / name, digest)
        return {'split': split, 'score_receipt_sha256': sha(output / 'SCORE_RECEIPT.json'), 'tables': len(receipt['tables']),
                'reused_verified_complete_scores': True}
    if output.exists():
        started = read_json(output / 'TARGET_ACCESS_STARTED.json')
        require(started['global_prediction_seal_sha256'] == sha(evaluation / 'PREDICTION_SEAL.json'),
                'Partial scoring belongs to a different prediction seal')
    else:
        output.mkdir(parents=True)
    atomic_json(output / 'TARGET_ACCESS_STARTED.json', {'global_prediction_seal_sha256': sha(evaluation / 'PREDICTION_SEAL.json'),
                                                       'all_dev_final_bundles_verified': 26, 'created_utc': utc()})
    target, target_receipt = scoring.read_targets(source, split, data)
    metric = read_json(source / replay.METRIC)
    sigma, fd = np.asarray(metric['target_sigma'], float), np.asarray(metric['safe_fd_scale'], float)
    require(sigma.shape == fd.shape == (28,) and np.all(sigma > 0) and np.all(fd > 0), 'Invalid metric scales')
    errors = {(p, c, b): scoring.components(v, target, data['anchor_native'], sigma, fd)
              for (s, p, c, b), (v, _) in bundles.items() if s == split}
    per_seed, per_task, per_episode, contrasts, contrast_tasks = [], [], [], [], []
    coverage, confusion, predicted_classes = [], [], []

    def add_result(system, condition, scope, a_seed, b_seed, values, mask):
        meta = {'system': system, 'condition': condition, 'scope': scope, 'a_seed': a_seed, 'b_seed': b_seed}
        avg, tasks, episodes = fixed_aggregate(values, mask, data)
        per_seed.append({**meta, **avg})
        per_task.extend({**meta, **r} for r in tasks)
        per_episode.extend({**meta, **r} for r in episodes)

    add_result('PERSISTENCE', 'persistence', 'full_original_population', None, None,
               errors['PERSISTENCE', 'persistence', None], full)
    common = np.logical_and.reduce([classifiers[a][1] for a, _ in PAIRS])
    for a_seed, b_seed in PAIRS:
        predicted, valid, audit = classifiers[a_seed]
        for task in (*TASKS, 'ALL'):
            population = full if task == 'ALL' else data['task_id'] == task
            accepted = population & valid
            row = {'a_seed': a_seed, 'b_seed': b_seed, 'true_task_id': task,
                   'anchors': int(population.sum()), 'accepted_anchors': int(accepted.sum()),
                   'coverage': float(accepted.sum() / population.sum()),
                   'episodes': len(np.unique(data['episode_id'][population])),
                   'accepted_episodes': len(np.unique(data['episode_id'][accepted])),
                   'unrestricted_correct_anchors': int((population & (predicted == data['task_id'])).sum())}
            for reason in ('unsupported_core_a_task', 'classification_observes_after_core_b_anchor', 'missing_core_a_prediction', 'missing_observation_cutoff'):
                row[reason] = sum(bool(population[i]) and r['status'] == reason for i, r in enumerate(audit))
            coverage.append(row)
        for task in TASKS:
            for guessed in np.unique(predicted[data['task_id'] == task]):
                mask = (data['task_id'] == task) & (predicted == guessed)
                confusion.append({'a_seed': a_seed, 'b_seed': b_seed, 'true_task_id': task, 'predicted_task_id': int(guessed),
                                  'anchors': int(mask.sum()), 'accepted_anchors': int((mask & valid).sum()),
                                  'episodes': len(np.unique(data['episode_id'][mask]))})
        for system in SYSTEMS:
            oracle, cascade = errors[system, 'oracle_task', b_seed], errors[system, 'core_a_task', b_seed]
            for condition, scope, values, mask in (
                ('oracle_task', 'full_original_population', oracle, full),
                ('oracle_task', 'pair_matched_accepted_rows', oracle, valid),
                ('core_a_task', 'pair_matched_accepted_rows', cascade, valid),
                ('oracle_task', 'all_seed_common_accepted_rows', oracle, common),
                ('core_a_task', 'all_seed_common_accepted_rows', cascade, common)):
                add_result(system, condition, scope, a_seed, b_seed, values, mask)
            for guessed in TASKS:
                mask = valid & (predicted == guessed)
                eps = np.unique(data['episode_id'][mask])
                for condition, values in (('oracle_task', oracle), ('core_a_task', cascade)):
                    means = [values[mask & (data['episode_id'] == ep)].mean(axis=0) for ep in eps]
                    avg = np.mean(means, axis=0) if means else [None] * 3
                    predicted_classes.append({'system': system, 'condition': condition, 'a_seed': a_seed, 'b_seed': b_seed,
                                              'predicted_task_id': guessed, 'anchor_support': int(mask.sum()), 'episode_support': len(eps),
                                              'weighting': 'equal_episode_within_predicted_class',
                                              **{m: None if v is None else float(v) for m, v in zip(METRICS, avg)}})
        add_result('PERSISTENCE', 'persistence', 'pair_matched_accepted_rows', a_seed, b_seed,
                   errors['PERSISTENCE', 'persistence', None], valid)
        comparisons = [
            ('EXPERT_minus_SHARED_oracle', 'full_original_population', errors['EXPERT_V0', 'oracle_task', b_seed] - errors['SHARED_V0', 'oracle_task', b_seed], full),
            ('EXPERT_minus_SHARED_core_a', 'pair_matched_accepted_rows', errors['EXPERT_V0', 'core_a_task', b_seed] - errors['SHARED_V0', 'core_a_task', b_seed], valid),
            ('EXPERT_minus_SHARED_oracle', 'pair_matched_accepted_rows', errors['EXPERT_V0', 'oracle_task', b_seed] - errors['SHARED_V0', 'oracle_task', b_seed], valid)]
        comparisons.extend((f'{p}_core_a_minus_oracle', 'pair_matched_accepted_rows', errors[p, 'core_a_task', b_seed] - errors[p, 'oracle_task', b_seed], valid) for p in SYSTEMS)
        for name, scope, values, mask in comparisons:
            avg, tasks, _ = fixed_aggregate(values, mask, data)
            meta = {'comparison': name, 'scope': scope, 'a_seed': a_seed, 'b_seed': b_seed,
                    'direction': 'negative favors first-named system/condition lower error'}
            contrasts.append({**meta, **avg})
            contrast_tasks.extend({**meta, **r} for r in tasks)
    ci_rows, ci_tasks = [], []
    for cohort, mask in (('full', full), ('common', common)):
        plan = read_plan(evaluation / f'BOOTSTRAP_PLAN_{split}.npz', cohort)
        averages = {(p, c): np.mean([errors[p, c, b] for b in SEEDS], axis=0)
                    for p in SYSTEMS for c in (('oracle_task',) if cohort == 'full' else CONDITIONS)}
        items = [(p + '_' + c, 'metric', v) for (p, c), v in averages.items()]
        items.append(('EXPERT_minus_SHARED_oracle', 'paired_delta', averages['EXPERT_V0', 'oracle_task'] - averages['SHARED_V0', 'oracle_task']))
        if cohort == 'common':
            items.append(('EXPERT_minus_SHARED_core_a', 'paired_delta', averages['EXPERT_V0', 'core_a_task'] - averages['SHARED_V0', 'core_a_task']))
            items.extend((p + '_core_a_minus_oracle', 'paired_delta', averages[p, 'core_a_task'] - averages[p, 'oracle_task']) for p in SYSTEMS)
        for name, kind, values in items:
            rows, tasks = interval_rows({'comparison_or_cell': name, 'kind': kind, 'cohort': cohort,
                                         'direction': 'negative delta favors first-named lower error' if kind == 'paired_delta' else 'lower error is better'},
                                        values, mask, data, plan)
            ci_rows.extend(rows)
            ci_tasks.extend(tasks)
    tables = {'PER_SEED.csv': per_seed, 'PER_TASK_SEED.csv': per_task, 'PER_EPISODE_SEED.csv': per_episode,
              'SUMMARY_MEAN_SD.csv': summarize(per_seed, ('system', 'condition', 'scope')),
              'PER_TASK_MEAN_SD.csv': summarize(per_task, ('system', 'condition', 'scope', 'task_id')),
              'PAIRED_DELTAS_PER_SEED.csv': contrasts, 'PAIRED_DELTAS_PER_TASK_SEED.csv': contrast_tasks,
              'PAIRED_DELTAS_MEAN_SD.csv': summarize(contrasts, ('comparison', 'scope')),
              'BOOTSTRAP_CI.csv': ci_rows, 'BOOTSTRAP_PER_TASK_CI.csv': ci_tasks,
              'COVERAGE_BY_TRUE_TASK.csv': coverage, 'CLASSIFICATION_CONFUSION.csv': confusion,
              'PER_PREDICTED_CLASS_SEED.csv': predicted_classes}
    for name, rows in tables.items():
        write_csv(output / name, rows)
    atomic_json(output / 'SCORE_RECEIPT.json', {
        'status': 'PASS', 'split': split, 'global_prediction_seal_sha256': sha(evaluation / 'PREDICTION_SEAL.json'),
        'target_receipt': target_receipt, 'metric_scale_sha256': METRIC_SHA, 'population_anchors': n,
        'population_episodes': len(np.unique(data['episode_id'])), 'common_accepted_anchors': int(common.sum()),
        'bootstrap_plan_sha256': sha(evaluation / f'BOOTSTRAP_PLAN_{split}.npz'),
        'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_draws': DRAWS,
        'estimands': 'Per-seed accepted means may use different rows; common-cohort intervals average the three fixed seed pairs on their acceptance intersection. All five true tasks required for fixed macro.',
        'capacity_qualification': contract['capacity_qualification'], 'history_qualification': contract['history_qualification'],
        'standardization': 'original training-derived scales, not physical joint-angle units',
        'score_only_no_model_or_hyperparameter_selection': True, 'tables': {name: sha(output / name) for name in tables},
        'completed_utc': utc()})
    return {'split': split, 'score_receipt_sha256': sha(output / 'SCORE_RECEIPT.json'), 'tables': len(tables)}


def write_capacity(evaluation, models):
    rows = []
    for seed in SEEDS:
        for family in ('shared', 'expert'):
            active = [r for r in models if r['seed'] == seed and r['family'] == family]
            rows.append({'family': family, 'seed': seed, 'models': len(active), 'epochs_per_model': 160,
                         'total_parameters': sum(r['parameter_count'] for r in active),
                         'parameters_active_per_prediction': active[0]['parameter_count'],
                         'total_optimizer_steps': sum(r['optimizer_steps'] for r in active),
                         'optimizer_steps_per_model_min': min(r['optimizer_steps'] for r in active),
                         'optimizer_steps_per_model_max': max(r['optimizer_steps'] for r in active),
                         'total_training_rows': sum(r['train_rows'] for r in active),
                         'total_training_presentations': 160 * sum(r['train_rows'] for r in active)})
    write_csv(evaluation / 'CAPACITY_AND_TRAINING_BUDGET.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'predict', 'score'), required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--historical-root', help='Existing two-split cascade root; default sibling coreB_class_conditioned')
    args = parser.parse_args()
    replay.require_external_paths()
    source, run_root = Path(args.source_root).resolve(), Path(args.run_root).resolve()
    historical_root = Path(args.historical_root).resolve() if args.historical_root else run_root.parent / 'coreB_class_conditioned'
    inputs = load_inputs(source, run_root, with_grid=args.stage in ('predict', 'preflight'))
    evaluation, contract = establish_lock(source, run_root, inputs, historical_root)
    if args.stage == 'preflight':
        expected = [run_root / 'models' / folder / f'seed{seed}' / 'RUN_COMPLETE.json'
                    for seed in SEEDS for folder in ('shared', *(f'task_{t}' for t in TASKS))]
        completed = sum(path.is_file() for path in expected)
        if completed != 18:
            print(json.dumps({'status': 'SOURCE_INPUT_PREFLIGHT_PASS_MODELS_PENDING', 'completed_fit_receipts': completed,
                              'required_fits': 18, 'future_targets_read': False,
                              'evaluation_lock_sha256': sha(evaluation / 'EVALUATION_LOCK.json')}))
            return
    models = freeze_models(source, run_root, evaluation)
    if args.stage == 'preflight':
        print(json.dumps({'status': 'PREFLIGHT_PASS_NO_TARGETS_READ', 'models': len(models), 'splits': list(inputs),
                          'evaluation_lock_sha256': sha(evaluation / 'EVALUATION_LOCK.json')}))
    elif args.stage == 'predict':
        predict(source, run_root, evaluation, contract, inputs, models, args.device)
    else:
        seal, bundles = verify_seal(evaluation, contract, inputs)
        # This is a CPU-only branch: no torch import, model execution or fitting.
        write_capacity(evaluation, models)
        results = [score_split(source, evaluation, split, inputs[split], bundles, seal, contract) for split in SPLITS]
        atomic_json(evaluation / 'EVALUATION_COMPLETE.json', {'status': 'PASS', 'scores': results,
                    'global_prediction_seal_sha256': sha(evaluation / 'PREDICTION_SEAL.json'),
                    'capacity_table_sha256': sha(evaluation / 'CAPACITY_AND_TRAINING_BUDGET.csv'), 'completed_utc': utc()})
        print(json.dumps({'status': 'PASS', 'evaluation': str(evaluation), 'results': results}))


if __name__ == '__main__':
    main()
