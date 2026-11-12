#!/usr/bin/env python3
"""Train independent five-task V0 experts and a same-environment shared control.

All formal fits start from the original hash-pinned fresh INIT, train for exactly
160 epochs, and save only the last-epoch model. No dev/final input or target is
opened. Expert epoch orders are stable filters of the original global sampler.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

OLD_ROOT = os.environ.get("EE6008_LEGACY_SOURCE_PREFIX", "").rstrip("/")
V9 = 'reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z'
MATRIX = 'reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z/protocol/FINAL_MODEL_MATRIX.csv'
METRIC = 'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z/protocol/METRIC_SCALE_LOCK.json'
TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
EPOCHS, BATCH = 160, 64
LR, ETA_MIN = 5e-4, 1e-5
PINNED_SOURCE = {
    'v2_model.py': '6e1e967c89afeeb6e676e9265a893fed54f66682f54af3bf04cf3d41766c8c24',
    'locked_source/D3_model.py': '6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589',
    'v9_common.py': 'f023fa7af55f681d1308b0165dbcb4b1fe75091fcf3f9ed1df59d456d1a12615',
    'locked_source/d3_eq_contract.py': '7359c956bf405092dade2e49e72f236759099330357652469c610d422c770b1c',
    'v2_scheduler.py': '4d81a77a5c983e0771c2c0d945b0bd7cc1467582139c767e58dbbffabbe1d80b',
    'prepare_v2_data.py': '64126bdf992037bb3455b4a78392269e7ecd9e62ae01a6b425ef6b090341e15d',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def verify(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch: {path}: {actual} != {expected}')
    return actual


def source_path(root, value):
    if not OLD_ROOT:
        raise ValueError("Set EE6008_LEGACY_SOURCE_PREFIX before relocating external assets")
    value = str(value)
    if value.startswith(OLD_ROOT + '/'):
        value = value[len(OLD_ROOT) + 1:]
    if Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Asset path escapes original source root')
    return Path(root) / value


def training_indices(task_ids, mode, task_id):
    task_ids = np.asarray(task_ids)
    if mode == 'shared':
        if task_id is not None:
            raise ValueError('Shared control cannot specify task_id')
        return np.arange(len(task_ids), dtype=np.int64)
    if mode != 'expert' or task_id not in TASKS:
        raise ValueError('Expert requires one of the five registered tasks')
    selected = np.flatnonzero(task_ids == task_id).astype(np.int64)
    if not len(selected):
        raise ValueError('Expert training population is empty')
    return selected


def training_weights(task_ids, episode_ids, selected):
    """Mean-one weights; equal episodes within each selected true task."""
    tasks = np.asarray(task_ids)[selected].astype(np.int64)
    episodes = np.asarray(episode_ids)[selected].astype(np.int64)
    if not len(tasks):
        raise ValueError('No training rows')
    classes = np.unique(tasks)
    weights = np.empty(len(selected), dtype=np.float64)
    for task in classes:
        local = np.flatnonzero(tasks == task)
        _, inverse, counts = np.unique(episodes[local], return_inverse=True, return_counts=True)
        weights[local] = len(selected) / (len(classes) * len(counts) * counts[inverse])
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), len(selected), atol=1e-10, rtol=0):
        raise ValueError('Training weights violate mean-one contract')
    return weights


def filter_epoch_order(global_order, selected, global_rows):
    order = np.asarray(global_order)
    if order.dtype != np.int64 or order.shape != (global_rows,) or not np.array_equal(np.sort(order), np.arange(global_rows)):
        raise ValueError('Global sampler is not a complete permutation')
    selected = np.asarray(selected, dtype=np.int64)
    if len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= global_rows):
        raise ValueError('Invalid selected training indices')
    mask = np.zeros(global_rows, dtype=bool)
    mask[selected] = True
    filtered = order[mask[order]]
    if not np.array_equal(np.sort(filtered), np.sort(selected)):
        raise ValueError('Filtered sampler does not cover each expert row exactly once')
    return filtered


def schedule_counts(rows):
    if rows < 1:
        raise ValueError('Nonempty training population required')
    steps = math.ceil(rows / BATCH)
    return steps, EPOCHS * steps


def freeze_cache_content(path, document):
    """Atomically freeze current cache hashes once, then require exact reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != document:
            raise ValueError('Training cache content differs from this study freeze')
        return
    temp = path.with_name(f'.{path.name}.{os.getpid()}.candidate')
    atomic_json(temp, document)
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            pass
        if read_json(path) != document:
            raise ValueError('Concurrent study cache freeze differs')
    finally:
        temp.unlink(missing_ok=True)


class PreloadedTrainGrid:
    def __init__(self, root, run_root, data, cache_path, expected_sha, q0_class):
        verify(cache_path, expected_sha)
        manifest = read_json(cache_path)
        adapted = dict(manifest)
        adapted['shards'] = [dict(row, array=str(source_path(root, row['array'])), meta=str(source_path(root, row['meta']))) for row in manifest['shards']]
        reader = q0_class(adapted)
        n = len(data['anchor_id'])
        indices = data['cache_index'].astype(np.int64)
        if len(np.unique(indices)) != n:
            raise ValueError('Training cache indices must be unique')
        used = set()
        for index in indices:
            shard_index = bisect.bisect_right(reader.starts, int(index)) - 1
            if shard_index < 0 or int(index) >= int(reader.shards[shard_index]['end_row']):
                raise ValueError('Training row absent from source Q0 shards')
            used.add(shard_index)
        receipts = []
        stats = {}
        for shard_index in sorted(used):
            row = reader.shards[shard_index]
            for kind in ('array', 'meta'):
                path = Path(row[kind])
                before = path.stat()
                digest = sha(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Source cache file changed while hashing')
                stats[str(path)] = (after.st_size, after.st_mtime_ns)
                receipts.append({'kind': kind, 'path': str(path), 'start_row': int(row['start_row']), 'end_row': int(row['end_row']),
                                 'bytes': after.st_size, 'sha256': digest})
        content_lock = {'schema_version': 1, 'source_manifest_sha256': expected_sha,
                        'hash_provenance': 'current-study byte hashes; historical manifest contains paths and metadata but no shard content hash',
                        'train_npz_anchor_ids_sha256': array_sha(data['anchor_id']), 'shards': receipts}
        freeze_cache_content(run_root / 'TRAIN_CACHE_CONTENT_LOCK.json', content_lock)
        self.values = np.empty((n, 2, 16, 4, 768), dtype=np.float32)
        self.lookup = {}
        for i, (index, anchor, episode) in enumerate(zip(indices, data['anchor_id'], data['episode_id'])):
            values = reader.row(int(index), str(anchor), str(episode))
            if values.shape != (2, 16, 4, 768) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError('Training feature row has invalid shape/dtype/values')
            self.values[i] = values
            self.lookup[int(index)] = (str(anchor), int(episode), i)
        for name, expected in stats.items():
            current = Path(name).stat()
            if (current.st_size, current.st_mtime_ns) != expected:
                raise ValueError('Source cache file changed during preload')
        self.values.setflags(write=False)
        self.receipt = {'source_manifest_path': str(cache_path), 'source_manifest_sha256': expected_sha,
                        'content_lock_path': str(run_root / 'TRAIN_CACHE_CONTENT_LOCK.json'),
                        'content_lock_sha256': sha(run_root / 'TRAIN_CACHE_CONTENT_LOCK.json'),
                        'preloaded_shape': list(self.values.shape), 'preloaded_dtype': str(self.values.dtype),
                        'preloaded_array_sha256': array_sha(self.values), 'identity_verified_rows': n,
                        'new_encoding_runs': 0, 'historical_shard_content_hash_available': False}
        del reader

    def row(self, cache_index, anchor_id, episode_id):
        anchor, episode, index = self.lookup[int(cache_index)]
        if anchor != str(anchor_id) or episode != int(episode_id):
            raise ValueError('Preloaded training grid row identity mismatch')
        return self.values[index]


def load_source(root, seed):
    code = root / V9 / 'code'
    for relative, digest in PINNED_SOURCE.items():
        verify(code / relative, digest)
    job_path = root / V9 / f'configs/fits/V0_seed{seed}.json'
    with (root / MATRIX).open(encoding='utf-8-sig', newline='') as f:
        matrix = [r for r in csv.DictReader(f) if r['core'] == 'B' and r['program'] == 'V0' and int(r['seed']) == seed]
    if len(matrix) != 1:
        raise ValueError('Original V0 configuration is not uniquely bound in accepted model matrix')
    verify(job_path, matrix[0]['config_sha256'])
    job = read_json(job_path)
    expected = {'seed': seed, 'program_id': 'V0', 'video_mode': 'TS11-D', 'numeric_context_drop_probability': 0.0,
                'epochs': EPOCHS, 'effective_batch': BATCH, 'microbatch_size': BATCH, 'accumulation_steps': 1,
                'lr': LR, 'target_mode': 'true_y', 'teacher_target_used': False}
    for key, value in expected.items():
        if job.get(key) != value:
            raise ValueError(f'Original V0 job differs from the retained recipe: {key}')
    verify(root / V9 / 'protocol/RECIPE_LOCK.json', job['recipe_lock_sha256'])
    verify(root / V9 / 'protocol/STUDY_LOCK.json', job['study_lock_sha256'])
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / 'locked_source'))
    modules = {name: importlib.import_module(name) for name in ('v2_model', 'v2_scheduler', 'v9_common', 'd3_eq_contract', 'prepare_v2_data')}
    return job, job_path, code, modules


def run(args):
    if not OLD_ROOT:
        raise ValueError("Set EE6008_LEGACY_SOURCE_PREFIX before reading external assets")
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from safetensors.torch import load_file, save_file

    root, run_root = Path(args.source_root).resolve(), Path(args.run_root).resolve()
    if args.seed not in SEEDS:
        raise ValueError('Seed outside the three fixed original seed slots')
    if args.mode == 'shared' and args.task_id is not None or args.mode == 'expert' and args.task_id not in TASKS:
        raise ValueError('Provide task-id only for an expert, using a registered task')
    label = 'shared' if args.mode == 'shared' else f'task_{args.task_id}'
    parent = 'preflight' if args.preflight_only else 'smoke' if args.stage == 'smoke' else 'models'
    output = run_root / parent / label / f'seed{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    started, timer = utc(), time.perf_counter()
    steps_done, presentations = 0, 0
    try:
        atomic_json(output / 'RUN_STATUS.json', {'status': 'PREPARING', 'started_utc': started, 'mode': args.mode, 'task_id': args.task_id, 'seed': args.seed})
        job, job_path, code, modules = load_source(root, args.seed)
        common, modelmod, eq, schedmod = (modules[x] for x in ('v9_common', 'v2_model', 'd3_eq_contract', 'v2_scheduler'))
        train_path = source_path(root, job['train_npz_path'])
        verify(train_path, job['train_npz_sha256'])
        with np.load(train_path, allow_pickle=False) as z:
            data = {key: z[key].copy() for key in z.files}
        n = len(data['anchor_id'])
        if n != 1280 or len(np.unique(data['episode_id'])) != 220 or len(set(data['anchor_id'].astype(str))) != n:
            raise ValueError('Original train population must be 1280 unique anchors / 220 episodes')
        if set(data['task_id'].tolist()) != set(TASKS):
            raise ValueError('Original train task population differs')
        for name, shape in {'history_native': (n, 32, 71), 'target_native_h30': (n, 30, 28),
                            'history_mask': (n, 32), 'relative_delta': (n, 32), 'anchor_native': (n, 28)}.items():
            if data[name].shape != shape or not np.isfinite(data[name]).all():
                raise ValueError('Original training tensor shape/values invalid: ' + name)
        selected = training_indices(data['task_id'], args.mode, args.task_id)
        weights = training_weights(data['task_id'], data['episode_id'], selected)
        weight_by_global = np.zeros(n, dtype=np.float64)
        weight_by_global[selected] = weights
        sampler_path = source_path(root, job['sampler_orders_path'])
        verify(sampler_path, job['sampler_orders_sha256'])
        with np.load(sampler_path, allow_pickle=False) as z:
            sampler = z[f'seed_{args.seed}'].copy()
        if sampler.shape != (EPOCHS, n) or sampler.dtype != np.int64 or array_sha(sampler) != job['sampler_sequence_sha256']:
            raise ValueError('Original sampler shape/content differs')
        orders = []
        for epoch, order in enumerate(sampler, 1):
            if not np.array_equal(order, eq.shuffled_epoch_order(n, args.seed, epoch)):
                raise ValueError('Original sampler differs from seed+epoch permutation')
            orders.append(filter_epoch_order(order, selected, n))
        orders = np.stack(orders)
        normalizer_path = source_path(root, job['normalizer_path'])
        verify(normalizer_path, job['normalizer_sha256'])
        normalizer = read_json(normalizer_path)
        metric_path = root / METRIC
        verify(metric_path, 'dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32')
        metric = read_json(metric_path)
        for key, job_key in (('target_sigma', 'metric_sigma'), ('safe_fd_scale', 'metric_safe_fd_scale')):
            if not np.array_equal(np.asarray(metric[key], dtype=np.float32), np.asarray(job[job_key], dtype=np.float32)):
                raise ValueError('Original training-log metric scales differ from metric lock')
        grid = PreloadedTrainGrid(root, run_root, data, source_path(root, job['feature_cache_manifest_path']),
                                  job['feature_cache_manifest_sha256'], modules['prepare_v2_data'].Q0ShardedGrid)
        init_path = source_path(root, job['init_path'])
        verify(init_path, job['init_file_sha256'])
        per_epoch, total_steps = schedule_counts(len(selected))
        resolved = {'schema_version': 1, 'stage': args.stage, 'mode': args.mode, 'task_id': args.task_id, 'seed': args.seed,
                    'epochs': EPOCHS, 'batch_size': BATCH, 'microbatch_size': BATCH, 'accumulation_steps': 1,
                    'steps_per_epoch': per_epoch, 'total_optimizer_steps': total_steps, 'train_rows': len(selected),
                    'train_episodes': len(np.unique(data['episode_id'][selected])),
                    'task_ids': [task for task in TASKS if np.any(data['task_id'][selected] == task)], 'task_order': list(TASKS),
                    'constant_task_rank': None if args.mode == 'shared' else TASKS.index(args.task_id),
                    'weight_formula': 'W_selected/(K_selected * E_true_task * windows_in_episode)',
                    'weight_mean': float(weights.mean()), 'sampler': 'original global epoch order filtered to selected true task rows',
                    'sampler_sequence_sha256': array_sha(orders), 'train_npz_path': str(train_path), 'train_npz_sha256': sha(train_path),
                    'normalizer_path': str(normalizer_path), 'normalizer_sha256': sha(normalizer_path),
                    'init_path': str(init_path), 'init_sha256': sha(init_path), 'expected_init_state_hash': job['expected_init_state_hash'],
                    'original_config_path': str(job_path), 'original_config_sha256': sha(job_path),
                    'source_code_root': str(code), 'source_code_hashes': PINNED_SOURCE,
                    'training_runner_sha256': sha(Path(__file__)), 'cache_receipt': grid.receipt,
                    'optimizer': {'name': 'AdamW', 'lr': LR, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 1e-4},
                    'scheduler': {'name': 'UpdateDrivenCosineAnnealingLR', 'T_max': total_steps, 'eta_min': ETA_MIN, 'step_location': 'optimizer post-step'},
                    'loss': 'sum(w_i * (standardized position_L1 + 0.1 * future_internal_safe_FD_L1))/B_actual',
                    'numeric_context_drop_probability': 0.0, 'new_train_from_fresh_init': True,
                    'dev_or_final_target_rows': 0, 'final_heldout_payload_access': 0,
                    'budget_note': '160 passes per training row; experts have fewer optimizer updates than the shared 1280-row control'}
        atomic_json(output / 'RESOLVED_CONFIG.json', resolved)
        write_csv(output / 'TRAINING_ROWS.csv', [{'global_train_row': int(i), 'anchor_id': str(data['anchor_id'][i]),
            'episode_id': int(data['episode_id'][i]), 'task_id': int(data['task_id'][i]), 'sample_weight': float(weight_by_global[i])} for i in selected])
        if args.preflight_only:
            atomic_json(output / 'PREFLIGHT_COMPLETE.json', {'status': 'PASS', **resolved})
            print(json.dumps({'status': 'PREFLIGHT_PASS', 'output': str(output), 'train_rows': len(selected), 'optimizer_steps': total_steps}), flush=True)
            return 0
        if not os.environ.get('GPU_WATCHER_PHYSICAL_GPU'):
            raise RuntimeError('GPU fitting must be launched by the project GPU lease watcher')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one watcher-mapped CUDA device is required')
        device = torch.device('cuda:0')
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        model = modelmod.V2TaskConditionedPredictor(seed=args.seed, video_mode='TS11-D', waypoint_mode=None)
        model.load_state_dict(load_file(str(init_path), device='cpu'), strict=True)
        if modelmod.state_hash(model) != job['expected_init_state_hash']:
            raise ValueError('Fresh INIT state hash differs')
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        scheduler = schedmod.UpdateDrivenCosineAnnealingLR(optimizer, T_max=total_steps, eta_min=ETA_MIN)
        torch.cuda.reset_peak_memory_stats(device)
        rng_before_fit = {'torch_cpu_sha256': array_sha(torch.get_rng_state().cpu().numpy()),
                          'torch_cuda_sha256': array_sha(torch.cuda.get_rng_state(device).cpu().numpy())}
        sigma = torch.as_tensor(np.asarray(normalizer['target_sigma'], dtype=np.float32), device=device)
        fdscale = torch.as_tensor(np.asarray(normalizer['safe_fd_scale'], dtype=np.float32), device=device)
        metric_sigma = torch.as_tensor(np.asarray(metric['target_sigma'], dtype=np.float32), device=device)
        metric_fd = torch.as_tensor(np.asarray(metric['safe_fd_scale'], dtype=np.float32), device=device)
        model.train()
        logs = []
        training_timer = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            scheduler.set_epoch(epoch)
            totals = np.zeros(4, dtype=np.float64)
            weight_sum = 0.0
            epoch_presentations = 0
            grad_norms = []
            for start in range(0, len(selected), BATCH):
                ix = orders[epoch - 1, start:start + BATCH]
                group_n = len(ix)
                optimizer.zero_grad(set_to_none=True)
                drop = np.zeros(group_n, dtype=np.bool_)
                prefix, task_rank, anchor = common.model_input(data, ix, normalizer, 'TS11-D', device, grid, drop_mask=drop)
                if args.mode == 'expert' and not torch.all(task_rank == TASKS.index(args.task_id)):
                    raise ValueError('Expert batch contains a different task token')
                target = torch.as_tensor(data['target_native_h30'][ix], device=device, dtype=torch.float32)
                result = model(prefix, task_rank)
                if result.goal_prediction is not None:
                    raise ValueError('Unexpected auxiliary task')
                losses = eq.per_sample_loss(result.action_residual, target, anchor, sigma, fdscale)
                w = torch.as_tensor(weight_by_global[ix], device=device, dtype=torch.float32)
                loss = torch.sum(losses * w) / float(group_n)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training objective')
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Nonfinite gradient: ' + name)
                metrics = common.metric_components(result.action_residual, target, anchor, sigma, fdscale, metric_sigma, metric_fd)
                for j, values in enumerate(metrics):
                    totals[j] += float(torch.sum(values * w).detach().cpu())
                weight_sum += float(w.sum().detach().cpu())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                grad_norms.append(float(norm.detach().cpu()))
                optimizer.step()
                steps_done += 1
                if scheduler.step_calls != steps_done:
                    raise ValueError('Optimizer and scheduler step counts diverged')
                presentations += group_n
                epoch_presentations += group_n
                if args.stage == 'smoke' and steps_done >= args.smoke_updates:
                    break
            logs.append({'epoch': epoch, 'optimizer_steps': steps_done, 'lr_after_epoch': scheduler.get_last_lr()[0],
                         'train_rows_presented': epoch_presentations, 'epoch_weight_sum': weight_sum,
                         'position_mae': float(totals[0] / weight_sum), 'safe_fd_l1': float(totals[1] / weight_sum),
                         'seam_safe_fd_l1': float(totals[2] / weight_sum), 'objective': float(totals[3] / weight_sum),
                         'preclip_gradient_norm_mean': float(np.mean(grad_norms)), 'preclip_gradient_norm_max': float(np.max(grad_norms)),
                         'clipped_updates': sum(v > 1.0 for v in grad_norms), 'training_elapsed_seconds': time.perf_counter() - training_timer})
            write_csv(output / 'EPOCH_LOG.csv', logs)
            atomic_json(output / 'RUN_STATUS.json', {'status': 'RUNNING', 'updated_utc': utc(), 'mode': args.mode, 'task_id': args.task_id,
                'seed': args.seed, 'epoch': epoch, 'epochs_target': EPOCHS, 'optimizer_steps': steps_done, 'target_optimizer_steps': total_steps,
                'lr': scheduler.get_last_lr()[0], 'train_rows': len(selected), 'training_elapsed_seconds': time.perf_counter() - training_timer})
            print(json.dumps({'stage': args.stage, 'mode': args.mode, 'task_id': args.task_id, 'seed': args.seed,
                              'epoch': epoch, 'steps': steps_done, 'objective': logs[-1]['objective']}), flush=True)
            if args.stage == 'smoke':
                if steps_done >= args.smoke_updates:
                    break
            elif epoch_presentations != len(selected) or not np.isclose(weight_sum, len(selected), atol=1e-3, rtol=0):
                raise ValueError('Epoch row/weight budget differs')
        torch.cuda.synchronize(device)
        resource = {'elapsed_seconds': time.perf_counter() - timer, 'training_seconds': time.perf_counter() - training_timer,
                    'peak_memory_allocated_mib': float(torch.cuda.max_memory_allocated(device) / 2**20),
                    'peak_memory_reserved_mib': float(torch.cuda.max_memory_reserved(device) / 2**20),
                    'hostname': socket.gethostname(), 'physical_gpu': os.environ.get('GPU_WATCHER_PHYSICAL_GPU'),
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'logical_device': 'cuda:0',
                    'gpu_name': torch.cuda.get_device_name(device), 'python_version': sys.version.split()[0],
                    'torch_version': torch.__version__, 'torch_cuda_version': torch.version.cuda,
                    'cudnn_version': torch.backends.cudnn.version(), 'dtype': 'float32', 'amp': False, 'tf32': False}
        atomic_json(output / 'RESOURCE_USAGE.json', resource)
        schedmod.write_lr_trace(output / 'LR_TRACE_ACTUAL.csv', scheduler.update_records)
        if args.stage == 'smoke':
            if steps_done != args.smoke_updates:
                raise ValueError('Smoke update count differs')
            atomic_json(output / 'SMOKE_COMPLETE.json', {'status': 'PASS', 'optimizer_steps': steps_done,
                'formal_training_steps': 0, 'formal_restart_from_fresh_INIT_required': True,
                'model_saved': False, 'dev_or_final_target_rows': 0, 'final_heldout_payload_access': 0, **resource})
            atomic_json(output / 'RUN_STATUS.json', {'status': 'SMOKE_PASS', 'completed_utc': utc()})
            return 0
        if steps_done != total_steps or presentations != EPOCHS * len(selected) or len(logs) != EPOCHS:
            raise ValueError('Formal fit did not reach its exact epoch/update/presentation budget')
        if not math.isclose(scheduler.get_last_lr()[0], ETA_MIN, abs_tol=2e-11, rel_tol=0):
            raise ValueError('Final cosine learning rate differs from eta_min')
        model_path = output / 'model.safetensors'
        temporary = output / f'.model.{os.getpid()}.tmp'
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temporary))
        os.replace(temporary, model_path)
        model.eval()
        final_hash = modelmod.state_hash(model)
        replay = modelmod.V2TaskConditionedPredictor(seed=args.seed, video_mode='TS11-D', waypoint_mode=None)
        replay.load_state_dict(load_file(str(model_path), device='cpu'), strict=True)
        if modelmod.state_hash(replay) != final_hash:
            raise ValueError('Final model state differs after save/reload')
        replay = replay.to(device).eval()
        check_ix = selected[:min(3, len(selected))]
        prefix, task_rank, _ = common.model_input(data, check_ix, normalizer, 'TS11-D', device, grid, drop_mask=np.zeros(len(check_ix), bool))
        with torch.no_grad():
            before, after = model(prefix, task_rank).action_residual, replay(prefix, task_rank).action_residual
        if not torch.equal(before, after):
            raise ValueError('Final model predictions differ after save/reload')
        manifest = {**resolved, 'status': 'PASS', 'created_utc': utc(), 'architecture': 'V2TaskConditionedPredictor',
                    'video_mode': 'TS11-D', 'program_id': 'V0_SHARED_RETRAIN' if args.mode == 'shared' else f'V0_EXPERT_{args.task_id}',
                    'model_file': str(model_path), 'model_file_sha256': sha(model_path), 'final_state_hash': final_hash,
                    'init_state_hash': job['expected_init_state_hash'], 'epochs_completed': EPOCHS, 'optimizer_steps': steps_done,
                    'parameter_count': modelmod.parameter_count(model), 'training_sample_presentations': presentations,
                    'checkpoint_selection': 'unique fixed final epoch 160; no dev/final selection',
                    'rng_before_fit': rng_before_fit, 'reload_check': {'status': 'PASS', 'rows': len(check_ix), 'prediction_bitwise_equal': True},
                    'resource_usage': resource}
        atomic_json(output / 'MODEL_MANIFEST.json', manifest)
        atomic_json(output / 'RUN_COMPLETE.json', {'schema_version': 1, 'status': 'PASS', 'completed_utc': utc(),
            'mode': args.mode, 'task_id': args.task_id, 'seed': args.seed, 'epochs_completed': EPOCHS,
            'optimizer_steps': steps_done, 'training_sample_presentations': presentations,
            'model_sha256': sha(model_path), 'model_state_hash': final_hash, 'model_manifest_sha256': sha(output / 'MODEL_MANIFEST.json'),
            'dev_or_final_target_rows': 0, 'final_heldout_payload_access': 0, 'elapsed_seconds': time.perf_counter() - timer})
        atomic_json(output / 'RUN_STATUS.json', {'status': 'COMPLETE', 'completed_utc': utc(), 'epochs': EPOCHS, 'optimizer_steps': steps_done})
        print(json.dumps({'status': 'PASS', 'output': str(output), 'model_sha256': sha(model_path)}), flush=True)
        return 0
    except Exception as exc:
        atomic_json(output / 'RUN_STATUS.json', {'status': 'FAILED', 'failed_utc': utc(), 'mode': args.mode, 'task_id': args.task_id,
            'seed': args.seed, 'optimizer_steps': steps_done, 'training_sample_presentations': presentations,
            'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
            'dev_or_final_target_rows': 0, 'final_heldout_payload_access': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--mode', choices=['shared', 'expert'], required=True)
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stage', choices=['smoke', 'train'], required=True)
    parser.add_argument('--smoke-updates', type=int, default=2)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.smoke_updates <= 8:
        parser.error('smoke-updates must be between 1 and 8')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
