#!/usr/bin/env python3
"""Score a completed, hash-sealed cascade replay; never used during inference."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from run_cascade import require_external_paths

from run_cascade import (DEV_TARGET, METRIC, METRICS, PAIRS, REC, RUN, TASKS,
                         csv_write, load_prefix, read_json, relocated, sha, verify, write_json)

PROGRAMS = ('A0', 'AD', 'V0', 'VD', 'S_OPT', 'MIX0', 'MIXD')
BOOTSTRAP_SEED = 2026101902
BOOTSTRAP_DRAWS = 5000


def validate_bundle_masks(bundles, audits, n, programs=PROGRAMS):
    """All model comparisons must use the audited classification population."""
    for a_seed, b_seed in PAIRS:
        expected = np.asarray([row['status'] == 'accepted' for row in audits[a_seed]], dtype=bool)
        if expected.shape != (n,):
            raise ValueError('Classification audit mask shape mismatch')
        for program in programs:
            oracle = bundles[('oracle_task', program, b_seed)][1]
            cascade = bundles[('core_a_task', program, b_seed)][1]
            if oracle.shape != (n,) or not np.all(oracle):
                raise ValueError(f'Oracle valid mask is not all True: {program}/{b_seed}')
            if not np.array_equal(cascade, expected):
                raise ValueError(f'Cascade valid mask differs from classification audit: {program}/{b_seed}')
    persistence = bundles[('persistence', 'PERSISTENCE', None)][1]
    if persistence.shape != (n,) or not np.all(persistence):
        raise ValueError('Persistence valid mask is not all True')


def shared_episode_resamples(data, common, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS):
    """Generate one immutable episode resampling plan shared by all programs."""
    episodes = [np.unique(data['episode_id'][common & (data['task_id'] == task)]) for task in TASKS]
    if any(len(values) == 0 for values in episodes):
        return [], []
    rng = np.random.default_rng(seed)
    indices = [rng.integers(0, len(values), size=(draws, len(values))) for values in episodes]
    for values in (*episodes, *indices):
        values.setflags(write=False)
    return episodes, indices


def bootstrap_from_shared_indices(episode_by_task, resample_indices):
    if len(episode_by_task) != len(TASKS) or len(resample_indices) != len(TASKS):
        raise ValueError('Bootstrap requires all five true tasks')
    return np.mean([values[index].mean(axis=1) for values, index in zip(episode_by_task, resample_indices)], axis=0)


def components(pred, target, anchor, sigma, fdscale):
    # Same accepted Full-28 metric as V9: equal Arm-14/Hand-14 average.
    pos = np.abs(pred - target) / sigma[None, None, :]
    fd = np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)) / fdscale[None, None, :]
    seam = np.abs((pred[:, 0] - anchor) - (target[:, 0] - anchor)) / fdscale[None, :]
    return np.stack((pos.mean(axis=(1, 2)), fd.mean(axis=(1, 2)), seam.mean(axis=1)), axis=1)


def aggregate(values, mask, data):
    task_rows, episode_rows = [], []
    for task in TASKS:
        ix_task = mask & (data['task_id'] == task)
        episodes = np.unique(data['episode_id'][ix_task])
        means = []
        for episode in episodes:
            ix = ix_task & (data['episode_id'] == episode)
            avg = values[ix].mean(axis=0)
            means.append(avg)
            episode_rows.append({'task_id': task, 'episode_id': int(episode), 'anchor_support': int(ix.sum()),
                                 **dict(zip(METRICS, map(float, avg)))})
        if means:
            avg = np.stack(means).mean(axis=0)
            task_rows.append({'task_id': task, 'episode_support': len(episodes), 'anchor_support': int(ix_task.sum()),
                              **dict(zip(METRICS, map(float, avg)))})
    if not task_rows:
        return {m: None for m in METRICS}, task_rows, episode_rows
    return {m: float(np.mean([r[m] for r in task_rows])) for m in METRICS}, task_rows, episode_rows


def read_targets(root, split, data):
    if split == 'dev':
        path = root / DEV_TARGET
        verify(path, '233a0cdb31d1ddd7cea6ecfaa5e9d02e9c42e9822af7fe488c7fe5a40febee27')
    else:
        path = root / RUN / 'scores/core_b/FINAL_TARGETS_AFTER_PREDICTION_SEAL.npz'
        exposure = read_json(root / REC / 'core_b/FINAL_TARGET_EXPOSURE.json')
        expected = exposure.get('target_archive_sha256')
        if not expected:
            raise ValueError('Original final target exposure receipt has no archive checksum')
        verify(path, expected)
    with np.load(path, allow_pickle=False) as z:
        ids = z['anchor_id'].astype(str)
        if len(ids) != len(set(ids.tolist())) or set(ids.tolist()) != set(data['anchor_id'].astype(str)):
            raise ValueError('Target anchors do not match the sealed fixed population')
        lookup = {value: i for i, value in enumerate(ids)}
        order = np.asarray([lookup[str(value)] for value in data['anchor_id']])
        for key in ('episode_id', 'task_id', 'source_cluster'):
            if not np.array_equal(z[key][order], data[key]):
                raise ValueError('Target identity mismatch: ' + key)
        target = np.asarray(z['target_native_h30'][order], dtype=np.float64)
    if target.shape != (len(ids), 30, 28) or not np.isfinite(target).all():
        raise ValueError('Invalid target shape/values')
    return target, {'path': str(path), 'sha256': sha(path)}


def run(args):
    require_external_paths()
    output = Path(args.output).resolve()
    seal_path = output / 'PREDICTION_SEAL.json'
    seal = read_json(seal_path)
    if seal.get('status') != 'PREDICTIONS_SEALED' or seal.get('future_targets_loaded') is not False:
        raise ValueError('Predictions were not completed and sealed before scoring')
    root, split = Path(seal['source_root']), seal['split']
    data = load_prefix(root, split)
    verify(Path(seal['prefix_path']), seal['prefix_sha256'])
    n = len(data['anchor_id'])
    bundles = {}
    for record in seal['predictions']:
        path = Path(record['path'])
        verify(path, record['sha256'])
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['canonical_ids'].astype(str), data['canonical_id'].astype(str)):
                raise ValueError('Prediction row order differs from sealed prefix')
            pred, valid = z['prediction_native'].copy(), z['valid'].copy()
        if pred.shape != (n, 30, 28) or valid.shape != (n,) or valid.dtype != np.bool_:
            raise ValueError('Invalid sealed prediction shape/mask')
        if not np.isfinite(pred[valid]).all() or np.isfinite(pred[~valid]).any():
            raise ValueError('Invalid sealed prediction finite/abstention contract')
        key = (record['condition'], record['program'], record['b_seed'])
        if key in bundles:
            raise ValueError('Duplicate sealed prediction identity')
        bundles[key] = (pred, valid)
    if len(bundles) != 43:
        raise ValueError(f'Expected 43 original and cascade prediction bundles, found {len(bundles)}')
    audits = {}
    for a_seed, _ in PAIRS:
        audit_record = seal['classification_audits'][str(a_seed)]
        audit_path = Path(audit_record['path'])
        verify(audit_path, audit_record['sha256'])
        with audit_path.open(encoding='utf-8', newline='') as f:
            audit = list(csv.DictReader(f))
        if len(audit) != n or [r['anchor_id'] for r in audit] != data['anchor_id'].astype(str).tolist():
            raise ValueError('Classification audit identity mismatch')
        audits[a_seed] = audit
    validate_bundle_masks(bundles, audits, n)
    score_dir = output / 'scores'
    score_dir.mkdir(exist_ok=False)
    # No target is opened before all prediction hashes and identities above pass.
    write_json(score_dir / 'TARGET_ACCESS_STARTED.json', {'prediction_seal_sha256': sha(seal_path), 'prediction_count': len(bundles)})
    target, target_receipt = read_targets(root, split, data)
    verify(root / METRIC, 'dbf36b2dda89f57f140e17afaf1c9f7adceb7b33b3130102e034923e6a7cdc32')
    metric = read_json(root / METRIC)
    sigma = np.asarray(metric['target_sigma'], dtype=np.float64)
    fdscale = np.asarray(metric['safe_fd_scale'], dtype=np.float64)
    if sigma.shape != (28,) or fdscale.shape != (28,) or np.any(sigma <= 0) or np.any(fdscale <= 0):
        raise ValueError('Original training-derived metric scales are invalid')
    errors = {key: components(p, target, data['anchor_native'], sigma, fdscale) for key, (p, _) in bundles.items()}
    seed_rows, task_rows, episode_rows, coverage, contrast_rows, class_rows, confusion = [], [], [], [], [], [], []
    programs = PROGRAMS
    persist_avg, persist_tasks, persist_episodes = aggregate(errors[('persistence', 'PERSISTENCE', None)], np.ones(n, bool), data)
    persist_meta = {'condition': 'persistence', 'scope': 'full_original_population', 'program': 'PERSISTENCE', 'a_seed': None, 'b_seed': None}
    seed_rows.append({**persist_meta, 'anchor_support': n, 'episode_support': len(np.unique(data['episode_id'])), 'task_support': len(persist_tasks), **persist_avg})
    task_rows.extend({**persist_meta, **r} for r in persist_tasks)
    episode_rows.extend({**persist_meta, **r} for r in persist_episodes)
    for a_seed, b_seed in PAIRS:
        audit = audits[a_seed]
        valid = bundles[('core_a_task', 'V0', b_seed)][1]
        predicted = np.asarray([int(r['predicted_task_id']) for r in audit])
        for true_task in TASKS:
            for predicted_task in np.unique(predicted[data['task_id'] == true_task]):
                ix = (data['task_id'] == true_task) & (predicted == predicted_task)
                confusion.append({'a_seed': a_seed, 'b_seed': b_seed, 'true_task_id': true_task,
                                  'predicted_task_id': int(predicted_task), 'anchors': int(ix.sum()),
                                  'episodes': len(np.unique(data['episode_id'][ix])), 'accepted_anchors': int((ix & valid).sum())})
        for task in (*TASKS, 'ALL'):
            population = np.ones(n, bool) if task == 'ALL' else data['task_id'] == task
            accepted = population & valid
            coverage.append({'a_seed': a_seed, 'b_seed': b_seed, 'true_task_id': task,
                             'anchors': int(population.sum()), 'accepted_anchors': int(accepted.sum()),
                             'coverage': float(accepted.sum() / population.sum()) if population.any() else None,
                             'episodes': len(np.unique(data['episode_id'][population])),
                             'accepted_episodes': len(np.unique(data['episode_id'][accepted])),
                             'unrestricted_correct_anchors': sum(population[i] and int(r['predicted_task_id']) == int(r['true_task_id']) for i, r in enumerate(audit)),
                             'unsupported_anchors': sum(population[i] and r['status'] == 'unsupported_core_a_task' for i, r in enumerate(audit)),
                             'post_anchor_rejections': sum(population[i] and r['status'] == 'classification_observes_after_core_b_anchor' for i, r in enumerate(audit))})
        for program in programs:
            oracle = errors[('oracle_task', program, b_seed)]
            cascade = errors[('core_a_task', program, b_seed)]
            for condition, scope, values, mask in (
                ('oracle_task', 'full_original_population', oracle, np.ones(n, bool)),
                ('oracle_task', 'matched_accepted_rows', oracle, valid),
                ('core_a_task', 'matched_accepted_rows', cascade, valid),
                ('persistence', 'matched_accepted_rows', errors[('persistence', 'PERSISTENCE', None)], valid),
            ):
                avg, tasks, episodes = aggregate(values, mask, data)
                meta = {'condition': condition, 'scope': scope, 'program': program if condition != 'persistence' else 'PERSISTENCE', 'a_seed': a_seed, 'b_seed': b_seed}
                # Persistence is identical across model programs; store each pair once.
                if condition == 'persistence' and program != programs[0]:
                    continue
                seed_rows.append({**meta, 'anchor_support': int(mask.sum()), 'episode_support': len(np.unique(data['episode_id'][mask])),
                                  'task_support': len(tasks), **avg})
                task_rows.extend({**meta, **r} for r in tasks)
                episode_rows.extend({**meta, **r} for r in episodes)
            delta, tasks, _ = aggregate(cascade - oracle, valid, data)
            contrast_rows.append({'program': program, 'a_seed': a_seed, 'b_seed': b_seed,
                                  'comparison': 'core_a_task_minus_oracle_task_on_same_rows',
                                  'anchor_support': int(valid.sum()), 'task_support': len(tasks), **delta})
            for predicted_task in TASKS:
                ix_class = valid & (predicted == predicted_task)
                episodes = np.unique(data['episode_id'][ix_class])
                for condition, values in (('core_a_task', cascade), ('oracle_task', oracle)):
                    means = [values[ix_class & (data['episode_id'] == ep)].mean(axis=0) for ep in episodes]
                    average = np.mean(means, axis=0) if means else [None] * 3
                    class_rows.append({'program': program, 'condition': condition, 'a_seed': a_seed, 'b_seed': b_seed,
                                       'predicted_task_id': predicted_task, 'anchor_support': int(ix_class.sum()),
                                       'episode_support': len(episodes), 'weighting': 'equal_episode_within_predicted_class',
                                       **{k: None if v is None else float(v) for k, v in zip(METRICS, average)}})
    csv_write(score_dir / 'COVERAGE_BY_TRUE_TASK.csv', coverage)
    csv_write(score_dir / 'PER_SEED.csv', seed_rows)
    csv_write(score_dir / 'PER_TASK_SEED.csv', task_rows)
    csv_write(score_dir / 'PER_EPISODE_SEED.csv', episode_rows)
    csv_write(score_dir / 'PAIRED_DELTAS_PER_SEED.csv', contrast_rows)
    csv_write(score_dir / 'PER_PREDICTED_CLASS_SEED.csv', class_rows)
    csv_write(score_dir / 'CLASSIFICATION_CONFUSION.csv', confusion)
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row['condition'], row['scope'], row['program'])].append(row)
    summary = []
    for (condition, scope, program), rows in grouped.items():
        row = {'condition': condition, 'scope': scope, 'program': program, 'pairs': len(rows),
               'anchor_support_min': min(r['anchor_support'] for r in rows), 'anchor_support_max': max(r['anchor_support'] for r in rows),
               'task_support_min': min(r['task_support'] for r in rows)}
        for name in METRICS:
            values = [r[name] for r in rows if r[name] is not None]
            row[name + '_mean'] = float(np.mean(values)) if values else None
            row[name + '_sample_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    csv_write(score_dir / 'SUMMARY_MEAN_SD.csv', summary)
    # Optional uncertainty on exactly the common accepted population across all
    # three predeclared seed pairs; resample episodes within each true task.
    common = np.logical_and.reduce([bundles[('core_a_task', 'V0', b)][1] for _, b in PAIRS])
    shared_episodes, shared_indices = shared_episode_resamples(data, common)
    resample_path = score_dir / 'PAIRED_BOOTSTRAP_SHARED_INDICES.npz'
    if shared_episodes:
        np.savez_compressed(resample_path, **{f'task{task}_{field}': values
            for task, episodes, indices in zip(TASKS, shared_episodes, shared_indices)
            for field, values in (('episode_ids', episodes), ('resample_indices', indices))})
    bootstrap = []
    for program in programs:
        avg_delta = np.mean([errors[('core_a_task', program, b)] - errors[('oracle_task', program, b)] for _, b in PAIRS], axis=0)
        episode_by_task = []
        for task, eps in zip(TASKS, shared_episodes):
            episode_by_task.append(np.stack([avg_delta[common & (data['task_id'] == task) & (data['episode_id'] == ep)].mean(axis=0) for ep in eps]))
        if episode_by_task:
            draws = bootstrap_from_shared_indices(episode_by_task, shared_indices)
            point = np.mean([v.mean(axis=0) for v in episode_by_task], axis=0)
            for j, name in enumerate(METRICS):
                low, high = np.quantile(draws[:, j], [0.025, 0.975])
                bootstrap.append({'program': program, 'metric': name, 'delta': float(point[j]), 'ci_low': float(low), 'ci_high': float(high),
                                  'common_accepted_anchors': int(common.sum()), 'common_accepted_episodes': sum(map(len, episode_by_task)),
                                  'draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
                                  'resample_indices_shared_across_programs': True,
                                  'scope': 'conditional_on_fixed_models_and_common_accepted_rows',
                                  'resampling': 'episodes_within_true_task; fixed_seed_pair_average; no_seed_resampling'})
    csv_write(score_dir / 'PAIRED_BOOTSTRAP_COMMON_COHORT.csv', bootstrap)
    write_json(score_dir / 'SCORE_RECEIPT.json', {'status': 'PASS', 'split': split, 'prediction_seal_sha256': sha(seal_path),
                'target_receipt': target_receipt, 'metric_scale_sha256': sha(root / METRIC), 'seed_pairs': [list(p) for p in PAIRS],
                'population_anchors': n, 'population_episodes': len(np.unique(data['episode_id'])),
                'interpretation': 'Task-token substitution with fixed original models. Coverage and matched-row errors must be reported together. Oracle full-population and cascade accepted-subset metrics have different populations.',
                'accepted_subset_summary_estimand': 'SUMMARY_MEAN_SD averages three fixed seed-pair estimates, each on that pair\'s accepted rows; accepted populations may differ across seed pairs.',
                'common_cohort_bootstrap_estimand': 'PAIRED_BOOTSTRAP_COMMON_COHORT uses the intersection of accepted rows across all three seed pairs, averages fixed-pair errors there, then gives equal episode and true-task weight. Its point estimates and intervals do not necessarily describe the seed-specific accepted-subset mean in SUMMARY_MEAN_SD.',
                'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_draws': BOOTSTRAP_DRAWS,
                'bootstrap_indices_shared_across_programs': True,
                'bootstrap_indices_sha256': sha(resample_path) if shared_episodes else None,
                'common_accepted_anchors': int(common.sum()),
                'standardization': 'original training-derived command and safe-FD scales; not physical joint-angle error'})
    print(json.dumps({'status': 'PASS', 'score_dir': str(score_dir)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    run(p.parse_args())
