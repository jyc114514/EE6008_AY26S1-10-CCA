#!/usr/bin/env python3
"""Re-score exactly 18 accepted N3 H30 dev predictions; no inference/training."""
from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
NEW = PROJECT / 'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'
STATE = PROJECT / 'repo/ee6008-humanoid-vjepa/reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z'
V3 = PROJECT / 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
N1 = PROJECT / 'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
SEEDS = (20260953, 20260954, 20260955)
STATES = ('FULL', 'OBS', 'ANCHOR')
VIDEOS = ('N', 'D')
CELLS = tuple(f'{s}-{v}' for s in STATES for v in VIDEOS)
BOOT = 10_000
BOOT_SEED = 20261011
METRICS = (
    'arm_mae_std', 'hand_mae_std', 'full_mae_std',
    'arm_mae_native', 'hand_mae_native', 'full_mae_native',
    'arm_fd_native', 'hand_fd_native', 'full_fd_native',
    'arm_fd_scaled', 'hand_fd_scaled', 'full_fd_scaled',
    'full_fd_safe_normalized', 'full_anchor_to_first', 'full_endpoint_mae_native',
    'full_late_mae_std', 'full_tv', 'target_full_tv', 'full_tv_ratio', 'full_tv_closeness',
)
SUMMARY_METRICS = ('full_mae_std', 'full_mae_native', 'full_fd_native', 'full_fd_safe_normalized', 'full_tv_ratio')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def load_checksums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split('  ', 1)
        if len(parts) == 2:
            out[parts[1].removeprefix('./')] = parts[0]
    return out


def metric_windows(pred: np.ndarray, target: np.ndarray, anchor: np.ndarray,
                   sigma: np.ndarray, safe_fd: np.ndarray) -> dict[str, np.ndarray]:
    error = np.abs(pred - target)
    std_error = error / sigma[None, None, :]
    pred_delta = np.diff(pred, axis=1)
    target_delta = np.diff(target, axis=1)
    fd_error = np.abs(pred_delta - target_delta)
    horizon = pred.shape[1]
    late_start = max(horizon // 2, horizon - 10)
    tv_pred = np.abs(pred_delta).sum(axis=1).mean(axis=1)
    tv_target = np.abs(target_delta).sum(axis=1).mean(axis=1)
    tv_ratio = np.divide(tv_pred, tv_target, out=np.full(len(pred), np.nan), where=tv_target > 1e-12)
    return {
        'arm_mae_std': std_error[:, :, :14].mean(axis=(1, 2)),
        'hand_mae_std': std_error[:, :, 14:].mean(axis=(1, 2)),
        'full_mae_std': std_error.mean(axis=(1, 2)),
        'arm_mae_native': error[:, :, :14].mean(axis=(1, 2)),
        'hand_mae_native': error[:, :, 14:].mean(axis=(1, 2)),
        'full_mae_native': error.mean(axis=(1, 2)),
        'arm_fd_native': fd_error[:, :, :14].mean(axis=(1, 2)),
        'hand_fd_native': fd_error[:, :, 14:].mean(axis=(1, 2)),
        'full_fd_native': fd_error.mean(axis=(1, 2)),
        # Exact original N3 accepted scorer uses target_sigma for full_fd_scaled.
        'arm_fd_scaled': (fd_error[:, :, :14] / sigma[:14]).mean(axis=(1, 2)),
        'hand_fd_scaled': (fd_error[:, :, 14:] / sigma[14:]).mean(axis=(1, 2)),
        'full_fd_scaled': (fd_error / sigma[None, None, :]).mean(axis=(1, 2)),
        # Requested safe-FD sensitivity uses the N3 train-fitted safe_fd_scale.
        'full_fd_safe_normalized': (fd_error / safe_fd[None, None, :]).mean(axis=(1, 2)),
        'full_anchor_to_first': np.abs((pred[:, 0] - anchor) - (target[:, 0] - anchor)).mean(axis=1),
        'full_endpoint_mae_native': error[:, -1].mean(axis=1),
        'full_late_mae_std': std_error[:, late_start:].mean(axis=(1, 2)),
        'full_tv': tv_pred,
        'target_full_tv': tv_target,
        'full_tv_ratio': tv_ratio,
        'full_tv_closeness': np.abs(tv_ratio - 1.0),
    }


def reduce_window_to_group(values: np.ndarray, episode_ids: np.ndarray, group_ids: np.ndarray,
                           ordered_groups: list[str]) -> np.ndarray:
    out = np.empty(len(ordered_groups), dtype=np.float64)
    for i, group in enumerate(ordered_groups):
        group_mask = group_ids == group
        episode_values = []
        for ep in np.unique(episode_ids[group_mask]):
            episode_values.append(np.nanmean(values[group_mask & (episode_ids == ep)]))
        out[i] = np.nanmean(np.asarray(episode_values, dtype=np.float64))
    return out


def bootstrap_ci(values_by_seed_group: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    # Fixed-seed inference: average the paired effect over the same 3 seeds per group,
    # then resample source groups with one shared draw matrix across all outcomes.
    group_effect = np.nanmean(values_by_seed_group, axis=0)
    boot = np.nanmean(group_effect[draws], axis=1)
    return float(np.nanmean(group_effect)), float(np.nanquantile(boot, .025)), float(np.nanquantile(boot, .975))


def seed_summary(seed_values: np.ndarray) -> dict[str, float]:
    seed_values = np.asarray(seed_values, dtype=np.float64)
    out = {'mean_over_fixed_seeds': float(seed_values.mean()),
           'sample_sd_over_seeds': float(seed_values.std(ddof=1)),
           'median_over_seeds': float(np.median(seed_values))}
    for i, seed in enumerate(SEEDS):
        keep = np.arange(len(SEEDS)) != i
        out[f'loo_mean_excluding_{seed}'] = float(seed_values[keep].mean())
    return out


def main() -> int:
    # Recheck all prediction inputs before opening canonical target arrays.
    checksums = load_checksums(STATE / 'qa/CHECKSUMS.sha256')
    acceptance_rows = [r for r in read_csv(STATE / 'analysis/PREDICTION_ACCEPTANCE.csv')
                       if int(r['horizon_steps']) == 30 and r['cell'] in CELLS]
    if len(acceptance_rows) != 18:
        raise RuntimeError(f'accepted H30 row count !=18: {len(acceptance_rows)}')
    preflight: list[dict[str, Any]] = []
    for row in acceptance_rows:
        run_id = row['run_id']
        source_dir = STATE / 'predictions' / run_id
        source = source_dir / 'primary.npz'
        copied = NEW / 'n3_refs/predictions' / run_id / 'primary.npz'
        marker = json.loads((source_dir / 'PREDICTION_EXPORT_COMPLETE.json').read_text(encoding='utf-8'))
        run_config_path = NEW / 'n3_refs/runs' / run_id / 'RUN_CONFIG_LOCKED.json'
        complete = json.loads((NEW / 'n3_refs/runs' / run_id / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
        source_sha, copied_sha = sha(source), sha(copied)
        if checksums.get(str(source.relative_to(STATE))) != source_sha or copied_sha != source_sha:
            raise RuntimeError(f'prediction SHA/checksum/copy gate failed: {run_id}')
        cp = Path(marker['checkpoint'])
        if sha(cp) != marker.get('checkpoint_sha256'):
            raise RuntimeError(f'checkpoint SHA gate failed: {run_id}')
        cfg = json.loads(run_config_path.read_text(encoding='utf-8'))
        if (row['status'] != 'PASS' or row['final_test_access'] != '0' or marker.get('status') != 'PASS'
                or marker.get('state_unchanged') is not True or marker.get('final_test_access') != 0
                or marker.get('encoder_updated') is not False or marker.get('new_encoder_forward_for_extraction') is not False
                or complete.get('status') != 'PASS' or complete.get('epoch') != 40
                or complete.get('optimizer_steps') != 11120 or complete.get('final_test_access') != 0
                or cfg.get('final_test_access') != 0 or cfg.get('encoder_updated') is not False):
            raise RuntimeError(f'prediction acceptance gate failed: {run_id}')
        preflight.append({'run_id': run_id, 'prediction_sha256': copied_sha,
                          'checkpoint_sha256': marker['checkpoint_sha256'], 'status': 'PASS', 'final_test_access': 0})
    if len(preflight) != 18:
        raise RuntimeError('N3 accepted prediction preflight incomplete')

    # Confirm the locked canonical train/dev data and normalizer. No split beyond 0/1 is read.
    input_path = V3 / 'data/V3_MAIN_INPUTS.npz'
    input_hash = sha(input_path)
    with np.load(input_path, allow_pickle=False) as archive:
        split_code = archive['split_code'].astype(np.int8, copy=True)
        if set(np.unique(split_code).tolist()) != {0, 1}:
            raise RuntimeError(f'input contains unexpected split code(s): {np.unique(split_code)}')
        dev = np.flatnonzero(split_code == 1)
        train = np.flatnonzero(split_code == 0)
        if len(dev) != 2322 or len(train) != 17757:
            raise RuntimeError(f'canonical train/dev counts changed: train={len(train)} dev={len(dev)}')
        dev_ids = archive['anchor_id'][dev].astype('U96', copy=True)
        episode_ids = archive['episode_index'][dev].astype(np.int64, copy=True)
        tasks = archive['original_task_index'][dev].astype(np.int64, copy=True)
        targets = archive['targets_native'][dev, :30].astype(np.float64, copy=True)
        anchors = archive['anchor_command_native'][dev].astype(np.float64, copy=True)
    normalizer_path = NEW / 'configs/n3/V3_NORMALIZERS.json'
    normalizer = json.loads(normalizer_path.read_text(encoding='utf-8'))
    if normalizer.get('final_test_access') != 0 or normalizer.get('fit_scope') != 'primary train windows only; episode equal':
        raise RuntimeError('N3 normalizer fit/final access contract mismatch')
    sigma = np.asarray(normalizer['target_sigma'], dtype=np.float64)
    safe_fd = np.asarray(normalizer['safe_fd_scale'], dtype=np.float64)
    if sigma.shape != (28,) or safe_fd.shape != (28,) or not np.isfinite(sigma).all() or not np.isfinite(safe_fd).all():
        raise RuntimeError('N3 train-fitted normalizer shapes/finite check failed')

    group_map: dict[int, tuple[str, str]] = {}
    with (N1 / 'inventory/SOURCE_GROUP_MAP.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ep = int(row['episode_index'])
            if ep in group_map and group_map[ep] != (row['split'], row['source_group_id']):
                raise RuntimeError(f'conflicting source-group map for episode {ep}')
            group_map[ep] = (row['split'], row['source_group_id'])
    group_ids = np.asarray([group_map[int(ep)][1] for ep in episode_ids], dtype='U128')
    if any(group_map[int(ep)][0] != 'dev' for ep in episode_ids):
        raise RuntimeError('dev episode mapped to non-dev split')
    unique_eps = np.unique(episode_ids)
    episode_to_group = {(int(ep), group_map[int(ep)][1]) for ep in unique_eps}
    if len(unique_eps) != 603 or len({g for _, g in episode_to_group}) != 603 or len(episode_to_group) != 603:
        raise RuntimeError('episode/source-group map is not 603-to-603 one-to-one')
    ordered_groups = sorted({str(x) for x in group_ids})
    if len(ordered_groups) != 603:
        raise RuntimeError('source group count mismatch')

    predictions: dict[tuple[int, str], np.ndarray] = {}
    per_group: dict[tuple[int, str, str], np.ndarray] = {}
    state_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for row in sorted(acceptance_rows, key=lambda x: (int(x['seed']), CELLS.index(x['cell']))):
        run_id = row['run_id']; seed = int(row['seed']); state, video = row['cell'].split('-')
        path = NEW / 'n3_refs/predictions' / run_id / 'primary.npz'
        with np.load(path, allow_pickle=False) as bundle:
            pred = bundle['predictions_native'].astype(np.float64, copy=True)
            ids = bundle['anchor_id'].astype('U96', copy=True)
            bundle_eps = bundle['episode_index'].astype(np.int64, copy=True) if 'episode_index' in bundle.files else None
        if pred.shape != (2322, 30, 28) or not np.isfinite(pred).all() or not np.array_equal(ids, dev_ids):
            raise RuntimeError(f'prediction shape/ID/finite acceptance failed: {run_id}')
        if bundle_eps is not None and not np.array_equal(bundle_eps, episode_ids):
            raise RuntimeError(f'prediction episode identity mismatch: {run_id}')
        predictions[(seed, row['cell'])] = pred
        windows = metric_windows(pred, targets, anchors, sigma, safe_fd)
        reduced = {m: reduce_window_to_group(windows[m], episode_ids, group_ids, ordered_groups) for m in METRICS}
        for m, vals in reduced.items():
            per_group[(seed, row['cell'], m)] = vals
        values = {m: float(vals.mean()) for m, vals in reduced.items()}
        state_rows.append({'run_id': run_id, 'seed': seed, 'state_level': state, 'video_level': video,
                           'horizon_steps': 30, 'n_windows': 2322, 'n_episodes': 603, 'n_source_groups': 603,
                           **values, 'final_test_access': 0})
        for gi, group in enumerate(ordered_groups):
            group_rows.append({'run_id': run_id, 'seed': seed, 'state_level': state, 'video_level': video,
                               'source_group_id': group, **{m: float(reduced[m][gi]) for m in METRICS},
                               'final_test_access': 0})

    # Exact reconciliation against the original accepted N3 scorer for all 18 cells.
    original_rows = [r for r in read_csv(STATE / 'analysis/RESULTS_BY_SEED.csv')
                     if int(r['horizon_steps']) == 30 and r['cell'] in CELLS]
    if len(original_rows) != 18:
        raise RuntimeError(f'original N3 score rows !=18: {len(original_rows)}')
    original = {(int(r['seed']), r['cell']): r for r in original_rows}
    compare_fields = ('arm_mae_std', 'hand_mae_std', 'full_mae_std', 'full_mae_native',
                      'full_fd_native', 'full_fd_scaled', 'full_tv', 'target_full_tv', 'full_tv_ratio')
    comparisons: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for row in state_rows:
        old = original[(int(row['seed']), f"{row['state_level']}-{row['video_level']}")]
        for field in compare_fields:
            old_value = float(old[field]); new_value = float(row[field]); diff = abs(old_value - new_value)
            max_abs_diff = max(max_abs_diff, diff)
            comparisons.append({'seed': row['seed'], 'cell': f"{row['state_level']}-{row['video_level']}",
                                'metric': field, 'original_accepted': old_value, 'recovered': new_value,
                                'absolute_difference': diff, 'status': 'PASS' if diff <= 5e-10 else 'FAIL'})
    if any(r['status'] != 'PASS' for r in comparisons):
        raise RuntimeError(f'N3 recovered scores disagree with accepted score table; max_abs_diff={max_abs_diff}')

    write_csv(NEW / 'results/N3_STATE_RESULTS.csv', state_rows)
    write_csv(NEW / 'figure_data/N3_GROUP_METRICS.csv', group_rows)
    write_csv(NEW / 'qa/N3_ORIGINAL_SCORE_RECONCILIATION.csv', comparisons)
    write_csv(NEW / 'qa/PREDICTION_ACCEPTANCE.csv', [
        {**r, 'shape_expected': '[2322,30,28]', 'ids_match_canonical_dev': 'true',
         'finite': 'true', 'n3_reuse_copy_sha256_verified': 'true'} for r in preflight
    ])
    write_json(NEW / 'qa/PREDICTION_ACCEPTANCE.json', {
        'status': 'PASS_N3_ACCEPTED_BUNDLES', 'accepted': '18/18', 'new_inference': 0,
        'windows_per_bundle': 2322, 'episodes': 603, 'source_groups': 603,
        'shape': [2322, 30, 28], 'all_ids_match_canonical_dev': True,
        'all_finite': True, 'source_checksum_match': True, 'copy_hash_match': True,
        'checkpoint_hash_match': True, 'final_test_access': 0,
        'target_arrays_opened_after_prediction_acceptance': True,
    })

    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(ordered_groups), size=(BOOT, len(ordered_groups)), dtype=np.uint16)
    draw_sha = hashlib.sha256(draws.tobytes()).hexdigest()
    bootstrap_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    loo_rows: list[dict[str, Any]] = []
    for metric in SUMMARY_METRICS:
        # Descriptive L values by state/video.
        for state in STATES:
            for video in VIDEOS:
                by_seed = np.stack([per_group[(seed, f'{state}-{video}', metric)] for seed in SEEDS])
                point, low, high = bootstrap_ci(by_seed, draws)
                bootstrap_rows.append({'metric': metric, 'estimand': f'L_{state}_{video}', 'point_estimate': point,
                                       'ci95_low': low, 'ci95_high': high, 'replicates': BOOT,
                                       'bootstrap_unit': 'paired source group; fixed seeds',
                                       'prng_seed': BOOT_SEED, 'shared_draw_sha256': draw_sha,
                                       'seed_count': 3, 'source_groups': 603, 'final_test_access': 0})
        effects: dict[str, np.ndarray] = {}
        for state in STATES:
            effects[f'g_{state}'] = np.stack([
                per_group[(seed, f'{state}-N', metric)] - per_group[(seed, f'{state}-D', metric)] for seed in SEEDS
            ])
        effects['h_command_at_N'] = np.stack([
            per_group[(seed, 'OBS-N', metric)] - per_group[(seed, 'FULL-N', metric)] for seed in SEEDS
        ])
        effects['h_observation_at_N'] = np.stack([
            per_group[(seed, 'ANCHOR-N', metric)] - per_group[(seed, 'OBS-N', metric)] for seed in SEEDS
        ])
        effects['J_OBS'] = effects['g_OBS'] - effects['g_FULL']
        effects['J_ANCHOR'] = effects['g_ANCHOR'] - effects['g_FULL']
        effects['J_ANCHOR_minus_OBS'] = effects['g_ANCHOR'] - effects['g_OBS']
        for name, arr in effects.items():
            per_seed = np.asarray([np.nanmean(arr[i]) for i in range(3)], dtype=np.float64)
            summary = seed_summary(per_seed)
            point, low, high = bootstrap_ci(arr, draws)
            row = {'metric': metric, 'estimand': name, 'definition': (
                    'L_N - L_D; positive favors dynamic video' if name.startswith('g_') else
                    'error after reduced history minus error with more history' if name.startswith('h_') else
                    'difference in video gain; positive means the lower-information condition has a less negative or more positive gain'),
                   'point_estimate_mean_fixed_seeds': point, 'group_bootstrap_ci95_low': low,
                   'group_bootstrap_ci95_high': high, 'group_bootstrap_replicates': BOOT,
                   'bootstrap_unit': 'paired source group; fixed seeds', 'prng_seed': BOOT_SEED,
                   'shared_draw_sha256': draw_sha, 'source_groups': 603, 'final_test_access': 0,
                   **{f'seed_{SEEDS[i]}': float(per_seed[i]) for i in range(3)}, **summary}
            if name.startswith('g_'):
                gain_rows.append(row)
            else:
                effect_rows.append(row)
            bootstrap_rows.append({'metric': metric, 'estimand': name, 'point_estimate': point,
                                   'ci95_low': low, 'ci95_high': high, 'replicates': BOOT,
                                   'bootstrap_unit': 'paired source group; fixed seeds',
                                   'prng_seed': BOOT_SEED, 'shared_draw_sha256': draw_sha,
                                   'seed_count': 3, 'source_groups': 603, 'final_test_access': 0})
            loo_rows.append({'metric': metric, 'estimand': name, **summary,
                             'seed_20260953': float(per_seed[0]), 'seed_20260954': float(per_seed[1]),
                             'seed_20260955': float(per_seed[2]), 'final_test_access': 0})
    write_csv(NEW / 'results/N3_VIDEO_GAINS.csv', gain_rows)
    write_csv(NEW / 'results/N3_INFORMATION_INTERACTIONS.csv', effect_rows)
    write_csv(NEW / 'results/BOOTSTRAP_CI.csv', bootstrap_rows)
    write_csv(NEW / 'results/SEED_MEDIAN_ALL_LOO.csv', loo_rows)
    write_csv(NEW / 'results/SUPPORT_AND_EXCLUSIONS.csv', [
        {'run_id': r['run_id'], 'seed': r['seed'], 'state_level': r['state_level'], 'video_level': r['video_level'],
         'horizon_steps': 30, 'windows': 2322, 'episodes': 603, 'source_groups': 603,
         'prediction_status': 'accepted_reuse', 'included': True, 'exclusion_reason': '',
         'final_test_access': 0} for r in state_rows
    ] + [{'run_id': 'FINAL_HELDOUT', 'seed': '', 'state_level': '', 'video_level': '', 'horizon_steps': '',
          'windows': 0, 'episodes': 0, 'source_groups': 0, 'prediction_status': 'not_accessed',
          'included': False, 'exclusion_reason': 'FINAL_TEST_ACCESS=0', 'final_test_access': 0}])
    write_json(NEW / 'qa/SCORER_AND_DECOMPOSITION_QA.json', {
        'status': 'PASS_N3_ORIGINAL_SCORER_RECONCILED', 'original_metric_rows_compared': len(original_rows),
        'metric_comparisons': len(comparisons), 'max_absolute_difference': max_abs_diff,
        'tolerance': 5e-10, 'source_group_count': 603, 'episode_to_group_one_to_one': True,
        'original_aggregation': 'window -> episode -> equal source group; 603 groups map one-to-one to 603 episodes',
        'main_metric': 'full_mae_std / target_sigma; lower is better',
        'original_n3_fd_metric': 'full_fd_scaled uses target_sigma as in accepted N3 scorer',
        'additional_safe_fd_metric': 'full_fd_safe_normalized uses the separate N3 train-fitted safe_fd_scale',
        'direction_metric': 'not present in the original N3 accepted scorer; not introduced here',
        'final_test_access': 0,
    })
    write_json(NEW / 'qa/ANALYSIS_ACCEPTANCE.json', {
        'status': 'PASS_N3_RESCORING', 'accepted_views': '18/18', 'original_scores_reconciled': True,
        'bootstrap_replicates': BOOT, 'bootstrap_prng_seed': BOOT_SEED,
        'fixed_seed_count': 3, 'all_seed_loo_reported': True,
        'new_training': 0, 'new_inference': 0, 'new_encoder_extraction': 0,
        'final_test_access': 0,
    })
    print(json.dumps({'status': 'PASS_N3_RESCORING', 'accepted_views': 18, 'score_comparisons': len(comparisons),
                      'max_abs_difference_vs_original': max_abs_diff, 'group_bootstrap_replicates': BOOT,
                      'final_test_access': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
