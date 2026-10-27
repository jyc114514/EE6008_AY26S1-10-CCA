#!/usr/bin/env python3
"""Complete the sealed V5 MIX50 baseline from already-scored rows; no target archive reads."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V5 = PROJECT / 'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
TASKS = [119, 94, 90, 141, 230]
SEEDS = [202610131, 202610132, 202610133]
METRICS = ['position_mae', 'safe_fd_l1', 'seam_safe_fd_l1']
REFERENCES = ['S', 'V', 'BIAS_STATE', 'BIAS_VIDEO']


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def bootstrap_group_stats(rows, metric, draws_by_task):
    """V5 accepted rule: sample source groups, carry all group episodes and anchors."""
    out = {}
    for task in TASKS:
        rr = [r for r in rows if int(r['task_id']) == task]
        by_episode = {}
        group_episode = {}
        for r in rr:
            ep = str(r['episode_id'])
            group = str(r['source_group'])
            by_episode.setdefault(ep, []).append(float(r[metric]))
            group_episode.setdefault(group, set()).add(ep)
        groups = sorted(group_episode)
        if not groups:
            out[task] = None
            continue
        group_sums = np.asarray([sum(float(np.mean(by_episode[e])) for e in sorted(group_episode[g])) for g in groups], dtype=np.float64)
        group_counts = np.asarray([len(group_episode[g]) for g in groups], dtype=np.float64)
        idx = draws_by_task[task]
        vals = group_sums[idx].sum(axis=1) / group_counts[idx].sum(axis=1)
        point = float(np.mean([np.mean(v) for v in by_episode.values()]))
        out[task] = {'draws': vals, 'point': point, 'n_groups': len(groups), 'n_episodes': len(by_episode)}
    return out


def main():
    seal = json.loads((V5 / 'qa/PREDICTION_SEAL.json').read_text())
    if seal.get('status') not in {'PASS', 'SEALED_BEFORE_TARGET_ACCESS'}:
        raise RuntimeError('V5 MIX50 seal is not accepted')
    exp = json.loads((V5 / 'inventory/OUTCOME_EXPOSURE.json').read_text())
    if exp.get('status') != 'complete' or int(exp.get('final_heldout_payload_access', -1)) != 0:
        raise RuntimeError('V5 outcome exposure receipt is not complete')
    primary = read_csv(V5 / 'results/PRIMARY_TASK_SEED.csv')
    if len(primary) != 15:
        raise RuntimeError('V5 primary task-seed table should have 15 rows')
    task_rows = read_csv(ROOT / 'results/V5_MIX50_TASK_SEED.csv')
    gain_rows = read_csv(ROOT / 'results/V5_MIX50_GAIN_TASK_SEED.csv')
    if len(task_rows) != 15 or len(gain_rows) != 60:
        raise RuntimeError('V6 copied V5 baseline rows are incomplete')

    # Reconstruct absolute per-window source rows only from V5's sealed scored tables.
    accepted = read_csv(V5 / 'results/PER_WINDOW_ACCEPTED_REFERENCES.csv')
    derived = read_csv(V5 / 'results/PER_WINDOW_DERIVED_LOSSES.csv')
    rows_by_method_seed = {}
    for method in REFERENCES + ['MIX50']:
        source_rows = derived if method == 'MIX50' else accepted
        source_method = 'MIX50' if method == 'MIX50' else method
        for seed in SEEDS:
            rr = [r for r in source_rows if r['method'] == source_method and int(r['seed']) == seed]
            if len(rr) != 185:
                raise RuntimeError(f'V5 scored windows incomplete {method}/{seed}: {len(rr)}')
            rows_by_method_seed[(method, seed)] = rr

    draw_rng = np.random.default_rng(20261015)
    draws_by_task, draw_meta = {}, {}
    for task in TASKS:
        rr = [r for r in rows_by_method_seed[('MIX50', SEEDS[0])] if int(r['task_id']) == task]
        groups = sorted({str(r['source_group']) for r in rr})
        if not groups:
            raise RuntimeError(f'V5 MIX50 has no source groups for task {task}')
        draws = draw_rng.integers(0, len(groups), size=(5000, len(groups)), dtype=np.int32)
        draws_by_task[task] = draws
        digest = hashlib.sha256(np.asarray(draws, dtype='<i4', order='C').tobytes()).hexdigest()
        draw_meta[str(task)] = {'groups': len(groups), 'group_ids': groups, 'sha256': digest}
    draw_manifest = {
        'schema_version': 1, 'status': 'PASS', 'rng': 'numpy.default_rng', 'seed': 20261015,
        'draws': 5000, 'same_draw_matrices_shared_across_all_methods_and_seeds': True,
        'within_task_source_groups': draw_meta, 'interval': 'percentile 95%', 'p_values': False,
        'source': 'V5 sealed source_group/episode/anchor scored rows', 'created_utc': utc(),
    }
    write_json(ROOT / 'qa/BOOTSTRAP_DRAW_MANIFEST.json', draw_manifest)

    # Full MIX50 absolute summary, including its three task-equal seed macros.
    absolute_rows = []
    for method in REFERENCES + ['MIX50']:
        by_seed = {}
        for seed in SEEDS:
            vals = {}
            for metric in METRICS:
                agg = bootstrap_group_stats(rows_by_method_seed[(method, seed)], metric, draws_by_task)
                vals[metric] = float(np.mean([agg[t]['point'] for t in TASKS]))
            by_seed[seed] = vals
        for metric in METRICS:
            per_seed = [by_seed[s][metric] for s in SEEDS]
            absolute_rows.append({
                'method': method, 'metric': metric,
                'seed_202610131_macro': per_seed[0], 'seed_202610132_macro': per_seed[1], 'seed_202610133_macro': per_seed[2],
                'mean_across_three_fixed_seeds': float(np.mean(per_seed)),
                'sample_sd_across_seeds': float(np.std(per_seed, ddof=1)),
            })
    write_csv(ROOT / 'results/V5_MIX50_FULL_SUMMARY.csv', absolute_rows)

    # Paired V5 source-reference gains: reference minus MIX50; intervals use shared V5 draws.
    ci_rows = []
    gain_macro = {}
    for reference in REFERENCES:
        contrast = f'{reference}_MINUS_MIX50'
        for seed in SEEDS:
            ref_rows = rows_by_method_seed[(reference, seed)]
            mix_rows = rows_by_method_seed[('MIX50', seed)]
            mix_by_id = {r['anchor_id']: r for r in mix_rows}
            if set(mix_by_id) != {r['anchor_id'] for r in ref_rows}:
                raise RuntimeError(f'V5 paired ID support differs for {reference}/{seed}')
            pair = []
            for r in ref_rows:
                m = mix_by_id[r['anchor_id']]
                if (r['episode_id'], r['task_id'], r['source_group']) != (m['episode_id'], m['task_id'], m['source_group']):
                    raise RuntimeError(f'V5 paired metadata mismatch for {reference}/{seed}/{r["anchor_id"]}')
                out = {k: r[k] for k in ['anchor_id', 'episode_id', 'task_id', 'source_group']}
                out['seed'] = str(seed)
                for metric in METRICS:
                    out[metric] = float(r[metric]) - float(m[metric])
                pair.append(out)
            for metric in METRICS:
                boot = bootstrap_group_stats(pair, metric, draws_by_task)
                for task in TASKS:
                    b = boot[task]
                    if b['n_groups'] < 2:
                        low = high = 'NA'
                    else:
                        low, high = float(np.quantile(b['draws'], .025)), float(np.quantile(b['draws'], .975))
                    ci_rows.append({'contrast': contrast, 'reference_method': reference, 'candidate_method': 'MIX50',
                                    'metric': metric, 'seed': seed, 'task_id': task, 'aggregation': 'within_task',
                                    'n_source_groups': b['n_groups'], 'draws': 5000, 'ci_level': .95,
                                    'point_estimate': b['point'], 'ci_low': low, 'ci_high': high,
                                    'interval': 'percentile_source_group_bootstrap' if b['n_groups'] >= 2 else 'NA_single_group'})
                macro_draw = np.mean(np.stack([boot[t]['draws'] for t in TASKS]), axis=0)
                macro_point = float(np.mean([boot[t]['point'] for t in TASKS]))
                gain_macro[(contrast, seed, metric)] = (macro_draw, macro_point)
                ci_rows.append({'contrast': contrast, 'reference_method': reference, 'candidate_method': 'MIX50',
                                'metric': metric, 'seed': seed, 'task_id': 'five_task_macro',
                                'aggregation': 'five_task_equal_within_task_source_group', 'n_source_groups': 47,
                                'draws': 5000, 'ci_level': .95, 'point_estimate': macro_point,
                                'ci_low': float(np.quantile(macro_draw, .025)), 'ci_high': float(np.quantile(macro_draw, .975)),
                                'interval': 'percentile_source_group_bootstrap'})
        for metric in METRICS:
            draws = np.mean(np.stack([gain_macro[(contrast, s, metric)][0] for s in SEEDS]), axis=0)
            point = float(np.mean([gain_macro[(contrast, s, metric)][1] for s in SEEDS]))
            ci_rows.append({'contrast': contrast, 'reference_method': reference, 'candidate_method': 'MIX50',
                            'metric': metric, 'seed': 'mean_across_formal_seeds', 'task_id': 'five_task_macro',
                            'aggregation': 'five_task_macro_mean_over_three_fixed_seeds', 'n_source_groups': 47,
                            'draws': 5000, 'ci_level': .95, 'point_estimate': point,
                            'ci_low': float(np.quantile(draws, .025)), 'ci_high': float(np.quantile(draws, .975)),
                            'interval': 'percentile_source_group_bootstrap_conditional_on_three_fixed_seeds'})
    write_csv(ROOT / 'results/V5_MIX50_PAIRED_GROUP_CI.csv', ci_rows)

    # Exact table-derived summary and complete audit receipt.
    mix_metrics = {r['metric']: {'mean_across_three_fixed_seeds': float(r['mean_across_three_fixed_seeds']),
                                'sample_sd_across_seeds': float(r['sample_sd_across_seeds'])}
                   for r in absolute_rows if r['method'] == 'MIX50'}
    summary = {
        'schema_version': 1, 'status': 'PASS', 'method': 'SV50 / V5 MIX50',
        'source_primary_table': str(V5 / 'results/PRIMARY_TASK_SEED.csv'),
        'source_per_window_derived': str(V5 / 'results/PER_WINDOW_DERIVED_LOSSES.csv'),
        'source_per_window_references': str(V5 / 'results/PER_WINDOW_ACCEPTED_REFERENCES.csv'),
        'task_seed_rows': len(task_rows), 'task_seed_metrics': mix_metrics,
        'paired_references': REFERENCES, 'paired_ci_rows': len(ci_rows),
        'bootstrap_seed': 20261015, 'bootstrap_draws': 5000,
        'exposure': {'new_target_archive_reads_this_reconciliation': 0,
                     'new_dev_loss_evaluations': 0, 'only_existing_sealed_v5_scored_rows_used': True},
        'final_heldout_payload_access': 0, 'created_utc': utc(),
    }
    write_json(ROOT / 'results/V5_MIX50_RECONCILIATION.json', summary)
    print(json.dumps({'status': 'PASS', 'task_seed_rows': len(task_rows), 'paired_ci_rows': len(ci_rows),
                      'mix50_macro': mix_metrics, 'new_dev_target_reads': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
