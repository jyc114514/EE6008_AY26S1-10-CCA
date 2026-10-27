#!/usr/bin/env python3
"""Synthetic scorer QA and one-seal-gated V6 dev scoring/statistics/figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / 'reports/benchmarks'
V2 = BENCH / 'CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V3 = BENCH / 'CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'
V4 = BENCH / 'CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V5 = BENCH / 'CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
TASKS = [119, 94, 90, 141, 230]
SEEDS = [202610131, 202610132, 202610133]
METRICS = ['position_mae', 'safe_fd_l1', 'seam_safe_fd_l1']
METHODS = ['S', 'A', 'V', 'BIAS_STATE', 'BIAS_VIDEO', 'PERSISTENCE', 'SV50', 'SS50', 'SV_CAL', 'SS_CAL']
CONTRASTS = {
    'VIDEO_SUBSTITUTION_50': ('SS50', 'SV50'),
    'VIDEO_SUBSTITUTION_CAL': ('SS_CAL', 'SV_CAL'),
    'SV_CALIBRATION_GAIN': ('SV50', 'SV_CAL'),
    'SS_CALIBRATION_GAIN': ('SS50', 'SS_CAL'),
    'SV_CAL_VS_S': ('S', 'SV_CAL'),
    'SV_CAL_VS_V': ('V', 'SV_CAL'),
    'SV_CAL_VS_BIAS_STATE': ('BIAS_STATE', 'SV_CAL'),
    'SV_CAL_VS_BIAS_VIDEO': ('BIAS_VIDEO', 'SV_CAL'),
    'SV50_VS_S': ('S', 'SV50'),
    'SV50_VS_V': ('V', 'SV50'),
    'SV50_VS_BIAS_STATE': ('BIAS_STATE', 'SV50'),
    'SV50_VS_BIAS_VIDEO': ('BIAS_VIDEO', 'SV50'),
    'SS50_VS_S': ('S', 'SS50'),
    'SS50_VS_A': ('A', 'SS50'),
}
ROLE_MAP = {'S': 'S_OPT', 'A': 'A_COMMON', 'V': 'V_MATCHED',
            'BIAS_STATE': 'BIAS_STATE', 'BIAS_VIDEO': 'BIAS_VIDEO'}
ATOL = 1e-12
RTOL = 1e-10


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
            for k in row:
                if k not in fields:
                    fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def score_native(pred, target, anchor, target_sigma, safe_fd_scale):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1:] != (30, 28):
        raise ValueError('score tensor must align as [N,30,28]')
    if anchor.shape != (len(pred), 28):
        raise ValueError('current command anchor must be [N,28]')
    pos = (np.abs(pred - target) / target_sigma[None, None, :]).mean(axis=(1, 2))
    fd = (np.abs(np.diff(pred, axis=1) - np.diff(target, axis=1)) / safe_fd_scale[None, None, :]).mean(axis=(1, 2))
    seam = (np.abs((pred[:, 0, :] - anchor) - (target[:, 0, :] - anchor)) / safe_fd_scale[None, :]).mean(axis=1)
    return {'position_mae': pos, 'safe_fd_l1': fd, 'seam_safe_fd_l1': seam}


def join_predictions_by_id(array, source_rows, canonical_rows):
    """Use explicit IDs and metadata; source row order is never used as a join key."""
    arr = np.asarray(array)
    if len(source_rows) != len(arr) or len({r['anchor_id'] for r in source_rows}) != len(source_rows):
        raise ValueError('prediction row table is not a unique one-to-one index')
    source_index = {r['anchor_id']: i for i, r in enumerate(source_rows)}
    if set(source_index) != {r['anchor_id'] for r in canonical_rows}:
        raise ValueError('source and canonical prediction ID sets differ')
    out = []
    for row in canonical_rows:
        src = source_rows[source_index[row['anchor_id']]]
        observed = (str(int(src['episode_id'])), str(int(src['task_id'])),
                    str(src.get('source_cluster', src.get('source_group', ''))))
        expected = (str(int(row['episode_id'])), str(int(row['task_id'])), str(row['source_group']))
        if observed != expected:
            raise ValueError(f'prediction metadata mismatch for anchor_id={row["anchor_id"]}')
        out.append(arr[source_index[row['anchor_id']]])
    return np.stack(out)


def window_rows_from_scores(method, seed, ids, metadata, scores, source):
    out = []
    for i, aid in enumerate(ids):
        m = metadata[aid]
        out.append({'method': method, 'seed': str(seed), 'anchor_id': aid,
                    'episode_id': str(m['episode_id']), 'task_id': str(m['task_id']),
                    'source_group': str(m['source_group']),
                    **{metric: float(scores[metric][i]) for metric in METRICS}, 'source': source})
    return out


def task_aggregate(rows, metric):
    result = {}
    for task in TASKS:
        rr = [r for r in rows if int(r['task_id']) == task]
        eps = {}
        for r in rr:
            eps.setdefault(str(r['episode_id']), []).append(float(r[metric]))
        result[task] = {
            'episode_macro': float(np.mean([np.mean(v) for v in eps.values()])) if eps else float('nan'),
            'anchor_pooled': float(np.mean([float(r[metric]) for r in rr])) if rr else float('nan'),
            'anchors': len(rr), 'episodes': len(eps),
            'source_groups': len({str(r['source_group']) for r in rr}),
        }
    return result


def validate_saved_loss_rows(rows, *, seal_sha, expected_methods, expected_seeds, expected_ids):
    if not rows:
        raise ValueError('saved loss table is empty')
    got = {(r['method'], int(r['seed'])) for r in rows}
    expected = {(m, int(s)) for m in expected_methods for s in expected_seeds}
    if got != expected:
        raise ValueError('saved loss table method/seed matrix differs from seal')
    by_key = {}
    for r in rows:
        key = (r['method'], int(r['seed']))
        by_key.setdefault(key, []).append(r)
        if r.get('prediction_seal_sha256') != seal_sha:
            raise ValueError('saved loss row is not bound to the accepted prediction seal')
    for key, rr in by_key.items():
        if len(rr) != len(expected_ids) or {r['anchor_id'] for r in rr} != set(expected_ids):
            raise ValueError(f'saved loss rows have incomplete IDs: {key}')
    return by_key


def bootstrap_group_stats(rows, metric, draws_by_task):
    out = {}
    for task in TASKS:
        rr = [r for r in rows if int(r['task_id']) == task]
        by_episode = {}
        group_episode = {}
        for r in rr:
            ep, group = str(r['episode_id']), str(r['source_group'])
            by_episode.setdefault(ep, []).append(float(r[metric]))
            group_episode.setdefault(group, set()).add(ep)
        groups = sorted(group_episode)
        if not groups:
            out[task] = None
            continue
        group_sums = np.asarray([sum(float(np.mean(by_episode[e])) for e in sorted(group_episode[g])) for g in groups], dtype=np.float64)
        group_counts = np.asarray([len(group_episode[g]) for g in groups], dtype=np.float64)
        draw_idx = draws_by_task[task]
        boot = group_sums[draw_idx].sum(axis=1) / group_counts[draw_idx].sum(axis=1)
        point = float(np.mean([np.mean(v) for v in by_episode.values()]))
        out[task] = {'draws': boot, 'point': point, 'n_groups': len(groups), 'n_episodes': len(by_episode)}
    return out


def synthetic_scorer_qa():
    # Three seeds, five task IDs, two episodes per task, two anchors per episode.
    rows = []
    for task in TASKS:
        for ep_ix in range(2):
            for anchor_ix in range(2):
                aid = f'syn_t{task}_e{ep_ix}_a{anchor_ix}'
                rows.append({'anchor_id': aid, 'episode_id': task * 100 + ep_ix,
                             'task_id': task, 'source_group': f'g{task}_{ep_ix}'})
    ids = [r['anchor_id'] for r in rows]
    n = len(rows)
    target = np.zeros((n, 30, 28), dtype=np.float64)
    anchor = np.zeros((n, 28), dtype=np.float64)
    sigma, fd = np.ones(28), np.ones(28)
    synthetic_scores = {}
    for seed in SEEDS:
        pred = {
            'S': np.full((n, 30, 28), .8 + (seed % 3) * .01),
            'A': np.full((n, 30, 28), .5), 'V': np.full((n, 30, 28), -.4),
            'BIAS_STATE': np.full((n, 30, 28), .7), 'BIAS_VIDEO': np.full((n, 30, 28), -.3),
            'PERSISTENCE': np.full((n, 30, 28), .2), 'SV50': np.full((n, 30, 28), .2),
            'SV_CAL': np.full((n, 30, 28), .2), 'SS50': np.full((n, 30, 28), .65),
            'SS_CAL': np.full((n, 30, 28), .66),
        }
        for method, arr in pred.items():
            # Deliberately permute a subset of sources; the explicit key join must restore canonical order.
            perm = np.arange(n)[::-1] if method in {'A', 'V'} else np.arange(n)
            source_rows = [rows[i] for i in perm]
            joined = join_predictions_by_id(arr[perm], source_rows, rows)
            loss = score_native(joined, target, anchor, sigma, fd)
            recs = window_rows_from_scores(method, seed, ids, {r['anchor_id']: r for r in rows}, loss, 'synthetic')
            synthetic_scores[(method, seed)] = recs
            if len(recs) != n or any(not np.isfinite(r[m]) for r in recs for m in METRICS):
                raise RuntimeError('synthetic scorer produced incomplete/nonfinite per-window table')
            if np.mean(loss['seam_safe_fd_l1']) <= 0 and method != 'SV_CAL':
                raise RuntimeError('synthetic seam branch was not exercised')
    # Ensure a corrupted ID set is rejected before the real target reader can run.
    bad_rows = list(rows)
    bad_rows[0] = {**bad_rows[0], 'anchor_id': 'wrong_id'}
    rejected_bad_id = False
    try:
        join_predictions_by_id(np.zeros((n, 30, 28)), bad_rows, rows)
    except ValueError:
        rejected_bad_id = True
    if not rejected_bad_id:
        raise RuntimeError('synthetic scorer failed to reject a wrong prediction ID')

    # Exercise accepted recovery-table validation, bound to a synthetic seal identity.
    seal_hash = 'a' * 64
    saved = []
    for method in ['SS50', 'SS_CAL']:
        for seed in SEEDS:
            for row in synthetic_scores[(method, seed)]:
                saved.append({**row, 'prediction_seal_sha256': seal_hash})
    resumed = validate_saved_loss_rows(saved, seal_sha=seal_hash,
                                        expected_methods=['SS50', 'SS_CAL'], expected_seeds=SEEDS,
                                        expected_ids=ids)
    if len(resumed) != 6:
        raise RuntimeError('synthetic saved loss recovery branch did not restore six method/seed tables')

    # Exercise primary task/seed aggregation and every fixed contrast branch.
    primary = {}
    for method in METHODS:
        for seed in SEEDS:
            rows_for = synthetic_scores[(method, seed)]
            primary[(method, seed)] = {m: task_aggregate(rows_for, m) for m in METRICS}
    pair_ct = 0
    for _, (reference, candidate) in CONTRASTS.items():
        for seed in SEEDS:
            for task in TASKS:
                rvals = primary[(reference, seed)]['position_mae'][task]['episode_macro']
                cvals = primary[(candidate, seed)]['position_mae'][task]['episode_macro']
                if not np.isfinite(rvals - cvals):
                    raise RuntimeError('synthetic fixed contrast branch returned nonfinite gain')
                pair_ct += 1
    # Recreate a shared source-group draw and exercise group/episode bootstrap.
    rng = np.random.default_rng(20261015)
    draws = {}
    for task in TASKS:
        groups = sorted({r['source_group'] for r in rows if int(r['task_id']) == task})
        draws[task] = rng.integers(0, len(groups), size=(32, len(groups)), dtype=np.int32)
    b = bootstrap_group_stats(synthetic_scores[('SS50', SEEDS[0])], 'position_mae', draws)
    if any(b[t] is None or b[t]['n_groups'] != 2 for t in TASKS):
        raise RuntimeError('synthetic group-bootstrap branch failed')
    if np.mean(synthetic_scores[('SS50', SEEDS[0])][0]['seam_safe_fd_l1']) <= 0:
        raise RuntimeError('synthetic seam loss branch failed')
    return {
        'schema_version': 1, 'status': 'PASS', 'created_utc': utc(),
        'synthetic_rows': n, 'tasks': len(TASKS), 'formal_seeds': len(SEEDS),
        'methods_exercised': METHODS, 'fixed_contrast_branches': pair_ct,
        'explicit_id_join_and_row_permutation': 'PASS', 'wrong_id_rejection': rejected_bad_id,
        'position_fd_seam_scoring': 'PASS', 'task_seed_aggregation': 'PASS',
        'saved_loss_recovery_branch': 'PASS', 'saved_loss_recovery_tables': len(resumed),
        'source_group_bootstrap': 'PASS', 'target_archive_opened': False,
    }


def ordered_receiver_metadata():
    ordered = read_csv(V5 / 'protocol/ORDERED_RECEIVER_IDS.csv')
    ordered.sort(key=lambda r: int(r['canonical_row_index']))
    ids = [r['anchor_id'] for r in ordered]
    with np.load(V2 / 'data/DEV_PREFIXES_NO_LABELS.npz', allow_pickle=False) as z:
        keys = ['anchor_id', 'episode_id', 'task_id', 'task_rank', 'source_cluster', 'anchor_frame', 'anchor_native']
        prefix = {k: z[k].copy() for k in keys}
    index = {str(x): i for i, x in enumerate(prefix['anchor_id'].astype(str).tolist())}
    if len(ids) != 185 or len(set(ids)) != 185 or set(ids) != set(index):
        raise RuntimeError('accepted label-free receiver IDs invalid')
    rows, meta = [], {}
    anchors = []
    for i, aid in enumerate(ids):
        j = index[aid]
        row = {'row': i, 'anchor_id': aid, 'episode_id': int(prefix['episode_id'][j]),
               'task_id': int(prefix['task_id'][j]), 'task_rank': int(prefix['task_rank'][j]),
               'source_group': str(prefix['source_cluster'][j]), 'anchor_frame': int(prefix['anchor_frame'][j])}
        rows.append(row)
        meta[aid] = row
        anchors.append(np.asarray(prefix['anchor_native'][j], dtype=np.float64))
    return ids, rows, meta, np.stack(anchors)


def source_prediction(role, seed, source_map, ids, rows):
    recs = [r for r in source_map['source_prediction_bundles'] if r.get('role') == role and int(r['seed']) == seed]
    if len(recs) != 1:
        raise RuntimeError(f'formal source prediction not unique for {role}/{seed}')
    rec = recs[0]
    path, row_path = Path(rec['prediction_path']), Path(rec['prediction_rows_path'])
    if sha_file(path) != rec['prediction_sha256'] or sha_file(row_path) != rec['prediction_rows_sha256']:
        raise RuntimeError(f'accepted source prediction hash mismatch for {role}/{seed}')
    tensors = load_file(str(path))
    if 'prediction_native' not in tensors:
        raise RuntimeError(f'accepted source tensor lacks prediction_native for {role}/{seed}')
    source_rows = read_csv(row_path)
    arr = join_predictions_by_id(np.asarray(tensors['prediction_native']), source_rows, rows)
    if arr.shape != (185, 30, 28) or not np.isfinite(arr).all():
        raise RuntimeError(f'accepted source native output invalid for {role}/{seed}')
    return arr, rec


def prediction_for_program(program, seed, seal_entries, ids, rows):
    matches = [x for x in seal_entries if x.get('program') == program and int(x['seed']) == seed]
    if len(matches) != 1:
        raise RuntimeError(f'sealed program prediction not unique: {program}/{seed}')
    rec = matches[0]
    path, row_path = Path(rec['prediction_path']), Path(rec['prediction_rows_path'])
    if sha_file(path) != rec['prediction_sha256'] or sha_file(row_path) != rec['prediction_rows_sha256']:
        raise RuntimeError(f'sealed program output hash mismatch: {program}/{seed}')
    tensors = load_file(str(path))
    if 'prediction_native' not in tensors:
        raise RuntimeError(f'sealed program tensor missing prediction_native: {program}/{seed}')
    src_rows = read_csv(row_path)
    arr = join_predictions_by_id(np.asarray(tensors['prediction_native']), src_rows, rows).astype(np.float64)
    if arr.shape != (185, 30, 28) or not np.isfinite(arr).all():
        raise RuntimeError(f'sealed program output shape/finite failure: {program}/{seed}')
    return arr, rec


def validate_prediction_seal():
    seal_path = ROOT / 'qa/PREDICTION_SEAL.json'
    if not seal_path.is_file():
        raise RuntimeError('dev target reader refuses missing prediction seal')
    seal = read_json(seal_path)
    if seal.get('status') != 'SEALED_BEFORE_TARGET_ACCESS' or seal.get('nonempty_seal') is not True:
        raise RuntimeError('dev target reader refuses empty or unaccepted prediction seal')
    if int(seal.get('accepted_logical_bundles', -1)) != 12 or int(seal.get('missing_bundles', -1)) != 0:
        raise RuntimeError('dev target reader refuses an incomplete logical prediction matrix')
    if seal.get('dev_target_reader_called') is not False or int(seal.get('preseal_dev_target_reads', -1)) != 0:
        raise RuntimeError('dev target was exposed before seal')
    alpha = read_json(ROOT / 'protocol/ALPHA_LOCK.json')
    if alpha.get('status') != 'LOCKED_BEFORE_V6_DEV_PREDICTION' or sha_file(ROOT / 'protocol/ALPHA_LOCK.json') != seal.get('alpha_lock_sha256'):
        raise RuntimeError('ALPHA_LOCK changed or is absent')
    if sha_file(ROOT / 'protocol/PROGRAM_LOCK.json') != seal.get('program_lock_sha256'):
        raise RuntimeError('PROGRAM_LOCK changed after prediction seal')
    return seal


def npy_header(zip_ref, member):
    with zip_ref.open(member, 'r') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise RuntimeError(f'unsupported NPY header version for {member}: {version}')
    return {'shape': list(shape), 'fortran_order': bool(fortran), 'dtype': str(dtype), 'version': list(version)}


def dev_target_schema_preflight(seal):
    path = V2 / 'data/DEV_TARGETS_SCORER_ONLY.npz'
    v5_exposure = read_json(V5 / 'inventory/OUTCOME_EXPOSURE.json')
    expected_sha = v5_exposure['target_archive_sha256']
    expected_size = int(v5_exposure['target_archive_bytes'])
    if not path.is_file() or path.stat().st_size != expected_size:
        raise RuntimeError('dev target scorer-only archive missing or size differs from sealed V5 evidence')
    required = ['anchor_id.npy', 'episode_id.npy', 'task_id.npy', 'source_cluster.npy', 'target_native_h30.npy']
    headers = {}
    with zipfile.ZipFile(path, 'r') as z:
        names = set(z.namelist())
        if not set(required).issubset(names):
            raise RuntimeError(f'dev target archive schema members are missing: {sorted(set(required)-names)}')
        for member in required:
            headers[member[:-4]] = npy_header(z, member)
    if headers['target_native_h30']['shape'] != [185, 30, 28]:
        raise RuntimeError(f'dev target future array shape invalid: {headers["target_native_h30"]}')
    for key in ['anchor_id', 'episode_id', 'task_id', 'source_cluster']:
        if not headers[key]['shape'] or headers[key]['shape'][0] != 185:
            raise RuntimeError(f'dev target metadata shape invalid for {key}: {headers[key]}')
    result = {
        'schema_version': 1, 'status': 'PASS', 'seal_sha256': sha_file(ROOT / 'qa/PREDICTION_SEAL.json'),
        'archive_path': str(path), 'expected_v5_archive_sha256': expected_sha, 'bytes': expected_size,
        'archive_member_metadata_open_count': 1, 'future_target_values_read': False,
        'members': headers, 'required_members': required,
        'accepted_prediction_bundles': seal['accepted_logical_bundles'],
        'final_heldout_payload_access': 0, 'created_utc': utc(),
    }
    write_json(ROOT / 'qa/DEV_TARGET_SCHEMA_PREFLIGHT.json', result)
    return result


def load_dev_outcomes_once(ids, metadata, expected_sha, expected_bytes):
    path = V2 / 'data/DEV_TARGETS_SCORER_ONLY.npz'
    blob = path.read_bytes()
    archive_sha = sha_bytes(blob)
    if len(blob) != expected_bytes or archive_sha != expected_sha:
        raise RuntimeError(f'dev target archive identity mismatch after seal: bytes={len(blob)} sha256={archive_sha}')
    required = {'anchor_id', 'episode_id', 'task_id', 'source_cluster', 'target_native_h30'}
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        if not required.issubset(set(z.files)):
            raise RuntimeError(f'dev target archive fields missing: {sorted(required-set(z.files))}')
        target_ids = z['anchor_id'].astype(str).tolist()
        target_eps = z['episode_id'].copy()
        target_tasks = z['task_id'].copy()
        target_groups = z['source_cluster'].astype(str)
        target = np.asarray(z['target_native_h30'], dtype=np.float64).copy()
    if len(target_ids) != 185 or len(set(target_ids)) != 185 or set(target_ids) != set(ids):
        raise RuntimeError('dev target receiver ID set differs from sealed prediction IDs')
    target_index = {aid: i for i, aid in enumerate(target_ids)}
    reordered = np.stack([target[target_index[aid]] for aid in ids])
    for aid in ids:
        i = target_index[aid]
        observed = (str(int(target_eps[i])), str(int(target_tasks[i])), str(target_groups[i]))
        expected = (str(int(metadata[aid]['episode_id'])), str(int(metadata[aid]['task_id'])), str(metadata[aid]['source_group']))
        if observed != expected:
            raise RuntimeError(f'dev target metadata join mismatch for anchor_id={aid}')
    if reordered.shape != (185, 30, 28) or not np.isfinite(reordered).all():
        raise RuntimeError('dev target future arrays shape/finite validation failed')
    return reordered, blob, archive_sha


def loss_rows_for_sources(ids, metadata, seed):
    refs = read_csv(V5 / 'results/PER_WINDOW_ACCEPTED_REFERENCES.csv')
    derived = read_csv(V5 / 'results/PER_WINDOW_DERIVED_LOSSES.csv')
    out = {}
    for method in ['S', 'A', 'V', 'BIAS_STATE', 'BIAS_VIDEO', 'PERSISTENCE']:
        rr = [r for r in refs if r['method'] == method and int(r['seed']) == seed]
        out[method] = convert_saved_rows(rr, method, seed, ids, metadata)
    mix = [r for r in derived if r['method'] == 'MIX50' and int(r['seed']) == seed]
    sv50 = convert_saved_rows(mix, 'SV50', seed, ids, metadata)
    svcal = convert_saved_rows(mix, 'SV_CAL', seed, ids, metadata)
    out['SV50'], out['SV_CAL'] = sv50, svcal
    return out


def convert_saved_rows(source_rows, method, seed, ids, metadata):
    by_id = {r['anchor_id']: r for r in source_rows}
    if len(by_id) != 185 or set(by_id) != set(ids):
        raise RuntimeError(f'accepted saved source scores incomplete for {method}/{seed}')
    out = []
    for aid in ids:
        src = by_id[aid]
        expected = metadata[aid]
        observed = (str(int(src['episode_id'])), str(int(src['task_id'])), str(src['source_group']))
        target = (str(int(expected['episode_id'])), str(int(expected['task_id'])), str(expected['source_group']))
        if observed != target:
            raise RuntimeError(f'saved score metadata mismatch for {method}/{seed}/{aid}')
        out.append({'method': method, 'seed': str(seed), 'anchor_id': aid,
                    'episode_id': str(int(expected['episode_id'])), 'task_id': str(int(expected['task_id'])),
                    'source_group': str(expected['source_group']),
                    **{m: float(src[m]) for m in METRICS}, 'source': 'accepted_v5_scored_window_reuse'})
    return out


def aggregate_primary(losses):
    aggregate = {}
    for method in METHODS:
        for seed in SEEDS:
            rr = losses[(method, seed)]
            for metric in METRICS:
                aggregate[(method, seed, metric)] = task_aggregate(rr, metric)
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            row = {'task_id': task, 'seed': seed, 'aggregation': 'episode_equal_within_task',
                   'persistence_seed_role': 'shared_reference' if seed == SEEDS[0] else 'shared_reference'}
            for method in METHODS:
                method_rows = [r for r in losses[(method, seed)] if int(r['task_id']) == task]
                stat = aggregate[(method, seed, 'position_mae')][task]
                row['anchors'] = stat['anchors']
                row['episodes'] = stat['episodes']
                row['source_groups'] = stat['source_groups']
                for metric in METRICS:
                    task_result = aggregate[(method, seed, metric)][task]
                    row[f'{method}_{metric}'] = task_result['episode_macro']
                    row[f'{method}_{metric}_anchor_pooled'] = task_result['anchor_pooled']
            rows.append(row)
    return rows, aggregate


def source_group_draws():
    manifest = read_json(ROOT / 'qa/BOOTSTRAP_DRAW_MANIFEST.json')
    if manifest.get('status') != 'PASS' or int(manifest.get('seed', -1)) != 20261015 or int(manifest.get('draws', -1)) != 5000:
        raise RuntimeError('V6 bootstrap draw manifest invalid')
    rng = np.random.default_rng(20261015)
    draws = {}
    for task in TASKS:
        meta = manifest['within_task_source_groups'][str(task)]
        groups = meta['group_ids']
        if groups != sorted(groups) or not groups:
            raise RuntimeError(f'bootstrap group list invalid for task {task}')
        idx = rng.integers(0, len(groups), size=(5000, len(groups)), dtype=np.int32)
        digest = hashlib.sha256(np.asarray(idx, dtype='<i4', order='C').tobytes()).hexdigest()
        if digest != meta['sha256']:
            raise RuntimeError(f'V6 bootstrap draw matrix hash mismatch for task {task}')
        draws[task] = idx
    return draws


def pair_window_rows(reference_rows, candidate_rows, metric_names=METRICS):
    ref, cand = ({r['anchor_id']: r for r in x} for x in (reference_rows, candidate_rows))
    if set(ref) != set(cand):
        raise RuntimeError('paired methods have different receiver ID sets')
    out = []
    for aid in ref:
        r, c = ref[aid], cand[aid]
        for key in ['episode_id', 'task_id', 'source_group']:
            if str(r[key]) != str(c[key]):
                raise RuntimeError(f'paired metadata differs for {key}/{aid}')
        row = {k: r[k] for k in ['anchor_id', 'episode_id', 'task_id', 'source_group']}
        row['seed'] = str(r['seed'])
        for metric in metric_names:
            row[metric] = float(r[metric]) - float(c[metric])
        out.append(row)
    return out


def compute_paired_and_ci(losses, draws):
    pair_rows, pair_maps = [], {}
    for name, (ref_method, cand_method) in CONTRASTS.items():
        for seed in SEEDS:
            recs = pair_window_rows(losses[(ref_method, seed)], losses[(cand_method, seed)])
            pair_maps[(name, seed)] = recs
            for task in TASKS:
                rr = [r for r in recs if int(r['task_id']) == task]
                first = rr[0]
                pair_rows.append({'contrast': name, 'reference_method': ref_method, 'candidate_method': cand_method,
                                  'seed': seed, 'task_id': task, 'anchors': len(rr),
                                  'episodes': len({r['episode_id'] for r in rr}),
                                  'source_groups': len({r['source_group'] for r in rr}),
                                  'aggregation': 'episode_equal_within_task',
                                  **{m: task_aggregate(rr, m)[task]['episode_macro'] for m in METRICS}})
    ci_rows = []
    macro_draws = {}
    for name, (ref_method, cand_method) in CONTRASTS.items():
        for seed in SEEDS:
            recs = pair_maps[(name, seed)]
            for metric in METRICS:
                boot = bootstrap_group_stats(recs, metric, draws)
                for task in TASKS:
                    b = boot[task]
                    ci_rows.append({'contrast': name, 'reference_method': ref_method, 'candidate_method': cand_method,
                                    'metric': metric, 'seed': seed, 'task_id': task, 'aggregation': 'within_task',
                                    'n_source_groups': b['n_groups'], 'draws': 5000, 'ci_level': .95,
                                    'point_estimate': b['point'], 'ci_low': float(np.quantile(b['draws'], .025)),
                                    'ci_high': float(np.quantile(b['draws'], .975)),
                                    'interval': 'percentile_source_group_bootstrap'})
                task_draw = np.mean(np.stack([boot[t]['draws'] for t in TASKS]), axis=0)
                task_point = float(np.mean([boot[t]['point'] for t in TASKS]))
                macro_draws[(name, seed, metric)] = task_draw
                ci_rows.append({'contrast': name, 'reference_method': ref_method, 'candidate_method': cand_method,
                                'metric': metric, 'seed': seed, 'task_id': 'five_task_macro',
                                'aggregation': 'five_task_equal_within_task_source_group',
                                'n_source_groups': sum(boot[t]['n_groups'] for t in TASKS), 'draws': 5000, 'ci_level': .95,
                                'point_estimate': task_point, 'ci_low': float(np.quantile(task_draw, .025)),
                                'ci_high': float(np.quantile(task_draw, .975)),
                                'interval': 'percentile_source_group_bootstrap'})
        for metric in METRICS:
            draws_mean = np.mean(np.stack([macro_draws[(name, s, metric)] for s in SEEDS]), axis=0)
            seed_macro_points = [float(np.mean([task_aggregate(pair_maps[(name, s)], metric)[t]['episode_macro'] for t in TASKS])) for s in SEEDS]
            point = float(np.mean(seed_macro_points))
            ci_rows.append({'contrast': name, 'reference_method': ref_method, 'candidate_method': cand_method,
                            'metric': metric, 'seed': 'mean_across_formal_seeds', 'task_id': 'five_task_macro',
                            'aggregation': 'five_task_macro_mean_over_three_fixed_seeds',
                            'n_source_groups': 47, 'draws': 5000, 'ci_level': .95, 'point_estimate': point,
                            'ci_low': float(np.quantile(draws_mean, .025)), 'ci_high': float(np.quantile(draws_mean, .975)),
                            'interval': 'percentile_source_group_bootstrap_conditional_on_three_fixed_seeds'})
    return pair_rows, ci_rows


def compute_seed_task_loo(primary_rows, pair_rows):
    seed_rows, task_rows = [], []
    by_primary = {(int(r['task_id']), int(r['seed'])): r for r in primary_rows}
    for method in METHODS:
        method_seeds = ['shared'] if method == 'PERSISTENCE' else list(map(str, SEEDS))
        for metric in METRICS:
            if method == 'PERSISTENCE':
                row = {'analysis': 'absolute_shared_reference', 'method_or_contrast': method,
                       'metric': metric, 'omitted_seed': 'NA', 'remaining_seeds': 'shared',
                       'n_remaining_seeds': 0, 'loo_five_task_macro': float(np.mean([
                           np.mean([float(by_primary[(t, SEEDS[0])][f'{method}_{metric}']) for t in TASKS])]))}
                seed_rows.append(row)
                continue
            per_seed = {s: float(np.mean([float(by_primary[(t, s)][f'{method}_{metric}']) for t in TASKS])) for s in SEEDS}
            for omitted in SEEDS:
                remaining = [s for s in SEEDS if s != omitted]
                seed_rows.append({'analysis': 'absolute', 'method_or_contrast': method, 'metric': metric,
                                  'omitted_seed': omitted, 'remaining_seeds': ','.join(map(str, remaining)),
                                  'n_remaining_seeds': 2,
                                  'loo_five_task_macro': float(np.mean([per_seed[s] for s in remaining]))})
        for seed in method_seeds:
            src_seed = SEEDS[0] if seed == 'shared' else int(seed)
            for omitted in TASKS:
                remaining = [t for t in TASKS if t != omitted]
                for metric in METRICS:
                    vals = [float(by_primary[(t, src_seed)][f'{method}_{metric}']) for t in remaining]
                    task_rows.append({'analysis': 'absolute', 'method_or_contrast': method, 'metric': metric,
                                      'seed': seed, 'omitted_task': omitted, 'remaining_tasks': ','.join(map(str, remaining)),
                                      'n_remaining_tasks': 4, 'loo_macro': float(np.mean(vals))})
    pair_lookup = {(r['contrast'], int(r['seed']), int(r['task_id'])): r for r in pair_rows}
    for name in CONTRASTS:
        for metric in METRICS:
            by_seed = {s: float(np.mean([float(pair_lookup[(name, s, t)][metric]) for t in TASKS])) for s in SEEDS}
            for omitted in SEEDS:
                remaining = [s for s in SEEDS if s != omitted]
                seed_rows.append({'analysis': 'paired_gain', 'method_or_contrast': name, 'metric': metric,
                                  'omitted_seed': omitted, 'remaining_seeds': ','.join(map(str, remaining)),
                                  'n_remaining_seeds': 2,
                                  'loo_five_task_macro': float(np.mean([by_seed[s] for s in remaining]))})
        for seed in SEEDS:
            for omitted in TASKS:
                remaining = [t for t in TASKS if t != omitted]
                for metric in METRICS:
                    vals = [float(pair_lookup[(name, seed, t)][metric]) for t in remaining]
                    task_rows.append({'analysis': 'paired_gain', 'method_or_contrast': name, 'metric': metric,
                                      'seed': seed, 'omitted_task': omitted, 'remaining_tasks': ','.join(map(str, remaining)),
                                      'n_remaining_tasks': 4, 'loo_macro': float(np.mean(vals))})
    return seed_rows, task_rows


def convexity_table(losses, alpha_lock):
    selected = {'SV50': (0.5, 'S', 'V'),
                'SV_CAL': (float(alpha_lock['selected_alphas']['SV']['alpha']), 'S', 'V'),
                'SS50': (0.5, 'S', 'A'),
                'SS_CAL': (float(alpha_lock['selected_alphas']['SS']['alpha']), 'S', 'A')}
    out = []
    for method, (alpha, left, right) in selected.items():
        for seed in SEEDS:
            lm = {r['anchor_id']: r for r in losses[(left, seed)]}
            rm = {r['anchor_id']: r for r in losses[(right, seed)]}
            cm = {r['anchor_id']: r for r in losses[(method, seed)]}
            if set(lm) != set(rm) or set(lm) != set(cm):
                raise RuntimeError(f'convexity ID support mismatch {method}/{seed}')
            for metric in METRICS:
                diffs, tolerances = [], []
                for aid in lm:
                    bound = (1-alpha)*float(lm[aid][metric]) + alpha*float(rm[aid][metric])
                    value = float(cm[aid][metric])
                    diffs.append(value-bound)
                    tolerances.append(ATOL + RTOL*abs(bound))
                violation = np.maximum(np.asarray(diffs)-np.asarray(tolerances), 0.0)
                out.append({'method': method, 'seed': seed, 'metric': metric, 'alpha': alpha,
                            'left_member': left, 'right_member': right, 'anchors': len(diffs),
                            'max_blend_minus_convex_bound': float(np.max(diffs)),
                            'max_allowed_abs_rel_tolerance': float(np.max(tolerances)),
                            'max_excess_over_tolerance': float(np.max(violation)),
                            'status': 'PASS' if float(np.max(violation)) == 0.0 else 'FAIL',
                            'atol': ATOL, 'rtol': RTOL})
    if any(r['status'] != 'PASS' for r in out):
        raise RuntimeError('locked per-window convexity inequality failed')
    return out


def mix_diagnostic(losses):
    out = []
    components = {'SV50': ('S', 'V'), 'SS50': ('S', 'A')}
    for program, (left, right) in components.items():
        for seed in SEEDS:
            maps = {m: {r['anchor_id']: r for r in losses[(m, seed)]} for m in [left, right, program]}
            for metric in METRICS:
                window = []
                for aid in maps[left]:
                    l, r, mix = (maps[m][aid] for m in [left, right, program])
                    window.append({'anchor_id': aid, 'episode_id': mix['episode_id'], 'task_id': mix['task_id'],
                                   'source_group': mix['source_group'],
                                   'mix_minus_mean_components': float(mix[metric]) - .5*(float(l[metric])+float(r[metric])),
                                   'gain_vs_better_single': min(float(l[metric]), float(r[metric])) - float(mix[metric])})
                for diagnostic in ['mix_minus_mean_components', 'gain_vs_better_single']:
                    vals_by_task = {}
                    for task in TASKS:
                        rr = [x for x in window if int(x['task_id']) == task]
                        eps = {}
                        for x in rr: eps.setdefault(str(x['episode_id']), []).append(float(x[diagnostic]))
                        vals_by_task[task] = float(np.mean([np.mean(v) for v in eps.values()]))
                        out.append({'program': program, 'metric': metric, 'diagnostic': diagnostic,
                                    'seed': seed, 'task_id': task, 'value': vals_by_task[task],
                                    'aggregation': 'episode_equal_within_task'})
                    out.append({'program': program, 'metric': metric, 'diagnostic': diagnostic,
                                'seed': seed, 'task_id': 'five_task_macro',
                                'value': float(np.mean([vals_by_task[t] for t in TASKS])),
                                'aggregation': 'five_task_equal_episode_equal'})
    return out


def residual_complementarity(predictions, target, sigma, fdscale, ids, metadata):
    out = []
    pairs = [('S', 'A'), ('S', 'V')]
    for seed in SEEDS:
        for left, right in pairs:
            for signal in ['level', 'internal_difference']:
                if signal == 'level':
                    e1 = (predictions[(left, seed)] - target) / sigma[None, None, :]
                    e2 = (predictions[(right, seed)] - target) / sigma[None, None, :]
                else:
                    e1 = (np.diff(predictions[(left, seed)], axis=1) - np.diff(target, axis=1)) / fdscale[None, None, :]
                    e2 = (np.diff(predictions[(right, seed)], axis=1) - np.diff(target, axis=1)) / fdscale[None, None, :]
                per_window = []
                for i, aid in enumerate(ids):
                    a, b = e1[i].ravel(), e2[i].ravel()
                    zero_a, zero_b = (a == 0), (b == 0)
                    nonzero = (~zero_a) & (~zero_b)
                    opposite = nonzero & ((a < 0) != (b < 0))
                    per_window.append({'anchor_id': aid, 'episode_id': str(metadata[aid]['episode_id']),
                                       'task_id': int(metadata[aid]['task_id']), 'source_group': metadata[aid]['source_group'],
                                       'opposite_all': float(opposite.mean()),
                                       'opposite_nonzero': float(opposite.sum()/nonzero.sum()) if nonzero.any() else 0.0,
                                       'zero_either': float((zero_a | zero_b).mean()),
                                       'both_zero': float((zero_a & zero_b).mean()),
                                       'both_nonzero': float(nonzero.mean())})
                task_results = {}
                for task in TASKS:
                    rr = [r for r in per_window if r['task_id'] == task]
                    eps = {}
                    for r in rr: eps.setdefault(r['episode_id'], []).append(r)
                    task_results[task] = {}
                    for col in ['opposite_all', 'opposite_nonzero', 'zero_either', 'both_zero', 'both_nonzero']:
                        task_results[task][col] = float(np.mean([np.mean([x[col] for x in vv]) for vv in eps.values()]))
                    out.append({'pair': f'{left}_VS_{right}', 'signal': signal, 'seed': seed, 'task_id': task,
                                'episodes': len(eps), 'anchors': len(rr), 'source_groups': len({x['source_group'] for x in rr}),
                                'aggregation': 'anchor_fraction_then_episode_equal_then_task_report',
                                'opposite_sign_fraction_all_coordinates': task_results[task]['opposite_all'],
                                'opposite_sign_fraction_nonzero_pairs': task_results[task]['opposite_nonzero'],
                                'zero_in_either_fraction': task_results[task]['zero_either'],
                                'both_zero_fraction': task_results[task]['both_zero'],
                                'both_nonzero_fraction': task_results[task]['both_nonzero']})
                out.append({'pair': f'{left}_VS_{right}', 'signal': signal, 'seed': seed, 'task_id': 'five_task_macro',
                            'episodes': 47, 'anchors': 185, 'source_groups': 47,
                            'aggregation': 'five_task_equal_episode_equal',
                            'opposite_sign_fraction_all_coordinates': float(np.mean([task_results[t]['opposite_all'] for t in TASKS])),
                            'opposite_sign_fraction_nonzero_pairs': float(np.mean([task_results[t]['opposite_nonzero'] for t in TASKS])),
                            'zero_in_either_fraction': float(np.mean([task_results[t]['zero_either'] for t in TASKS])),
                            'both_zero_fraction': float(np.mean([task_results[t]['both_zero'] for t in TASKS])),
                            'both_nonzero_fraction': float(np.mean([task_results[t]['both_nonzero'] for t in TASKS]))})
    return out


def make_figures(alpha_curves, primary_rows, pair_rows, trajectory_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    alpha_lock = read_json(ROOT / 'protocol/ALPHA_LOCK.json')
    alpha_data = [{'family':r['family'],'alpha':r['alpha'],'J':r['J'],'position_mae_macro':r['position_mae_macro'],
                   'safe_fd_l1_macro':r['safe_fd_l1_macro'],'selected_alpha':float(alpha_lock['selected_alphas'][r['family']]['alpha'])}
                  for r in alpha_curves]
    write_csv(ROOT / 'figure_data/ALPHA_SELECTION_CURVES.csv', alpha_data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    colors = {'SV': '#0072B2', 'SS': '#D55E00'}
    for ax, family in zip(axes, ['SV', 'SS']):
        rr = [r for r in alpha_curves if r['family'] == family]
        rr.sort(key=lambda r: float(r['alpha']))
        x = [float(r['alpha']) for r in rr]
        y = [float(r['J']) for r in rr]
        selected = float(alpha_lock['selected_alphas'][family]['alpha'])
        ax.plot(x, y, marker='o', color=colors[family], linewidth=1.7)
        ax.axvline(selected, color='#222222', linestyle='--', linewidth=1.2, label=f'locked alpha={selected:.2f}')
        ax.scatter([selected], [alpha_lock['selected_alphas'][family]['selected_J']], color='#111111', s=55, zorder=3)
        ax.set_title(f'{family} family: inner-tune J')
        ax.set_xlabel('Global coefficient alpha')
        ax.set_ylabel('Position MAE + 0.1 × Safe-FD L1')
        ax.grid(alpha=.22)
        ax.legend(frameon=False)
    fig.savefig(ROOT / 'figures/ALPHA_SELECTION_CURVES.png', dpi=180, facecolor='white')
    plt.close(fig)

    methods = ['S', 'A', 'V', 'BIAS_STATE', 'BIAS_VIDEO', 'PERSISTENCE', 'SV50', 'SS50', 'SV_CAL', 'SS_CAL']
    palette = dict(zip(methods, ['#0072B2','#009E73','#D55E00','#56B4E9','#CC79A7','#777777','#000000','#E69F00','#6A3D9A','#17BECF']))
    macro_records=read_csv(ROOT / 'results/FIVE_TASK_MACRO.csv')
    macro_rows=[]
    for method in methods:
        seed_values=['shared'] if method=='PERSISTENCE' else [str(s) for s in SEEDS]
        for seed in seed_values:
            values={r['metric']:float(r['five_task_equal_episode_equal_macro'])
                    for r in macro_records if r['method']==method and str(r['seed'])==seed}
            if set(values)!=set(METRICS):
                raise RuntimeError(f'figure macro rows incomplete for {method}/{seed}')
            macro_rows.append({'method':method,'seed':seed,**values})
    write_csv(ROOT / 'figure_data/FIVE_TASK_SEED_DUAL_METRICS.csv',macro_rows)
    dual_data=[]
    for method in methods:
        for r in primary_rows:
            dual_data.append({'method':method,'task_id':int(r['task_id']),'seed':int(r['seed']),
                              'position_mae':float(r[f'{method}_position_mae']),
                              'safe_fd_l1':float(r[f'{method}_safe_fd_l1']),
                              'seam_safe_fd_l1':float(r[f'{method}_seam_safe_fd_l1'])})
    write_csv(ROOT / 'figure_data/DUAL_METRIC_AND_SEAM.csv', dual_data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), layout='constrained')
    seed_markers={str(SEEDS[0]):'o',str(SEEDS[1]):'s',str(SEEDS[2]):'^','shared':'D'}
    for method in methods:
        for rec in [r for r in macro_rows if r['method']==method]:
            seed=str(rec['seed'])
            axes[0].scatter(rec['position_mae'],rec['safe_fd_l1'],s=64 if seed=='shared' else 46,
                            color=palette[method],marker=seed_markers[seed],edgecolor='black' if seed=='shared' else 'none',
                            linewidth=.5,zorder=3)
            y=methods.index(method)+({'202610131':-.16,'202610132':0.0,'202610133':.16,'shared':0.0}[seed])
            axes[1].scatter(rec['seam_safe_fd_l1'],y,s=64 if seed=='shared' else 46,
                            color=palette[method],marker=seed_markers[seed],edgecolor='black' if seed=='shared' else 'none',
                            linewidth=.5,zorder=3)
    axes[0].set_xlabel('Position MAE (outer-train scale)')
    axes[0].set_ylabel('Safe-FD L1 (outer-train scale)')
    axes[0].set_title('Five-task macro by fixed seed')
    axes[0].grid(alpha=.22)
    axes[1].set_xlabel('Seam Safe-FD L1 (outer-train scale)')
    axes[1].set_title('Anchor-to-first-step seam by method and seed')
    axes[1].set_yticks(range(len(methods)))
    axes[1].set_yticklabels(methods)
    axes[1].invert_yaxis()
    axes[1].grid(axis='x',alpha=.22)
    from matplotlib.lines import Line2D
    method_handles=[Line2D([0],[0],marker='o',linestyle='none',color=palette[m],label=m,markersize=7) for m in methods]
    seed_handles=[Line2D([0],[0],marker=seed_markers[str(s)],linestyle='none',color='#333333',label=f'seed {s}',markersize=7) for s in SEEDS]
    seed_handles.append(Line2D([0],[0],marker='D',linestyle='none',color='#777777',markeredgecolor='black',label='Persistence shared',markersize=7))
    fig.legend(handles=method_handles+seed_handles,loc='outside lower center',ncol=7,frameon=False,fontsize=8)
    fig.suptitle('Position, internal FD, and seam; points show fixed-seed task-equal macros')
    fig.savefig(ROOT / 'figures/DUAL_METRIC_AND_SEAM.png', dpi=180, facecolor='white')
    plt.close(fig)

    main_contrasts = ['VIDEO_SUBSTITUTION_50','VIDEO_SUBSTITUTION_CAL','SV_CALIBRATION_GAIN','SS_CALIBRATION_GAIN',
                      'SV_CAL_VS_S','SV_CAL_VS_V','SV_CAL_VS_BIAS_STATE','SV_CAL_VS_BIAS_VIDEO']
    pair_lookup = {(r['contrast'], int(r['seed']), int(r['task_id'])): r for r in pair_rows}
    plot_data=[]
    for contrast in main_contrasts:
        for task in TASKS:
            for seed in SEEDS:
                r=pair_lookup[(contrast,seed,task)]
                plot_data.append({'contrast':contrast,'task_id':task,'seed':seed,
                                  **{m:float(r[m]) for m in METRICS}})
    write_csv(ROOT / 'figure_data/PRIMARY_CONTRASTS_BY_TASK.csv', plot_data)
    fig, axes = plt.subplots(1, 3, figsize=(16, 11), sharey=True, layout='constrained')
    markers = {SEEDS[0]:'o', SEEDS[1]:'s', SEEDS[2]:'^'}
    task_colors = {119:'#0072B2',94:'#D55E00',90:'#009E73',141:'#CC79A7',230:'#E69F00'}
    for ax, metric, title in zip(axes, METRICS, ['Position gain', 'Safe-FD gain', 'Seam gain']):
        ylabels=[];yticks=[];y=0
        for contrast in main_contrasts:
            for task in TASKS:
                ylabels.append(f'{contrast} | task {task}')
                yticks.append(y)
                for seed in SEEDS:
                    r=pair_lookup[(contrast,seed,task)]
                    ax.scatter(float(r[metric]), y+(seed-SEEDS[0])/(SEEDS[-1]-SEEDS[0]+1)*.2-.1,
                               color=task_colors[task], marker=markers[seed], s=28, alpha=.82)
                y+=1
        ax.axvline(0,color='#222222',linewidth=.9)
        ax.set_title(title)
        ax.set_xlabel('Gain = L(reference) − L(candidate); positive favors candidate')
        ax.grid(axis='x',alpha=.2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels,fontsize=6)
    from matplotlib.lines import Line2D
    seed_handles=[Line2D([0],[0],marker=markers[s],linestyle='none',color='#333333',label=f'seed {s}',markersize=7) for s in SEEDS]
    fig.legend(handles=seed_handles,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    fig.savefig(ROOT / 'figures/PRIMARY_CONTRASTS_BY_TASK.png', dpi=180, facecolor='white')
    plt.close(fig)

    # The trajectory table is metadata-selected from the exact V5 cases and has 0/14 coordinates by seed.
    methods_t = ['GT','S','A','V','SV50','SS50','SV_CAL','SS_CAL','PERSISTENCE']
    styles = {'GT':('#111111','-'),'S':('#0072B2','--'),'A':('#009E73','--'),'V':('#D55E00','-.'),
              'SV50':('#000000',':'),'SS50':('#E69F00','-'),'SV_CAL':('#6A3D9A','--'),
              'SS_CAL':('#17BECF','-.'),'PERSISTENCE':('#888888',':')}
    fig, axs = plt.subplots(10, 3, figsize=(16, 24), sharex=True, layout='constrained')
    for task_i, task in enumerate(TASKS):
        case = next(x for x in trajectory_rows if int(x['task_id']) == task)
        aid = case['anchor_id']
        for axis_i, dim in enumerate([0,14]):
            ridx=task_i*2+axis_i
            for seed_i, seed in enumerate(SEEDS):
                ax=axs[ridx,seed_i]
                rows=[x for x in trajectory_rows if 'seed' in x and int(x['task_id'])==task and int(x['seed'])==seed and int(x['action_index'])==dim]
                for method in methods_t:
                    rr=sorted([x for x in rows if x['method']==method],key=lambda x:int(x['future_step']))
                    if len(rr)!=30: continue
                    color,ls=styles[method]
                    ax.plot([int(x['future_step']) for x in rr],[float(x['native_command_value']) for x in rr],
                            color=color,linestyle=ls,linewidth=1.1,label=method)
                if seed_i==0: ax.set_ylabel(f'task {task}\naction dim {dim}')
                if ridx==0: ax.set_title(f'seed {seed}')
                ax.grid(alpha=.16)
    axs[-1,1].set_xlabel('Future step (1–30); numeric native command (physical units unresolved)')
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,fontsize=8)
    fig.suptitle('V5 metadata-selected receiver trajectories; targets are shown only after prediction seal')
    fig.savefig(ROOT / 'figures/TRAJECTORY_CASES.png', dpi=150, facecolor='white')
    plt.close(fig)

    captions = [
        '# V6 figure captions', '',
        '- `ALPHA_SELECTION_CURVES.png`: the two fixed 21-point inner-tune J curves; dashed lines show the locked global coefficients.',
        '- `DUAL_METRIC_AND_SEAM.png`: fixed-seed five-task macro Position MAE/Safe-FD scatter and seam points; color identifies method, marker identifies seed, Persistence is one shared diamond.',
        '- `PRIMARY_CONTRASTS_BY_TASK.png`: eight preregistered paired gains by task and fixed seed; task IDs are in row labels, marker legend identifies seed, and positive favors the candidate.',
        '- `TRAJECTORY_CASES.png`: V5 metadata-selected cases for tasks 119/94/90/141/230, dimensions 0/14, columns separated by seed; values are numeric native commands, physical units unresolved.',
        '', 'All outcome-derived plots use the single post-seal dev scoring pass. CIs are conditional source-group bootstrap intervals and are not plotted as seed-population uncertainty.',
    ]
    (ROOT / 'figures/CAPTIONS.zh-CN.md').write_text('\n'.join(captions) + '\n', encoding='utf-8')


def run_synthetic():
    result = synthetic_scorer_qa()
    write_json(ROOT / 'qa/SYNTHETIC_SCORER_QA.json', result)
    print(json.dumps(result, sort_keys=True))


def read_primary_values(rows, method, metric, task, seed):
    row = next(r for r in rows if int(r['task_id']) == task and int(r['seed']) == seed)
    return float(row[f'{method}_{metric}'])


def make_target_access_receipt(seal, schema, archive_sha, archive_bytes, selected_rows, scored_rows):
    v5 = read_json(V5 / 'inventory/OUTCOME_EXPOSURE.json')
    correction_path = ROOT / 'qa/DEV_SCORE_ACCESS_CORRECTIONS.json'
    failed_reads = int(read_json(correction_path).get('failed_dev_archive_content_reads_before_success', 0)) if correction_path.exists() else 0
    archive_loads = 1 + failed_reads
    exposure = {
        'schema_version': 1, 'status': 'complete', 'study': 'post_V5_controlled_late_fusion_development',
        'prior_v5_target_archive_load_operations_total': int(v5['target_archive_load_operations_total']),
        'prior_v5_target_window_rows_loaded_total': int(v5['target_array_window_rows_loaded_total']),
        'prior_v5_unique_dev_receiver_windows': int(v5['unique_dev_receiver_windows']),
        'prior_v5_primary_window_loss_evaluations': int(v5['new_dev_future_window_logical_reads']),
        'prior_v5_l2_diagnostic_window_evaluations': int(v5['l2_diagnostic_window_evaluations']),
        'reported_prior_counter': v5.get('reported_prior_counter'),
        'reported_prior_counter_unit_status': v5.get('reported_prior_counter_unit_status'),
        'prior_counter_added_to_v6_totals': False,
        'inner_tune': {
            'train_tune_archive_sha_passes': 2, 'npz_load_operations': 2,
            'target_member_full_rows_materialized_per_load': 1280,
            'inner_tune_rows_used_for_alpha': 270,
            'failed_alignment_attempt_rows_used_in_risk_scoring': 0,
            'candidate_family_risk_rows': 42, 'logical_inner_tune_anchor_loss_evaluations': 11340,
            'all_inner_fit_target_scores_used_by_alpha': 0,
            'all_outer_train_formal_targets_used_by_alpha': 0,
        },
        'v6_dev': {
            'prediction_seal_sha256': sha_file(ROOT / 'qa/PREDICTION_SEAL.json'),
            'target_archive_metadata_preflight_opens': 1,
            'target_archive_content_reads': archive_loads,
            'np_load_operations_over_in_memory_blob': archive_loads,
            'failed_postseal_dev_archive_content_reads_before_success': failed_reads,
            'target_archive_sha256': archive_sha, 'target_archive_bytes': archive_bytes,
            'unique_dev_receiver_windows': 185, 'target_array_windows_loaded': 185 * archive_loads,
            'new_logical_prediction_bundles': 9, 'new_unique_prediction_tensor_payloads': 6,
            'shared_payload_aliases': 3, 'reused_v5_sv50_bundles': 3,
            'new_dev_window_loss_evaluations': 6 * 185,
            'new_dev_task_seed_views': 6 * 5,
            'shared_v5_sv50_or_sv_cal_loss_rows_reused': 2 * 3 * 185,
            'figure_target_archive_reads_after_score': 0,
            'scored_new_payload_rows': scored_rows, 'target_rows_loaded': selected_rows * archive_loads,
        'unique_target_rows_scored': selected_rows,
        },
        'v5_plus_v6_dev_target_archive_load_operations': int(v5['target_archive_load_operations_total']) + archive_loads,
        'v5_plus_v6_target_array_window_rows_loaded_total': int(v5['target_array_window_rows_loaded_total']) + 185 * archive_loads,
        'v5_plus_v6_unique_dev_receiver_windows': 185,
        'v5_plus_v6_new_primary_window_loss_evaluations_known_grain': int(v5['new_dev_future_window_logical_reads']) + 6 * 185,
        'v5_and_v6_l2_or_auxiliary_evaluations_kept_separate': int(v5['l2_diagnostic_window_evaluations']),
        'final_heldout_payload_access': 0, 'created_utc': utc(),
    }
    write_json(ROOT / 'inventory/OUTCOME_EXPOSURE.json', exposure)
    return exposure


def run_real_preflight():
    if (ROOT / 'qa/PREFLIGHT.json').exists():
        raise RuntimeError('real preflight receipt already exists; do not rewrite it')
    seal = validate_prediction_seal()
    synthetic_path = ROOT / 'qa/SYNTHETIC_SCORER_QA.json'
    synthetic = read_json(synthetic_path)
    if synthetic.get('status') != 'PASS' or synthetic.get('target_archive_opened') is not False:
        raise RuntimeError('synthetic scorer QA must pass before real archive preflight')
    schema = dev_target_schema_preflight(seal)
    ids, rows, metadata, anchors = ordered_receiver_metadata()
    for seed in SEEDS:
        for method in ['SV50','SS50','SV_CAL','SS_CAL']:
            arr, _ = prediction_for_program(method, seed, seal['entries'], ids, rows)
            if arr.shape != (185,30,28) or not np.isfinite(arr).all():
                raise RuntimeError(f'real target-free preflight prediction failed {method}/{seed}')
    for method in ['S','A','V','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','SV50']:
        for seed in SEEDS:
            saved = read_csv(V5 / ('results/PER_WINDOW_DERIVED_LOSSES.csv' if method=='SV50' else 'results/PER_WINDOW_ACCEPTED_REFERENCES.csv'))
            source_method = 'MIX50' if method=='SV50' else method
            rr = [r for r in saved if r['method']==source_method and int(r['seed'])==seed]
            if len(rr)!=185 or {r['anchor_id'] for r in rr} != set(ids):
                raise RuntimeError(f'accepted V5 scored reference table incomplete for {method}/{seed}')
    metric = read_json(V2 / 'protocol/METRIC_SCALE_LOCK.json')
    if metric.get('status')!='LOCKED' or int(metric.get('final_heldout_payload_access',-1))!=0:
        raise RuntimeError('accepted outer-train metric scale lock invalid')
    preflight={'schema_version':1,'status':'PASS','created_utc':utc(),
        'prediction_seal_sha256':sha_file(ROOT/'qa/PREDICTION_SEAL.json'),
        'alpha_lock_sha256':sha_file(ROOT/'protocol/ALPHA_LOCK.json'),
        'synthetic_scorer_qa':'PASS','target_archive_schema_preflight':'PASS',
        'target_archive_values_read':False,'dev_target_values_read':False,
        'accepted_program_seed_payloads_verified':12,'receiver_rows':len(ids),
        'receiver_ids_unique':len(set(ids))==len(ids),'source_reference_score_tables_verified':7*3,
        'formal_metric_scale_lock_sha256':sha_file(V2/'protocol/METRIC_SCALE_LOCK.json'),
        'final_heldout_payload_access':0}
    write_json(ROOT/'qa/PREFLIGHT.json',preflight)
    owner_path=ROOT/'coordination/OWNER_STATE.json';owner=read_json(owner_path)
    owner.update({'phase':'real_target_schema_preflight_complete','status':'running','dev_target_values_read':False,'updated_utc':utc()})
    write_json(owner_path,owner)
    print(json.dumps(preflight,sort_keys=True))

def score_dev():
    if (ROOT / 'qa/FINAL_ACCEPTANCE.json').exists():
        raise RuntimeError('final acceptance already exists; dev scorer is write-once')
    seal = validate_prediction_seal()
    preflight=read_json(ROOT/'qa/PREFLIGHT.json')
    if preflight.get('status')!='PASS' or preflight.get('prediction_seal_sha256')!=sha_file(ROOT/'qa/PREDICTION_SEAL.json'):
        raise RuntimeError('separate real-target preflight absent or not bound to accepted seal')
    synthetic=read_json(ROOT/'qa/SYNTHETIC_SCORER_QA.json')
    if synthetic.get('status')!='PASS':
        raise RuntimeError('synthetic scorer QA did not pass')
    schema=read_json(ROOT/'qa/DEV_TARGET_SCHEMA_PREFLIGHT.json')
    if schema.get('status')!='PASS':
        raise RuntimeError('target-free archive schema preflight is missing')
    ids, rows, metadata, anchors = ordered_receiver_metadata()
    v5_exposure = read_json(V5 / 'inventory/OUTCOME_EXPOSURE.json')
    scale_path = V2 / 'protocol/METRIC_SCALE_LOCK.json'
    scale_lock = read_json(scale_path)
    if scale_lock.get('status') != 'LOCKED' or int(scale_lock.get('final_heldout_payload_access', -1)) != 0:
        raise RuntimeError('accepted outer-train primary-dev metric scale lock is invalid')
    sigma = np.asarray(scale_lock['target_sigma'], dtype=np.float64)
    fdscale = np.asarray(scale_lock['safe_fd_scale'], dtype=np.float64)
    if sigma.shape != (28,) or fdscale.shape != (28,) or np.any(sigma <= 0) or np.any(fdscale <= 0):
        raise RuntimeError('formal dev metric scales invalid')

    # Validate all prediction payloads and saved source score tables before the target reader starts.
    source_map = read_json(ROOT / 'inventory/SOURCE_MAP.json')
    program_lock = read_json(ROOT / 'protocol/PROGRAM_LOCK.json')
    alpha_lock = read_json(ROOT / 'protocol/ALPHA_LOCK.json')
    seal_entries = seal['entries']
    predictions = {}
    source_records = {}
    for seed in SEEDS:
        for method, role in ROLE_MAP.items():
            predictions[(method, seed)], source_records[(method, seed)] = source_prediction(role, seed, source_map, ids, rows)
        for method in ['SV50', 'SS50', 'SV_CAL', 'SS_CAL']:
            predictions[(method, seed)], source_records[(method, seed)] = prediction_for_program(method, seed, seal_entries, ids, rows)
        predictions[('PERSISTENCE', seed)] = np.repeat(anchors[:, None, :], 30, axis=1)
    losses = {}
    for seed in SEEDS:
        source_losses = loss_rows_for_sources(ids, metadata, seed)
        for method, rr in source_losses.items():
            losses[(method, seed)] = rr
    v5_primary = read_csv(V5 / 'results/PRIMARY_TASK_SEED.csv')
    for method in ['S','A','V','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','SV50']:
        v5_name='MIX50' if method=='SV50' else method
        for seed in SEEDS:
            for task in TASKS:
                saved=next(r for r in v5_primary if int(r['seed'])==seed and int(r['task_id'])==task)
                for metric in METRICS:
                    observed=task_aggregate(losses[(method,seed)],metric)[task]['episode_macro']
                    if abs(observed-float(saved[f'{v5_name}_{metric}']))>1e-12:
                        raise RuntimeError(f'accepted source score mismatch before target reader: {method}/{seed}/{task}/{metric}')

    target_started = utc()
    try:
        target, target_blob, target_sha = load_dev_outcomes_once(ids, metadata,
            v5_exposure['target_archive_sha256'], int(v5_exposure['target_archive_bytes']))
        target_bytes = len(target_blob)
        del target_blob
        # Seal-bound window losses are the recovery boundary after this target read.
        loss_path = ROOT / 'results/PER_WINDOW_NEW_DEV_LOSSES.csv'
        loss_receipt_path = ROOT / 'qa/SAVED_NEW_LOSS_TABLE_RECEIPT.json'
        seal_sha = sha_file(ROOT / 'qa/PREDICTION_SEAL.json')
        alpha_sha = sha_file(ROOT / 'protocol/ALPHA_LOCK.json')
        expected_methods = ['SS50', 'SS_CAL']
        new_loss_rows = []
        if loss_path.exists():
            if not loss_receipt_path.is_file():
                raise RuntimeError('preserved V6 new-loss table lacks its seal-bound receipt; refusing recomputation')
            receipt = read_json(loss_receipt_path)
            if receipt.get('status') != 'PASS' or receipt.get('prediction_seal_sha256') != seal_sha or receipt.get('alpha_lock_sha256') != alpha_sha:
                raise RuntimeError('preserved V6 loss table is not bound to this seal and alpha lock')
            if sha_file(loss_path) != receipt.get('sha256') or int(receipt.get('rows', -1)) != 1110:
                raise RuntimeError('preserved V6 loss table identity/count mismatch')
            saved = read_csv(loss_path)
            validate_saved_loss_rows(saved, seal_sha=seal_sha, expected_methods=expected_methods,
                                     expected_seeds=SEEDS, expected_ids=ids)
            for r in saved:
                method, seed = r['method'], int(r['seed'])
                losses.setdefault((method, seed), []).append({k:(float(v) if k in METRICS else v)
                    for k,v in r.items() if k != 'prediction_seal_sha256'})
            new_loss_rows = saved
            new_loss_evaluations_this_attempt = 0
            loss_recovery_status = 'REUSED_SEAL_BOUND_WINDOW_LOSSES'
        else:
            for seed in SEEDS:
                for method in expected_methods:
                    score = score_native(predictions[(method, seed)], target, anchors, sigma, fdscale)
                    losses[(method, seed)] = window_rows_from_scores(method, seed, ids, metadata, score, 'V6_postseal_target_score')
            new_loss_evaluations_this_attempt = sum(len(losses[(m, s)]) for m in expected_methods for s in SEEDS)
            if new_loss_evaluations_this_attempt != 1110:
                raise RuntimeError('new dev window-loss count is not six unique payloads × 185')
            for method in expected_methods:
                for seed in SEEDS:
                    for r in losses[(method, seed)]:
                        new_loss_rows.append({**r, 'prediction_seal_sha256': seal_sha})
            write_csv(loss_path, new_loss_rows)
            write_json(loss_receipt_path, {'schema_version':1,'status':'PASS','path':str(loss_path),
                'sha256':sha_file(loss_path),'rows':len(new_loss_rows),'methods':expected_methods,'seeds':SEEDS,
                'ids_per_method_seed':185,'prediction_seal_sha256':seal_sha,'alpha_lock_sha256':alpha_sha,
                'source':'postseal V6 target score','created_utc':utc()})
            loss_recovery_status = 'NEWLY_SCORED_AND_SEAL_BOUND'
        write_json(ROOT / 'qa/LOSS_TABLE_RECOVERY_QA.json', {'schema_version':1,'status':'PASS',
            'recovery_status':loss_recovery_status,'new_loss_evaluations_this_attempt':new_loss_evaluations_this_attempt,
            'preserved_loss_rows':len(new_loss_rows),'prediction_seal_sha256':seal_sha,'alpha_lock_sha256':alpha_sha,
            'created_utc':utc()})

        # Check saved references line up with the accepted V5 wide table before using them.
        source_reconciliation = []
        for method in ['S', 'A', 'V', 'BIAS_STATE', 'BIAS_VIDEO', 'PERSISTENCE', 'SV50']:
            v5_name = 'MIX50' if method == 'SV50' else method
            for seed in SEEDS:
                for task in TASKS:
                    saved = next(r for r in v5_primary if int(r['seed']) == seed and int(r['task_id']) == task)
                    for metric in METRICS:
                        calculated = task_aggregate(losses[(method, seed)], metric)[task]['episode_macro']
                        expected = float(saved[f'{v5_name}_{metric}'])
                        delta = abs(calculated - expected)
                        source_reconciliation.append({'method': method, 'seed': seed, 'task_id': task,
                                                      'metric': metric, 'v6_reused_value': calculated,
                                                      'v5_accepted_value': expected, 'abs_delta': delta,
                                                      'pass': delta <= 1e-12})
        if not all(r['pass'] for r in source_reconciliation):
            raise RuntimeError('V5 reused per-window scores do not reconcile to the accepted primary table')
        write_csv(ROOT / 'qa/SOURCE_SCORE_RECONCILIATION.csv', source_reconciliation)

        primary_rows, task_aggregate_cache = aggregate_primary(losses)
        write_csv(ROOT / 'results/PRIMARY_TASK_SEED.csv', primary_rows)
        macro_rows = []
        for method in METHODS:
            for metric in METRICS:
                if method == 'PERSISTENCE':
                    macro = float(np.mean([task_aggregate_cache[(method, SEEDS[0], metric)][t]['episode_macro'] for t in TASKS]))
                    macro_rows.append({'method': method, 'metric': metric, 'seed': 'shared',
                                       'five_task_equal_episode_equal_macro': macro, 'sample_sd_across_seeds': 'NA',
                                       'n_seeds': 1, 'aggregation': 'shared_reference'})
                    macro_rows.append({'method': method, 'metric': metric, 'seed': 'mean_across_formal_seeds',
                                       'five_task_equal_episode_equal_macro': macro, 'sample_sd_across_seeds': 'NA',
                                       'n_seeds': 1, 'aggregation': 'shared_reference'})
                    continue
                per_seed = []
                for seed in SEEDS:
                    vals = [task_aggregate_cache[(method, seed, metric)][t]['episode_macro'] for t in TASKS]
                    macro = float(np.mean(vals))
                    per_seed.append(macro)
                    macro_rows.append({'method': method, 'metric': metric, 'seed': seed,
                                       'five_task_equal_episode_equal_macro': macro, 'n_seeds': 3,
                                       'aggregation': 'five_tasks_equal_within_task_episode_equal'})
                macro_rows.append({'method': method, 'metric': metric, 'seed': 'mean_across_formal_seeds',
                                   'five_task_equal_episode_equal_macro': float(np.mean(per_seed)),
                                   'sample_sd_across_seeds': float(np.std(per_seed, ddof=1)), 'n_seeds': 3,
                                   'aggregation': 'mean_and_sample_sd_over_three_fixed_seeds'})
        write_csv(ROOT / 'results/FIVE_TASK_MACRO.csv', macro_rows)

        # Unified per-window table is an auditable copy of accepted scores plus V6's newly scored rows.
        all_rows = [r for method in METHODS for seed in SEEDS for r in losses[(method, seed)]]
        write_csv(ROOT / 'results/PER_WINDOW_ALL_METHODS.csv', all_rows)
        write_csv(ROOT / 'results/PER_WINDOW_ACCEPTED_REFERENCE_REUSE.csv',
                  [r for m in ['S','A','V','BIAS_STATE','BIAS_VIDEO','PERSISTENCE','SV50','SV_CAL'] for s in SEEDS for r in losses[(m,s)]])

        draws = source_group_draws()
        paired_rows, ci_rows = compute_paired_and_ci(losses, draws)
        write_csv(ROOT / 'results/PAIRED_EFFECTS.csv', paired_rows)
        write_csv(ROOT / 'results/PAIRED_GROUP_CI.csv', ci_rows)
        seed_loo, task_loo = compute_seed_task_loo(primary_rows, paired_rows)
        write_csv(ROOT / 'results/SEED_LOO.csv', seed_loo)
        write_csv(ROOT / 'results/TASK_LOO.csv', task_loo)

        convexity = convexity_table(losses, alpha_lock)
        write_csv(ROOT / 'results/CONVEXITY_CHECK.csv', convexity)
        mix_diag = mix_diagnostic(losses)
        write_csv(ROOT / 'results/MIXTURE_CONVEXITY_DIAGNOSTIC.csv', mix_diag)

        # Residual complementarity and metadata-selected 0/14 trajectories use already loaded arrays.
        complementarity = residual_complementarity(predictions, target, sigma, fdscale, ids, metadata)
        write_csv(ROOT / 'results/RESIDUAL_COMPLEMENTARITY.csv', complementarity)
        v5_cases = read_csv(V5 / 'figure_data/TRAJECTORY_CASES.csv')
        cases = {int(r['task_id']): r for r in v5_cases}
        trajectory_case_rows, trajectory_rows = [], []
        for task in TASKS:
            case = cases[task]
            aid = case['anchor_id']
            if aid not in metadata or int(metadata[aid]['task_id']) != task:
                raise RuntimeError(f'accepted V5 trajectory case ID is not in locked V6 receiver set: {aid}')
            trajectory_case_rows.append({**case, 'reused_v5_case_id': True})
            i = ids.index(aid)
            for seed in SEEDS:
                track = {'GT': target, 'S': predictions[('S', seed)], 'A': predictions[('A', seed)],
                         'V': predictions[('V', seed)], 'SV50': predictions[('SV50', seed)],
                         'SS50': predictions[('SS50', seed)], 'SV_CAL': predictions[('SV_CAL', seed)],
                         'SS_CAL': predictions[('SS_CAL', seed)], 'PERSISTENCE': predictions[('PERSISTENCE', seed)]}
                for dim in [0, 14]:
                    for method, matrix in track.items():
                        for step in range(30):
                            trajectory_rows.append({'task_id': task, 'episode_id': int(metadata[aid]['episode_id']),
                                'anchor_id': aid, 'source_group': metadata[aid]['source_group'], 'seed': seed,
                                'action_index': dim, 'future_step': step + 1, 'method': method,
                                'native_command_value': float(matrix[i, step, dim]),
                                'unit_note': 'numeric native command; physical units unresolved'})
        write_csv(ROOT / 'figure_data/TRAJECTORY_CASES.csv', trajectory_case_rows)
        write_csv(ROOT / 'figure_data/TRAJECTORY_DATA.csv', trajectory_rows)

        # V5 baseline paired contrasts are included in V6 plots and the direct handoff.
        v5_mix_ci = read_csv(ROOT / 'results/V5_MIX50_PAIRED_GROUP_CI.csv')
        if len(v5_mix_ci) != 228:
            raise RuntimeError('V5 MIX50 conditional interval table is incomplete')
        v5mix = read_json(ROOT / 'results/V5_MIX50_RECONCILIATION.json')
        if v5mix.get('status') != 'PASS':
            raise RuntimeError('V5 MIX50 full baseline reconciliation is not accepted')

        alpha_curves = read_csv(ROOT / 'results/INNER_TUNE_ALPHA_CURVES.csv')
        make_figures(alpha_curves, primary_rows, paired_rows, trajectory_case_rows + trajectory_rows)

        # This receipt records target exposure and per-window evals only after all scoring completes.
        exposure = make_target_access_receipt(seal, schema, target_sha, target_bytes, 185, 1110)
        write_json(ROOT / 'qa/DEV_TARGET_ACCESS.json', {
            'schema_version': 1, 'status': 'PASS', 'seal_sha256': seal_sha,
            'target_archive_path': str(V2 / 'data/DEV_TARGETS_SCORER_ONLY.npz'),
            'target_archive_sha256': target_sha, 'target_archive_bytes': target_bytes,
            'target_archive_content_read_started_utc': target_started,
            'target_rows_loaded': exposure['v6_dev']['target_array_windows_loaded'], 'unique_receiver_windows': 185,
            'per_window_new_loss_evaluations': 1110, 'target_reader_invocations': archive_loads,
            'failed_pre_score_target_reads': failed_reads,
            'metadata_only_preflight_opens': schema['archive_member_metadata_open_count'],
            'final_heldout_payload_access': 0,
        })
        make_report_json(primary_rows, paired_rows, ci_rows, alpha_lock, exposure, v5mix,
                         seal, target_sha, new_loss_evaluations_this_attempt, len(convexity))
        write_report_md(primary_rows, paired_rows, ci_rows, alpha_lock, exposure, v5mix, seal, target_sha)
        write_final_acceptance(seal, primary_rows, paired_rows, ci_rows, convexity, synthetic, exposure)
        update_owner_after_score(exposure)
        print(json.dumps({'status': 'SCORED', 'alpha_sv': alpha_lock['selected_alphas']['SV']['alpha'],
                          'alpha_ss': alpha_lock['selected_alphas']['SS']['alpha'],
                          'new_target_loss_rows': len(new_loss_rows), 'logical_bundles': 12,
                          'final_heldout_payload_access': 0}, sort_keys=True))
    except Exception as exc:
        attempt = {
            'attempt': 'dev_score_001', 'phase': 'postseal_dev_scoring', 'status': 'FAILED_RECORDED',
            'error_type': type(exc).__name__, 'error': str(exc), 'prediction_seal_sha256': sha_file(ROOT / 'qa/PREDICTION_SEAL.json'),
            'target_reader_started_utc': target_started, 'target_archive_sha256_if_read': locals().get('target_sha'),
            'target_archive_content_read': 'target_sha' in locals(),
            'target_array_rows_loaded': 185 if 'target_sha' in locals() else 0,
            'new_dev_loss_evaluations_completed': 0, 'final_heldout_payload_access': 0,
            'created_utc': utc(),
        }
        p = ROOT / 'qa/FAILED_ATTEMPTS.jsonl'
        with p.open('a', encoding='utf-8') as f:
            f.write(json.dumps(attempt, sort_keys=True) + '\n')
        raise


def write_final_acceptance(seal, primary_rows, paired_rows, ci_rows, convexity, synthetic, exposure):
    result = {
        'schema_version': 1, 'status': 'complete', 'study': 'post_V5_controlled_late_fusion_development',
        'prediction_seal_status': seal['status'], 'logical_prediction_bundles_accepted': 12,
        'new_logical_prediction_bundles': 9, 'new_unique_prediction_tensor_payloads': 6,
        'shared_payload_aliases': 3, 'reused_v5_sv50_bundles': 3,
        'alpha_families': 2, 'alpha_grid_points_per_family': 21,
        'alpha_selected_using_dev': False,
        'alpha_sv': read_json(ROOT / 'protocol/ALPHA_LOCK.json')['selected_alphas']['SV']['alpha'],
        'alpha_ss': read_json(ROOT / 'protocol/ALPHA_LOCK.json')['selected_alphas']['SS']['alpha'],
        'inner_tune_risk_rows': 42, 'primary_task_seed_rows': len(primary_rows),
        'paired_effect_task_seed_rows': len(paired_rows), 'paired_group_ci_rows': len(ci_rows),
        'convexity_checks': len(convexity), 'convexity_status': 'PASS',
        'synthetic_scorer_qa': synthetic['status'], 'positive_gain_completion_gate': False,
        'minimum_support_completion_gate': False, 'new_inner_fit_neural_fits': 0,
        'new_formal_neural_fits': 0, 'reused_inner_fit_models': 3,
        'inner_tune_gpu_replay_attempts': 4, 'accepted_inner_tune_prediction_bundles': 3,
        'cpu_scalar_calibrations': 2, 'new_dev_target_loss_evaluations': 1110,
        'final_heldout_payload_access': 0, 'old_root_writes': 0,
        'V5_reference_reconciliation': 'PASS', 'dev_prediction_seal_preceded_target_reader': True,
        'outcome_exposure_path': str(ROOT / 'inventory/OUTCOME_EXPOSURE.json'),
        'created_utc': utc(),
    }
    write_json(ROOT / 'qa/FINAL_ACCEPTANCE.json', result)


def update_owner_after_score(exposure):
    p = ROOT / 'coordination/OWNER_STATE.json'
    d = read_json(p)
    d.update({'phase': 'scoring_complete', 'status': 'running',
              'active_gpu_jobs': 0, 'active_jobs': 0, 'all_gpu_leases_released': True,
              'new_dev_loss_evaluations': exposure['v6_dev']['new_dev_window_loss_evaluations'],
              'final_heldout_payload_access': 0, 'updated_utc': utc()})
    write_json(p, d)


def make_report_json(primary_rows, pair_rows, ci_rows, alpha_lock, exposure, v5mix, seal, target_sha, new_rows, convexity_count):
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in METRICS:
            if method == 'PERSISTENCE':
                seed_rows = [r for r in primary_rows if int(r['seed']) == SEEDS[0]]
                value = float(np.mean([float(r[f'{method}_{metric}']) for r in seed_rows]))
                summary[method][metric] = {'mean': value, 'sample_sd': None, 'n_seeds': 1, 'per_seed': [value], 'seed_role': 'shared_reference'}
                continue
            seed_values = []
            for seed in SEEDS:
                seed_rows = [r for r in primary_rows if int(r['seed']) == seed]
                seed_values.append(float(np.mean([float(r[f'{method}_{metric}']) for r in seed_rows])))
            summary[method][metric] = {'mean': float(np.mean(seed_values)), 'sample_sd': float(np.std(seed_values, ddof=1)),
                                       'n_seeds': 3, 'per_seed': seed_values}
    main_gains = {}
    for name in ['VIDEO_SUBSTITUTION_50','VIDEO_SUBSTITUTION_CAL','SV_CALIBRATION_GAIN','SS_CALIBRATION_GAIN',
                 'SV_CAL_VS_S','SV_CAL_VS_V','SV_CAL_VS_BIAS_STATE','SV_CAL_VS_BIAS_VIDEO']:
        rr = [r for r in pair_rows if r['contrast'] == name]
        main_gains[name] = {}
        for metric in METRICS:
            vals = [float(r[metric]) for r in rr]
            main_gains[name][metric] = {'mean_across_task_seed': float(np.mean(vals)),
                                        'sample_sd_across_task_seed': float(np.std(vals, ddof=1))}
    report = {
        'schema_version': 1, 'status': 'complete', 'study': 'post_V5_controlled_late_fusion_development',
        'V5_MIX50': v5mix['task_seed_metrics'], 'V6_alpha': alpha_lock['selected_alphas'],
        'V6_method_macro_mean_and_sample_sd': summary, 'main_paired_gains': main_gains,
        'prediction_bundle_counts': {'logical_total': 12, 'new_logical': 9, 'new_unique_payload': 6,
                                     'shared_alias': 3, 'reused_V5_SV50': 3},
        'paired_ci_rows': len(ci_rows), 'new_v6_loss_rows': new_rows,
        'new_dev_window_loss_evaluations': exposure['v6_dev']['new_dev_window_loss_evaluations'],
        'target_archive_sha256': target_sha, 'dev_unique_receiver_windows': 185,
        'final_heldout_payload_access': 0, 'development_only': True,
        'source_group_ci_scope': 'conditional on fixed bases, selected coefficients, and already exposed dev outcomes',
        'created_utc': utc(),
    }
    write_json(ROOT / 'RETURN_TO_CHATGPT.json', report)


def write_report_md(primary_rows, pair_rows, ci_rows, alpha_lock, exposure, v5mix, seal, target_sha):
    def get_ci(name, metric, seed='mean_across_formal_seeds'):
        rows = read_csv(ROOT / 'results/PAIRED_GROUP_CI.csv')
        return next(r for r in rows if r['contrast'] == name and r['metric'] == metric and str(r['seed']) == str(seed)
                    and r['task_id'] == 'five_task_macro')
    lines = [
        '# V6 有对照的晚期融合与 inner-tune 标量校准', '',
        '**状态：complete；开发实验。** 固定矩阵已评分。正收益和最低样本量不是完成门；final/heldout payload access 为 0。', '',
        '## V5 MIX50 基线补全', '',
        '| 指标 | macro mean | 三个固定 seed 的 sample SD |',
        '|---|---:|---:|',
    ]
    for metric, label in [('position_mae','Position MAE'),('safe_fd_l1','Safe-FD L1'),('seam_safe_fd_l1','Seam Safe-FD L1')]:
        item=v5mix['task_seed_metrics'][metric]
        lines.append(f'| {label} | {item["mean_across_three_fixed_seeds"]:.9f} | {item["sample_sd_across_seeds"]:.9f} |')
    lines += ['', '15 个 task×seed 行与 V5 对 S/V/BIAS_STATE/BIAS_VIDEO 的 paired gains 见 `results/V5_MIX50_TASK_SEED.csv` 与 `results/V5_MIX50_GAIN_TASK_SEED.csv`；228 个 task、seed、macro 条件区间见 `results/V5_MIX50_PAIRED_GROUP_CI.csv`。这些数值直接复用 V5 已封存的窗口分数，没有重新读 dev archive。', '',
              '## Inner-tune 系数', '',
              '| 家族 | alpha | inner-tune Position MAE | inner-tune Safe-FD L1 | inner-tune seam | J |',
              '|---|---:|---:|---:|---:|---:|']
    for family in ['SV','SS']:
        a=alpha_lock['selected_alphas'][family]
        lines.append(f'| {family} | {a["alpha"]:.2f} | {a["position_mae_macro"]:.9f} | {a["safe_fd_l1_macro"]:.9f} | {a["seam_safe_fd_l1_macro"]:.9f} | {a["selected_J"]:.9f} |')
    lines += ['', '系数从 screen seed 202610130 的三个 inner-fit-only checkpoint 预测上，以 46 episodes / 270 anchors、task 等权、episode 等权和 `Position MAE + 0.1×Safe-FD L1` 在固定 21 点选择。该 inner-tune split 已参与 V2 recipe 选择，因此本结果是开发元校准，不是独立确认。`alpha_SV=0.50` 使 `SV_CAL` 与 V5 `SV50/MIX50` payload 共享；三条逻辑程序按原计划报告，但不算成额外独立样本。', '',
              '## 四个程序', '',
              '| 方法 | Position MAE mean ± seed SD | Safe-FD L1 mean ± seed SD | Seam Safe-FD L1 mean ± seed SD |',
              '|---|---:|---:|---:|']
    summary_rows = read_csv(ROOT / 'results/FIVE_TASK_MACRO.csv')
    for method in METHODS:
        vals={r['metric']:r for r in summary_rows if r['method']==method and r['seed']=='mean_across_formal_seeds'}
        cells=[]
        for metric in METRICS:
            mean=float(vals[metric]['five_task_equal_episode_equal_macro'])
            sd=vals[metric].get('sample_sd_across_seeds','NA')
            cells.append(f'{mean:.9f} (shared)' if method=='PERSISTENCE' or sd in ('NA',None,'') else f'{mean:.9f} ± {float(sd):.9f}')
        lines.append(f'| {method} | {cells[0]} | {cells[1]} | {cells[2]} |')
    lines += ['', '### 预先锁定的 paired gains', '', '定义均为 `gain = L(reference) − L(candidate)`；正数有利于 candidate。CI 对固定三个 seed 取均值，使用 within-task source-group paired bootstrap 5000 次。', '',
              '| 对比 | Position gain [95% CI] | Safe-FD gain [95% CI] | Seam gain [95% CI] |',
              '|---|---:|---:|---:|']
    names=['VIDEO_SUBSTITUTION_50','VIDEO_SUBSTITUTION_CAL','SV_CALIBRATION_GAIN','SS_CALIBRATION_GAIN',
           'SV_CAL_VS_S','SV_CAL_VS_V','SV_CAL_VS_BIAS_STATE','SV_CAL_VS_BIAS_VIDEO']
    for name in names:
        cells=[]
        for metric in METRICS:
            ci=get_ci(name,metric)
            cells.append(f'{float(ci["point_estimate"]):+.6f} [{float(ci["ci_low"]):+.6f}, {float(ci["ci_high"]):+.6f}]')
        lines.append(f'| {name} | {cells[0]} | {cells[1]} | {cells[2]} |')
    lines += ['', '全部 15 行 task×seed、seed-LOO、task-LOO、14 组固定 paired effects、条件 CI 和逐窗口 convexity 检查见 `results/`。MAE、Safe-FD 与 seam 并列；Position 改善不代表机器人成功，numeric command 的物理单位仍未确定。', '',
              '## 计算与暴露', '',
              f'- inner-fit models reused: 3；new neural fits: 0；new formal fits: 0；CPU scalar calibrations: 2。',
              f'- inner-tune GPU replay attempts: 4，accepted tune bundles: 3；三模型在同一 screen seed，V_FIT 的第一次内存峰值估计偏低已保留并按 134 MiB 更正后重放。',
              f'- V6 逻辑预测 bundles: 12；新增逻辑程序×seed: 9；唯一新 tensor payloads: 6；3 个 SV_CAL bundle 与 V5 MIX50 共享 payload；V5 SV50 的 3 个 bundle 原样复用。',
              f'- dev target archive SHA-256: `{target_sha}`；V5 之后 V6 新窗口损失计算 1110；target access 发生在 12-bundle 非空 seal 之后。图形未额外读取 target archive。',
              f'- final/heldout payload access: **0**。session/trial 独立性未知，extractor 历史绑定不完整，CI 条件于固定 base 和已暴露 dev。', '',
              '## 产物', '',
              '- `protocol/PROGRAM_LOCK.json`, `protocol/ALPHA_LOCK.json`, `qa/INNER_TUNE_PREDICTION_SEAL.json`, `qa/PREDICTION_SEAL.json`',
              '- `results/V5_MIX50_*`, `results/PRIMARY_TASK_SEED.csv`, `results/PAIRED_EFFECTS.csv`, `results/PAIRED_GROUP_CI.csv`, `results/SEED_LOO.csv`, `results/TASK_LOO.csv`, `results/CONVEXITY_CHECK.csv`',
              '- `figures/` 与 `figure_data/`；人工图审状态为 pending，不影响数字完成。',
              '- `inventory/OUTCOME_EXPOSURE.json`, `qa/FINAL_ACCEPTANCE.json`, `CHECKSUMS.sha256`', '',
              '固定矩阵到此停止；未按 dev 追加网格点、方法、seed、gate 或训练。', '']
    (ROOT / 'RETURN_TO_CHATGPT.zh-CN.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--synthetic',action='store_true')
    group.add_argument('--preflight-real',action='store_true')
    group.add_argument('--score-dev',action='store_true')
    args=parser.parse_args()
    if args.synthetic:
        run_synthetic()
    elif args.preflight_real:
        run_real_preflight()
    else:
        score_dev()


if __name__ == '__main__':
    main()
