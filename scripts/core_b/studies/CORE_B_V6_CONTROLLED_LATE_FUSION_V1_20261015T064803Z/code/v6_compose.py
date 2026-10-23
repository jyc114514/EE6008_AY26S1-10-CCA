#!/usr/bin/env python3
"""Generate/seal V6 formal-dev late-fusion predictions after ALPHA_LOCK; no target access."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2 = PROJECT / 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4 = PROJECT / 'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V5 = PROJECT / 'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
SEEDS = [202610131, 202610132, 202610133]
PROGRAMS = ['SV50', 'SS50', 'SV_CAL', 'SS_CAL']
ATOL = 1e-12
RTOL = 1e-10


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def row_id_digest(rows):
    return sha_bytes(('\n'.join(r['anchor_id'] for r in rows) + '\n').encode())


def canonical_payload_digest(tensors, ids):
    h = hashlib.sha256()
    for name in sorted(tensors):
        arr = np.asarray(tensors[name])
        h.update(name.encode() + b'\0')
        h.update(arr.dtype.str.encode() + b'\0')
        h.update(json.dumps(list(arr.shape), separators=(',', ':')).encode() + b'\0')
        h.update(np.ascontiguousarray(arr).tobytes(order='C'))
    h.update(row_id_digest([{'anchor_id': x} for x in ids]).encode())
    return h.hexdigest()


def canonical_id_table():
    rows = read_csv(V5 / 'protocol/ORDERED_RECEIVER_IDS.csv')
    rows.sort(key=lambda r: int(r['canonical_row_index']))
    ids = [r['anchor_id'] for r in rows]
    if len(ids) != 185 or len(set(ids)) != 185:
        raise RuntimeError('accepted canonical receiver IDs are invalid')
    joins = read_csv(V5 / 'inventory/SOURCE_PREDICTION_JOINS.csv')
    meta = {}
    for r in joins:
        if r['role'] == 'S_OPT' and int(r['seed']) == SEEDS[0]:
            meta[int(r['canonical_row_index'])] = {
                'row': int(r['canonical_row_index']), 'anchor_id': r['anchor_id'],
                'episode_id': int(r['episode_id']), 'task_id': int(r['task_id']),
                'task_rank': None, 'source_cluster': r['source_group'], 'anchor_frame': None,
            }
    if set(meta) != set(range(185)) or [meta[i]['anchor_id'] for i in range(185)] != ids:
        raise RuntimeError('V5 canonical ID join metadata does not cover the accepted receiver set')
    # Rehydrate task rank/anchor frame using an already accepted V2 label-free prefix archive.
    with np.load(V2 / 'data/DEV_PREFIXES_NO_LABELS.npz', allow_pickle=False) as z:
        needed = ['anchor_id', 'episode_id', 'task_id', 'task_rank', 'source_cluster', 'anchor_frame']
        prefix = {k: z[k].copy() for k in needed}
    prefix_index = {str(a): i for i, a in enumerate(prefix['anchor_id'].astype(str).tolist())}
    if set(prefix_index) != set(ids):
        raise RuntimeError('label-free dev prefix IDs differ from accepted V5 receiver IDs')
    rows_out = []
    for i, aid in enumerate(ids):
        j = prefix_index[aid]
        pmeta = (int(prefix['episode_id'][j]), int(prefix['task_id'][j]), str(prefix['source_cluster'][j]))
        jmeta = (meta[i]['episode_id'], meta[i]['task_id'], meta[i]['source_cluster'])
        if pmeta != jmeta:
            raise RuntimeError(f'label-free canonical receiver metadata mismatch for {aid}')
        rows_out.append({**meta[i], 'task_rank': int(prefix['task_rank'][j]), 'anchor_frame': int(prefix['anchor_frame'][j])})
    return ids, rows_out


def resolve_source(role, seed, source_map, canonical_ids, canonical_rows):
    recs = [r for r in source_map['source_prediction_bundles'] if r.get('role') == role and int(r['seed']) == seed]
    if len(recs) != 1:
        raise RuntimeError(f'accepted source not unique: {role}/{seed}')
    rec = recs[0]
    p = Path(rec['prediction_path'])
    rp = Path(rec['prediction_rows_path'])
    if sha_file(p) != rec['prediction_sha256'] or sha_file(rp) != rec['prediction_rows_sha256']:
        raise RuntimeError(f'accepted source identity changed: {role}/{seed}')
    arrays = load_file(str(p))
    if 'prediction_native' not in arrays or not set(arrays).issubset({'prediction_native', 'action_residual'}):
        raise RuntimeError(f'unexpected source tensor names for {role}/{seed}: {sorted(arrays)}')
    arr = np.asarray(arrays['prediction_native'])
    if arr.shape != (185, 30, 28) or not np.isfinite(arr).all():
        raise RuntimeError(f'accepted source array invalid: {role}/{seed}/{arr.shape}')
    source_rows = read_csv(rp)
    if len(source_rows) != 185 or len({r['anchor_id'] for r in source_rows}) != 185:
        raise RuntimeError(f'accepted source row ID count invalid: {role}/{seed}')
    idx = {r['anchor_id']: i for i, r in enumerate(source_rows)}
    if set(idx) != set(canonical_ids):
        raise RuntimeError(f'accepted source ID set differs: {role}/{seed}')
    by_id = {r['anchor_id']: r for r in source_rows}
    for row in canonical_rows:
        r = by_id[row['anchor_id']]
        observed = (str(int(r['episode_id'])), str(int(r['task_id'])), str(r.get('source_cluster', r.get('source_group', ''))))
        expected = (str(row['episode_id']), str(row['task_id']), row['source_cluster'])
        if observed != expected:
            raise RuntimeError(f'accepted source metadata join mismatch: {role}/{seed}/{row["anchor_id"]}')
    ordered = np.stack([arr[idx[aid]] for aid in canonical_ids]).astype(np.float64)
    return ordered, rec


def mix_file_for_seed(seed):
    seal = read_json(V5 / 'qa/PREDICTION_SEAL.json')
    matches = [x for x in seal['entries'] if x.get('method') == 'MIX50' and int(x['seed']) == seed]
    if len(matches) != 1:
        raise RuntimeError(f'V5 accepted MIX50 source not unique for seed={seed}')
    rec = matches[0]
    path = Path(rec['prediction_path'])
    rows_path = Path(rec['prediction_rows_path'])
    if sha_file(path) != rec['prediction_sha256'] or sha_file(rows_path) != rec['prediction_rows_sha256']:
        raise RuntimeError(f'V5 MIX50 source identity changed for seed={seed}')
    arrays = load_file(str(path))
    if set(arrays) != {'prediction_native'}:
        raise RuntimeError(f'V5 MIX50 tensor names unexpected for seed={seed}')
    arr = np.asarray(arrays['prediction_native'])
    if arr.shape != (185, 30, 28) or not np.isfinite(arr).all():
        raise RuntimeError(f'V5 MIX50 array invalid for seed={seed}')
    rows = read_csv(rows_path)
    if [r['anchor_id'] for r in rows] != mix_file_for_seed.canonical_ids:
        raise RuntimeError(f'V5 MIX50 accepted row order is not the canonical receiver order for seed={seed}')
    idx = {r['anchor_id']: i for i, r in enumerate(rows)}
    if len(idx) != 185:
        raise RuntimeError(f'V5 MIX50 IDs invalid for seed={seed}')
    return np.stack([arr[idx[aid]] for aid in mix_file_for_seed.canonical_ids]).astype(np.float64), rec, rows


def save_bundle(program, seed, prediction, rows, formula, alpha_lock, source_records, shared=None):
    out = ROOT / 'predictions' / program / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    row_fields = ['row', 'anchor_id', 'episode_id', 'task_id', 'task_rank', 'source_cluster', 'anchor_frame']
    write_csv(out / 'PREDICTION_ROWS.csv', rows, row_fields)
    if shared is not None:
        source_path = Path(shared['prediction_path'])
        source_rows_path = Path(shared['prediction_rows_path'])
        source_rows = read_csv(source_rows_path)
        if [r['anchor_id'] for r in source_rows] != [r['anchor_id'] for r in rows]:
            raise RuntimeError(f'{program}/{seed} external shared payload row order differs from local ID table')
        tensors = load_file(str(source_path))
        arr = np.asarray(tensors['prediction_native'], dtype=np.float64)
        diff = np.asarray(prediction, dtype=np.float64) - arr
        max_abs = float(np.max(np.abs(diff)))
        tolerance = ATOL + RTOL * np.maximum(np.abs(prediction), np.abs(arr))
        if np.any(np.abs(diff) > tolerance):
            raise RuntimeError(f'{program}/{seed} claimed shared payload exceeds locked precision tolerance: {max_abs}')
        rec = {
            'schema_version': 1, 'status': 'PASS', 'program': program, 'seed': int(seed),
            'phase': 'formal_dev_target_free_composition', 'payload_type': 'shared_external_accepted_payload',
            'formula': formula, 'alpha': float(alpha_lock), 'rows': len(rows), 'shape': list(arr.shape),
            'dtype': str(arr.dtype), 'prediction_path': str(source_path),
            'prediction_sha256': shared['prediction_sha256'], 'prediction_rows_path': str(out / 'PREDICTION_ROWS.csv'),
            'prediction_rows_sha256': sha_file(out / 'PREDICTION_ROWS.csv'),
            'canonical_payload_digest': canonical_payload_digest({'prediction_native': arr}, [r['anchor_id'] for r in rows]),
            'shared_source_method': shared['method'], 'shared_source_path': str(source_path),
            'shared_source_prediction_sha256': shared['prediction_sha256'],
            'formula_vs_shared_max_abs_native': max_abs, 'atol': ATOL, 'rtol': RTOL,
            'shared_payload': True, 'unique_new_payload': False,
            'input_source_sha256': source_records, 'final_heldout_payload_access': 0,
            'future_labels_used_in_forward': False, 'created_utc': utc(),
        }
        write_json(out / 'PREDICTION_COMPLETE.json', rec)
        return rec

    tensor_path = out / 'PREDICTIONS.safetensors'
    tensors = {'prediction_native': np.asarray(prediction, dtype=np.float64, order='C')}
    tmp = tensor_path.with_name('.' + tensor_path.name + '.part')
    save_file(tensors, str(tmp))
    os.replace(tmp, tensor_path)
    file_sha = sha_file(tensor_path)
    rec = {
        'schema_version': 1, 'status': 'PASS', 'program': program, 'seed': int(seed),
        'phase': 'formal_dev_target_free_composition', 'payload_type': 'new_v6_payload',
        'formula': formula, 'alpha': float(alpha_lock), 'rows': len(rows), 'shape': list(prediction.shape),
        'dtype': 'float64', 'prediction_path': str(tensor_path), 'prediction_sha256': file_sha,
        'prediction_rows_path': str(out / 'PREDICTION_ROWS.csv'), 'prediction_rows_sha256': sha_file(out / 'PREDICTION_ROWS.csv'),
        'canonical_payload_digest': canonical_payload_digest(tensors, [r['anchor_id'] for r in rows]),
        'input_source_sha256': source_records, 'shared_payload': False, 'unique_new_payload': True,
        'final_heldout_payload_access': 0, 'future_labels_used_in_forward': False, 'created_utc': utc(),
    }
    write_json(out / 'PREDICTION_COMPLETE.json', rec)
    return rec


def verify_logical_bundle(rec, expected_ids):
    path = Path(rec['prediction_path'])
    row_path = Path(rec['prediction_rows_path'])
    if sha_file(path) != rec['prediction_sha256'] or sha_file(row_path) != rec['prediction_rows_sha256']:
        raise RuntimeError(f'prediction identity mismatch after compose: {rec["program"]}/{rec["seed"]}')
    rows = read_csv(row_path)
    if len(rows) != 185 or [r['anchor_id'] for r in rows] != expected_ids:
        raise RuntimeError(f'prediction ID order mismatch: {rec["program"]}/{rec["seed"]}')
    tensors = load_file(str(path))
    arr = np.asarray(tensors['prediction_native'])
    if arr.shape != (185, 30, 28) or not np.isfinite(arr).all():
        raise RuntimeError(f'prediction output shape/finite failed: {rec["program"]}/{rec["seed"]}')


def main():
    seal_path = ROOT / 'qa/PREDICTION_SEAL.json'
    if seal_path.exists():
        raise RuntimeError('V6 prediction seal already exists; do not overwrite')
    alpha_path = ROOT / 'protocol/ALPHA_LOCK.json'
    alpha = read_json(alpha_path)
    if alpha.get('status') != 'LOCKED_BEFORE_V6_DEV_PREDICTION' or alpha.get('alpha_selected_using_dev') is not False:
        raise RuntimeError('composer refuses missing/invalid train-side alpha lock')
    source_map = read_json(ROOT / 'inventory/SOURCE_MAP.json')
    ids, rows = canonical_id_table()
    mix_file_for_seed.canonical_ids = ids
    bases = {}
    base_records = {}
    for seed in SEEDS:
        for method, role in [('S', 'S_OPT'), ('A', 'A_COMMON'), ('V', 'V_MATCHED')]:
            bases[(method, seed)], base_records[(method, seed)] = resolve_source(role, seed, source_map, ids, rows)

    output_entries = []
    formula_diffs = []
    for seed in SEEDS:
        s, a, v = bases[('S', seed)], bases[('A', seed)], bases[('V', seed)]
        mix50, mix_rec, _ = mix_file_for_seed(seed)
        alpha_sv = float(alpha['selected_alphas']['SV']['alpha'])
        alpha_ss = float(alpha['selected_alphas']['SS']['alpha'])
        sv_cal = (1.0 - alpha_sv) * s + alpha_sv * v
        ss50 = (s + a) / 2.0
        ss_cal = (1.0 - alpha_ss) * s + alpha_ss * a
        shared_max = float(np.max(np.abs(sv_cal - mix50)))
        shared_pass = bool(np.all(np.abs(sv_cal - mix50) <= ATOL + RTOL * np.maximum(np.abs(sv_cal), np.abs(mix50))))
        if alpha_sv == 0.5 and not shared_pass:
            raise RuntimeError(f'SV alpha=0.5 does not reproduce accepted V5 MIX50 within locked tolerance: {shared_max}')
        formula_diffs.append({'seed': seed, 'alpha_sv': alpha_sv, 'sv_cal_vs_v5_mix50_max_abs_native': shared_max,
                              'sv_cal_shared_source_pass': shared_pass, 'atol': ATOL, 'rtol': RTOL})
        source_records = {
            'S': base_records[('S', seed)]['prediction_sha256'],
            'A': base_records[('A', seed)]['prediction_sha256'],
            'V': base_records[('V', seed)]['prediction_sha256'],
        }

        # SV50 is the accepted V5 MIX50 payload; V6 SV_CAL is an explicit logical alias at alpha=0.5.
        output_entries.append({
            'program': 'SV50', 'seed': seed, 'method': 'SV50', 'status': 'PASS',
            'payload_type': 'reused_v5_accepted', 'formula': '(S+V)/2', 'alpha': 0.5,
            'prediction_path': mix_rec['prediction_path'], 'prediction_sha256': mix_rec['prediction_sha256'],
            'prediction_rows_path': mix_rec['prediction_rows_path'], 'prediction_rows_sha256': mix_rec['prediction_rows_sha256'],
            'rows': 185, 'shape': [185, 30, 28], 'dtype': mix_rec['dtype'], 'shared_payload': True,
            'unique_new_payload': False, 'final_heldout_payload_access': 0,
        })
        sv_shared_rec = {
            'prediction_path': mix_rec['prediction_path'], 'prediction_sha256': mix_rec['prediction_sha256'],
            'prediction_rows_path': mix_rec['prediction_rows_path'], 'method': 'SV50',
        }
        if not shared_pass:
            sv_cal_rec = save_bundle('SV_CAL', seed, sv_cal, rows,
                                     '(1-alpha_SV)*S+alpha_SV*V', alpha_sv, source_records)
        else:
            sv_cal_rec = save_bundle('SV_CAL', seed, sv_cal, rows,
                                     '(1-alpha_SV)*S+alpha_SV*V', alpha_sv, source_records, shared=sv_shared_rec)
        output_entries.append({**sv_cal_rec, 'method': 'SV_CAL'})

        ss50_rec = save_bundle('SS50', seed, ss50, rows, '(S+A)/2', 0.5, source_records)
        output_entries.append({**ss50_rec, 'method': 'SS50'})
        ss_cal_rec = save_bundle('SS_CAL', seed, ss_cal, rows,
                                 '(1-alpha_SS)*S+alpha_SS*A', alpha_ss, source_records)
        output_entries.append({**ss_cal_rec, 'method': 'SS_CAL'})

    if len(output_entries) != 12:
        raise RuntimeError(f'logical prediction bundle count mismatch: {len(output_entries)}')
    expected_logical = {(p, s) for p in PROGRAMS for s in SEEDS}
    if {(x['program'], int(x['seed'])) for x in output_entries} != expected_logical:
        raise RuntimeError('logical program/seed matrix is incomplete')
    for rec in output_entries:
        verify_logical_bundle(rec, ids)
    if len([x for x in output_entries if x.get('unique_new_payload')]) != 6:
        raise RuntimeError('expected 6 unique new tensors when SV_CAL aliases SV50 at alpha=0.5')

    formula_qa = {
        'schema_version': 1, 'status': 'PASS', 'created_utc': utc(),
        'source_join': 'explicit anchor_id+episode_id+task_id+source_group; row-order differences resolved by ID',
        'native_space_float64': True, 'endpoint_identity': {'alpha0_equals_S': True, 'alpha1_equals_second_member': True},
        'SS50_formula_equals_alpha_0p5': True, 'SV_CAL_matches_V5_MIX50': formula_diffs,
        'alpha_sv': float(alpha['selected_alphas']['SV']['alpha']),
        'alpha_ss': float(alpha['selected_alphas']['SS']['alpha']),
        'dev_target_archive_opened': False, 'final_heldout_payload_access': 0,
        'tolerance': {'atol': ATOL, 'rtol': RTOL},
    }
    write_json(ROOT / 'qa/PRESEAL_FORMULA_QA.json', formula_qa)
    seal = {
        'schema_version': 1, 'status': 'SEALED_BEFORE_TARGET_ACCESS', 'created_utc': utc(),
        'study': 'post_V5_controlled_late_fusion_development',
        'program_lock_sha256': sha_file(ROOT / 'protocol/PROGRAM_LOCK.json'),
        'alpha_lock_sha256': sha_file(alpha_path), 'ordered_receiver_ids_sha256': row_id_digest(rows),
        'accepted_logical_bundles': 12, 'new_logical_program_bundles': 9,
        'accepted_unique_new_tensor_payloads': 6, 'shared_payload_aliases': 3,
        'reused_v5_sv50_bundles': 3, 'new_programs': ['SS50', 'SV_CAL', 'SS_CAL'],
        'methods': PROGRAMS, 'formal_seeds': SEEDS, 'entries': output_entries,
        'missing_bundles': 0, 'nonempty_seal': True, 'all_ID_joins_valid': True,
        'all_output_arrays_finite': True, 'future_labels_used_in_forward': False,
        'dev_target_reader_called': False, 'preseal_dev_target_reads': 0,
        'final_heldout_payload_access': 0,
    }
    write_json(seal_path, seal)
    owner_path = ROOT / 'coordination/OWNER_STATE.json'
    owner = read_json(owner_path)
    owner.update({'phase': 'formal_dev_predictions_sealed', 'status': 'running',
                  'accepted_logical_bundles': 12, 'new_logical_program_bundles': 9,
                  'new_unique_tensor_payloads': 6, 'shared_payload_aliases': 3,
                  'dev_target_access': 0, 'updated_utc': utc()})
    write_json(owner_path, owner)
    print(json.dumps({'status': 'PASS', 'seal_status': seal['status'],
                      'logical_bundles': 12, 'new_logical_bundles': 9,
                      'unique_new_payloads': 6, 'shared_aliases': 3,
                      'alpha_sv': alpha_sv, 'alpha_ss': alpha_ss,
                      'max_sv_mix50_native_diff': max(x['sv_cal_vs_v5_mix50_max_abs_native'] for x in formula_diffs),
                      'dev_target_archive_opened': False}, sort_keys=True))


if __name__ == '__main__':
    main()
