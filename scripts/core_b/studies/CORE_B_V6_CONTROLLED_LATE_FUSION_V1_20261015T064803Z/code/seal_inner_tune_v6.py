#!/usr/bin/env python3
"""Seal three target-free inner-tune replay bundles before calibration labels are read."""
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
ROLES = ['S_FIT', 'A_FIT', 'V_FIT']
CORRECTED_V_OUTPUT = ROOT / 'predictions/inner_tune/V_FIT/attempt_02'


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
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


def id_digest(ids):
    return sha_bytes(('\n'.join(ids) + '\n').encode())


def array_payload_digest(tensors, ordered_ids):
    h = hashlib.sha256()
    for name in sorted(tensors):
        arr = np.asarray(tensors[name])
        h.update(name.encode() + b'\0')
        h.update(arr.dtype.str.encode() + b'\0')
        h.update(json.dumps(list(arr.shape), separators=(',', ':')).encode() + b'\0')
        h.update(np.ascontiguousarray(arr).tobytes(order='C'))
    h.update(id_digest(ordered_ids).encode())
    return h.hexdigest()


def normalize_row(r):
    return {
        'anchor_id': str(r['anchor_id']), 'episode_id': str(int(r['episode_id'])),
        'task_id': str(int(r['task_id'])), 'source_cluster': str(r['source_cluster']),
    }


def output_for_role(role):
    if role == 'V_FIT':
        return CORRECTED_V_OUTPUT
    return ROOT / 'predictions/inner_tune' / role


def verify_bundle(role, split_membership):
    out = output_for_role(role)
    complete_path = out / 'PREDICTION_COMPLETE.json'
    rows_path = out / 'PREDICTION_ROWS.csv'
    tensor_path = out / 'PREDICTIONS.safetensors'
    reload_path = out / 'RELOAD_QA.json'
    complete, reload_qa = read_json(complete_path), read_json(reload_path)
    if complete.get('status') != 'PASS' or reload_qa.get('status') != 'PASS':
        raise RuntimeError(f'{role} replay completion/reload QA failed')
    if complete.get('dataset_kind') != 'train' or complete.get('dataset_role') != 'inner_tune':
        raise RuntimeError(f'{role} replay is not screen-seed inner-tune inference')
    if complete.get('future_target_arrays_requested') is not False or reload_qa.get('future_target_arrays_requested') is not False:
        raise RuntimeError(f'{role} inference requested future target arrays')
    if int(complete.get('final_heldout_payload_access', -1)) != 0:
        raise RuntimeError(f'{role} replay final/heldout access is nonzero')
    if sha_file(tensor_path) != complete.get('prediction_sha256') or sha_file(rows_path) != complete.get('row_ids_sha256'):
        raise RuntimeError(f'{role} replay output identity hash mismatch')
    rows_raw = read_csv(rows_path)
    rows = [normalize_row(r) for r in rows_raw]
    if len(rows) != 270 or len({r['anchor_id'] for r in rows}) != 270:
        raise RuntimeError(f'{role} inner-tune row count/unique ID mismatch')
    if tuple(complete.get('prediction_shape', [])) != (270, 30, 28):
        raise RuntimeError(f'{role} tensor shape mismatch')
    tensors = load_file(str(tensor_path))
    if set(tensors) != {'action_residual', 'prediction_native'}:
        raise RuntimeError(f'{role} tensor names mismatch: {sorted(tensors)}')
    pred = np.asarray(tensors['prediction_native'])
    if pred.shape != (270, 30, 28) or not np.isfinite(pred).all():
        raise RuntimeError(f'{role} prediction shape/finite check failed')
    if np.asarray(tensors['action_residual']).shape != (270, 30, 28):
        raise RuntimeError(f'{role} residual shape mismatch')
    role_keys = {(x['task_id'], x['episode_id'], x['source_cluster']) for x in rows}
    if not role_keys.issubset(split_membership):
        raise RuntimeError(f'{role} rows include groups outside accepted inner_tune split')
    counts = {}
    for r in rows:
        key = (r['task_id'], r['episode_id'], r['source_cluster'])
        counts[key] = counts.get(key, 0) + 1
    if counts != {key: count for key, count in split_membership.items()}:
        raise RuntimeError(f'{role} row counts differ from literal inner_tune anchor support')
    ordered_ids = [r['anchor_id'] for r in rows]
    return {
        'role': role, 'prediction_path': str(tensor_path), 'prediction_sha256': sha_file(tensor_path),
        'rows_path': str(rows_path), 'rows_sha256': sha_file(rows_path), 'completion_path': str(complete_path),
        'completion_sha256': sha_file(complete_path), 'reload_qa_path': str(reload_path),
        'model_path': complete['model_path'], 'model_sha256': complete['model_sha256'],
        'model_state_hash': complete['model_state_hash'], 'normalizer_path': complete['normalizer_path'],
        'normalizer_sha256': complete['normalizer_sha256'], 'data_path': complete['data_path'],
        'data_sha256': complete['data_sha256'], 'feature_cache_manifest_sha256': complete['feature_cache_manifest_sha256'],
        'rows': len(rows), 'shape': list(pred.shape), 'dtype': str(pred.dtype),
        'ordered_ids_sha256': id_digest(ordered_ids), 'tensor_payload_digest': array_payload_digest(tensors, ordered_ids),
        'physical_gpu': str(complete.get('physical_gpu')), 'cuda_visible_devices': str(complete.get('cuda_visible_devices')),
        'logical_device': str(complete.get('logical_device')), 'peak_memory_reserved_mib': complete.get('peak_memory_reserved_mib'),
        'peak_memory_allocated_mib': complete.get('peak_memory_allocated_mib'),
        'elapsed_seconds': complete.get('elapsed_seconds'), 'future_target_arrays_requested': False,
        'final_heldout_payload_access': 0, 'rows_data': rows,
    }


def main():
    if (ROOT / 'qa/INNER_TUNE_PREDICTION_SEAL.json').exists():
        raise RuntimeError('inner-tune prediction seal already exists; do not rewrite it')
    split_rows = read_csv(V2 / 'protocol/INNER_SPLIT.csv')
    split_membership = {}
    for r in split_rows:
        if r['role'] != 'inner_tune':
            continue
        task = str(int(r['task_id']))
        episode = str(int(r['episode_ids']))
        group = str(r['cluster_id'])
        key = (task, episode, group)
        if key in split_membership:
            raise RuntimeError(f'duplicate accepted inner_tune split cluster: {key}')
        split_membership[key] = int(r['anchors'])
    if len(split_membership) != 46 or sum(split_membership.values()) != 270:
        raise RuntimeError('accepted V2 inner_tune membership count mismatch')

    bundles = [verify_bundle(role, split_membership) for role in ROLES]
    base_ids = [r['anchor_id'] for r in bundles[0]['rows_data']]
    base_meta = [(r['anchor_id'], r['episode_id'], r['task_id'], r['source_cluster']) for r in bundles[0]['rows_data']]
    for bundle in bundles[1:]:
        ids = [r['anchor_id'] for r in bundle['rows_data']]
        meta = [(r['anchor_id'], r['episode_id'], r['task_id'], r['source_cluster']) for r in bundle['rows_data']]
        if ids != base_ids or meta != base_meta:
            raise RuntimeError(f'{bundle["role"]} ordered inner_tune IDs differ from S_FIT')

    # The first V attempt is kept as evidence; select the corrected-lease replay only after tensor identity passes.
    first_v_path = ROOT / 'predictions/inner_tune/V_FIT/PREDICTIONS.safetensors'
    first_v_rows = ROOT / 'predictions/inner_tune/V_FIT/PREDICTION_ROWS.csv'
    first_v_complete_path = ROOT / 'predictions/inner_tune/V_FIT/PREDICTION_COMPLETE.json'
    first_v_complete = read_json(first_v_complete_path)
    first_v = load_file(str(first_v_path))
    retry_v = load_file(str(CORRECTED_V_OUTPUT / 'PREDICTIONS.safetensors'))
    if set(first_v) != set(retry_v) or any(not np.array_equal(np.asarray(first_v[k]), np.asarray(retry_v[k])) for k in first_v):
        raise RuntimeError('corrected V_FIT replay payload differs from preserved first attempt')
    if read_csv(first_v_rows) != read_csv(CORRECTED_V_OUTPUT / 'PREDICTION_ROWS.csv'):
        raise RuntimeError('corrected V_FIT replay ID rows differ from the preserved first attempt')

    # Drop private row data from the seal and create a concise ID/support audit.
    seal_bundles = [{k: v for k, v in b.items() if k != 'rows_data'} for b in bundles]
    task_support = {}
    for task in TASKS:
        task_rows = [r for r in bundles[0]['rows_data'] if int(r['task_id']) == task]
        task_support[str(task)] = {'anchors': len(task_rows), 'episodes': len({r['episode_id'] for r in task_rows}),
                                   'source_groups': len({r['source_cluster'] for r in task_rows})}
    if sum(x['anchors'] for x in task_support.values()) != 270 or sum(x['episodes'] for x in task_support.values()) != 46:
        raise RuntimeError(f'inner_tune task support mismatch: {task_support}')

    join_rows = []
    for bundle in bundles:
        for index, row in enumerate(bundle['rows_data']):
            join_rows.append({'role': bundle['role'], 'canonical_row_index': index,
                              'anchor_id': row['anchor_id'], 'episode_id': row['episode_id'],
                              'task_id': row['task_id'], 'source_group': row['source_cluster'],
                              'prediction_path': bundle['prediction_path'], 'prediction_sha256': bundle['prediction_sha256'],
                              'ordered_ids_sha256': bundle['ordered_ids_sha256']})
    write_csv(ROOT / 'inventory/INNER_TUNE_REPLAY_PROVENANCE.csv', join_rows,
              ['role', 'canonical_row_index', 'anchor_id', 'episode_id', 'task_id', 'source_group',
               'prediction_path', 'prediction_sha256', 'ordered_ids_sha256'])
    seal = {
        'schema_version': 1, 'status': 'SEALED_BEFORE_TUNE_LABEL_READER', 'created_utc': utc(),
        'study': 'post_V5_controlled_late_fusion_development', 'screen_seed': 202610130,
        'accepted_roles': ROLES, 'accepted_bundle_count': 3, 'accepted_rows_per_bundle': 270,
        'task_support': task_support, 'ordered_inner_tune_ids_sha256': bundles[0]['ordered_ids_sha256'],
        'bundles': seal_bundles,
        'V_FIT_first_attempt': {'prediction_path': str(first_v_path), 'prediction_sha256': sha_file(first_v_path),
                                'completion_path': str(first_v_complete_path), 'completion_sha256': sha_file(first_v_complete_path),
                                'physical_gpu': first_v_complete.get('physical_gpu'),
                                'registered_peak_mib': 102, 'observed_peak_reserved_mib': first_v_complete.get('peak_memory_reserved_mib'),
                                'lease_released': True, 'status': 'PRESERVED_RESOURCE_ESTIMATE_DEVIATION'},
        'V_FIT_corrected_attempt': {'prediction_path': str(CORRECTED_V_OUTPUT / 'PREDICTIONS.safetensors'),
                                   'prediction_sha256': sha_file(CORRECTED_V_OUTPUT / 'PREDICTIONS.safetensors'),
                                   'completion_path': str(CORRECTED_V_OUTPUT / 'PREDICTION_COMPLETE.json'),
                                   'completion_sha256': sha_file(CORRECTED_V_OUTPUT / 'PREDICTION_COMPLETE.json'),
                                   'payload_equal_to_first_attempt': True, 'chosen_for_calibration': True,
                                   'registered_peak_mib': 134, 'observed_peak_reserved_mib': read_json(CORRECTED_V_OUTPUT / 'PREDICTION_COMPLETE.json').get('peak_memory_reserved_mib')},
        'target_archive_opened': False, 'target_native_h30_member_requested': False,
        'dev_target_archive_opened': False, 'final_heldout_payload_access': 0,
    }
    write_json(ROOT / 'qa/INNER_TUNE_PREDICTION_SEAL.json', seal)
    write_json(ROOT / 'inventory/INNER_TUNE_TARGET_FREE_ID_AUDIT.json', {
        'schema_version': 1, 'status': 'PASS', 'v2_literal_split_sha256': sha_file(V2 / 'protocol/INNER_SPLIT.csv'),
        'source_of_expected_membership': 'V2 accepted protocol/INNER_SPLIT.csv rows with role=inner_tune',
        'prediction_id_join': 'anchor_id+episode_id+task_id+source_cluster',
        'inner_tune_groups': len(split_membership), 'inner_tune_anchors': sum(split_membership.values()),
        'predicted_id_overlap_by_role': {'S_FIT': 270, 'A_FIT': 270, 'V_FIT': 270},
        'all_three_ordered_id_lists_equal': True, 'target_archive_opened': False,
        'target_array_requested': False, 'dev_target_archive_opened': False,
        'created_utc': utc(),
    })

    audit_path = ROOT / 'protocol/CALIBRATION_BASE_ROLE_AUDIT.json'
    audit = read_json(audit_path)
    audit['inner_tune_prediction_seal_path'] = str(ROOT / 'qa/INNER_TUNE_PREDICTION_SEAL.json')
    audit['inner_tune_prediction_seal_status'] = seal['status']
    audit['inner_tune_replay_predictions_accepted'] = 3
    audit['inner_tune_gpu_replay_attempts'] = 4
    audit['V_FIT_attempt_01_resource_estimate_deviation_preserved'] = True
    audit['V_FIT_attempt_02_selected_after_resource_correction'] = True
    audit['target_archive_opened_for_id_seal'] = False
    backup = audit_path.with_name(audit_path.name + '.pre-inner-tune-seal-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.bak')
    backup.write_bytes(audit_path.read_bytes())
    write_json(audit_path, audit)

    owner_path = ROOT / 'coordination/OWNER_STATE.json'
    owner = read_json(owner_path)
    owner.update({'phase': 'inner_tune_predictions_complete', 'status': 'running',
                  'inner_tune_predictions_accepted': 3, 'inner_tune_gpu_replay_attempts': 4,
                  'active_gpu_jobs': 0, 'active_jobs': 0, 'all_gpu_leases_released': True,
                  'updated_utc': utc()})
    write_json(owner_path, owner)
    print(json.dumps({'status': 'PASS', 'accepted_tune_bundles': 3, 'rows_each': 270,
                      'screen_target_arrays_requested': False, 'dev_target_archive_opened': False,
                      'V_replay_equal': True, 'final_heldout_payload_access': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
