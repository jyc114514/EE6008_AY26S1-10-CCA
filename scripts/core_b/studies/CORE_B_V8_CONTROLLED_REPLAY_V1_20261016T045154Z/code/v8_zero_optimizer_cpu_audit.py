#!/usr/bin/env python3
"""Full-row, train-only V7 target/identity audit; CPU only, no optimizer."""
from __future__ import annotations
import os

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2 = PROJECT / 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4 = PROJECT / 'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V7 = PROJECT / 'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
SEEDS = (202610131, 202610132, 202610133)
TASKS = (119, 94, 90, 141, 230)
SHARED_INPUTS = ('anchor_id', 'episode_id', 'task_id', 'task_rank', 'source_cluster',
                 'cache_index', 'anchor_frame', 'history_native', 'history_mask',
                 'relative_delta', 'anchor_native')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def checked_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def prediction(role: str, seed: int, data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    bundle = V7 / 'teacher_predictions/source' / role / f'seed{seed}'
    receipt = json_read(bundle / 'PREDICTION_COMPLETE.json')
    if receipt.get('status') != 'PASS' or receipt.get('teacher_role') != role or int(receipt.get('seed', -1)) != seed:
        raise RuntimeError(f'unaccepted/mismatched teacher replay: {role}/{seed}')
    pred_path = Path(receipt['prediction_path'])
    rows_path = Path(receipt['prediction_rows_path'])
    if sha(pred_path) != receipt['prediction_sha256'] or sha(rows_path) != receipt['prediction_rows_sha256']:
        raise RuntimeError(f'teacher prediction source hash mismatch: {role}/{seed}')
    tensors = load_file(str(pred_path), device='cpu')
    if 'prediction_native' not in tensors:
        raise RuntimeError(f'prediction_native absent: {role}/{seed}')
    values = tensors['prediction_native'].numpy()
    if values.shape != (1280, 30, 28) or values.dtype != np.float64 or not np.isfinite(values).all():
        raise RuntimeError(f'teacher replay schema failure: {role}/{seed}: {values.shape}/{values.dtype}')
    with rows_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(data['anchor_id']) or len({r['anchor_id'] for r in rows}) != len(rows):
        raise RuntimeError(f'teacher rows missing/duplicated: {role}/{seed}')
    by_id = {r['anchor_id']: r for r in rows}
    ids = data['anchor_id'].astype(str)
    if set(by_id) != set(ids):
        raise RuntimeError(f'teacher anchor population mismatch: {role}/{seed}')
    order = np.empty(len(ids), dtype=np.int64)
    for i, aid in enumerate(ids.tolist()):
        row = by_id[aid]
        for key, field in [('episode_id', 'episode_id'), ('task_id', 'task_id'), ('source_cluster', 'source_cluster')]:
            if str(row[field]) != str(data[key][i]):
                raise RuntimeError(f'{role}/{seed} identity mismatch {field} at {aid}')
        order[i] = int(row['row'])
    if not np.array_equal(np.sort(order), np.arange(len(ids))):
        raise RuntimeError(f'teacher row mapping is not bijective: {role}/{seed}')
    return np.asarray(values[order], dtype=np.float64), receipt


def target_bank(program: str, seed: int, data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any], list[dict[str, str]]]:
    base = V7 / 'teacher_targets' / program / f'seed{seed}'
    manifest = json_read(base / 'TARGET_MANIFEST.json')
    path, rows_path = base / 'Z_TRAIN_NATIVE.npy', base / 'TARGET_ROWS.csv'
    if manifest.get('status') != 'PASS' or manifest.get('program_id') != program or int(manifest.get('seed', -1)) != seed:
        raise RuntimeError(f'target manifest role/seed mismatch: {program}/{seed}')
    if sha(path) != manifest.get('target_sha256') or sha(rows_path) != manifest.get('rows_sha256'):
        raise RuntimeError(f'target bank checksum mismatch: {program}/{seed}')
    z = np.load(path, mmap_mode='r', allow_pickle=False)
    if z.shape != (1280, 30, 28) or z.dtype != np.float64 or not np.isfinite(z).all():
        raise RuntimeError(f'target bank schema failure: {program}/{seed}: {z.shape}/{z.dtype}')
    with rows_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1280 or len({r['anchor_id'] for r in rows}) != 1280:
        raise RuntimeError(f'target row map missing/duplicated: {program}/{seed}')
    by_id = {r['anchor_id']: r for r in rows}
    ids = data['anchor_id'].astype(str)
    if set(by_id) != set(ids):
        raise RuntimeError(f'target ID set differs from accepted train rows: {program}/{seed}')
    order = np.empty(1280, dtype=np.int64)
    for i, aid in enumerate(ids.tolist()):
        row = by_id[aid]
        for field in ('episode_id', 'task_id', 'source_cluster'):
            if str(row[field]) != str(data[field][i]):
                raise RuntimeError(f'target identity mismatch {field}: {program}/{seed}/{aid}')
        order[i] = int(row['target_row'])
    if not np.array_equal(np.sort(order), np.arange(1280)):
        raise RuntimeError(f'target row mapping is not bijective: {program}/{seed}')
    return np.asarray(z[order], dtype=np.float64), manifest, rows


def loss_components(pred: np.ndarray, target: np.ndarray, sigma: np.ndarray, fdscale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos_err = np.abs(pred - target) / sigma[None, None, :]
    pos = 0.5 * (pos_err[..., :14].mean(axis=(1, 2)) + pos_err[..., 14:].mean(axis=(1, 2)))
    # Exactly 29 future-internal differences; the anchor-to-first-target seam is excluded.
    fd_err = np.abs(np.diff(pred - target, axis=1) / fdscale[None, None, :])
    fd = 0.5 * (fd_err[..., :14].mean(axis=(1, 2)) + fd_err[..., 14:].mean(axis=(1, 2)))
    return pos, fd


def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    train_path = V2 / 'data/TRAIN_SELECTED_H30.npz'
    prefix_path = V7 / 'data/TRAIN_PREFIXES_NO_LABELS.npz'
    norm_path = V7 / 'normalizers/NORMALIZER_FORMAL.json'
    data, prefix = checked_npz(train_path), checked_npz(prefix_path)
    if len(data['anchor_id']) != 1280 or len(set(data['anchor_id'].astype(str))) != 1280:
        raise RuntimeError('accepted train population is not 1280 unique anchor IDs')
    if len(prefix['anchor_id']) != 1280:
        raise RuntimeError('V7 no-label prefix population is not 1280 rows')
    prefix_index = {str(a): i for i, a in enumerate(prefix['anchor_id'].astype(str))}
    if set(prefix_index) != set(data['anchor_id'].astype(str)):
        raise RuntimeError('V7 prefix IDs differ from V2 accepted train IDs')
    prefix_order = np.asarray([prefix_index[str(a)] for a in data['anchor_id'].astype(str)], dtype=np.int64)
    input_equal = {k: bool(np.array_equal(data[k], prefix[k][prefix_order])) for k in SHARED_INPUTS}
    if not all(input_equal.values()):
        raise RuntimeError(f'V2/V7 shared input fields are not identical: {input_equal}')
    norm = json_read(norm_path)
    sigma = np.asarray(norm['target_sigma'], dtype=np.float64)
    fdscale = np.asarray(norm['safe_fd_scale'], dtype=np.float64)
    if sigma.shape != (28,) or fdscale.shape != (28,) or np.any(sigma <= 0) or np.any(fdscale <= 0):
        raise RuntimeError('formal normalizer has invalid target scales')
    y = np.asarray(data['target_native_h30'], dtype=np.float64)
    anchor = np.asarray(data['anchor_native'], dtype=np.float64)
    if y.shape != (1280, 30, 28) or not np.isfinite(y).all() or not np.isfinite(anchor).all():
        raise RuntimeError('accepted train Y/anchor shape or finite check failed')
    ids = data['anchor_id'].astype(str)
    frame_pattern = re.compile(r'_f(\d+)$')
    rows_out: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    program_roles = {'VD_SELF': 'V_MATCHED', 'VD_SV': 'SV50'}
    for seed in SEEDS:
        teachers: dict[str, np.ndarray] = {}
        teacher_receipts: dict[str, dict[str, Any]] = {}
        for role in ('S_OPT', 'V_MATCHED'):
            teachers[role], teacher_receipts[role] = prediction(role, seed, data)
        for program, role in program_roles.items():
            if program == 'VD_SELF':
                t = teachers['V_MATCHED']
            else:
                t = 0.5 * teachers['S_OPT'] + 0.5 * teachers['V_MATCHED']
            z, manifest, target_rows = target_bank(program, seed, data)
            expected_z = y + 0.5 * (t - y)
            delta = np.abs(z - expected_z)
            max_formula = float(delta.max())
            formula_ix = np.unravel_index(int(delta.argmax()), delta.shape)
            if not np.allclose(z, expected_z, atol=1e-12, rtol=1e-10):
                raise RuntimeError(f'V7 native Y/T/Z mixture mismatch: {program}/{seed}; max={max_formula}')
            py, fy = loss_components(t, y, sigma, fdscale)
            pz, fz = loss_components(t, z, sigma, fdscale)
            identity = {'T_equals_T_max': float(max(np.abs(loss_components(t, t, sigma, fdscale)[0]).max(), np.abs(loss_components(t, t, sigma, fdscale)[1]).max())),
                        'position_half_identity_max': float(np.max(np.abs(pz - 0.5 * py))),
                        'fd_half_identity_max': float(np.max(np.abs(fz - 0.5 * fy)))}
            if identity['T_equals_T_max'] > 1e-12 or identity['position_half_identity_max'] > 1e-12 or identity['fd_half_identity_max'] > 1e-12:
                raise RuntimeError(f'FP64 SELF algebra identity failed: {program}/{seed}: {identity}')
            target_rows_by_id = {r['anchor_id']: r for r in target_rows}
            for i, aid in enumerate(ids.tolist()):
                frame_match = frame_pattern.search(aid)
                if frame_match is None:
                    raise RuntimeError(f'anchor ID lacks accepted frame suffix: {aid}')
                frame = int(frame_match.group(1))
                if frame != int(data['anchor_frame'][i]):
                    raise RuntimeError(f'anchor ID/frame mismatch: {aid}')
                hist_frames = data['history_selected_frame_indices'][i]
                if hist_frames.shape != (32,) or np.any(np.diff(hist_frames) < 0) or int(hist_frames[-1]) != frame or int(hist_frames.max()) > frame:
                    raise RuntimeError(f'history frame mapping failure: {aid}')
                r = target_rows_by_id[aid]
                ry, rt, rz = y[i], t[i], z[i]
                residual_y, residual_t, residual_z = tuple((x - anchor[i]) / sigma for x in (ry, rt, rz))
                roundtrip_y = anchor[i] + sigma * residual_y
                roundtrip_t = anchor[i] + sigma * residual_t
                roundtrip_z = anchor[i] + sigma * residual_z
                errors = [float(np.max(np.abs(roundtrip_y - ry))), float(np.max(np.abs(roundtrip_t - rt))), float(np.max(np.abs(roundtrip_z - rz)))]
                rows_out.append({
                    'program_id': program, 'teacher_role': role, 'seed': seed, 'anchor_id': aid,
                    'episode_id': int(data['episode_id'][i]), 'task_id': int(data['task_id'][i]),
                    'source_cluster': str(data['source_cluster'][i]), 'accepted_row': i,
                    'target_row': int(r['target_row']), 'cache_index': int(data['cache_index'][i]),
                    'anchor_frame': frame, 'target_start_frame_contract': frame + 1,
                    'target_end_frame_contract': frame + 30, 'target_rows_joined_by_anchor_id': True,
                    'all_shared_prefix_fields_exact': True, 'history_frame_indices_valid': True,
                    'Y_native_to_model_to_native_max_abs': errors[0],
                    'T_native_to_model_to_native_max_abs': errors[1],
                    'Z_native_to_model_to_native_max_abs': errors[2],
                    'native_mix_formula_max_abs': max_formula,
                    'native_mix_formula_max_row_anchor': ids[formula_ix[0]],
                    'T_equals_Y_position_loss': float(py[i]), 'T_equals_Y_29diff_loss': float(fy[i]),
                    'T_vs_Z_position_loss': float(pz[i]), 'T_vs_Z_29diff_loss': float(fz[i]),
                })
            checks.append({
                'program_id': program, 'seed': seed, 'teacher_role': role,
                'target_sha256': manifest['target_sha256'], 'target_rows_sha256': manifest['rows_sha256'],
                'source_S_bundle_sha256': teacher_receipts['S_OPT']['prediction_sha256'],
                'source_V_bundle_sha256': teacher_receipts['V_MATCHED']['prediction_sha256'],
                'rows': 1280, 'target_formula_max_abs': max_formula,
                'target_formula_worst_id': str(ids[formula_ix[0]]),
                'algebra_identities': identity,
                'teacher_prediction_is_oof': False,
                'train_targets_only': True,
            })
    fields = list(rows_out[0].keys())
    with (root / 'FULL_ROW_TARGET_ROUNDTRIP.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows_out)
    summary = {
        'status': 'PASS', 'schema_version': 1, 'created_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),
        'train_npz_path': str(train_path), 'train_npz_sha256': sha(train_path),
        'prefix_archive_path': str(prefix_path), 'prefix_archive_sha256': sha(prefix_path),
        'normalizer_path': str(norm_path), 'normalizer_sha256': sha(norm_path),
        'train_anchors': 1280, 'train_episodes': int(len(np.unique(data['episode_id']))),
        'task_ids': sorted(int(x) for x in np.unique(data['task_id'])),
        'shared_prefix_fields_exact': input_equal,
        'program_seed_rows_audited': len(rows_out), 'program_seeds': checks,
        'outer_train_only': True, 'dev_target_archive_opened': False, 'final_heldout_payload_access': 0,
        'physical_timestamp_clock_calibration': 'UNKNOWN; this audit verifies accepted archive frame indices and H30 construction source only',
        'target_windows': 'source construction uses anchor+1 through anchor+30; no seam included in 29-difference loss',
    }
    (root / 'FULL_ROW_TARGET_ROUNDTRIP.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'rows': len(rows_out), 'input_equal': input_equal,
                      'max_formula_delta': max(c['target_formula_max_abs'] for c in checks),
                      'max_roundtrip': max(float(max(r[k] for k in ('Y_native_to_model_to_native_max_abs','T_native_to_model_to_native_max_abs','Z_native_to_model_to_native_max_abs'))) for r in rows_out)}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
