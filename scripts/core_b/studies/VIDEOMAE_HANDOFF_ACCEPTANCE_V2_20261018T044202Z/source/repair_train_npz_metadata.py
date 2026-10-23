#!/usr/bin/env python3
"""Repair only misindexed train-bundle metadata after proving predictions unchanged."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temp = path.with_name('.' + path.name + '.part')
    temp.write_text(payload, encoding='utf-8')
    os.replace(temp, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def save_npz(path: Path, arrays: dict[str, np.ndarray], temp_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix='.metadata-repair.', suffix='.npz', dir=temp_dir, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        with temp.open('wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, required=True)
    ap.add_argument('--v2-root', type=Path, required=True)
    ap.add_argument('--audit-root', type=Path, required=True)
    args = ap.parse_args()
    project, v2, audit = [p.resolve() for p in (args.project_root, args.v2_root, args.audit_root)]
    input_path = v2 / 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz'
    contract_path = audit / 'REPLAY_CONTRACT.json'
    cells_path = audit / 'EXPECTED_CELLS.csv'
    with np.load(input_path, allow_pickle=False) as z:
        split = z['split_code'].copy()
        ids = z['anchor_id'].astype('U96', copy=True)
        episodes = z['episode_index'].astype(np.int64, copy=True)
        task = z['original_task_index'].astype(np.int64, copy=True)
        frames = z['anchor_frame'].astype(np.int64, copy=True)
        timestamps = z['anchor_timestamp'].astype(np.float64, copy=True)
    train_rows = np.flatnonzero(split == 0).astype(np.int64)
    dev_rows = np.flatnonzero(split == 1).astype(np.int64)
    if (len(train_rows), len(dev_rows)) != (17757, 2322):
        raise AssertionError('canonical train/dev split sizes changed')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    refs = {(x['cell'], x['split']): x for x in contract['references']}
    archive = json.loads((v2 / 'ARCHIVE.json').read_text(encoding='utf-8'))
    archive_cells = {x['name']: x for x in archive['cells']}
    expected = list(csv.DictReader(cells_path.open(newline='', encoding='utf-8')))
    if len(expected) != 12:
        raise AssertionError('expected exactly 12 locked model cells')

    # Preflight every cell before writing any output.
    for row in expected:
        cell = row['cell']
        cell_dir = audit / 'replay/gpu' / cell
        npz_path = cell_dir / 'predictions_train.npz'
        receipt_path = cell_dir / 'REPLAY_RECEIPT.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('status') != 'PASS_GPU_FROZEN_REPLAY' or receipt.get('target_arrays_loaded') is not False or receipt.get('final_test_access') != 0:
            raise AssertionError(f'{cell}: original child receipt scope/status changed')
        train_side = next(x for x in receipt['outputs'] if x['split'] == 'train')
        if digest(npz_path) != train_side.get('sha256'):
            raise AssertionError(f'{cell}: original train bundle no longer matches child receipt')
        data = load_npz(npz_path)
        old_run = v2 / archive_cells[cell]['run']
        reference = Path(refs[(cell, 'train')]['path'])
        if not reference.is_file() or digest(reference) != refs[(cell, 'train')]['sha256']:
            raise AssertionError(f'{cell}: locked historical train reference hash mismatch')
        old_pred = load_npz(reference)['predictions_native']
        pred = data['predictions_native']
        if pred.dtype != np.float32 or pred.shape != (17757, int(row['horizon']), 28) or not np.isfinite(pred).all():
            raise AssertionError(f'{cell}: prediction schema is not repairable')
        if not np.array_equal(pred, old_pred):
            raise AssertionError(f'{cell}: prediction values are not bitwise equal to locked historical GPU output')
        if not np.array_equal(data['anchor_id'].astype('U96'), ids[train_rows]) or not np.array_equal(data['episode_index'].astype(np.int64), episodes[train_rows]):
            raise AssertionError(f'{cell}: train IDs/episodes differ from canonical rows')
        # The observed bug leaves the final (dev) row selection in the three metadata fields.
        for key, expected_dev in (('original_task_index', task[dev_rows]), ('anchor_frame', frames[dev_rows]), ('anchor_timestamp', timestamps[dev_rows])):
            if not np.array_equal(data[key], expected_dev):
                raise AssertionError(f'{cell}: unexpected pre-repair {key}; stop rather than normalize')
        if len(data['original_task_index']) != len(dev_rows):
            raise AssertionError(f'{cell}: unexpected pre-repair metadata lengths')
        if row['cell'] not in archive_cells:
            raise AssertionError(f'{cell}: not present in frozen archive')

    failure_dir = audit / 'diagnostics/first_gpu_validation_failure'
    failure_dir.mkdir(parents=True, exist_ok=False)
    for name in ('CHECKPOINT_REPLAY.csv', 'PREDICTION_DIFF_DETAILS.csv', 'PREDICTION_VALIDATION_SUMMARY.json', 'RUN_RECEIPT.json'):
        source = audit / name
        if source.is_file():
            shutil.copy2(source, failure_dir / name)
    (audit / 'replay/gpu_initial_metadata').mkdir(parents=True, exist_ok=False)
    original_contract_sha = digest(contract_path)
    input_sha = digest(input_path)
    repaired_cells = []

    for row in expected:
        cell = row['cell']
        cell_dir = audit / 'replay/gpu' / cell
        original_path = cell_dir / 'predictions_train.npz'
        receipt_path = cell_dir / 'REPLAY_RECEIPT.json'
        original_sha = digest(original_path)
        original_receipt_sha = digest(receipt_path)
        backup_dir = audit / 'replay/gpu_initial_metadata' / cell
        backup_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(original_path, backup_dir / 'predictions_train.npz')
        shutil.copy2(receipt_path, backup_dir / 'REPLAY_RECEIPT.json')
        arrays = load_npz(original_path)
        prediction_sha = array_sha(arrays['predictions_native'])
        arrays['original_task_index'] = task[train_rows]
        arrays['anchor_frame'] = frames[train_rows]
        arrays['anchor_timestamp'] = timestamps[train_rows]
        save_npz(original_path, arrays, audit / 'tmp')
        new_sha = digest(original_path)
        if array_sha(load_npz(original_path)['predictions_native']) != prediction_sha:
            raise AssertionError(f'{cell}: prediction values changed during metadata-only repair')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        for out in receipt['outputs']:
            if out['split'] == 'train':
                out.update({'sha256': new_sha, 'bytes': original_path.stat().st_size,
                            'shape': list(arrays['predictions_native'].shape), 'rows': len(train_rows)})
        repair = {
            'status': 'PASS_METADATA_ONLY_REPAIR', 'cell': cell,
            'repair_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'cause': 'runner serialization referenced the last loop variable rows (dev) instead of per-split train row indices',
            'original_train_bundle_path': str(backup_dir / 'predictions_train.npz'),
            'original_train_bundle_sha256': original_sha,
            'repaired_train_bundle_path': str(original_path),
            'repaired_train_bundle_sha256': new_sha,
            'original_child_receipt_path': str(backup_dir / 'REPLAY_RECEIPT.json'),
            'original_child_receipt_sha256': original_receipt_sha,
            'canonical_input_path': str(input_path), 'canonical_input_sha256': input_sha,
            'replay_contract_sha256': original_contract_sha,
            'repaired_fields': ['original_task_index', 'anchor_frame', 'anchor_timestamp'],
            'original_metadata_rows': len(dev_rows), 'repaired_metadata_rows': len(train_rows),
            'predictions_native_array_sha256_before': prediction_sha,
            'predictions_native_array_sha256_after': array_sha(arrays['predictions_native']),
            'predictions_bitwise_equal_to_locked_historical_gpu_reference': True,
            'anchor_ids_and_episode_indices_unchanged': True,
            'target_arrays_loaded': False, 'final_test_access': 0,
            'training': 0, 'optimizer_updates': 0, 'encoder_extraction': 0
        }
        repair_path = cell_dir / 'METADATA_REPAIR.json'
        write_json(repair_path, repair)
        receipt['metadata_only_repair'] = {
            'status': repair['status'], 'manifest_path': str(repair_path),
            'manifest_sha256': digest(repair_path), 'original_child_receipt_sha256': original_receipt_sha,
            'original_train_bundle_sha256': original_sha, 'repaired_train_bundle_sha256': new_sha,
            'predictions_native_array_sha256_unchanged': prediction_sha,
            'target_arrays_loaded': False, 'final_test_access': 0
        }
        write_json(receipt_path, receipt)
        repaired_cells.append(repair)

    summary = {
        'status': 'PASS_METADATA_ONLY_REPAIR', 'cell_count': len(repaired_cells),
        'train_bundle_count': len(repaired_cells), 'prediction_arrays_changed': 0,
        'metadata_fields_repaired': ['original_task_index', 'anchor_frame', 'anchor_timestamp'],
        'source_bug_fixed_for_future_replays': True,
        'initial_failure_evidence_dir': str(failure_dir),
        'repaired_cells': repaired_cells,
        'target_arrays_loaded': False, 'final_test_access': 0,
        'training': 0, 'optimizer_updates': 0, 'encoder_extraction': 0
    }
    write_json(audit / 'PREDICTION_METADATA_REPAIR_SUMMARY.json', summary)
    print(json.dumps({'status': summary['status'], 'cells': len(repaired_cells), 'prediction_arrays_changed': 0}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
