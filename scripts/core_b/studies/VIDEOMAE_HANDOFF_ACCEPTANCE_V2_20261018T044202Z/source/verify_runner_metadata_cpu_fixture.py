#!/usr/bin/env python3
"""CPU-only regression for replay_one_cell's shared split-bundle serializer."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def array_sha(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-root', type=Path, required=True)
    args = ap.parse_args()
    audit = args.audit_root.resolve()
    runner_path = audit / 'source/replay_one_cell.py'
    runner_sha = file_sha(runner_path)
    spec = importlib.util.spec_from_file_location('videomae_replay_runner_under_test', runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('could not load final replay runner source')
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    if not hasattr(runner, 'write_prediction_bundles'):
        raise AssertionError('final runner does not expose its shared split-bundle serializer')

    # Interleaved canonical indices make stale last-loop-row reuse observable.
    anchor_ids = np.asarray([
        'TRAIN-ID-A0', 'DEV-ID-B0', 'TRAIN-ID-A1', 'DEV-ID-B1',
        'TRAIN-ID-A2', 'DEV-ID-B2', 'TRAIN-ID-A3', 'TRAIN-ID-A4'
    ], dtype='U96')
    episodes = np.asarray([101, 201, 102, 202, 103, 203, 104, 105], dtype=np.int64)
    task_index = np.asarray([1101, 2201, 1102, 2202, 1103, 2203, 1104, 1105], dtype=np.int64)
    anchor_frame = np.asarray([10001, 20001, 10002, 20002, 10003, 20003, 10004, 10005], dtype=np.int64)
    anchor_timestamp = np.asarray([1.001, 2.001, 1.002, 2.002, 1.003, 2.003, 1.004, 1.005], dtype=np.float64)
    rows_by_split = {
        'train': np.asarray([0, 2, 4, 6, 7], dtype=np.int64),
        'dev': np.asarray([1, 3, 5], dtype=np.int64)
    }
    assert len(rows_by_split['train']) == 5 and len(rows_by_split['dev']) == 3
    train_id_set = set(anchor_ids[rows_by_split['train']].tolist())
    dev_id_set = set(anchor_ids[rows_by_split['dev']].tolist())
    assert train_id_set.isdisjoint(dev_id_set)

    predictions = {
        'train': (np.arange(5 * 3 * 28, dtype=np.float32).reshape(5, 3, 28) / 17.0),
        'dev': (1000.0 + np.arange(3 * 3 * 28, dtype=np.float32).reshape(3, 3, 28) / 19.0)
    }
    prediction_hashes_before = {key: array_sha(value) for key, value in predictions.items()}
    input_shas = {}
    for split, rows in rows_by_split.items():
        input_shas[split] = {
            'ids_sha256': array_sha(anchor_ids[rows].astype('U96')),
            'episode_ids_sha256': array_sha(episodes[rows]),
            'row_indices_sha256': array_sha(rows.astype(np.int64))
        }

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_root = audit / 'replay' / f'runner_metadata_cpu_fixture_{run_id}'
    output_root.mkdir(parents=True, exist_ok=False)
    if any(output_root.iterdir()):
        raise AssertionError('CPU fixture output directory was not empty before bundle generation')
    temp_dir = audit / 'tmp'
    if not temp_dir.is_dir():
        raise FileNotFoundError(temp_dir)

    records = runner.write_prediction_bundles(
        output_root, temp_dir, predictions, rows_by_split, anchor_ids, episodes,
        task_index, anchor_frame, anchor_timestamp, 3, 'CPU_FIXTURE', 314159, input_shas
    )
    if {x['split'] for x in records} != {'train', 'dev'}:
        raise AssertionError('writer did not emit exactly the train/dev fixture bundles')

    split_checks = []
    for split, rows in rows_by_split.items():
        path = output_root / f'predictions_{split}.npz'
        with np.load(path, allow_pickle=False) as saved:
            checks = {
                'anchor_id': np.array_equal(saved['anchor_id'].astype('U96'), anchor_ids[rows]),
                'episode_index': np.array_equal(saved['episode_index'].astype(np.int64), episodes[rows]),
                'original_task_index': np.array_equal(saved['original_task_index'].astype(np.int64), task_index[rows]),
                'anchor_frame': np.array_equal(saved['anchor_frame'].astype(np.int64), anchor_frame[rows]),
                'anchor_timestamp': np.array_equal(saved['anchor_timestamp'].astype(np.float64), anchor_timestamp[rows]),
                'predictions_native': np.array_equal(saved['predictions_native'], predictions[split]),
                'prediction_shape': tuple(saved['predictions_native'].shape) == tuple(predictions[split].shape),
                'prediction_dtype': saved['predictions_native'].dtype == predictions[split].dtype,
                'metadata_split': str(saved['split'].item()) == split,
                'metadata_horizon': int(saved['horizon_steps'].item()) == 3,
                'metadata_config': str(saved['config'].item()) == 'CPU_FIXTURE',
                'metadata_seed': int(saved['seed'].item()) == 314159
            }
            prediction_hash_after = array_sha(saved['predictions_native'])
        if not all(checks.values()):
            raise AssertionError(f'{split} fixture failed: {checks}')
        if prediction_hash_after != prediction_hashes_before[split]:
            raise AssertionError(f'{split} prediction array hash changed during serialization')
        record = next(x for x in records if x['split'] == split)
        if record['sha256'] != file_sha(path) or record['rows'] != len(rows):
            raise AssertionError(f'{split} output record does not bind the generated bundle')
        split_checks.append({
            'split': split, 'rows': len(rows), 'ids': anchor_ids[rows].tolist(),
            'ids_sha256': input_shas[split]['ids_sha256'],
            'task_index_sha256': array_sha(task_index[rows]),
            'anchor_frame_sha256': array_sha(anchor_frame[rows]),
            'anchor_timestamp_sha256': array_sha(anchor_timestamp[rows]),
            'prediction_array_sha256_before': prediction_hashes_before[split],
            'prediction_array_sha256_after': prediction_hash_after,
            'metadata_checks': checks, 'bundle_path': str(path),
            'bundle_sha256': file_sha(path), 'bundle_bytes': path.stat().st_size
        })

    receipt = {
        'status': 'PASS_POST_FIX_RUNNER_CPU_REGRESSION',
        'created_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'runner_path': str(runner_path), 'runner_sha256': runner_sha,
        'test_source_path': str(Path(__file__).resolve()), 'test_source_sha256': file_sha(Path(__file__).resolve()),
        'test_method': 'call the final runner shared write_prediction_bundles function used by main()',
        'fixture_kind': 'small synthetic canonical-row CPU fixture; no checkpoint, target, model forward, or GPU',
        'output_directory_empty_before_generation': True,
        'train_rows': 5, 'dev_rows': 3, 'train_dev_lengths_differ': True,
        'train_dev_ids_disjoint': True, 'metadata_fields_checked': ['original_task_index', 'anchor_frame', 'anchor_timestamp'],
        'all_metadata_checks_pass': True, 'prediction_arrays_bitwise_unchanged': True,
        'prediction_arrays_changed': 0, 'splits': split_checks,
        'gpu_accessed': False, 'targets_loaded': False, 'final_test_access': 0,
        'training': 0, 'optimizer_updates': 0, 'encoder_extraction': 0
    }
    receipt_path = audit / 'RUNNER_METADATA_CPU_REGRESSION.json'
    write_json(receipt_path, receipt)
    print(json.dumps({'status': receipt['status'], 'runner_sha256': runner_sha,
                      'train_rows': 5, 'dev_rows': 3, 'metadata': 'PASS',
                      'predictions_bitwise_unchanged': True, 'receipt': str(receipt_path)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
