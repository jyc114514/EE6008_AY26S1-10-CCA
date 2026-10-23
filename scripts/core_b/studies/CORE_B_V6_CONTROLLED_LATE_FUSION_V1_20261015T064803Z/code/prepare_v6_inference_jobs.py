#!/usr/bin/env python3
"""Build V6-owned watcher jobs for the three accepted screen-seed tune replays."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2 = PROJECT / 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
REPO = PROJECT / 'repo/ee6008-humanoid-vjepa'
PYTHON = PROJECT / 'envs/jyc/bin/python'
WATCHER = REPO / 'tools/gpu_lease_watcher.py'
WATCHER_SHA = 'b703ef125ae481819db4af20c66a2a13906098e6522fe24fe2ce1735164355a5'
ROLES = {
    'S_FIT': ('v2_screening_G_S_lr0p001_s202610130', 'STATE-CTRL', 'G-S'),
    'A_FIT': ('v2_screening_G_S_lr0p0005_s202610130', 'STATE-CTRL', 'G-S'),
    'V_FIT': ('v2_screening_G_V_lr0p0005_s202610130', 'TS11-D', 'G-V'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def main():
    if sha(WATCHER) != WATCHER_SHA:
        raise RuntimeError('shared watcher identity hash mismatch')
    if not (ROOT / 'protocol/PROGRAM_LOCK.json').is_file():
        raise RuntimeError('PROGRAM_LOCK must precede tune prediction jobs')
    source = json.loads((ROOT / 'inventory/SOURCE_MAP.json').read_text())
    manifest = json.loads((V2 / 'inventory/RESOLVED_PATHS.json').read_text())
    if manifest['feature_cache_manifest_sha256'] != source['feature_cache']['manifest_sha256']:
        raise RuntimeError('accepted V2 Q0 cache manifest changed')

    rows = []
    for role, (training_job, video_mode, model_id) in ROLES.items():
        model_dir = V2 / 'model_only/screening' / training_job
        model_path = model_dir / 'model.safetensors'
        model_manifest = model_dir / 'MODEL_MANIFEST.json'
        if sha(model_path) != json.loads(model_manifest.read_text())['model_file_sha256']:
            raise RuntimeError(f'{role} model hash changed after preflight')
        job_id = f'v6_inner_tune_{role}_s202610130'
        output = ROOT / 'predictions/inner_tune' / role
        inference_fit = {
            'schema_version': 1, 'job_id': job_id, 'phase': 'v6_inner_tune_calibration_replay',
            'model_id': role, 'seed': 202610130, 'scope': 'shared', 'task_filter': None,
            'training_job_id': training_job, 'dataset_kind': 'train', 'dataset_role': 'inner_tune',
            'data_path': str(V2 / 'data/TRAIN_SELECTED_H30.npz'),
            'model_path': str(model_path), 'model_manifest_path': str(model_manifest),
            'normalizer_path': str(V2 / 'screening/NORMALIZER_SCREENING.json'),
            'video_mode': video_mode, 'waypoint_mode': None, 'batch_size': 32,
            'output_root': str(output), 'final_heldout_payload_access': 0,
            'future_target_arrays_requested': False,
        }
        fit_path = ROOT / 'coordination/jobs' / f'{job_id}.fit.json'
        write_json(fit_path, inference_fit)
        # The worst prior accepted same-script screen replay peak was 102 MiB at batch=32.
        # The 270-row split increases host-side row count, while the GPU batch contract is unchanged.
        queue_job = {
            'schema_version': 1, 'job_id': job_id,
            'protocol_id': 'CORE_B_V6_INNER_TUNE_FROZEN_REPLAY',
            'output_root': str(output),
            'command': [str(PYTHON), str(V2 / 'code/v2_inference.py'), '--job-json', str(fit_path)],
            'resource_estimate': {
                'measured_peak_mib': 102,
                'estimate_basis': 'accepted V2 same-script same-batch G-V inference observed 102 MiB reserved; V6 batch remains 32',
                'measurement_source': str(V2 / 'predictions/dev/v2_formal_G_V_lr0p0005_s202610131/PREDICTION_COMPLETE.json'),
            },
            'log_path': str(ROOT / 'logs' / f'{job_id}.child.log'),
            'environment': {'PYTHONDONTWRITEBYTECODE': '1'},
        }
        job_path = ROOT / 'coordination/jobs' / f'{job_id}.watcher.json'
        write_json(job_path, queue_job)
        rows.append({
            'role': role, 'job_id': job_id, 'training_job_id': training_job,
            'video_mode': video_mode, 'model_id': model_id, 'seed': 202610130,
            'fit_config_path': str(fit_path), 'fit_config_sha256': sha(fit_path),
            'watcher_job_path': str(job_path), 'watcher_job_sha256': sha(job_path),
            'output_root': str(output), 'resource_estimate_mib': 102,
            'resource_basis': 'same-script accepted peak at same batch size',
        })
    write_json(ROOT / 'protocol/INNER_TUNE_REPLAY_PLAN.json', {
        'schema_version': 1, 'status': 'LOCKED', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'watcher_path': str(WATCHER), 'watcher_sha256': WATCHER_SHA,
        'jobs': rows, 'serial_submission': True,
        'reused_checkpoints': True, 'new_neural_fits': 0,
        'tune_target_arrays_requested_by_inference': False,
        'screen_seed': 202610130, 'tune_anchors': 270,
        'future_target_access': 'deferred to V6 CPU calibration after frozen prediction bundle completion',
    })
    print(json.dumps({'status': 'PASS', 'jobs': len(rows), 'outputs': [r['output_root'] for r in rows],
                      'watcher_sha256': WATCHER_SHA}, sort_keys=True))


if __name__ == '__main__':
    main()
