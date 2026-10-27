#!/usr/bin/env python3
"""Prepare scoped N2/N3 provenance and copy accepted source artifacts."""
from __future__ import annotations
import os

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
NEW = PROJECT / 'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'
N1 = PROJECT / 'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
NEXT = PROJECT / 'reports/benchmarks/CORE_B_POST_P1P2_MECHANISMS_V1_20261008T045413Z'
SCHED = PROJECT / 'reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z'
STATE = PROJECT / 'repo/ee6008-humanoid-vjepa/reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z'
V3 = PROJECT / 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
SEEDS_N1 = (20261001, 20261002, 20261003)
SEEDS_N3 = (20260953, 20260954, 20260955)
N3_CELLS = ('FULL-N', 'FULL-D', 'OBS-N', 'OBS-D', 'ANCHOR-N', 'ANCHOR-D')
source_rows: list[dict[str, str]] = []


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


def register_ref(src: Path, purpose: str, expected_sha: str | None = None) -> str:
    if not src.is_file():
        raise FileNotFoundError(src)
    digest = sha(src)
    if expected_sha and digest != expected_sha:
        raise RuntimeError(f'source hash mismatch: {src}')
    source_rows.append({'source_path': str(src), 'new_root_path': '', 'sha256': digest,
                        'copy_status': 'REFERENCED_READ_ONLY', 'purpose': purpose})
    return digest


def copy_ref(src: Path, rel: str, purpose: str, expected_sha: str | None = None) -> str:
    if not src.is_file():
        raise FileNotFoundError(src)
    digest = sha(src)
    if expected_sha and digest != expected_sha:
        raise RuntimeError(f'source hash mismatch: {src}')
    dst = NEW / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copy2(src, dst)
    if sha(dst) != digest:
        raise RuntimeError(f'copy hash mismatch: {dst}')
    source_rows.append({'source_path': str(src), 'new_root_path': str(dst.relative_to(NEW)),
                        'sha256': digest, 'copy_status': 'COPIED_HASH_MATCH', 'purpose': purpose})
    return digest


def parse_checksum_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split('  ', 1)
        if len(parts) == 2:
            out[parts[1].removeprefix('./')] = parts[0]
    return out


def main() -> int:
    if not NEW.is_dir():
        raise RuntimeError(f'new root missing: {NEW}')

    n1_manifest = NEXT / 'n1_refs/N1_ARTIFACT_MANIFEST.csv'
    n1_rows = read_csv(n1_manifest)
    n1_checkpoint_rows = [r for r in n1_rows if r['artifact_type'] == 'checkpoint' and r['scope'] in ('N1_Q0', 'N1_Q1')]
    if len(n1_checkpoint_rows) != 12:
        raise RuntimeError(f'expected 12 fixed N1 checkpoints, got {len(n1_checkpoint_rows)}')
    n1_matrix = read_csv(N1 / 'protocol/MATRIX12.csv')
    matrix_by_source = {r['source_run_id']: r for r in n1_matrix if r['condition'].startswith('Q0-')}
    model_registry: list[dict[str, Any]] = []
    n2_matrix: list[dict[str, Any]] = []
    asrun_rows: list[dict[str, Any]] = []
    for row in n1_checkpoint_rows:
        cp = Path(row['path'])
        actual_sha = sha(cp)
        if actual_sha != row['sha256']:
            raise RuntimeError(f'N1 checkpoint artifact manifest mismatch: {cp}')
        q = 'Q0' if row['scope'] == 'N1_Q0' else 'Q1'
        qvariant = 'old_fixed_sinusoidal_time_query' if q == 'Q0' else 'canonical_Hadamard_H30_RMS_matched'
        source_run = row['run_id']
        q0matrix = matrix_by_source.get(source_run) if q == 'Q0' else None
        if q == 'Q0' and q0matrix is None:
            raise RuntimeError(f'Q0 source run absent from locked matrix: {source_run}')
        n1_run_id = q0matrix['run_id'] if q0matrix else source_run
        model_registry.append({
            'n1_scope': row['scope'], 'model': row['model'], 'query': q, 'query_variant': qvariant,
            'seed': int(row['seed']), 'horizon_steps': 30, 'n1_run_id': n1_run_id,
            'checkpoint_source_run_id': source_run, 'checkpoint_path': str(cp),
            'checkpoint_sha256': actual_sha, 'manifest_status': row['status'],
            'safe_weights_only_load': 'BLOCKED_UNSUPPORTED_NUMPY_GLOBAL',
            'safe_model_export_found': 'false', 'final_test_access': 0,
        })
        for profile in ('CPU_COMMON', 'CUDA_COMMON'):
            n2_matrix.append({
                'view_id': f"{row['model']}__{q}__S{row['seed']}__{profile}",
                'model': row['model'], 'query': q, 'query_variant': qvariant, 'seed': int(row['seed']),
                'horizon_steps': 30, 'profile': profile, 'profile_role': 'primary' if profile == 'CUDA_COMMON' else 'sensitivity',
                'dev_windows_expected': 2322, 'dev_source_groups_expected': 603,
                'checkpoint_path': str(cp), 'checkpoint_sha256': actual_sha,
                'status': 'SOURCE_BLOCKED_SAFE_MODEL_EXPORT_UNAVAILABLE',
                'safe_checkpoint_load': 'BLOCKED_WEIGHTS_ONLY_12_OF_12', 'new_inference_started': 'false',
                'final_test_access': 0,
            })
        # Historical N1 profile rows are descriptive only; do not promote them to COMMON_PROFILE.
        if q == 'Q0':
            pred_base = SCHED / f"predictions/dev/{source_run}"
            sidecar = pred_base.with_suffix('.json')
        else:
            sidecar_entry = next((x for x in n1_rows if x['run_id'] == source_run and x['artifact_type'] == 'dev_sidecar'), None)
            sidecar = Path(sidecar_entry['path']) if sidecar_entry else Path('/nonexistent')
        profile_meta: dict[str, Any] = {}
        if sidecar.is_file():
            profile_meta = json.loads(sidecar.read_text(encoding='utf-8'))
            side_sha = sha(sidecar)
            copy_ref(sidecar, f"n1_refs/sidecars/{q}/{sidecar.name}", 'N1 accepted as-run profile sidecar')
        else:
            side_sha = ''
        runtime = profile_meta.get('runtime', {}) if isinstance(profile_meta.get('runtime'), dict) else {}
        asrun_rows.append({
            'model': row['model'], 'query': q, 'seed': int(row['seed']), 'n1_run_id': n1_run_id,
            'backend': profile_meta.get('device', 'CPU' if q == 'Q0' else 'CUDA from accepted N1 report'),
            'physical_gpu': profile_meta.get('physical_gpu', '0 from accepted N1 report' if q == 'Q1' else ''),
            'batch_size': profile_meta.get('batch_size', ''), 'dtype': profile_meta.get('dtype', 'float32' if profile_meta else ''),
            'tf32_matmul': runtime.get('tf32_matmul_enabled', 'unknown'),
            'tf32_cudnn': runtime.get('tf32_cudnn_enabled', 'unknown'),
            'sdpa_flash': runtime.get('sdpa_flash_enabled', 'unknown'),
            'sdpa_math': runtime.get('sdpa_math_enabled', 'unknown'),
            'deterministic_algorithms': runtime.get('deterministic_algorithms', 'unknown'),
            'torch_num_threads': runtime.get('torch_num_threads', 'unknown'),
            'sidecar_path': str(sidecar) if sidecar.is_file() else '', 'sidecar_sha256': side_sha,
            'role': 'AS_RUN_REPLAY_REFERENCE_ONLY', 'common_profile_equivalence': 'NOT_ESTABLISHED',
        })

    # Copy exact N1 locks, accepted metrics and implementation used as read-only references.
    for src, rel, why in (
        (n1_manifest, 'n1_refs/N1_ARTIFACT_MANIFEST.csv', 'accepted N1 artifact manifest'),
        (N1 / 'protocol/PROTOCOL_LOCKED.json', 'n1_refs/PROTOCOL_LOCKED.json', 'accepted N1 lock'),
        (N1 / 'protocol/MATRIX12.csv', 'n1_refs/MATRIX12.csv', 'accepted N1 matrix'),
        (N1 / 'qa/FINAL_ACCEPTANCE.json', 'n1_refs/FINAL_ACCEPTANCE.json', 'accepted N1 completion status'),
        (N1 / 'RETURN_TO_CHATGPT.zh-CN.md', 'n1_refs/RETURN_TO_CHATGPT.zh-CN.md', 'prior N1 report'),
        (N1 / 'results/TRAIN_DEV_BY_SEED.csv', 'n1_refs/TRAIN_DEV_BY_SEED.csv', 'accepted N1 train/dev metrics'),
        (N1 / 'results/TRAIN_CURVES.csv', 'n1_refs/TRAIN_CURVES.csv', 'accepted N1 training monitor curves'),
        (N1 / 'results/QUERY_GAINS.csv', 'n1_refs/QUERY_GAINS.csv', 'accepted N1 per-seed query gains'),
        (N1 / 'results/QUERY_MODEL_INTERACTION.csv', 'n1_refs/QUERY_MODEL_INTERACTION.csv', 'accepted N1 system interaction'),
        (N1 / 'results/BOOTSTRAP_CI.csv', 'n1_refs/BOOTSTRAP_CI.csv', 'accepted N1 group bootstrap'),
        (N1 / 'code/d3_model.py', 'code/n1_reference/d3_model.py', 'N1 fixed model implementation'),
        (N1 / 'code/fixed_future_query_model.py', 'code/n1_reference/fixed_future_query_model.py', 'N1 Q1 query wrapper'),
        (N1 / 'code/infer_full_query.py', 'code/n1_reference/infer_full_query.py', 'N1 target-blind inference source'),
        (N1 / 'code/analyze_full_query.py', 'code/n1_reference/analyze_full_query.py', 'N1 locked scoring source'),
        (N1 / 'code/train_full_query.py', 'code/n1_reference/train_full_query.py', 'N1 accepted model/checkpoint source'),
        (STATE / 'PROTOCOL_LOCKED.json', 'n3_refs/PROTOCOL_LOCKED.json', 'accepted N3 protocol'),
        (STATE / 'INPUT_MASK_CONTRACT.json', 'n3_refs/INPUT_MASK_CONTRACT.json', 'accepted N3 state/video mask contract'),
        (STATE / 'MATRIX_54.csv', 'n3_refs/MATRIX_54.csv', 'accepted N3 original matrix'),
        (STATE / 'analysis/PREDICTION_ACCEPTANCE.csv', 'n3_refs/PREDICTION_ACCEPTANCE.csv', 'accepted N3 prediction status'),
        (STATE / 'analysis/PREDICTION_ACCEPTANCE.json', 'n3_refs/PREDICTION_ACCEPTANCE.json', 'accepted N3 prediction summary'),
        (STATE / 'analysis/ANALYSIS_METADATA.json', 'n3_refs/ANALYSIS_METADATA.json', 'accepted N3 analysis metadata'),
        (STATE / 'analysis/RESULTS_BY_SEED.csv', 'n3_refs/RESULTS_BY_SEED.csv', 'accepted N3 score reference'),
        (STATE / 'analysis/VIDEO_GAINS_BY_STATE_LEVEL.csv', 'n3_refs/VIDEO_GAINS_BY_STATE_LEVEL.csv', 'accepted N3 video-gain reference'),
        (STATE / 'analysis/INFORMATION_INTERACTIONS.csv', 'n3_refs/INFORMATION_INTERACTIONS.csv', 'accepted N3 interaction reference'),
        (STATE / 'qa/CHECKSUMS.sha256', 'n3_refs/CHECKSUMS.sha256', 'accepted N3 source checksum manifest'),
        (STATE / 'code/analyze_state_video.py', 'code/n3_reference/analyze_state_video.py', 'N3 accepted scorer source'),
        (STATE / 'code/state_video_model.py', 'code/n3_reference/state_video_model.py', 'N3 locked masks source'),
        (STATE / 'code/model_v3_source.py', 'code/n3_reference/model_v3_source.py', 'N3 model source'),
        (V3 / 'code/prepare_v3_inputs.py', 'code/n3_reference/prepare_v3_inputs.py', 'N3 raw/schema feature assembly source'),
        (V3 / 'data/V3_NORMALIZERS.json', 'configs/n3/V3_NORMALIZERS.json', 'N3 train-fitted normalizers'),
        (PROJECT / 'repo/ee6008-humanoid-vjepa/docs/operations/GPU_WATCHER_POLICY.md', 'configs/GPU_WATCHER_POLICY.md', 'shared watcher policy reference'),
    ):
        copy_ref(src, rel, why)

    # Scoped file references: input NPZ is read only after accepted N3 predictions pass.
    input_path = V3 / 'data/V3_MAIN_INPUTS.npz'
    input_sha = register_ref(input_path, 'accepted canonical train/dev inputs; dev targets opened only after 18 bundle hash gates')
    norm_path = V3 / 'data/V3_NORMALIZERS.json'
    norm_sha = sha(norm_path)
    if norm_sha != '343c4428c2b42c14f5bec7eadb2c571541374ccfa485122f28d745c2d13c826c':
        raise RuntimeError('N3 train-fitted normalizer hash mismatch')
    group_map = N1 / 'inventory/SOURCE_GROUP_MAP.csv'
    group_sha = register_ref(group_map, 'canonical train/dev episode to source-group map')
    n3_acceptance = read_csv(STATE / 'analysis/PREDICTION_ACCEPTANCE.csv')
    rows = [r for r in n3_acceptance if int(r['horizon_steps']) == 30 and r['cell'] in N3_CELLS]
    if len(rows) != 18 or any(r['status'] != 'PASS' or r['final_test_access'] != '0' for r in rows):
        raise RuntimeError('N3 accepted H30 view count/status mismatch')
    checksum = parse_checksum_file(STATE / 'qa/CHECKSUMS.sha256')
    audit_rows: list[dict[str, Any]] = []
    n3_matrix: list[dict[str, Any]] = []
    pred_acceptance_rows: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: (int(x['seed']), N3_CELLS.index(x['cell']))):
        run_id = r['run_id']
        pred_dir = STATE / 'predictions' / run_id
        marker_path = pred_dir / 'PREDICTION_EXPORT_COMPLETE.json'
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        primary = pred_dir / 'primary.npz'
        primary_sha = sha(primary)
        rel_primary = str(primary.relative_to(STATE))
        if checksum.get(rel_primary) != primary_sha:
            raise RuntimeError(f'N3 primary checksum mismatch: {run_id}')
        checkpoint = Path(marker['checkpoint'])
        checkpoint_sha = sha(checkpoint)
        if checkpoint_sha != marker['checkpoint_sha256']:
            raise RuntimeError(f'N3 checkpoint/export hash mismatch: {run_id}')
        complete = json.loads((checkpoint.parent / 'RUN_COMPLETE.json').read_text(encoding='utf-8'))
        run_config = json.loads((checkpoint.parent / 'RUN_CONFIG_LOCKED.json').read_text(encoding='utf-8'))
        if (marker.get('status') != 'PASS' or marker.get('state_unchanged') is not True or
                marker.get('encoder_updated') is not False or marker.get('new_encoder_forward_for_extraction') is not False or
                marker.get('final_test_access') != 0 or complete.get('status') != 'PASS' or
                complete.get('epoch') != 40 or complete.get('optimizer_steps') != 11120 or complete.get('final_test_access') != 0):
            raise RuntimeError(f'N3 acceptance metadata mismatch: {run_id}')
        out_dir = NEW / 'n3_refs/predictions' / run_id
        copy_ref(primary, f'n3_refs/predictions/{run_id}/primary.npz', 'accepted N3 H30 dev predictions reused for scoring', primary_sha)
        copy_ref(marker_path, f'n3_refs/predictions/{run_id}/PREDICTION_EXPORT_COMPLETE.json', 'accepted N3 prediction export receipt')
        copy_ref(checkpoint.parent / 'RUN_COMPLETE.json', f'n3_refs/runs/{run_id}/RUN_COMPLETE.json', 'accepted N3 checkpoint completion receipt')
        copy_ref(checkpoint.parent / 'RUN_CONFIG_LOCKED.json', f'n3_refs/runs/{run_id}/RUN_CONFIG_LOCKED.json', 'accepted N3 run lock')
        cell = r['cell']; state_level, video_level = cell.split('-')
        seed = int(r['seed'])
        audit_rows.append({
            'run_id': run_id, 'state_level': state_level, 'video_level': video_level, 'seed': seed,
            'horizon_steps': 30, 'prediction_source': str(primary), 'new_root_copy': str((out_dir / 'primary.npz').relative_to(NEW)),
            'prediction_sha256': primary_sha, 'checkpoint_path': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
            'checkpoint_epoch': complete['epoch'], 'optimizer_steps': complete['optimizer_steps'],
            'state_hash_before': marker['state_hash_before'], 'state_hash_after': marker['state_hash_after'],
            'status': 'PASS_ACCEPTED_REUSE', 'final_test_access': 0,
        })
        n3_matrix.append({
            'view_id': run_id, 'state_level': state_level, 'video_level': video_level, 'seed': seed,
            'horizon_steps': 30, 'dev_windows': 2322, 'dev_episodes': 603, 'source_groups': 603,
            'action_columns_masked': 'none' if state_level == 'FULL' else '43-70' if state_level == 'OBS' else '0-70',
            'anchor_token_kept': 'true', 'prediction_sha256': primary_sha, 'checkpoint_sha256': checkpoint_sha,
            'reuse_status': 'accepted_existing_prediction', 'new_inference': 'false', 'new_training': 'false',
            'final_test_access': 0,
        })
        pred_acceptance_rows.append({'run_id': run_id, 'seed': seed, 'state_level': state_level, 'video_level': video_level,
                                     'horizon_steps': 30, 'rows': 2322, 'prediction_sha256': primary_sha,
                                     'checkpoint_sha256': checkpoint_sha, 'status': 'PASS_HASH_AND_RECEIPT',
                                     'target_arrays_opened_after_acceptance': 'yes', 'final_test_access': 0})
    write_csv(NEW / 'protocol/N2_MODEL_REGISTRY.csv', model_registry)
    write_csv(NEW / 'protocol/N2_MATRIX24.csv', n2_matrix)
    write_csv(NEW / 'protocol/AS_RUN_PROFILE_REGISTRY.csv', asrun_rows)
    write_csv(NEW / 'protocol/N3_MATRIX18.csv', n3_matrix)
    write_csv(NEW / 'protocol/N3_REUSE_AUDIT.csv', audit_rows)
    write_csv(NEW / 'inventory/SOURCE_REGISTRY.csv', source_rows)
    write_csv(NEW / 'qa/PREDICTION_ACCEPTANCE.csv', pred_acceptance_rows)
    write_json(NEW / 'qa/CHECKPOINT_SAFE_LOAD.json', {
        'status': 'BLOCKED_NO_SAFE_MODEL_EXPORT', 'torch': '2.12.1+cu130',
        'attempt': 'torch.load(map_location="cpu", weights_only=True)', 'n1_fixed_checkpoints_checked': 12,
        'safe_load_failures': 12, 'safe_exports_found': 0,
        'failure': 'UnpicklingError: unsupported global numpy._core.multiarray._reconstruct',
        'unsafe_weights_only_false_used': False, 'numpy_allowlist_added': False,
        'impact': 'N2 common-profile CPU/CUDA inference was not started; no N2 outputs are accepted',
        'final_test_access': 0,
    })
    write_json(NEW / 'protocol/N2_PROFILE_LOCK.json', {
        'status': 'LOCKED_BUT_NOT_EXECUTED_SOURCE_BLOCKED', 'primary_profile': 'CUDA_COMMON',
        'sensitivity_profile': 'CPU_COMMON', 'torch': '2.12.1+cu130', 'model_dtype': 'float32',
        'batch_size': 64, 'row_order': 'canonical accepted dev row order', 'eval_mode': True,
        'inference_mode': True, 'amp': False, 'compile': False,
        'cpu': {'threads': 4, 'interop_threads': 1, 'profile_status': 'planned_not_executed'},
        'cuda': {'tf32_matmul': False, 'tf32_cudnn': False, 'cudnn_benchmark': False,
                 'deterministic_algorithms': True, 'sdpa_backend': 'math_only_when_supported',
                 'cuda_visible_devices': 'set by shared watcher', 'physical_gpu_pool': 'dynamic_all_visible',
                 'same_gpu_for_all_12_checkpoints': 'preferred'},
        'historical_q0': 'CPU; exact thread/BLAS profile not fully recorded in sidecar',
        'historical_q1': 'CUDA physical GPU 0; FP32/no AMP/no compile; TF32 matmul false, TF32 cuDNN true, deterministic algorithms false, flash and memory-efficient SDPA enabled',
        'resource_peak_source': 'N1 accepted full inference peak 86 MiB at batch 64; applies as prior estimate only, not new profile measurement',
        'blocker': 'all 12 accepted checkpoints fail weights_only=True and no safe model-only export is available',
        'final_test_access': 0,
    })
    write_json(NEW / 'protocol/PROTOCOL_LOCKED.json', {
        'protocol_id': 'CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1', 'created_utc': '2026-10-11T14:05:43Z',
        'scope': 'N2 fixed checkpoint backend bridge plus N3 accepted H30 state-information recovery',
        'new_training': 0, 'optimizer_steps': 0, 'new_encoder_extraction': 0,
        'n0_rerun': False, 'n1_retrained': False, 'n2_expected_views': 24,
        'n3_expected_views': 18, 'n3_seed_source': 'STATE_ROOT/SEED_REGISTRY.json',
        'n3_seeds': list(SEEDS_N3), 'n3_cells': list(N3_CELLS),
        'final_test_access': 0, 'old_root_writes': 0,
        'status': 'PARTIAL_N2_SAFE_CHECKPOINT_SOURCE_BLOCKER_N3_REUSE_ACCEPTED',
    })
    write_json(NEW / 'protocol/ANALYSIS_LOCKED.json', {
        'N2': {'primary': 'CUDA_COMMON', 'sensitivity': 'CPU_COMMON',
               'definitions': {'G_m': 'L_m,Q0 - L_m,Q1', 'V_q': 'L_STATE,q - L_TS11D,q',
                               'I': 'G_TS11D - G_STATE_CTRL', 'backend_drift': 'B_m,q = L_m,q,CUDA - L_m,q,CPU'},
               'status': 'NOT_COMPUTABLE_SOURCE_BLOCKED'},
        'N3': {'main_metric': 'full_mae_std from original accepted N3 scorer',
               'video_gain': 'g_level = L_level,N - L_level,D; positive favors dynamic video',
               'h_command': 'L_OBS,N - L_FULL,N', 'h_observation': 'L_ANCHOR,N - L_OBS,N',
               'J_OBS': 'g_OBS - g_FULL', 'J_ANCHOR': 'g_ANCHOR - g_FULL',
               'aggregation': 'window mean within episode, episode mean within source group, equal source-group mean',
               'bootstrap': {'unit': 'paired source group', 'replicates': 10000, 'prng_seed': 20261011,
                             'draws_shared_across_all_cells_seeds_metrics': True,
                             'scope': 'conditional on fixed three accepted seeds and previously exposed dev'},
               'seed_summary': 'per-seed, mean, sample SD, median, and all three leave-one-seed-out means',
               'final_test_access': 0},
    })
    write_json(NEW / 'protocol/OUTCOME_EXPOSURE.json', {
        'development_split': 'previously exposed primary dev', 'windows': 2322, 'episodes': 603,
        'source_groups': 603, 'exploratory_rerevaluation': True, 'independent_confirmation': False,
        'final_test_access': 0, 'no_new_model_selection': True,
    })
    write_csv(NEW / 'inventory/N1_TRAIN_METRIC_RECOVERY.csv', [
        {**r, 'source_file_sha256': sha(N1 / 'results/TRAIN_DEV_BY_SEED.csv'),
         'interpretation': 'accepted in-sample training score; not held-out generalization'}
        for r in read_csv(N1 / 'results/TRAIN_DEV_BY_SEED.csv') if r['split'] == 'train'
    ])
    n1_boot = read_csv(N1 / 'results/BOOTSTRAP_CI.csv')
    n1_effect_rows = [r for r in n1_boot if r['split'] == 'dev']
    write_csv(NEW / 'results/N1_AS_RUN_REFERENCE.csv', [
        {**r, 'historical_backend': 'Q0 CPU; Q1 CUDA physical GPU 0',
         'common_profile_status': 'NOT_MEASURED_THIS_RUN',
         'interpretation': 'previous accepted N1 result; cannot identify CPU/CUDA drift'}
        for r in n1_effect_rows
    ])
    write_csv(NEW / 'results/N2_PROFILE_DRIFT_DECOMPOSITION.csv', [
        {**r, 'common_profile_estimate': '', 'profile_drift_estimate': '',
         'decomposition_status': 'NOT_COMPUTABLE_SAFE_CHECKPOINT_EXPORT_MISSING'}
        for r in n1_effect_rows
    ])
    write_csv(NEW / 'results/N2_CPU_CUDA_DRIFT.csv', [
        {'model': r['model'], 'query': r['query'], 'seed': r['seed'], 'checkpoint_sha256': r['checkpoint_sha256'],
         'historical_backend': 'CPU' if r['query'] == 'Q0' else 'CUDA physical GPU 0',
         'CPU_COMMON_status': 'NOT_RUN_SOURCE_BLOCKED', 'CUDA_COMMON_status': 'NOT_RUN_SOURCE_BLOCKED',
         'CPU_to_CUDA_delta': '', 'common_profile_status': 'UNAVAILABLE'}
        for r in model_registry
    ])
    write_csv(NEW / 'results/N2_QUERY_SYSTEM_EFFECTS.csv', [
        {**r, 'controlled_common_profile_estimate': '', 'status': 'HISTORICAL_AS_RUN_ONLY'}
        for r in n1_effect_rows
    ])

    # VideoMAE is only an accepted background status; never load its checkpoint here.
    vm_root = PROJECT / 'reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z'
    vm_index = PROJECT / 'tmp/project-team/videomae_handoff_20261009T020906Z/HANDOFF_INDEX.json'
    vm_accept = vm_root / 'FINAL_ACCEPTANCE.json'
    vm_status = 'background_unknown'
    vm_accept_status = ''
    if vm_index.is_file() and vm_accept.is_file():
        vm = json.loads(vm_accept.read_text(encoding='utf-8'))
        vm_status = 'accepted_background_not_comparable_leaderboard'
        vm_accept_status = vm.get('status', '')
        register_ref(vm_index, 'VideoMAE handoff index; background only')
        register_ref(vm_accept, 'VideoMAE accepted handoff status; background only')
    write_csv(NEW / 'inventory/BACKGROUND_BASELINE_STATUS.csv', [{
        'model': 'VideoMAE', 'status': vm_status, 'acceptance_status': vm_accept_status,
        'handoff_index': str(vm_index), 'acceptance_root': str(vm_root),
        'comparability': 'different feature layout/projection/pretraining; not merged into N1/N3 contrasts',
        'checkpoint_unpickle_attempted': 'false', 'final_test_access': 0,
    }])

    write_csv(NEW / 'inventory/SOURCE_REGISTRY.csv', source_rows)
    write_json(NEW / 'SOURCE_ACCEPTANCE.json', {
        'status': 'PARTIAL_N2_SAFE_CHECKPOINT_SOURCE_BLOCKER_N3_SOURCE_ACCEPTED',
        'N1_fixed_checkpoints': '12/12 manifest hashes match; all 12 fail weights_only safe load',
        'N3_primary_dev_predictions': '18/18 original checksum + export marker + checkpoint hash + RUN_COMPLETE checks pass',
        'N3_input_sha256': input_sha, 'N3_normalizer_sha256': norm_sha,
        'source_group_map_sha256': group_sha, 'final_test_access': 0,
        'old_root_mutation': 'none by this task; scoped source reads/copies only',
    })
    write_csv(NEW / 'coordination/RESOURCE_LOG.csv', [
        {'observed_utc': '2026-10-11T14:04:29Z', 'host': os.environ.get("EE6008_HOST_LABEL", "shared-host"), 'host_type': 'shared GPU host; no Slurm allocation detected',
         'gpu_physical': 0, 'uuid': 'GPU-3c8b646d-d1dc-1bc4-6e82-d96422d9f36b', 'model': 'NVIDIA GeForce RTX 3090', 'used_mib': 21, 'free_mib': 24107, 'utilization_pct': 0},
        {'observed_utc': '2026-10-11T14:04:29Z', 'host': os.environ.get("EE6008_HOST_LABEL", "shared-host"), 'host_type': 'shared GPU host; no Slurm allocation detected',
         'gpu_physical': 1, 'uuid': 'GPU-ceb12ae6-6ff4-4685-6309-544becf58fe6', 'model': 'NVIDIA GeForce RTX 3090', 'used_mib': 16, 'free_mib': 24111, 'utilization_pct': 0},
        {'observed_utc': '2026-10-11T14:04:29Z', 'host': os.environ.get("EE6008_HOST_LABEL", "shared-host"), 'host_type': 'shared GPU host; no Slurm allocation detected',
         'gpu_physical': 2, 'uuid': 'GPU-00465dc6-e96e-67f1-285a-84ed61565ac2', 'model': 'NVIDIA GeForce RTX 3090', 'used_mib': 20, 'free_mib': 24107, 'utilization_pct': 0},
        {'observed_utc': '2026-10-11T14:04:29Z', 'host': os.environ.get("EE6008_HOST_LABEL", "shared-host"), 'host_type': 'shared GPU host; no Slurm allocation detected',
         'gpu_physical': 3, 'uuid': 'GPU-b9c07d60-9a04-0693-af83-6976ba6d4dfb', 'model': 'NVIDIA GeForce RTX 3090', 'used_mib': 22310, 'free_mib': 1817, 'utilization_pct': 0},
        {'observed_utc': '2026-10-11T14:04:29Z', 'host': os.environ.get("EE6008_HOST_LABEL", "shared-host"), 'host_type': 'shared GPU host; no Slurm allocation detected',
         'gpu_physical': '', 'uuid': '', 'model': 'Torch 2.12.1+cu130 CUDA init', 'used_mib': '', 'free_mib': '', 'utilization_pct': '',
         'runtime_usable': 'true; 4 CUDA devices visible', 'note': 'project env host preflight only; no workload started'},
        {'observed_utc': '2026-10-11T14:04:29Z', 'host': os.environ.get("EE6008_HOST_LABEL", "shared-host"), 'host_type': 'shared GPU host; no Slurm allocation detected',
         'gpu_physical': '', 'uuid': '', 'model': 'Filesystem /usr1', 'used_mib': '', 'free_mib': '', 'utilization_pct': '',
         'runtime_usable': '', 'note': '5.8 TiB free; quota CLI unavailable; user quota unknown'},
    ])
    write_json(NEW / 'coordination/JOB_LEDGER.json', {
        'jobs': [
            {'module': 'N2', 'status': 'source_blocked_before_child_spawn', 'gpu_child_started': False,
             'reason': '12/12 checkpoints fail weights_only=True; no safe model-only exports; unsafe load and allowlists prohibited',
             'shared_watcher_modified': False, 'final_test_access': 0},
            {'module': 'N3', 'status': 'accepted_predictions_reused; CPU rescoring pending', 'new_training': 0,
             'new_inference': 0, 'new_encoder_extraction': 0, 'final_test_access': 0},
        ], 'max_project_gpu_children_policy': 2, 'actual_new_gpu_children': 0,
    })
    write_csv(NEW / 'coordination/RESOURCE_USAGE.csv', [{
        'n2_gpu_child_hours': 0, 'n3_gpu_child_hours': 0, 'total_new_gpu_child_hours': 0,
        'n2_new_inference_views': 0, 'n3_new_inference_views': 0, 'new_training': 0,
        'new_encoder_extraction': 0, 'note': 'No GPU workload launched because N2 safe checkpoint source gate failed; N3 uses accepted predictions'
    }])
    (NEW / 'coordination/RESUME_COMMANDS.md').write_text(
        '# Safe continuation\n\nN3 scoring is independent and can complete now. N2 can resume without training once the trusted producer supplies a tensor-only or safetensors model export for each of the 12 accepted checkpoints, with source checkpoint SHA and parameter/buffer hashes tied to the N1 acceptance manifest. Then revalidate identity and run the locked 24-view CPU_COMMON/CUDA_COMMON matrix through the existing shared watcher. Do not use `weights_only=False` or add unreviewed pickle allowlists.\n',
        encoding='utf-8')
    (NEW / 'CONTEXT_RESTORED.zh-CN.md').write_text(
        '# N2/N3 续跑状态\n\n- 已恢复 N1、STATE、N3 18 个 H30 dev bundle 的接受清单。STATE H30 的 18 个目标视图均通过旧 root checksum、checkpoint hash、RUN_COMPLETE 和 final-test denial 的逐项检查。\n- N2 仍保留原 12 checkpoint × CPU/CUDA 计划，但 12 个 checkpoint 全部被 `torch.load(weights_only=True)` 拒绝，原因是 checkpoint 内含未默认允许的 NumPy global，且无安全 model-only export。未运行 unsafe pickle、未加 allowlist、未启动 CUDA 子进程。\n- N3 可沿原 train-fitted normalizer 和接受 scorer 在 primary dev 重评分。source-group 与 episode 映射为 603 对 603 一一对应。\n- 输出只写本新 root；N1、SCHED、STATE、P1、NEXT 根按只读方式使用。\n', encoding='utf-8')
    (NEW / 'protocol/DEVIATIONS.jsonl').write_text(
        json.dumps({'time_utc':'2026-10-11T14:05:43Z','deviation':'initial nested mkdir failed because parents were absent; completed with mkdir -p in the same new root','impact':'no prior files changed'}, ensure_ascii=False)+'\n'+
        json.dumps({'time_utc':datetime.now(timezone.utc).isoformat(),'deviation':'N2 safely blocked before inference because all 12 checkpoints reject weights_only=True and no safe model-only export is present','impact':'24 common-profile views not produced; N3 continues using accepted predictions'}, ensure_ascii=False)+'\n',
        encoding='utf-8')
    print(json.dumps({'status':'PASS_PREPARED','root':str(NEW),'N2_checkpoints':len(model_registry),
                      'N2_matrix_views':len(n2_matrix),'N3_accepted_views':len(audit_rows),
                      'source_registry_rows':len(source_rows),'target_arrays_opened':False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
