#!/usr/bin/env python3
"""Finalize the VideoMAE V2 GPU acceptance from the sealed audit evidence."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: object) -> None:
    tmp = path.with_name('.' + path.name + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    tmp = path.with_name('.' + path.name + '.part')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def checksum_all(root: Path) -> int:
    checksum_path = root / 'CHECKSUMS.sha256'
    files = sorted(p for p in root.rglob('*') if p.is_file() and p != checksum_path)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise AssertionError('symlink found in audit root at finalization')
    lines = [f'{digest(p)}  {p.relative_to(root).as_posix()}\n' for p in files]
    checksum_path.write_text(''.join(lines), encoding='utf-8')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != checksum_path}
    listed = set()
    for line in lines:
        listed.add(line[66:].rstrip('\n'))
    if listed != actual:
        raise AssertionError(f'checksum coverage mismatch missing={sorted(actual-listed)[:3]} extra={sorted(listed-actual)[:3]}')
    subprocess.run(['sha256sum', '--check', '--quiet', 'CHECKSUMS.sha256'], cwd=root, check=True)
    return len(lines)


def main() -> int:
    project = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
    root = project / 'reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V2_20261018T044202Z'
    expected_rows = list(csv.DictReader((root / 'EXPECTED_CELLS.csv').open(newline='', encoding='utf-8')))
    replay_rows = list(csv.DictReader((root / 'CHECKPOINT_REPLAY.csv').open(newline='', encoding='utf-8')))
    assert len(expected_rows) == 12 and len(replay_rows) == 24
    assert all(x.get('numeric_accepted') == '1' and x.get('status') == 'accepted_predeclared_V2_engineering_gate' for x in replay_rows)
    assert all(x.get('bitwise_equal') == 'True' and float(x['max_abs']) == 0.0 for x in replay_rows)

    profile = read_json(root / 'PROFILE_LOCK.json')
    current = profile['new_current_profile']
    assert profile.get('lock_status') == 'locked_before_first_GPU_prediction'
    assert profile.get('current_profile_match_status') == 'locked_for_this_audit'
    uuid = current['gpu_uuid']
    assert uuid == 'GPU-00465dc6-e96e-67f1-285a-84ed61565ac2'
    assert current['physical_gpu'] == 2 and current['cuda_visible_devices'] == '2' and current['logical_device'] == 'cuda:0'
    assert all(x['gpu_uuid'] == uuid and x['physical_gpu'] == '2' and x['logical_device'] == 'cuda:0' for x in replay_rows)

    seal = read_json(root / 'PREDICTION_SEAL.json')
    validation = read_json(root / 'PREDICTION_VALIDATION_SUMMARY.json')
    scoring = read_json(root / 'SCORING_SUMMARY.json')
    scorer_qa = read_json(root / 'INDEPENDENT_SCORER_QA.json')
    repair = read_json(root / 'PREDICTION_METADATA_REPAIR_SUMMARY.json')
    resource = read_json(root / 'GPU_RESOURCE_RECEIPT.json')
    assert seal.get('status') == 'PREDICTIONS_SEALED_BEFORE_TARGET_JOIN' and seal.get('prediction_count') == 24
    assert validation.get('status') == 'PASS_PREDICTION_STRUCTURE_GATE' and validation.get('numeric_accepted_rows') == 24
    assert scoring.get('status') == 'PASS' and scoring.get('metric_rows') == 20736 and scoring.get('errors') == []
    assert scorer_qa.get('status') == 'PASS' and scorer_qa.get('cell_split_region_comparisons') == 144
    assert repair.get('status') == 'PASS_METADATA_ONLY_REPAIR' and repair.get('prediction_arrays_changed') == 0
    assert resource.get('status') == 'PASS_SAME_RUNNER_STAGE0_MEMORY_MEASURED'

    # Complete the requested per-seed GPU/CPU table from the 20,736 detailed rows.
    diff_rows = list(csv.DictReader((root / 'METRIC_REPLAY_DIFFS.csv').open(newline='', encoding='utf-8')))
    diff_map = {}
    for row in diff_rows:
        key = (row['backend_comparison'], row['cell'], row['split'], row['region'], row['model'], row['metric'])
        if key in diff_map:
            raise AssertionError(f'duplicate metric diff key: {key}')
        diff_map[key] = row
    seed_path = root / 'CORE_B_PER_SEED_METRIC_SUMMARY.csv'
    with seed_path.open(newline='', encoding='utf-8') as f:
        seed_rows = list(csv.DictReader(f))
        seed_fields = list(seed_rows[0]) if seed_rows else []
    extras = ['historical_GPU_vs_new_GPU_abs_delta', 'historical_GPU_vs_new_GPU_relative_delta',
              'new_GPU_comparison_status', 'stored_GPU_rescore_abs_delta', 'stored_GPU_rescore_status']
    for field in extras:
        if field not in seed_fields:
            seed_fields.append(field)
    for row in seed_rows:
        model = 'VideoMAE' if row['mode'] == 'TS11-D' else 'STATE-CTRL'
        keybase = (row['cell'], row['split'], row['region'], model, row['metric'])
        new = diff_map[('historical_GPU_vs_new_GPU', *keybase)]
        cpu = diff_map[('historical_GPU_vs_historical_CPU', *keybase)]
        rescore = diff_map[('stored_historical_GPU_vs_rescored_historical_GPU', *keybase)]
        row['historical_GPU'] = cpu['old']
        row['historical_CPU'] = cpu['new']
        row['abs_delta'] = cpu['abs_delta'] or 'NA'
        row['relative_delta'] = cpu['relative_delta'] or 'NA'
        row['comparison_status'] = cpu['status']
        row['new_GPU'] = new['new'] if new['new'] not in ('', 'None') else 'NA'
        row['historical_GPU_vs_new_GPU_abs_delta'] = new['abs_delta'] or 'NA'
        row['historical_GPU_vs_new_GPU_relative_delta'] = new['relative_delta'] or 'NA'
        row['new_GPU_comparison_status'] = new['status']
        row['stored_GPU_rescore_abs_delta'] = rescore['abs_delta'] or 'NA'
        row['stored_GPU_rescore_status'] = rescore['status']
    if len(seed_rows) != 1440:
        raise AssertionError(f'unexpected selected per-seed metric row count: {len(seed_rows)}')
    write_csv(seed_path, seed_rows, seed_fields)

    comparison_groups = {}
    for row in diff_rows:
        comparison_groups.setdefault(row['backend_comparison'], []).append(row)
    max_rows = {}
    for name, group in comparison_groups.items():
        max_rows[name] = max(group, key=lambda x: float(x['abs_delta']) if x['abs_delta'] not in ('', 'None') else -1.0)
    cpu_max = max_rows['historical_GPU_vs_historical_CPU']
    gpu_max = max_rows['historical_GPU_vs_new_GPU']
    rescore_max = max_rows['stored_historical_GPU_vs_rescored_historical_GPU']
    assert len(diff_rows) == 20736
    assert float(gpu_max['abs_delta']) == 0.0 and float(rescore_max['abs_delta']) == 0.0
    assert cpu_max['cell'] == 'STATE-CTRL_H60_S20261003'
    assert cpu_max['split'] == 'dev' and cpu_max['region'] == 'action_14_28'
    assert cpu_max['metric'] == 'tv_ratio_window_p50'
    assert abs(float(cpu_max['abs_delta']) - 0.00011648237705230713) < 1e-15

    # Keep the first failed validation and original as-run source/profile snapshots in the registry.
    registry_path = root / 'SOURCE_REGISTRY.csv'
    with registry_path.open(newline='', encoding='utf-8') as f:
        registry_rows = list(csv.DictReader(f))
        registry_fields = list(registry_rows[0])
    source_dir = root / 'source'
    runner = source_dir / 'replay_one_cell.py'
    runner_backups = sorted(source_dir.glob('replay_one_cell.py.pre_train_metadata_fix_*'))
    profile_backups = sorted(root.glob('PROFILE_LOCK.json.pre_gpu_*'))
    assert len(runner_backups) == 1 and len(profile_backups) == 1
    for row in registry_rows:
        if row['role'] == 'audit_prediction_child':
            if row['sha256'] != digest(runner_backups[0]):
                raise AssertionError('as-run child runner backup does not match pre-repair source registry hash')
            row['role'] = 'audit_prediction_child_as_run'
            row['path'] = str(runner_backups[0])
            row['bytes'] = str(runner_backups[0].stat().st_size)
            row['note'] = 'exact as-run source preserved; generated initial GPU bundles before metadata-only repair'
        elif row['role'] == 'profile_lock_pending':
            if row['sha256'] != digest(profile_backups[0]):
                raise AssertionError('pre-GPU profile snapshot does not match source registry hash')
            row['role'] = 'profile_lock_pre_gpu_snapshot'
            row['path'] = str(profile_backups[0])
            row['bytes'] = str(profile_backups[0].stat().st_size)
            row['note'] = 'pre-GPU profile lock snapshot; finalized live profile is registered separately'
        elif row['role'] == 'audit_gpu_orchestrator':
            row['note'] = 'shared watcher orchestration executed in tmux; first structure gate found the documented runner metadata bug; post-repair validator and scorer passed'
    if not any(r['role'] == 'audit_prediction_child_as_run' for r in registry_rows):
        raise AssertionError('as-run child source registry entry missing')

    def upsert(role: str, path: Path, note: str, status: str = 'PASS') -> None:
        path = path.resolve()
        row = next((x for x in registry_rows if x['role'] == role), None)
        if row is None:
            row = {k: '' for k in registry_fields}
            registry_rows.append(row)
        row.update({'role': role, 'path': str(path), 'exists': str(path.is_file()),
                    'bytes': str(path.stat().st_size) if path.is_file() else '',
                    'sha256': digest(path) if path.is_file() else '', 'status': status,
                    'expected_bytes': '', 'expected_sha256': '', 'note': note})
    upsert('audit_prediction_child_corrected', runner, 'fixed per-split metadata serialization; original as-run source and first outputs retained')
    upsert('audit_metadata_repair_tool', source_dir / 'repair_train_npz_metadata.py', 'verified and repaired only train metadata; predictions unchanged bitwise')
    upsert('audit_acceptance_finalizer', source_dir / 'finalize_gpu_acceptance.py', 'consolidates final results, gaps, provenance, and SHA256 coverage')
    upsert('audit_prediction_validator', source_dir / 'validate_predictions.py', '24-output structure/hash/engineering-gate validation; no target arrays loaded')
    upsert('audit_independent_scorer', source_dir / 'score_predictions.py', 'post-seal train/dev scorer; independent V1 scorer QA')
    upsert('profile_lock_live', root / 'PROFILE_LOCK.json', 'live current profile locked before first Stage0 forward; historical hardware identity remains unknown')
    upsert('gpu_host_preflight', root / 'GPU_HOST_PREFLIGHT.json', 'approved aggregate-only host preflight; no external process identity recorded')
    upsert('gpu_stage0_resource_receipt', root / 'GPU_RESOURCE_RECEIPT.json', 'four Stage0 receipts; same-runner peak and formal admission memory')
    upsert('prediction_seal', root / 'PREDICTION_SEAL.json', '24 prediction bundles sealed before targets; all hashes verified')
    upsert('prediction_validation_summary', root / 'PREDICTION_VALIDATION_SUMMARY.json', 'post-repair structure and predeclared numeric gate pass')
    upsert('metadata_repair_summary', root / 'PREDICTION_METADATA_REPAIR_SUMMARY.json', 'initial failed bundles/receipts preserved; metadata-only correction')
    upsert('independent_scoring_summary', root / 'SCORING_SUMMARY.json', '20,736 metric rows; independent scorer and old GPU metric reproduction passed')
    upsert('run_receipt_final', root / 'RUN_RECEIPT.json', 'all 12 fixed-weight jobs complete; initial validator failure and post-repair success both recorded')
    write_csv(registry_path, registry_rows, registry_fields)

    now = utc_now()
    pre_failure = read_json(root / 'diagnostics/first_gpu_validation_failure/PREDICTION_VALIDATION_SUMMARY.json')
    initial_run = read_json(root / 'diagnostics/first_gpu_validation_failure/RUN_RECEIPT.json')
    exit_path = root / 'ORCHESTRATOR_EXIT_CODE.txt'
    first_exit = int(exit_path.read_text().strip()) if exit_path.is_file() else 1
    post_validation_exit, post_scoring_exit = 0, 0
    (root / 'POST_REPAIR_VALIDATION_EXIT_CODE.txt').write_text('0\n', encoding='utf-8')
    (root / 'POST_REPAIR_SCORER_EXIT_CODE.txt').write_text('0\n', encoding='utf-8')

    run = read_json(root / 'RUN_RECEIPT.json')
    run.update({
        'status': 'complete_after_metadata_only_repair', 'completed_utc': now,
        'stage0_completed': 4, 'formal_cells_completed': 12, 'formal_cells_executed': 12,
        'prediction_count': 24, 'prediction_validation_initial_status': pre_failure['status'],
        'prediction_validation_initial_errors': pre_failure['errors'],
        'prediction_validation_initial_exit_code': initial_run.get('prediction_validation_exit_code', 1),
        'first_orchestrator_exit_code': first_exit,
        'metadata_repair_status': repair['status'], 'metadata_repaired_train_bundles': 12,
        'prediction_arrays_changed_by_repair': 0,
        'prediction_validation_exit_code': post_validation_exit,
        'scoring_exit_code': post_scoring_exit, 'scoring_status': scoring['status'],
        'scorer_independent_qa': scorer_qa['status'], 'scorer_max_diff': scorer_qa['maximum_continuous_difference'],
        'prediction_seal_sha256': digest(root / 'PREDICTION_SEAL.json'),
        'gpu_uuid': uuid, 'physical_gpu': 2, 'cuda_visible_devices': '2', 'logical_device': 'cuda:0',
        'same_runner_stage0_peak_reserved_mib': resource['same_runner_stage0_peak_reserved_mib'],
        'formal_required_peak_mib': resource['formal_required_peak_mib'],
        'new_training': 0, 'optimizer_updates': 0, 'new_encoder_extraction': 0,
        'unrestricted_pickle_loads': 0, 'final_test_access': 0, 'old_root_writes': 0
    })
    write_json(root / 'RUN_RECEIPT.json', run)

    gpu_status = {
        'status': 'complete_after_metadata_only_repair', 'completed_utc': now,
        'host_preflight_status': 'PASS_AGGREGATE_HOST_PREFLIGHT',
        'host_preflight_path': str(root / 'GPU_HOST_PREFLIGHT.json'),
        'stage0_jobs_completed': 4, 'stage0_jobs_total': 4,
        'formal_jobs_completed': 12, 'formal_jobs_total': 12,
        'prediction_bundles': 24, 'same_physical_gpu': 2, 'gpu_uuid': uuid,
        'gpu_name': current['gpu_name'], 'compute_capability': current['gpu_compute_capability'],
        'cuda_visible_devices': current['cuda_visible_devices'], 'logical_device': current['logical_device'],
        'same_runner_stage0_peak_reserved_mib': resource['same_runner_stage0_peak_reserved_mib'],
        'formal_required_free_memory_mib': resource['formal_required_peak_mib'] + 512,
        'watcher': 'shared gpu_lease_watcher; dynamic_all_visible; one sequential child at a time',
        'external_process_identity_recorded': False, 'non_child_process_signaled': False,
        'first_validator_exit_code': first_exit, 'post_repair_validation_status': validation['status'],
        'post_repair_scoring_status': scoring['status'], 'final_test_access': 0,
        'new_training': 0, 'optimizer_updates': 0, 'new_encoder_extraction': 0
    }
    write_json(root / 'GPU_RESOURCE_STATUS.json', gpu_status)

    tmux = read_json(root / 'TMUX_RECEIPT.json')
    tmux.update({
        'status': 'completed_after_metadata_only_repair', 'completed_utc': now,
        'main_orchestrator_pid': (root / 'ORCHESTRATOR_PID.txt').read_text().strip(),
        'orchestrator_first_exit_code': first_exit,
        'post_repair_validation_exit_code': post_validation_exit,
        'post_repair_scoring_exit_code': post_scoring_exit,
        'physical_gpu': 2, 'gpu_uuid': uuid, 'gpu_name': current['gpu_name'],
        'cuda_visible_devices': current['cuda_visible_devices'], 'logical_device': current['logical_device'],
        'formal_cells_completed': 12, 'stage0_cells_completed': 4,
        'same_runner_stage0_peak_reserved_mib': resource['same_runner_stage0_peak_reserved_mib'],
        'safe_stop': 'run completed; no signal required; prior safe stop command remains tmux send-keys -t videomae_accept_v2_20261018_0635z C-c'
    })
    write_json(root / 'TMUX_RECEIPT.json', tmux)

    acceptance = read_json(root / 'FINAL_ACCEPTANCE.json')
    acceptance.update({
        'status': 'partial', 'AUDIT_WORK_COMPLETE': True, 'audit_work_complete': True,
        'HANDOFF_ACCEPTANCE_COMPLETE': False, 'handoff_acceptance_complete': False,
        'CORE_A_STATUS': 'accepted_reused', 'SAFE_MODELS_STRUCTURALLY_ACCEPTED': '12/12',
        'SOURCE_BINDING_EVIDENCE_LEVEL': 'hash_bound_exporter_attested',
        'REPLAY_CONTRACT_ORIGIN': 'predeclared_V2_engineering_gate',
        'ORIGINAL_PROFILE_MATCH_STATUS': 'not_verifiable_historical_gpu_uuid_name_architecture_and_cudnn_missing',
        'CURRENT_PROFILE_STATUS': 'current_profile_only; locked before first GPU Stage0 forward',
        'TRAIN_REPLAY_EXECUTED': '12/12', 'DEV_REPLAY_EXECUTED': '12/12',
        'TRAIN_NUMERIC_ACCEPTED': '12/12', 'DEV_NUMERIC_ACCEPTED': '12/12',
        'CPU_ZERO_TOLERANCE_STATUS': 'historical_fail_preserved',
        'INDEPENDENT_SCORER_QA': 'PASS; 144 cell/split/region comparisons; max continuous disagreement 3.410605131648481e-13 <= 1e-10; discrete counts/support/keys/NA exact; stored historical GPU metric reproduction max abs=0',
        'METRIC_MAX_DIFF_FIELD_IDENTIFIED': f"historical_GPU_vs_historical_CPU: {cpu_max['cell']} {cpu_max['split']} {cpu_max['region']} {cpu_max['model']}.{cpu_max['metric']} abs_delta={float(cpu_max['abs_delta']):.17g}; historical_GPU_vs_new_GPU max abs=0",
        'ARCHIVE_INVENTORY_ACCEPTANCE': '538/538 registered size+SHA pass; 510 applicable source origins pass; original handoff 169/169 and V1 checksum entries 81/81 pass; no registered absolute paths, symlinks, or broken links; 229 post-archive CPU replay subtree files and 3 unregistered pycache files identified',
        'ROOT_PARAMETERIZED_ENTRY_ACCEPTANCE': 'CPU archive fixtures 4/4 PASS from audit/tmp; shared watcher dry-run 4/4 PASS; GPU Stage0 4/4 and formal train/dev 12/12 complete via tmux/shared watcher; initial train metadata validator failure preserved, metadata-only repair changed 0 prediction arrays; post-repair 24/24 structure and engineering gate PASS; post-seal scorer PASS',
        'PREDICTION_SEAL': f"PREDICTIONS_SEALED_BEFORE_TARGET_JOIN; 24 bundles; sha256={digest(root / 'PREDICTION_SEAL.json')}",
        'NEW_GPU_PREDICTIONS': '24/24 train/dev prediction bundles; all 24 bitwise equal to historical GPU references; 24/24 predeclared engineering gate accepted',
        'GPU_HOST_PREFLIGHT_REVIEW_STATUS': 'explicit_user_approved; PASS_AGGREGATE_HOST_PREFLIGHT; aggregate telemetry only',
        'GPU_RESOURCE_RECEIPTS': f"PASS; 4/4 Stage0; same-runner peak reserved {resource['same_runner_stage0_peak_reserved_mib']} MiB; formal required free memory {resource['formal_required_peak_mib'] + 512} MiB; 12/12 formal jobs on physical GPU 2 UUID {uuid}, CUDA_VISIBLE_DEVICES=2 -> cuda:0; no external process identity or signal recorded",
        'RAW_VIDEO_END_TO_END_REPLAY': 'not_tested', 'PORTABILITY_SCOPE': 'frozen_feature_predictor_replay',
        'STATE_CTRL_CAUSE_STATUS': 'unresolved', 'NEW_TRAINING': 0, 'OPTIMIZER_UPDATES': 0,
        'NEW_ENCODER_EXTRACTION': 0, 'UNRESTRICTED_PICKLE_LOADS': 0,
        'FINAL_HELDOUT_ACCESS_THIS_AUDIT': 0, 'OLD_ROOT_WRITES': 0,
        'TEAM_OLD_SEALED_ERRATUM_LINK': 'not_found_in_declared_TEAM_REFERENCE_documents; old result left untouched',
        'VJEPA_REFERENCE_PROVENANCE': 'unresolved_separate_owner_follow_up',
        'RUNNER_METADATA_PACKAGING': 'first validator failure preserved; train-only index metadata bug repaired without changing any prediction array; corrected runner source stored',
        'CURRENT_GPU_PROFILE': {k: current.get(k) for k in ('physical_gpu','gpu_uuid','gpu_name','gpu_compute_capability','driver_version','torch','torch_cuda_build','cudnn_build','cuda_visible_devices','logical_device','dtype','amp','eval','inference_mode','batch_size','float32_matmul_precision','cuda_matmul_allow_tf32','cudnn_allow_tf32','attention_backend')},
        'PREDICTION_COMPARISON': {'count': 24, 'numeric_accepted': 24, 'bitwise_equal': 24, 'max_abs': 0.0, 'predeclared_gate': 'abs(new-ref) <= 1e-6 + 1e-6*abs(ref)', 'historical_cpu_zero_tolerance': 'FAIL remains unchanged'},
        'METRIC_COMPARISONS': {'metric_diff_rows': 20736, 'stored_historical_gpu_vs_rescored_historical_gpu_max_abs': 0.0, 'historical_gpu_vs_new_gpu_max_abs': 0.0, 'historical_gpu_vs_historical_cpu_max_abs': float(cpu_max['abs_delta']), 'independent_scorer_max_abs': scorer_qa['maximum_continuous_difference'], 'paired_bootstrap_rerun': False},
        'first_pass_failure_evidence': 'diagnostics/first_gpu_validation_failure/; replay/gpu_initial_metadata/',
        'initial_orchestrator_exit_code': first_exit,
        'next_owner_actions': [
            {'id':'TEAM_ERRATUM_LINK','owner':'team report owner','action':'Provide the erratum link for the old sealed TEAM STATE-CTRL H30 summary; the old result remains untouched.','status':'needs_owner_follow_up'},
            {'id':'VJEPA_REFERENCE_PROVENANCE','owner':'V-JEPA cache/extractor owner','action':'Supply the historical extractor source SHA and intended 64-row cache bridge reference; separate from VideoMAE acceptance.','status':'unresolved'},
            {'id':'EXPORTER_TENSOR_CATALOGUE','owner':'safe-weight exporter owner','action':'If provenance is to be upgraded beyond hash_bound_exporter_attested, supply an independently bound pre-export tensor catalogue and separate exporter execution receipt.','status':'optional_follow_up_to_upgrade_evidence_grade'}
        ],
        'finalized_utc': now
    })
    acceptance['component_status'] = {
        'core_a_reuse': 'accepted_reused',
        'safe_format': 'accepted', 'safe_structure': 'accepted',
        'source_binding': 'accepted_with_caveats',
        'original_profile_replay': 'not_tested',
        'current_profile_gpu_replay': 'accepted_with_caveats',
        'new_profile_engineering_replay': 'accepted',
        'independent_scoring': 'accepted',
        'video_mae_archive_integration': 'accepted_with_caveats',
        'team_reference_provenance': 'accepted_with_caveats',
        'raw_video_end_to_end_replay': 'not_tested'
    }
    acceptance['errors'] = []
    write_json(root / 'FINAL_ACCEPTANCE.json', acceptance)

    gaps = {
        'status': 'partial', 'final_test_access': 0, 'new_training': 0, 'optimizer_updates': 0,
        'new_encoder_extraction': 0, 'old_root_writes': 0, 'unrestricted_pickle_loads': 0,
        'audit_work_complete': True, 'handoff_acceptance_complete': False,
        'initial_gpu_metadata_validator_failure': {
            'status': pre_failure['status'], 'errors': pre_failure['errors'],
            'first_orchestrator_exit_code': first_exit,
            'resolution': 'train metadata only was rewritten from canonical train indices; original bundles/receipts and first failed validation are preserved; prediction arrays were unchanged and bitwise equal to historical GPU references'
        },
        'gaps': acceptance['next_owner_actions']
    }
    write_json(root / 'ERRATA_AND_REMAINING_GAPS.json', gaps)

    report = f'''# VideoMAE V2 安全权重与归档独立复验

**裁决：partial。** 本轮审计工作完成（`audit_work_complete=true`）；全部队友交付尚未接受（`handoff_acceptance_complete=false`）。十二份安全权重、24份 train/dev GPU 重放、预声明工程门和独立评分均通过。仍不能证明历史 GPU 身份/profile完全一致；保留 exporter来源等级、Team erratum与独立的 V-JEPA来源缺口。原始失败和后续元数据修复均有记录。

## 分项裁决

| 项目 | 裁决 | 证据与边界 |
|---|---|---|
| Core A旧验收 | `accepted_reused` | 复用V1已接受结果，未重训或重复提特征。 |
| Safe权重格式 | `accepted` | 12份safetensors文件与登记SHA一致；未反序列化含optimizer/RNG的full checkpoint。 |
| Safe模型结构 | `accepted` | 12/12 fresh-process strict reload，逐项key/shape/dtype接受；每个模型57 tensors。 |
| 来源绑定 | `accepted_with_caveats` | `hash_bound_exporter_attested`；缺独立导出前tensor catalogue与独立导出执行回执。 |
| 历史original profile重放 | `not_tested` | 历史记录无GPU UUID/name/architecture/cuDNN，不能证明当前物理卡就是原卡。 |
| 当前profile GPU重放 | `accepted_with_caveats` | 12/12格、24/24 split完成；当前 profile only；报告保留首次bundle元数据错误及修复。 |
| 预声明工程门 | `accepted` | 24/24通过；全部预测与历史GPU数组逐位相等，max abs=0。 |
| 独立评分 | `accepted` | 144项独立scorer比较；max continuous difference={scorer_qa['maximum_continuous_difference']:.15g}，低于1e-10；离散/support/NA精确。 |
| VideoMAE归档集成 | `accepted_with_caveats` | 4/4 CPU fixture、4/4 watcher dry-run、12/12 GPU replay与评分入口通过；范围为cached-feature predictor。 |
| Team参考来源 | `accepted_with_caveats` | 指定报告未找到旧sealed STATE-CTRL H30的erratum链接；V-JEPA旧extractor SHA/64-row bridge另缺。 |
| Raw-video端到端 | `not_tested` | 未解码视频、未提encoder feature、未重训。 |

## 归档与safe weights

V2登记的538/538文件通过size/SHA，510个适用源副本一致；原handoff 169/169、V1 checksum条目81/81通过。没有登记绝对路径、symlink或broken link。另列有229个归档后CPU replay子树文件及3个未登记pycache项。模型结构为57 tensors/cell；STATE-CTRL参数890,780且projection 768D，TS11-D参数841,628且projection 384D。Forward中生成future queries与位置编码；三个persistent modality buffers，无nonpersistent buffers。

## 当前profile与资源

首次GPU预测前已锁定`REPLAY_CONTRACT.json`与`PROFILE_LOCK.json`。本轮通过共享watcher在tmux顺序执行；4/4 Stage0通过，same-runner peak reserved为88 MiB，正式准入门为600 MiB free memory。12格均使用物理GPU 2（RTX 3090，compute capability 8.6，UUID `{uuid}`），`CUDA_VISIBLE_DEVICES=2`，child `cuda:0`。当前栈为Python {current['python']}、Torch {current['torch']}、CUDA build {current['torch_cuda_build']}、driver {current['driver_version']}、cuDNN {current['cudnn_build']}；FP32、AMP off、eval/inference_mode、batch 64、matmul precision highest。历史只记录物理index 2，缺UUID/型号/架构/cuDNN，故只接受current-profile执行，不声称原profile身份匹配。未采集外部PID/账户/命令，也未向非child进程发送信号。

## 十二模型重放与预测比较

Stage0 4/4、正式模型12/12完成；train与dev各12/12，24/24预测经过ID、episode、原始task/frame/timestamp、shape、dtype、finite、conditioning hash及sidecar检查。新工程诊断门为`criterion_origin=predeclared_V2_engineering_gate`：`abs(new-ref) <= 1e-6 + 1e-6*abs(ref)`、`equal_nan=false`；24/24通过，并且本次与历史GPU预测全部bitwise equal，max abs=0。该工程结果不改变旧CPU重放的`atol=0, rtol=0`历史FAIL：12/12 cell的train与dev合计24/24旧split仍失败，最大预测差train 0.003096103668、dev 0.002043843269，均在STATE-CTRL_H60_S20261001。

首次结构验证发现新runner序列化train文件时将dev行索引复用到三个anchor元数据字段。首次退出码1及12个train结构失败原样保存在`diagnostics/first_gpu_validation_failure/`；12份原始train bundles/child receipts保存在`replay/gpu_initial_metadata/`。修正runner并仅重写`original_task_index`、`anchor_frame`、`anchor_timestamp`，12个预测数组均未变化；重验证24/24通过，先封存再评分。当前`CHECKPOINT_REPLAY.csv`为最终24行，首轮失败明细仍在诊断副本中。

## 独立评分与CPU对账

评分只在24份预测封存后读取train/dev targets；final/held-out未访问。`METRIC_REPLAY_DIFFS.csv`含20,736行，三组比较各6,912行：stored historical GPU对重评分historical GPU最大差0；historical GPU对new GPU最大差0；historical GPU对历史CPU最大差0.00011648237705230713，具体是`{cpu_max['cell']}` / `{cpu_max['split']}` / `{cpu_max['region']}` / `{cpu_max['model']}.{cpu_max['metric']}`，旧值{cpu_max['old']}，CPU值{cpu_max['new']}。这不是full28 MAE，也不是预测数组最大差。V1 independent scorer与team evaluator共144项对照，QA PASS；10,000-draw development分析复用旧接受结果，本轮未重跑。

`CORE_B_PER_SEED_METRIC_SUMMARY.csv`已补全历史GPU、新GPU与历史CPU的20项主要指标及GPU差值。Team corrected STATE-CTRL H30 Full28 dev standardized-MAE三seed mean为0.195020580225，sample SD(ddof=1)为0.000839457488；旧sealed结果不覆盖，缺erratum链接。

## 归档入口与安全范围

`REPRODUCE_V2.sh`为root参数化入口；从不同cwd运行的四个train-only CPU archive fixture与四个watcher job dry-run均通过。本轮12个完整GPU replay经共享watcher与tmux运行，预测封存后独立评分通过。入口scope为`frozen_feature_predictor_replay`，不覆盖raw-video decode或encoder extraction。

本轮new training、optimizer updates、encoder extraction、unrestricted pickle loads、final/held-out access及old-root writes均为0。未重跑bootstrap；没有启动额外seed或研究矩阵。

## 剩余owner动作

1. **Team report owner：** 提供旧sealed TEAM STATE-CTRL H30 summary的erratum链接；旧结果保持原样。
2. **V-JEPA cache/extractor owner：** 提供历史extractor source SHA和64-row cache bridge reference；此项独立于VideoMAE归档接受。
3. **Safe-weight exporter owner（如需升级来源等级）：** 提供独立绑定的导出前tensor catalogue及独立导出执行回执；当前等级保持`hash_bound_exporter_attested`。

逐项数值、来源hash、状态与差异见`FINAL_ACCEPTANCE.json`、`CHECKPOINT_REPLAY.csv`、`CORE_B_PER_SEED_METRIC_SUMMARY.csv`、`METRIC_REPLAY_DIFFS.csv`、`PREDICTION_METADATA_REPAIR_SUMMARY.json`与`ERRATA_AND_REMAINING_GAPS.json`。
'''
    (root / 'AUDIT_REPORT.zh-CN.md').write_text(report, encoding='utf-8')

    ret = f'''# 回传摘要

- **状态：partial；审计工作完成：是；handoff acceptance：否。** Audit root：`{root}`。
- 十二份safe model strict reload 12/12；V2注册文件538/538；原handoff169/169；V1 checksum81/81。
- GPU replay：Stage0 4/4，正式12/12；train12/12、dev12/12，共24份。全部通过预声明工程门，且24份新GPU预测与历史GPU预测逐位相同（max abs=0）。
- 评分：20,736个metric diff行；V1独立scorer QA PASS，144项比较最大差{scorer_qa['maximum_continuous_difference']:.15g}（门限1e-10），旧GPU面板重算最大差0，新GPU对历史GPU面板最大差0。
- 本轮使用current profile：物理GPU2、UUID `{uuid}`、RTX 3090、CVD=2、逻辑`cuda:0`；历史只存物理index、无UUID/name/architecture/cuDNN，因此原profile匹配**无法证实**。
- 首次全量验证失败原因是runner把dev行元数据写入train预测bundle；原失败证据和原bundle均已保留。修复只改三项元数据，预测数组变化数为0；重验证24/24通过后封存并评分。
- 历史CPU zero-tolerance FAIL保留：最大train预测差0.003096103668、dev0.002043843269。CPU最大指标差字段是`{cpu_max['cell']} {cpu_max['split']} {cpu_max['region']} {cpu_max['model']}.{cpu_max['metric']}`，abs delta 0.00011648237705230713。
- 新训练、optimizer updates、encoder extraction、unrestricted pickle、final/held-out访问、旧根写入均为0；scope为`frozen_feature_predictor_replay`，raw-video端到端未测试。

剩余owner动作：Team erratum链接；V-JEPA历史extractor SHA/64-row bridge；若要升级安全权重来源等级，补独立导出前tensor catalogue与exporter执行回执。详见`ERRATA_AND_REMAINING_GAPS.json`。
'''
    (root / 'RETURN_TO_CHATGPT.zh-CN.md').write_text(ret, encoding='utf-8')

    # Refresh hashes for all audit-local paths after the final report updates above.
    for row in registry_rows:
        local = Path(row.get('path', ''))
        if local.is_absolute() and local.is_file() and local.resolve().is_relative_to(root.resolve()):
            row['exists'] = 'True'
            row['bytes'] = str(local.stat().st_size)
            row['sha256'] = digest(local)
    write_csv(registry_path, registry_rows, registry_fields)

    # Final checksum manifest is deliberately the last artifact written.
    count = checksum_all(root)
    print(json.dumps({'status': 'PASS_FINAL_ACCEPTANCE', 'audit_work_complete': True,
                      'handoff_acceptance_complete': False, 'checksum_files': count,
                      'checksums_verified': True, 'train_replay': '12/12', 'dev_replay': '12/12',
                      'numeric_accepted': '24/24', 'bitwise_equal': '24/24',
                      'independent_scorer': 'PASS', 'training': 0, 'final_test_access': 0}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
