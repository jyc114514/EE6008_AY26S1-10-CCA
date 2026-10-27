#!/usr/bin/env python3
"""Build the final V6 resource ledger and run-context receipt from watcher/artifact evidence."""
from __future__ import annotations

import csv
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / 'repo/ee6008-humanoid-vjepa'
V2 = PROJECT / 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V5 = PROJECT / 'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
PYTHON = PROJECT / 'envs/jyc/bin/python'
WATCHER = REPO / 'tools/gpu_lease_watcher.py'


def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


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


def gpu_row(job_id, role, output, lease_dir, resource_note):
    watcher = read_json(ROOT / 'coordination/watcher_status' / f'{job_id}.json')
    complete = read_json(Path(output) / 'PREDICTION_COMPLETE.json')
    if watcher.get('status') != 'complete' or watcher.get('lease_released') is not True:
        raise RuntimeError(f'GPU job not complete/released: {job_id}')
    return {
        'phase': 'screen_seed_inner_tune_frozen_replay', 'job_id': job_id,
        'role': role, 'status': 'PASS', 'device_type': 'GPU', 'visible_gpu_pool': watcher.get('visible_gpu_pool'),
        'physical_gpu': watcher.get('physical_gpu'), 'cuda_visible_devices': complete.get('cuda_visible_devices'),
        'logical_device': complete.get('logical_device'),
        'registered_peak_mib': watcher.get('measured_peak_mib'),
        'required_free_memory_mib': watcher.get('required_free_memory_mib'),
        'observed_peak_allocated_mib': complete.get('peak_memory_allocated_mib'),
        'observed_peak_reserved_mib': complete.get('peak_memory_reserved_mib'),
        'child_elapsed_seconds': complete.get('elapsed_seconds'),
        'lease_path': watcher.get('lease_path'), 'lease_directory': str(lease_dir),
        'lease_released': watcher.get('lease_released'), 'output_rows': complete.get('rows'),
        'prediction_sha256': complete.get('prediction_sha256'), 'final_heldout_payload_access': 0,
        'resource_note': resource_note,
    }


def main():
    gpu = []
    gpu.append(gpu_row('v6_inner_tune_S_FIT_s202610130', 'S_FIT',
        ROOT/'predictions/inner_tune/S_FIT', ROOT/'coordination/gpu_lease',
        'accepted 102 MiB prior same-script peak; observed 50 MiB reserved'))
    gpu.append(gpu_row('v6_inner_tune_A_FIT_s202610130', 'A_FIT',
        ROOT/'predictions/inner_tune/A_FIT', ROOT/'coordination/gpu_lease',
        'accepted 102 MiB prior same-script peak; observed 50 MiB reserved'))
    gpu.append(gpu_row('v6_inner_tune_V_FIT_s202610130', 'V_FIT_attempt_01',
        ROOT/'predictions/inner_tune/V_FIT', ROOT/'coordination/watcher_status',
        'output preserved but not used for alpha after observed 134 MiB exceeded the registered 102 MiB; run-local lease directory argument was watcher_status'))
    gpu.append(gpu_row('v6_inner_tune_V_FIT_s202610130_peak134_retry02', 'V_FIT_selected_attempt_02',
        ROOT/'predictions/inner_tune/V_FIT/attempt_02', ROOT/'coordination/gpu_lease',
        'accepted for alpha; registered 134 MiB from prior attempt and exact prediction payload matched attempt_01'))

    rows = list(gpu)
    rows.extend([
        {'phase':'v5_mix50_reconciliation','job_id':'cpu_v5_saved_score_reconciliation','status':'PASS',
         'device_type':'CPU','wall_seconds':'not_instrumented','new_dev_target_reads':0,
         'reused_task_seed_rows':15,'reused_paired_ci_rows':228,'new_window_loss_evaluations':0,
         'resource_note':'read V5 sealed scored tables and per-window results only'},
        {'phase':'inner_tune_alpha_calibration','job_id':'cpu_scalar_calibration_SV_SS','status':'PASS',
         'device_type':'CPU','scalar_programs':2,'grid_points_per_program':21,
         'logical_anchor_loss_evaluations':11340,'inner_tune_archive_npz_loads':2,
         'target_member_rows_materialized_per_load':1280,'inner_tune_rows_scored':270,
         'failed_first_alignment_attempts':1,'risk_rows_saved':42,'dev_target_reads':0,
         'resource_note':'one global scalar per family selected from inner-fit-only base models'},
        {'phase':'formal_dev_composition_and_seal','job_id':'cpu_native_float64_composition','status':'PASS',
         'device_type':'CPU','logical_bundles':12,'new_logical_bundles':9,
         'new_unique_tensor_payloads':6,'shared_payload_aliases':3,'reused_v5_sv50_bundles':3,
         'dev_target_reads':0,'final_heldout_payload_access':0,
         'resource_note':'all source predictions ID-joined; native-space outputs; seal precedes target reader'},
        {'phase':'dev_schema_and_synthetic_preflight','job_id':'cpu_synthetic_and_header_preflight','status':'PASS',
         'device_type':'CPU','synthetic_scorer_qa':'PASS','target_archive_metadata_opens':1,
         'target_values_read':False,'dev_target_reads':0,
         'resource_note':'checked NPZ member list and NPY headers only'},
        {'phase':'postseal_dev_scoring','job_id':'cpu_dev_score_attempt_01','status':'FAILED_RECORDED',
         'device_type':'CPU','target_archive_content_read':True,'target_windows_loaded':185,
         'new_window_loss_evaluations':0,'final_heldout_payload_access':0,
         'resource_note':'stopped on accepted BIAS_STATE companion tensor schema after target load; retained and corrected'},
        {'phase':'postseal_dev_scoring','job_id':'cpu_dev_score_attempt_02','status':'SCORES_COMPLETE_FIGURE_FINALIZATION_FAILED',
         'device_type':'CPU','target_archive_content_read':True,'target_windows_loaded':185,
         'new_window_loss_evaluations':1110,'seal_bound_loss_table_rows':1110,
         'final_heldout_payload_access':0,
         'resource_note':'all numeric tables, diagnostics, and trajectory data saved; figure failure finalized from saved rows without archive reopen'},
        {'phase':'saved_score_finalization','job_id':'cpu_finalize_saved_score','status':'PASS',
         'device_type':'CPU','target_archive_reopened':False,'new_window_loss_evaluations':0,
         'figures_regenerated':4,'final_heldout_payload_access':0,
         'resource_note':'reused seal-bound losses and saved trajectory data'},
    ])
    fields=['phase','job_id','role','status','device_type','visible_gpu_pool','physical_gpu','cuda_visible_devices','logical_device',
            'registered_peak_mib','required_free_memory_mib','observed_peak_allocated_mib','observed_peak_reserved_mib',
            'child_elapsed_seconds','lease_path','lease_directory','lease_released','output_rows','prediction_sha256',
            'scalar_programs','grid_points_per_program','logical_anchor_loss_evaluations','inner_tune_archive_npz_loads',
            'target_member_rows_materialized_per_load','inner_tune_rows_scored','failed_first_alignment_attempts','risk_rows_saved',
            'logical_bundles','new_logical_bundles','new_unique_tensor_payloads','shared_payload_aliases','reused_v5_sv50_bundles',
            'synthetic_scorer_qa','target_archive_metadata_opens','target_values_read','target_archive_content_read',
            'target_windows_loaded','new_window_loss_evaluations','seal_bound_loss_table_rows','target_archive_reopened',
            'figures_regenerated','new_dev_target_reads','final_heldout_payload_access','resource_note','wall_seconds',
            'reused_task_seed_rows','reused_paired_ci_rows']
    write_csv(ROOT/'results/RESOURCE_LEDGER.csv',rows,fields)
    max_reserved=max(float(r['observed_peak_reserved_mib']) for r in gpu)
    summary={
        'schema_version':1,'status':'PASS','host':os.environ.get("EE6008_HOST_LABEL", "shared-host"),'physical_gpus_used':[2],
        'logical_gpu':'cuda:0','cuda_visible_devices':'2','visible_gpu_pool':'dynamic_all_visible',
        'gpu_jobs':len(gpu),'accepted_screen_tune_prediction_bundles':3,
        'replay_attempts':4,'max_observed_reserved_mib':max_reserved,
        'all_leases_released':all(r['lease_released'] for r in gpu),
        'new_neural_fits':0,'new_formal_fits':0,'scalar_calibrations':2,
        'new_logical_dev_bundles':9,'new_unique_tensor_payloads':6,
        'dev_target_archive_reads_after_seal':2,'new_window_loss_evaluations':1110,
        'final_heldout_payload_access':0,'created_utc':utc(),
    }
    write_json(ROOT/'inventory/RESOURCE_SUMMARY.json',summary)
    context={
        'schema_version':1,'created_utc':utc(),'unix_uid':os.getuid(),'host':socket.gethostname(),
        'working_directory':str(ROOT),'project_root':str(PROJECT),'output_root':str(ROOT),
        'project_python':str(PYTHON),'shared_watcher':str(WATCHER),
        'source_root_write_policy':'V2/V3/V4/V5 read-only; V6 unique output root only',
        'repository_git_state':'repo path is the project placeholder; no commit/branch claim used',
        'gpu_scheduler_environment':'no SLURM/PBS/LSB job variables were present in preflight shell',
        'gpu_selection':'shared watcher dynamic_all_visible lease',
        'physical_gpu_ids_used':[2],'CUDA_VISIBLE_DEVICES':'2','logical_device':'cuda:0',
        'active_gpu_jobs':0,'all_gpu_leases_released':True,
        'inner_fit_models_reused':3,'new_model_fits':0,'final_heldout_payload_access':0,
    }
    write_json(ROOT/'RUN_CONTEXT.json',context)
    print(json.dumps({'status':'PASS','gpu_jobs':len(gpu),'physical_gpus_used':[2],
                      'max_reserved_mib':max_reserved,'all_leases_released':summary['all_leases_released']},sort_keys=True))


if __name__=='__main__':
    main()
