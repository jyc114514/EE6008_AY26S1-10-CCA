#!/usr/bin/env python3
"""Finish figures, exposure, report, and acceptance from seal-bound V6 score artifacts."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import v6_score as score

ROOT = Path(__file__).resolve().parents[1]
METRICS = score.METRICS


def main():
    if (ROOT / 'qa/FINAL_ACCEPTANCE.json').exists():
        raise RuntimeError('V6 final acceptance already exists')
    seal = score.validate_prediction_seal()
    if seal.get('status') != 'SEALED_BEFORE_TARGET_ACCESS':
        raise RuntimeError('saved-score finalizer refuses unsealed outcomes')
    loss_path = ROOT / 'results/PER_WINDOW_NEW_DEV_LOSSES.csv'
    receipt_path = ROOT / 'qa/SAVED_NEW_LOSS_TABLE_RECEIPT.json'
    if not loss_path.is_file() or not receipt_path.is_file():
        raise RuntimeError('saved-score finalizer requires a seal-bound new loss table')
    receipt = score.read_json(receipt_path)
    seal_sha = score.sha_file(ROOT / 'qa/PREDICTION_SEAL.json')
    alpha_sha = score.sha_file(ROOT / 'protocol/ALPHA_LOCK.json')
    if (receipt.get('status') != 'PASS' or receipt.get('prediction_seal_sha256') != seal_sha
            or receipt.get('alpha_lock_sha256') != alpha_sha or score.sha_file(loss_path) != receipt.get('sha256')):
        raise RuntimeError('saved new loss table is not bound to current seal/alpha lock')
    saved = score.read_csv(loss_path)
    score.validate_saved_loss_rows(saved, seal_sha=seal_sha, expected_methods=['SS50','SS_CAL'],
                                   expected_seeds=score.SEEDS,
                                   expected_ids=[r['anchor_id'] for r in sorted(score.read_csv(score.V5/'protocol/ORDERED_RECEIVER_IDS.csv'), key=lambda r:int(r['canonical_row_index']))])
    if len(saved) != 1110:
        raise RuntimeError(f'saved V6 loss table row count mismatch: {len(saved)}')

    # Confirm all downstream numerical outputs were already written from the same scoring pass.
    expected_rows = {
        'results/PRIMARY_TASK_SEED.csv': 15,
        'results/PAIRED_EFFECTS.csv': 14 * 15,
        'results/PAIRED_GROUP_CI.csv': 14 * 3 * 19,
        'results/CONVEXITY_CHECK.csv': 4 * 3 * 3,
        'results/RESIDUAL_COMPLEMENTARITY.csv': 72,
        'figure_data/TRAJECTORY_CASES.csv': 5,
        'figure_data/TRAJECTORY_DATA.csv': 8100,
    }
    for rel, expected in expected_rows.items():
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f'saved scoring artifact is missing: {rel}')
        actual = sum(1 for _ in path.open(encoding='utf-8')) - 1 if path.suffix == '.csv' else -1
        if actual != expected:
            raise RuntimeError(f'saved scoring artifact row count mismatch {rel}: {actual} != {expected}')
    convexity = score.read_csv(ROOT / 'results/CONVEXITY_CHECK.csv')
    if any(r['status'] != 'PASS' for r in convexity):
        raise RuntimeError('saved convexity check contains a failure')

    synthetic = score.read_json(ROOT / 'qa/SYNTHETIC_SCORER_QA.json')
    preflight = score.read_json(ROOT / 'qa/PREFLIGHT.json')
    schema = score.read_json(ROOT / 'qa/DEV_TARGET_SCHEMA_PREFLIGHT.json')
    if synthetic.get('status') != 'PASS' or preflight.get('status') != 'PASS' or schema.get('status') != 'PASS':
        raise RuntimeError('synthetic/real target-free preflight is not accepted')

    alpha_lock = score.read_json(ROOT / 'protocol/ALPHA_LOCK.json')
    alpha_curves = score.read_csv(ROOT / 'results/INNER_TUNE_ALPHA_CURVES.csv')
    primary_rows = score.read_csv(ROOT / 'results/PRIMARY_TASK_SEED.csv')
    pair_rows = score.read_csv(ROOT / 'results/PAIRED_EFFECTS.csv')
    ci_rows = score.read_csv(ROOT / 'results/PAIRED_GROUP_CI.csv')
    trajectories = score.read_csv(ROOT / 'figure_data/TRAJECTORY_DATA.csv')
    cases = score.read_csv(ROOT / 'figure_data/TRAJECTORY_CASES.csv')

    # No target archive is opened here. Figures use stored figure_data and sealed tables.
    score.make_figures(alpha_curves, primary_rows, pair_rows, cases + trajectories)
    v5mix = score.read_json(ROOT / 'results/V5_MIX50_RECONCILIATION.json')
    if v5mix.get('status') != 'PASS' or len(score.read_csv(ROOT / 'results/V5_MIX50_PAIRED_GROUP_CI.csv')) != 228:
        raise RuntimeError('completed V5 MIX50 reconciliation is missing')

    # The first target-bearing scoring call stopped before scoring; the next wrote all rows and downstream data.
    corrections = score.read_json(ROOT / 'qa/DEV_SCORE_ACCESS_CORRECTIONS.json')
    if corrections.get('failed_dev_archive_content_reads_before_success') != 1:
        raise RuntimeError('post-seal target access correction receipt is missing or inconsistent')
    v5_exposure = score.read_json(score.V5 / 'inventory/OUTCOME_EXPOSURE.json')
    exposure = score.make_target_access_receipt(seal, schema,
        v5_exposure['target_archive_sha256'], int(v5_exposure['target_archive_bytes']), 185, 1110)
    score.write_json(ROOT / 'qa/DEV_TARGET_ACCESS.json', {
        'schema_version': 1, 'status': 'PASS', 'seal_sha256': seal_sha,
        'target_archive_path': str(score.V2 / 'data/DEV_TARGETS_SCORER_ONLY.npz'),
        'target_archive_sha256': exposure['v6_dev']['target_archive_sha256'],
        'target_archive_bytes': exposure['v6_dev']['target_archive_bytes'],
        'target_rows_loaded': exposure['v6_dev']['target_array_windows_loaded'],
        'unique_receiver_windows': 185, 'per_window_new_loss_evaluations': 1110,
        'new_window_loss_evaluations_completed_before_figure_failure': 1110,
        'saved_loss_rows_reused_for_finalization': True,
        'target_reader_invocations': exposure['v6_dev']['np_load_operations_over_in_memory_blob'],
        'failed_pre_score_target_reads': corrections['failed_dev_archive_content_reads_before_success'],
        'metadata_only_preflight_opens': schema['archive_member_metadata_open_count'],
        'finalizer_opened_target_archive': False, 'final_heldout_payload_access': 0,
    })
    score.write_json(ROOT / 'qa/LOSS_TABLE_RECOVERY_QA.json', {
        'schema_version': 1, 'status': 'PASS',
        'recovery_status': 'FINALIZED_FROM_SEAL_BOUND_LOSS_TABLE_AND_SAVED_FIGURE_DATA',
        'new_loss_evaluations_in_scoring_attempt': 1110,
        'new_loss_evaluations_in_finalization': 0,
        'loss_table_rows': len(saved), 'loss_table_sha256': score.sha_file(loss_path),
        'prediction_seal_sha256': seal_sha, 'alpha_lock_sha256': alpha_sha,
        'finalizer_target_archive_access': 0, 'created_utc': score.utc(),
    })
    score.make_report_json(primary_rows, pair_rows, ci_rows, alpha_lock, exposure, v5mix,
                           seal, exposure['v6_dev']['target_archive_sha256'], 1110, len(convexity))
    score.write_report_md(primary_rows, pair_rows, ci_rows, alpha_lock, exposure, v5mix, seal,
                          exposure['v6_dev']['target_archive_sha256'])
    score.write_final_acceptance(seal, primary_rows, pair_rows, ci_rows, convexity, synthetic, exposure)
    score.update_owner_after_score(exposure)
    score.write_json(ROOT / 'qa/FINALIZATION_FROM_SAVED_SCORE.json', {
        'schema_version':1,'status':'PASS','created_utc':score.utc(),
        'target_archive_reopened':False,'new_loss_rows_recomputed':0,
        'saved_loss_table_path':str(loss_path),'saved_loss_table_sha256':score.sha_file(loss_path),
        'saved_trajectory_data_reused':True,'figures_regenerated_from_saved_rows':True,
        'final_heldout_payload_access':0,
    })
    print(json.dumps({'status':'COMPLETE','logical_bundles':12,'primary_rows':len(primary_rows),
        'paired_rows':len(pair_rows),'ci_rows':len(ci_rows),'target_archive_reopened':False,
        'target_loads_v6':exposure['v6_dev']['np_load_operations_over_in_memory_blob'],
        'new_loss_evaluations':1110,'final_heldout_payload_access':0},sort_keys=True))


if __name__=='__main__':
    main()
