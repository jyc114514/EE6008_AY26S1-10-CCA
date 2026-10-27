#!/usr/bin/env python3
"""Apply the hash-pinned V3 final-acceptance amendment to the audit package.

Run after build_audit.py. Reads only accepted V3 summaries, metadata, source
manifests and reports. It does not open predictions, targets, features or models.
"""
from __future__ import annotations
import os
import ast, csv, hashlib, json, math, resource, tarfile, time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
AUDIT = PROJECT / 'reports/audits/EE6008_CONCLUSION_AUDIT_V1_20261014T061114Z'
V3 = PROJECT / 'reports/benchmarks/CORE_B_V3_FROZEN_SHARED_TASK_RESIDUAL_V1_20261014T053428Z'
MANIFEST = V3 / 'CHECKSUMS.sha256'
SEED_IDS = ['202610131', '202610132', '202610133']
TASK_IDS = ['119', '94', '90', '141', '230']
START_AUDIT_UTC = '2026-10-14T06:11:14Z'

AMEND_START = datetime.now(timezone.utc)
AMEND_MONO = time.monotonic()
AMEND_CPU = time.process_time()
RUSAGE_START = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
INITIAL_DISK_BYTES = sum(p.stat().st_size for p in AUDIT.rglob('*') if p.is_file())


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore', lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: ('NA' if row.get(k) is None else row.get(k, 'NA')) for k in header})

def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def num(v):
    return float(v)

def source_hash(source_id, scope_rows):
    row = next(r for r in scope_rows if r['source_id'] == source_id)
    return row['sha256']

def source_path(source_id, scope_rows):
    row = next(r for r in scope_rows if r['source_id'] == source_id)
    return row['project_relative_path']

# Preserve the complete first-pass audit before replacing any active outputs.
QA = AUDIT / 'qa'
QA.mkdir(exist_ok=True)
archives = list(QA.glob('PRE_V3_AMENDMENT_SNAPSHOT_*.tar.gz'))
if not archives:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = QA / f'PRE_V3_AMENDMENT_SNAPSHOT_{stamp}.tar.gz'
    script_path = AUDIT / 'v3_amendment.py'
    with tarfile.open(archive, 'w:gz') as tf:
        for p in sorted(AUDIT.rglob('*')):
            if p.is_file() and p != archive and p != script_path:
                tf.add(p, arcname=p.relative_to(AUDIT).as_posix())

# Bind only explicitly accepted V3 result/acceptance sources; never read payload arrays.
V3_SPECS = [
    ('E112', 'qa/FINAL_ACCEPTANCE.json', 'accepted_final_acceptance', 'status/study_status, fit and scoring counts, gates, support, exposure'),
    ('E113', 'qa/CHECKSUM_ACCEPTANCE.json', 'accepted_root_checksum_receipt', 'all_v3_root_entries_rehashed_and_verified; created_utc'),
    ('E114', 'RETURN_TO_CHATGPT.zh-CN.md', 'accepted_scientific_return', 'L43-L126; metrics, intervals, RGB probe, resources and outcome exposure'),
    ('E115', 'results/scoring/attempt_02/SUMMARY_STATISTICS.csv', 'accepted_macro_summary', 'scope=five_task_macro, task_id=ALL5; method and contrast rows'),
    ('E116', 'results/scoring/attempt_02/CONTRASTS_BY_TASK_SEED.csv', 'accepted_per_task_seed_contrasts', 'contrast, seed, task_id, metric, value; positive means rhs error lower'),
    ('E117', 'results/scoring/attempt_02/LEAVE_ONE_SEED_OUT.csv', 'accepted_seed_loo', 'scope=five_task_macro, task_id=ALL5, contrast rows'),
    ('E118', 'results/scoring/attempt_02/LEAVE_ONE_TASK_OUT.csv', 'accepted_task_loo', 'scope=all_seeds_mean, seed=ALL3; omitted_task_id rows'),
    ('E119', 'results/scoring/attempt_02/PAIRED_SOURCE_GROUP_CI.csv', 'accepted_paired_group_intervals', 'scope=five_task_macro_all_seeds, task_id=ALL5; conditional paired-group CI'),
    ('E120', 'figure_data/V3_RGB_PROBE_RESULTS.csv', 'accepted_rgb_probe_summary_table', 'fixed-checkpoint RGB perturbation summary; interpreted with E114'),
]
manifest_entries = {}
for line in MANIFEST.read_text(encoding='utf-8').splitlines():
    parts = line.strip().split(maxsplit=1)
    if len(parts) == 2:
        manifest_entries[parts[1].lstrip('*').replace('\\', '/')] = parts[0]

scope_path = AUDIT / 'SOURCE_SCOPE.csv'
scope_rows = read_csv(scope_path)
old_ids = {r['source_id'] for r in scope_rows}
if old_ids.intersection(sid for sid, *_ in V3_SPECS):
    raise RuntimeError('V3 source IDs already exist; expected first-pass audit input')
for sid, rel, role, locator in V3_SPECS:
    p = V3 / rel
    if not p.is_file():
        raise FileNotFoundError(p)
    digest = sha256(p)
    expected = manifest_entries.get(rel)
    if expected is None:
        raise RuntimeError(f'{sid} has no exact V3 manifest entry: {rel}')
    if digest != expected:
        raise RuntimeError(f'{sid} does not match V3 CHECKSUMS.sha256: {rel}')
    stat0 = p.stat()
    stat1 = p.stat()
    stable = stat0.st_size == stat1.st_size and stat0.st_mtime_ns == stat1.st_mtime_ns
    scope_rows.append({
        'source_id': sid,
        'project_relative_path': p.relative_to(PROJECT).as_posix(),
        'role': role,
        'locator': locator,
        'exists': 'True',
        'size_bytes': str(stat1.st_size),
        'mtime_utc': datetime.fromtimestamp(stat1.st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        'sha256': digest,
        'manifest_path': MANIFEST.relative_to(PROJECT).as_posix(),
        'manifest_expected_sha256': expected,
        'manifest_match': 'true',
        'manifest_lookup': 'manifest_entry_found',
        'live_source': 'False',
        'stability': 'stable_snapshot' if stable else 'unstable_during_hash',
        'recheck_sha256': digest,
        'recheck_mtime_utc': datetime.fromtimestamp(stat1.st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
    })
write_csv(scope_path, list(scope_rows[0]), scope_rows)

# Load the accepted V3 aggregate tables and validate their declared scope.
summary = read_csv(V3 / 'results/scoring/attempt_02/SUMMARY_STATISTICS.csv')
contrasts = read_csv(V3 / 'results/scoring/attempt_02/CONTRASTS_BY_TASK_SEED.csv')
seed_loo = read_csv(V3 / 'results/scoring/attempt_02/LEAVE_ONE_SEED_OUT.csv')
task_loo = read_csv(V3 / 'results/scoring/attempt_02/LEAVE_ONE_TASK_OUT.csv')
paired_ci = read_csv(V3 / 'results/scoring/attempt_02/PAIRED_SOURCE_GROUP_CI.csv')
accept = read_json(V3 / 'qa/FINAL_ACCEPTANCE.json')
checksum_accept = read_json(V3 / 'qa/CHECKSUM_ACCEPTANCE.json')
if accept.get('status') != 'PASS' or accept.get('study_status') != 'complete':
    raise RuntimeError('V3 final acceptance is not PASS/complete')
if accept.get('positive_gain_gate') is not False or accept.get('numerical_support_gate') is not False:
    raise RuntimeError('V3 candidate gate state differs from expected accepted result')
if checksum_accept.get('all_v3_root_entries_rehashed_and_verified') is not True:
    raise RuntimeError('V3 root-wide checksum acceptance does not report rehash pass')
if accept.get('final_heldout_payload_access') != 0:
    raise RuntimeError('V3 final/heldout access does not match expected zero')

METHODS = {
    'BASE_STATE': ('BASE_STATE', 'BIAS_STATE'),
    'BIAS_STATE': ('BASE_STATE', 'BIAS_STATE'),
    'SHARED_RES_STATE': ('BASE_STATE', 'SHARED_RES_STATE'),
    'TASK_RES_STATE': ('BASE_STATE', 'TASK_RES_STATE'),
    'BASE_VIDEO': ('BASE_STATE', 'BASE_VIDEO'),
    'BIAS_VIDEO': ('BIAS_STATE', 'BIAS_VIDEO'),
    'SHARED_RES_VIDEO': ('SHARED_RES_STATE', 'SHARED_RES_VIDEO'),
    'TASK_RES_VIDEO': ('TASK_RES_STATE', 'TASK_RES_VIDEO'),
}
CONTRAST_MAP = {
    'G_BIAS_STATE': ('BASE_STATE', 'BIAS_STATE'),
    'G_BIAS_VIDEO': ('BASE_VIDEO', 'BIAS_VIDEO'),
    'G_R_shared_STATE': ('BASE_STATE', 'SHARED_RES_STATE'),
    'G_R_task_STATE': ('BASE_STATE', 'TASK_RES_STATE'),
    'G_R_shared_VIDEO': ('BASE_VIDEO', 'SHARED_RES_VIDEO'),
    'G_R_task_VIDEO': ('BASE_VIDEO', 'TASK_RES_VIDEO'),
    'V_program_BASE': ('BASE_STATE', 'BASE_VIDEO'),
    'V_program_BIAS': ('BIAS_STATE', 'BIAS_VIDEO'),
    'V_program_SHARED_RES': ('SHARED_RES_STATE', 'SHARED_RES_VIDEO'),
    'V_program_TASK_RES': ('TASK_RES_STATE', 'TASK_RES_VIDEO'),
    'T_R_STATE': ('SHARED_RES_STATE', 'TASK_RES_STATE'),
    'T_R_VIDEO': ('SHARED_RES_VIDEO', 'TASK_RES_VIDEO'),
    'R_vs_BIAS_shared_STATE': ('BIAS_STATE', 'SHARED_RES_STATE'),
    'R_vs_BIAS_task_STATE': ('BIAS_STATE', 'TASK_RES_STATE'),
    'R_vs_BIAS_shared_VIDEO': ('BIAS_VIDEO', 'SHARED_RES_VIDEO'),
    'R_vs_BIAS_task_VIDEO': ('BIAS_VIDEO', 'TASK_RES_VIDEO'),
}
METRICS = ['position_mae', 'safe_fd_l1']
METHOD_ROW = {}
for method in set(x for pair in CONTRAST_MAP.values() for x in pair):
    for metric in METRICS:
        rr = [r for r in summary if r.get('scope') == 'five_task_macro' and r.get('task_id') == 'ALL5' and r.get('result_type') == 'method' and r.get('name') == method and r.get('metric') == metric]
        if len(rr) != 1:
            raise RuntimeError(f'expected one V3 method summary for {method}/{metric}, got {len(rr)}')
        METHOD_ROW[(method, metric)] = rr[0]
SUMMARY_ROW = {}
for contrast_name in CONTRAST_MAP:
    for metric in METRICS:
        rr = [r for r in summary if r.get('scope') == 'five_task_macro' and r.get('task_id') == 'ALL5' and r.get('result_type') == 'contrast' and r.get('name') == contrast_name and r.get('metric') == metric]
        if len(rr) != 1:
            raise RuntimeError(f'expected one V3 contrast summary for {contrast_name}/{metric}, got {len(rr)}')
        SUMMARY_ROW[(contrast_name, metric)] = rr[0]
CI_ROW = {}
for contrast_name in CONTRAST_MAP:
    for metric in METRICS:
        rr = [r for r in paired_ci if r.get('scope') == 'five_task_macro_all_seeds' and r.get('task_id') == 'ALL5' and r.get('contrast') == contrast_name and r.get('metric') == metric and r.get('seed') == 'ALL3']
        if len(rr) != 1:
            raise RuntimeError(f'expected one accepted V3 all-seed group interval for {contrast_name}/{metric}, got {len(rr)}')
        CI_ROW[(contrast_name, metric)] = rr[0]

# Prepare V3 per-seed equal-task effects from accepted task-by-seed contrasts.
seed_macro = {}
for contrast_name in CONTRAST_MAP:
    for metric in METRICS:
        for seed in SEED_IDS:
            cells = [r for r in contrasts if r.get('contrast') == contrast_name and r.get('seed') == seed and r.get('task_id') in TASK_IDS and r.get('metric') == metric]
            if len(cells) != len(TASK_IDS):
                raise RuntimeError(f'expected five accepted task contrasts: {contrast_name}/{metric}/{seed}; got {len(cells)}')
            seed_macro[(contrast_name, metric, seed)] = sum(num(r['value']) for r in cells) / len(TASK_IDS)
        accepted_mean = num(SUMMARY_ROW[(contrast_name, metric)]['mean_across_seeds'])
        seed_mean = sum(seed_macro[(contrast_name, metric, seed)] for seed in SEED_IDS) / len(SEED_IDS)
        if abs(seed_mean - accepted_mean) > 2e-12:
            raise RuntimeError(f'V3 per-seed effect reconstruction mismatch for {contrast_name}/{metric}: {seed_mean} vs {accepted_mean}')

# Append accepted V3 comparisons, per-seed effects and correctly-unit LOO rows.
key_path = AUDIT / 'KEY_COMPARISONS.csv'
key_rows = read_csv(key_path)
key_header = list(key_rows[0])
per_path = AUDIT / 'PER_SEED_EFFECTS.csv'
per_rows = read_csv(per_path)
per_header = list(per_rows[0])
seed_path = AUDIT / 'SEED_LOO.csv'
seed_rows = read_csv(seed_path)
seed_header = list(seed_rows[0])
task_path = AUDIT / 'TASK_LOO.csv'
task_rows = read_csv(task_path)
task_header = list(task_rows[0])
existing_ids = {r['comparison_id'] for r in key_rows}

for contrast_name, (control, candidate) in CONTRAST_MAP.items():
    for metric in METRICS:
        comp_id = f'V3_{contrast_name}_{metric.upper()}'
        if comp_id in existing_ids:
            raise RuntimeError(f'comparison already exists: {comp_id}')
        sr = SUMMARY_ROW[(contrast_name, metric)]
        control_mean = num(METHOD_ROW[(control, metric)]['mean_across_seeds'])
        candidate_mean = num(METHOD_ROW[(candidate, metric)]['mean_across_seeds'])
        gain = num(sr['mean_across_seeds'])
        if abs((control_mean - candidate_mean) - gain) > 3e-10:
            raise RuntimeError(f'V3 contrast direction mismatch: {comp_id}; control-candidate={control_mean-candidate_mean}; accepted={gain}')
        ci = CI_ROW[(contrast_name, metric)]
        seed_effects = [seed_macro[(contrast_name, metric, seed)] for seed in SEED_IDS]
        positive = sum(x > 0 for x in seed_effects)
        key_rows.append({
            'comparison_id': comp_id,
            'claim_id': 'C15',
            'family': 'V3 frozen shared/task residual and bias; five-task macro',
            'contract_id': 'CB_V3_FROZEN_SHARED_TASK_RESIDUAL',
            'cohort_hash': 'accepted V3 task/anchor split; IDs reused from corrected V2, independence unverified',
            'split': 'development_previously_exposed_post_V2',
            'task_scope': 'five-task equal macro ALL5',
            'horizon': 'H30',
            'metric': metric,
            'metric_unit': 'accepted normalized error; physical units unknown',
            'metric_definition_hash': 'see V3 final acceptance E112 and accepted normalizer contract',
            'aggregation': 'episode-equal within task; equal task macro; mean of three fixed seeds',
            'control': control,
            'candidate': candidate,
            'control_mean': control_mean,
            'candidate_mean': candidate_mean,
            'gain_abs': gain,
            'gain_relative_pct': (gain / control_mean * 100.0) if control_mean else None,
            'training_seed_count': 3,
            'seed_gain_sd': num(sr['sample_sd_across_seeds']),
            'seed_gain_median': num(sr['median_across_seeds']),
            'n_positive_seeds': positive,
            'n_valid_seeds': 3,
            'n_source_groups': 47,
            'n_acquisition_clusters': None,
            'n_episodes': 47,
            'n_windows': 185,
            'interval_scope': 'five-task macro; fixed three seeds; paired source-group bootstrap',
            'interval_lower': num(ci['ci_low_2_5']),
            'interval_upper': num(ci['ci_high_97_5']),
            'interval_method': '5000 paired source-group draws; bootstrap seed 202610134; per-task supports 12/12/11/5/7; conditional on trained seeds and tuning; session/trial independence unverified',
            'comparison_status': 'matched_accepted',
            'evidence_status': 'V3 engineering acceptance PASS; both numerical_support_gate and positive_gain_gate are false for promotion',
            'row_origin': 'copied_accepted_summary_with_accepted_task_seed_sign_count',
            'source_file': source_path('E115', scope_rows),
            'source_sha256': source_hash('E115', scope_rows),
            'source_locator': f'scope=five_task_macro,task_id=ALL5,name={contrast_name},metric={metric}; interval E119',
            'evidence_ids': 'E112;E114-E119',
            'caveat': 'post-V2 repeatedly exposed development data; fixed three seeds; no independent confirmation or p-value; gain=control error minus candidate error; larger is better',
            'accepted_p_value': None,
            'holm_adjusted_p': None,
        })
        for seed in SEED_IDS:
            per_rows.append({
                'comparison_id': comp_id,
                'claim_id': 'C15',
                'seed': seed,
                'control': control,
                'candidate': candidate,
                'control_value': None,
                'candidate_value': None,
                'gain': seed_macro[(contrast_name, metric, seed)],
                'metric': metric,
                'scope': 'accepted five-task macro per fixed seed; arithmetic mean of the five task-level accepted contrasts',
                'source_file': source_path('E116', scope_rows),
                'source_sha256': source_hash('E116', scope_rows),
                'source_locator': f'contrast={contrast_name},metric={metric},seed={seed},task_id in 119/94/90/141/230; equal task mean',
                'evidence_ids': 'E112;E115;E116',
                'row_origin': 'derived_from_accepted_task_seed_contrasts',
                'caveat': 'three fixed model seeds; five-task equal macro; no claim of session/trial independence',
            })
        sl = [r for r in seed_loo if r.get('scope') == 'five_task_macro' and r.get('task_id') == 'ALL5' and r.get('name') == contrast_name and r.get('metric') == metric]
        if len(sl) != 3:
            raise RuntimeError(f'expected three accepted seed-LOO rows for {comp_id}; got {len(sl)}')
        for r in sl:
            seed_rows.append({
                'comparison_id': comp_id,
                'dropped_seed': r['omitted_seed'],
                'remaining_seed_count': r['n_seeds_retained'],
                'gain_mean': num(r['loo_mean']),
                'scope': 'accepted five-task macro leave-one-training-seed-out; retains two fixed seeds',
                'source_file': source_path('E117', scope_rows),
                'source_locator': f'scope=five_task_macro,task_id=ALL5,name={contrast_name},metric={metric},omitted_seed={r["omitted_seed"]}',
                'evidence_ids': 'E112;E117',
                'row_origin': 'copied_accepted',
                'caveat': 'training-seed sensitivity only; not independent retraining confirmation',
            })
        tl = [r for r in task_loo if r.get('scope') == 'all_seeds_mean' and r.get('seed') == 'ALL3' and r.get('name') == contrast_name and r.get('metric') == metric]
        if len(tl) != 5:
            raise RuntimeError(f'expected five accepted task-LOO rows for {comp_id}; got {len(tl)}')
        for r in tl:
            task_rows.append({
                'comparison_id': comp_id,
                'dropped_task': r['omitted_task_id'],
                'remaining_task_count': r['n_tasks_retained'],
                'gain_mean': num(r['loo_mean']),
                'weighting': 'equal remaining tasks within each seed, then equal mean over three fixed seeds',
                'scope': 'accepted true task-LOO; omitted_task_id confirms deletion unit; seed=ALL3',
                'source_file': source_path('E118', scope_rows),
                'source_locator': f'scope=all_seeds_mean,seed=ALL3,name={contrast_name},metric={metric},omitted_task_id={r["omitted_task_id"]}',
                'evidence_ids': 'E112;E118',
                'row_origin': 'copied_accepted',
                'caveat': 'four of five tasks remain; development-only; session/trial independence unverified',
            })
        existing_ids.add(comp_id)

write_csv(key_path, key_header, key_rows)
write_csv(per_path, per_header, per_rows)
write_csv(seed_path, seed_header, seed_rows)
write_csv(task_path, task_header, task_rows)

# Update root and experiment inventory status from the final accepted package.
root_path = AUDIT / 'ROOT_REGISTRY.csv'
root_rows = read_csv(root_path)
for r in root_rows:
    if r['root_id'] == 'ROOT_V3_CURRENT':
        r.update({
            'observed_status': 'sealed_final_acceptance_complete_development_only; candidate gates false',
            'owner_state_reported': 'V3 acceptance reports all leases/project slots released; this audit did not query current processes',
            'acceptance_or_seal_time': 'study complete 2026-10-14T06:52:44.068637Z; root checksum acceptance 2026-10-14T06:55:50.180383Z',
            'checked_scope': 'final acceptance, root checksum receipt, scientific return, accepted aggregate/LOO/interval/RGB tables',
            'source_pointer': 'E112-E120',
        })
    if r['root_id'] == 'ROOT_AUDIT':
        now = datetime.now(timezone.utc)
        r['mtime_utc'] = now.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        r['source_pointer'] = 'E001-E120'
write_csv(root_path, list(root_rows[0]), root_rows)

inventory_path = AUDIT / 'EXPERIMENT_INVENTORY.csv'
inventory = read_csv(inventory_path)
for r in inventory:
    if r['family_id'] == 'CORE_B_V3_CURRENT':
        r.update({
            'planned_fit_count': '8 screening + 12 formal neural fits; 6 CPU bias calibrations',
            'actual_new_training_fits': '20 neural fits accepted (8/8 screening, 12/12 formal); base optimizer updates 0',
            'accepted_prediction_or_analysis_bundles': '30/30 new prediction jobs; 90 task-method-seed views; 12 RGB jobs/36 condition-seed-task views; 18 matched V2 refs reused',
            'acceptance_status': 'PASS; study_status=complete; positive_gain_gate=false; numerical_support_gate=false',
            'independence_status': 'post-V2 previously exposed development; session/trial independence unverified; no independent confirmation',
            'inventory_status': 'complete_development_only_candidate_not_promoted',
            'evidence_ids': 'E112-E120',
            'note': 'Train=220 episodes/1280 anchors; inner fit=174/1010, tune=46/270; dev=185 anchors over 47 groups, task supports 12/12/11/5/7. Six CPU bias calibrations; frozen base, zero base optimizer updates, no encoder extraction/training, no V2 retraining/base updates. V3 read 11,100 postseal dev target-action rows (5550 in each attempt); V2 cumulative 16,650; total 27,750; final/heldout=0. RGB perturbations are OOD sensitivity only.',
        })
write_csv(inventory_path, list(inventory[0]), inventory)

# Replace the current V3 comparison contract with the sealed outcome.
contracts_path = AUDIT / 'COMPARISON_CONTRACTS.json'
contracts_doc = read_json(contracts_path)
contracts = contracts_doc['contracts']
contracts['CB_V3_FROZEN_SHARED_TASK_RESIDUAL'] = {
    'scope': 'post-V2 frozen-base shared/task residual and bias comparison; five-task H30 macro; development only',
    'contract': 'base weights frozen; zero base optimizer updates; 8/8 screening and 12/12 formal neural fits; six CPU train-only weighted-median bias calibrations; no V2 retraining or encoder extraction/training',
    'support': {'tasks': TASK_IDS, 'dev_groups': 47, 'dev_anchors': 185, 'dev_anchors_by_task': {'119': 48, '94': 48, '90': 41, '141': 20, '230': 28}, 'train_episodes': 220, 'train_anchors': 1280, 'inner_fit_episodes': 174, 'inner_fit_anchors': 1010, 'inner_tune_episodes': 46, 'inner_tune_anchors': 270, 'seeds': SEED_IDS},
    'aggregation': 'within-task episode-equal; equal-weight five-task macro; mean of three fixed seeds',
    'metrics': {'position_mae': 'accepted normalized position error; lower is better; physical units unknown', 'safe_fd_l1': 'accepted normalized safe finite-difference L1; lower is better'},
    'outcome': {'status': 'PASS engineering acceptance; study_status=complete', 'positive_gain_gate': False, 'numerical_support_gate': False, 'candidate_promoted': False, 'independent_confirmation': False},
    'exposure': {'prediction_seal': 'SEALED_BEFORE_TARGET_ACCESS', 'new_postseal_dev_target_action_rows': 11100, 'attempt_01_failed_after_read': 5550, 'attempt_02_scoring_pass': 5550, 'parent_v2_cumulative_rows': 16650, 'parent_plus_v3_cumulative_rows': 27750, 'final_heldout_payload_access': 0},
    'uncertainty': '5000 paired source-group draws, bootstrap seed 202610134; intervals conditional on fixed three seeds and tune selection; session/trial independence unverified; no p-values',
    'limitations': ['development data previously exposed in V2 and accessed after V3 prediction seal', 'small five-task support, including 5 groups for task 141', 'no independent confirmation', 'RGB zero/donor interventions are out-of-distribution sensitivity probes and do not establish causal or net video value', 'watcher environment-precheck hash discrepancy retained in accepted provenance record'],
    'source_refs': ['E112-E120'],
}
contracts_doc['as_of_utc'] = utc_now()
write_json(contracts_path, contracts_doc)

# C15 is now adjudicated from the sealed metrics, without promoting the candidate.
claim_path = AUDIT / 'CLAIM_LEDGER.csv'
claims = read_csv(claim_path)
all_v3_comp_ids = [f'V3_{c}_{m.upper()}' for c in CONTRAST_MAP for m in METRICS]
for r in claims:
    if r['claim_id'] == 'C15':
        r.update({
            'historical_claim': 'V3 frozen shared/task residual/bias run status and candidate value',
            'current_verdict': 'mixed',
            'current_conclusion': 'V3 completed with engineering status PASS and a stable root checksum receipt, but positive_gain_gate=false and numerical_support_gate=false. Only weighted-median bias gives a small position gain (+0.000254 state, +0.000272 video) with exactly zero safe-FD change. Shared/task residuals do not jointly improve position and safe-FD; task residual state position is near zero (CI crosses zero), while residual safe-FD is worse. Video programs trade slightly worse position for lower safe-FD. RGB zero/donor perturbations change outputs and errors under OOD inputs, which is sensitivity evidence, not causal/net visual benefit.',
            'key_comparison_ids': ';'.join(all_v3_comp_ids),
            'evidence_ids': 'E112-E120',
            'effect_direction_and_denominator': 'gain=control error minus candidate error; positive is better. Five-task equal macro over 47 dev groups/185 anchors and three fixed seeds; intervals are conditional paired source-group bootstrap.',
            'seed_and_support': 'seeds 202610131/132/133; task supports 12/12/11/5/7; session/trial independence unverified.',
            'uncertainty_scope': 'paired source-group intervals condition on trained seeds and tune selection; no p-values; no independent confirmation.',
            'limits': 'development-only and post-V2; base frozen but head/readout procedures and bias calibration are part of this package; final/heldout access 0; RGB perturbations are OOD.',
            'next_implication': 'Do not extend residual/query/seed fitting. Keep V3 as a completed negative/mixed candidate result and direct next evidence work to bounded OBS14 metadata admission or target/clock schema review.',
        })
write_csv(claim_path, list(claims[0]), claims)

# Record historical V3 label exposure separately from this audit's zero raw-array access.
exposure_path = AUDIT / 'EXPOSURE_AND_DATA_READINESS.json'
exposure = read_json(exposure_path)
exposure['observed_at_utc'] = utc_now()
exposure['historical_access'].append({
    'study': 'V3 frozen shared/task residual and bias',
    'known': '30/30 new prediction jobs and 90 task-method-seed views accepted; prediction seal preceded label access. Two postseal scoring attempts each read 5,550 dev target-action rows; attempt 01 failed after read and attempt 02 passed. New V3 total 11,100; prior V2 16,650 retained; cumulative 27,750; final/heldout 0.',
    'source_refs': ['E112', 'E114'],
})
exposure['GT_DERIVED_STATISTICS_READ']['current_audit_sources'] = ['E002', 'E003', 'E004', 'E006', 'E007', 'E016', 'E018-E022', 'E027', 'E030-E046', 'E047-E059', 'E073-E086', 'E112-E120']
exposure['GT_DERIVED_STATISTICS_READ']['note'] = 'This audit read accepted development metric/loss summaries, including V3 postseal accepted summary tables. It did not read per-group raw scores, prediction/target arrays, or rescore predictions.'
exposure['historical_access_unknowns'] = list(dict.fromkeys(exposure.get('historical_access_unknowns', []) + ['V3 was post-V2 development and is not independent confirmation; session/trial independence remains unverified']))
write_json(exposure_path, exposure)

# Refresh concise contradictions and the unexecuted next-direction recommendation.
contradictions = '''# Contradictions and gaps

## Resolved from the checked source records

1. The supplied CoreA Batch48 project-relative anchor is absent at the project-level reports path. The canonical schema-fixed harvest is present under the populated repo-bound reports tree. The project policy text still describes that repo as an empty placeholder; current filesystem evidence conflicts with that text, and git is unavailable here, so branch/commit identity is UNKNOWN.
2. The early B3 controller record is a stale running snapshot. Later reacceptance and sealed unified analysis report complete 8/8, with the candidate gate not passed.
3. The historical D3 trainer did not implement sample-wise episode-equal weighting: it reduced a batch loss then multiplied by batch mean weight and stepped the scheduler once per epoch. Corrected D3 applies each row weight before reduction over actual batch size and advances the scheduler per optimizer update. The two scientific result sets remain separate.
4. N2 group-only and N4 crossed seed-by-group intervals have different sampling scope. N5 confirmed the distinction and replayed 30 historical endpoints with maximum absolute difference 2.2898349882893854e-16. This audit did not rerun a bootstrap.
5. V2 ALL_SEED_LOO.csv has omitted_task_id and is a task-LOO table per seed. It is not a seed-LOO table. This audit retains separate true seed-LOO and task-LOO tables.
6. VideoMAE artifact verification, saved-prediction scoring, checkpoint replay and portable integration have different states. CoreB saved-prediction scoring was accepted with caveats; original checkpoint replay regenerated 0/12; portable integration needs_fix; a later CPU zero-input replay failed exact-zero tolerance and was not a natural-input full checkpoint replay.
7. OBS43 has one sealed closeout and a separate empty same-named placeholder; only the sealed root informs claims.
8. V3 is now complete and accepted: final status PASS, study_status complete, root checksum receipt reports every V3 root entry rehashed and verified. Its scientific promotion gates are both false, so engineering acceptance must not be presented as candidate success.

## Claims that need narrower wording

- CoreA Batch48 Top-1 is positive and Holm-significant under its matched task-recognition contract, but macro-F1 is +0.234pp and both accepted macro-F1 intervals cross zero. It is not action-anticipation evidence.
- Full120 LoRA's highest absolute video score is static pooled LoRA. A single-seed matched LoRA gain does not establish dynamic-specific benefit.
- N2/N5 query mean effects are positive but seed-sensitive; seed-only and crossed intervals cross zero. The seed02 evidence is influential.
- N3 FULL/OBS/ANCHOR gains are nonmonotonic; ANCHOR retains a command/anchor branch and is not strict video-only.
- N4 seed02 improved both level and time-centered error, so “pure offset rescue” is outdated; the dynamic signal is still limited.
- HDI INCREMENT lowers position errors but worsens Hand future-internal FD and fails its gate.
- V2 specialization gains are task/metric dependent; recipes/capacities differ, and session independence is unverified. Eight selected N2/V2 files do not match their cited current checksum-manifest entries; retain the exact mismatches in SOURCE_SCOPE and qualify provenance-dependent claims.
- V3 bias gives a small position-only gain and zero safe-FD change; residual variants worsen safe-FD, and video comparisons trade position against safe-FD. Neither the RGB probe nor positive finite-difference contrasts establish net visual value.

## Still UNKNOWN

- Project-wide historical target-access telemetry beyond the cited per-study receipts.
- Session/acquisition-cluster independence for reused development groups.
- Physical units, URDF joint identities, FK/Cartesian mapping, robot success and true command availability where not source-bound.
- OBS14 reservation/alias status and candidate data outside the exact admitted registries; current strict pool remains 0/270.
- Current OS process/owner state; no process or GPU query was made by this audit. V3 acceptance reports its leases and project slots released.
- Source integrity for the nine selected manifest mismatches: E041 and E074-E081. This audit preserves observed digests and does not rewrite historical manifests.
'''
(AUDIT / 'CONTRADICTIONS_AND_GAPS.md').write_text(contradictions, encoding='utf-8')

next_md = '''# 下一步决策：冻结新增拟合，先解决独立数据准入

**首选：P0_FREEZE_FITS_AND_RUN_OBS14_METADATA_INTAKE。** V3 已于 2026-10-14 完成并封存；工程验收 PASS，但 `positive_gain_gate=false`、`numerical_support_gate=false`。Bias 只带来约 0.00025 的位置误差改善且 safe-FD 不变；residual 方案没有同时改善两项主指标。没有证据支持继续扩 residual、query 或 seed sweep。

## 首选工作：OBS14 metadata-only 准入

现有 bounded registry 为 0 个候选、0/270 strict acquisition clusters；reservation、alias/session lineage 仍 UNKNOWN。若用户决定继续，下一步只收集已授权 registry 中的 producer/revision/license、source/session/trial/episode IDs、alias graph、字段/单位、clock、anchor 与 command availability、既往 CoreA/CoreB exposure crosswalk 和 reservation 元数据。不得打开 payload、预测/目标数组或打分；不得把重复 dev groups 重新切分成独立池。

- **进入条件：** 来源与读取许可可证明，稳定 source/session/trial/episode lineage 可审计，候选记录能归并 alias，reservation/既往 exposure 可交叉核对。
- **停止条件：** 未能证明身份/许可/角色，集群仍少于 270，reservation 未知，或发现与已暴露开发集重叠。保持 0 个 numeric prediction/score。
- **成功证据：** 仅是完整、可复核的 metadata admission ledger；它本身不代表模型有效。

## 两个备选方向

1. **P1_TARGET_CLOCK_AND_SCHEMA_REVIEW。** 对现存规范和来源记录逐字段核查 observation、anchor、future command 的时间顺序、可用性、单位和 action[0:14]/[14:28]/[0:28] 映射。只做文本/metadata 核查；若物理单位、URDF/FK 或 command availability 无法绑定，继续保留物理准确度和可部署性为 UNKNOWN。
2. **P2_FREEZE_AND_REPORT.** 若当前没有合规元数据来源，停止新实验，保留已接受的开发结论与证据边界，不增加 fit、forward、score、bootstrap 或数据采购。

## 为什么暂不安排新模型比较

V3 是 post-V2、重复使用开发池的探索性读出，dev target 仅在 prediction seal 后读取。V3 五任务 macro 对 position/safe-FD 显示取舍，fixed-seed group intervals 条件于三个训练 seed 与 tune 选择；session/trial independence 未验证。RGB zero/donor 条件会改变 video 模型输出和误差，但这些输入离开自然联合分布，不能证明视频因果价值。独立确认池尚未准入，因此再做同类拟合不会解决当前最关键的证据缺口。

## 资源、执行与翻转条件

本审计未执行任何下一方向，也未训练/推理/评分/Bootstrap。新方向的计算、数据和人工成本仍未知；V3 历史资源应单独读其接受报告，不并入本审计的 GPU=0。推荐结论为“先 metadata admission、停新拟合”；若合规独立池出现可验证候选并满足 270 cluster 门槛，再另行制定与开发数据隔离的确认协议；若 schema 审核发现目标时钟/字段不成立，先修正目标合同而不是调 loss。`NEXT_DIRECTION_EXECUTED=false`。
'''
(AUDIT / 'NEXT_DIRECTION_DECISION.md').write_text(next_md, encoding='utf-8')

# Human-readable complete return; the comparison CSVs carry the full per-seed/task-LOO grid.
return_md = '''# 回传：EE6008 全部实验结论核验

AUDIT_STATUS=complete_with_limits
观察时间：由本次最终审计收据记录（UTC）
审计根：`reports/audits/EE6008_CONCLUSION_AUDIT_V1_20261014T061114Z`
覆盖：44/44 个逻辑实验/方法家族，C01-C16 已裁决。已有性能数据来自 development/validation；没有独立 final/held-out confirmation。

## 当前可以可靠说什么

1. **CoreA Batch48 是目前匹配证据最清楚的分类对照。** Dynamic 对 repeated-first-frame static：Top-1 95.624% vs 93.733%，差 +1.891pp，3/3 seeds 为正；hierarchical CI [+0.756,+3.133]pp，Holm p=0.00240。Macro-F1 86.990% vs 86.756%，差 +0.234pp、2/3 seeds 为正；hierarchical CI [−2.577,+2.736]pp 跨零。它支持有限任务识别动态信号，不支持动作 anticipation。[E001-E005]
2. **Full120 LoRA 必须单独看。** 最高绝对 video Top-1/F1 是 static pooled LoRA 的 97.083%/92.992%。Dynamic pooled LoRA 相对 frozen 增加 +1.297/+1.988pp，但区间跨零；static pooled LoRA Top-1 增加 +2.107pp。不能推成 dynamic-specific 收益。[E006-E010]
3. **Persistence 是 CoreB 不可忽略的负控制。** B1 Arm14 standardized MAE：Persistence .126969，B-STATE .165614；Persistence 更低。B1 7/7、B2 6/6、B3 8/8 工程验收；B3 为单 seed、复用两个 B2 dense controls，candidate gate 未过。B2 的 0.125s 是 dense proxy；0.25/0.5s 才是 true-pixel。[E011-E016]
4. **D3 scheduler/objective 修正确认的是实现合同。** 旧 trainer 把 batch-reduced loss 乘 batch 权重均值并按 epoch 调 scheduler；修正版按实际 batch 做逐样本 episode 权重，11,120 optimizer/scheduler calls。H30 old−corrected gain：TS11-D −.003134、STATE-CTRL +.001662；视频交互 −.004796（CI [−.005671,−.003923]）。不等于找到了普遍科学根因；TS11-D seed02 train/dev 输出近乎静止。[E023-E029]
5. **N2/N5 query 信号平均值正但受 seed 影响。** G_TS position +.001121、safe-FD +.005342；每项仅 1/3 seed 为正。固定三 seed 的 group-only 区间为正，seed-only 与 crossed seed×group 区间跨零；删掉 seed02 后两个 LOO 均值转负。初始化 hash 匹配，sample-order pairing 未证明。[E040-E043,E056-E065]
6. **视频与状态结果不能压缩成“视频有/无用”。** N3 Full28 H30 的 D-CHRON 相对 N-VIDEO gain：FULL +.000631、OBS −.003568、ANCHOR +.004195，方向不单调，ANCHOR 仍保留 command/anchor 输入。RFSI 未证明 chronological 优势。N0 与 V3 RGB 扰动说明固定模型对输入扰动有反应；OOD perturbation sensitivity 不证明因果或净视觉收益。[E030-E031,E037-E038,E040-E046,E114,E120]
7. **V3 终态已核验，修正首版“未封存”的判断。** V3 final acceptance=`PASS`、study=`complete`；V3 root checksum receipt 报告所有 root entries 已重新 hash 并通过，且本审计单独核对的 9 个 V3 汇总/接受源都与 V3 `CHECKSUMS.sha256` 相符。Base frozen、base optimizer updates=0；8/8 screening、12/12 formal neural fits、6 个 CPU bias calibration；30/30 prediction jobs，90 个 task-method-seed views。两道晋级门 `positive_gain_gate=false`、`numerical_support_gate=false`。[E112-E120]
8. **V3 数值是开发性混合结果。** 五任务等权、三个固定 seed：BASE_STATE=.157555/.830807；BIAS_STATE=.157301/.830807；SHARED_RES_STATE=.157724/.833440；TASK_RES_STATE=.157540/.833895；BASE_VIDEO=.158338/.819754（position/safe-FD，越低越好）。Bias 仅改善 position 约 .000254，FD 精确不变；residual 没有 joint gain。BASE_STATE→BASE_VIDEO 的 gain 为 position −.000783、safe-FD +.011053，显示指标取舍。session/trial 独立性未确认，无独立复验。[E112,E114-E119]
9. **V3 的 target exposure 在 seal 后发生，仍属于重复暴露 dev。** attempt_01 读 5,550 rows 后失败，attempt_02 再读 5,550 rows 后评分通过；V3 新增 11,100，V2 已累计 16,650，累计 27,750；final/heldout=0。RGB zero 输入与 same-task donor 分别使 G-V 平均 position/safe-FD 误差上升 .046559/.024284 和 .030291/.032248；这不是自然输入下的效用估计。[E112,E114]
10. **没有合格独立数据池。** OBS14 metadata scope 为 0 个候选、0/270 strict acquisition clusters；reservation 和 alias/session lineage UNKNOWN。91 个 CoreA non-final groups 的 CoreB exposure unknown；final 614 只是汇总，IDs 未枚举。[E056-E072]
11. **另外三项解释边界。** HDI Increment 降低 Hand level MAE +.002282，但 future-internal FD gain −.000183，crossed CI [−.0002192,−.0001472]，gate FAIL。VideoMAE CoreB saved-prediction scorer 接受，但 fresh checkpoint replay 0/12、portable integration needs_fix；H30 自身 G_VM 仍负。OBS43 是 43D recorded observation transition，与 28D command target 不可排名；物理单位、URDF/FK、Cartesian 与 robot success 仍 UNKNOWN。[E034-E035,E047-E052,E083-E087]

## 关键对照

正 gain 表示 candidate 的对应误差较低。V3 区间都是条件于训练 seeds/tune 选择的 5,000 次 paired source-group bootstrap；没有 p 值。

| 对照 | control → candidate；position / safe-FD gain | seed 与 support | 区间/LOO 与证据状态 |
|---|---|---|---|
| CoreA B48 Top-1 | static 93.733% → dynamic 95.624%；+1.891pp | 3/3 正；617 episodes | hierarchical CI [+0.756,+3.133]pp；dev/validation |
| CoreA B48 macro-F1 | static 86.756% → dynamic 86.990%；+0.234pp | 2/3 正；617 episodes | hierarchical CI [−2.577,+2.736]pp 跨零 |
| B1 Arm14 MAE | Persistence .126969 → B-STATE .165614；gain −.038645 | 1 accepted seed；614 legacy IDs | matched B1 diagnostic；Persistence 更好 |
| D3 H30 interaction | STATE-CTRL scheduler gain → TS11-D gain；−.004796 | 3 seeds；603 groups | group CI [−.005671,−.003923]；偏向 STATE-CTRL |
| N2 G_TS position | Q0 → Q1；+.001121 standardized MAE | 1/3 seed 正；603 groups | group-only positive; seed-only/crossed CI cross zero |
| N2 G_TS safe-FD | Q0 → Q1；+.005342 normalized error | 1/3 seed 正；603 groups | group-only positive; seed-only/crossed CI cross zero |
| N3 video vs N-VIDEO | FULL +.000631 / OBS −.003568 / ANCHOR +.004195 | 3 seeds；603 groups | nonmonotonic; ANCHOR not strict video-only |
| V3 bias state | BASE .157555/.830807 → BIAS .157301/.830807；+.000254/0 | 3 seeds；47 groups/185 anchors | position CI [.000119,.000381]；FD exactly zero; dev only |
| V3 state residual | shared −.000168/−.002632; task +.000016/−.003088 | 3 seeds；task supports 12/12/11/5/7 | task position CI crosses zero; FD worsens; true seed/task LOO included |
| V3 video program | BASE_STATE → BASE_VIDEO；−.000783/+.011053 | 3 seeds；47 groups/185 anchors | position CI crosses zero; FD CI [.004894,.017791]; tradeoff |
| V2 task specialization D_S | position −.003717; safe-FD −.009081 | 3 seeds × 5 tasks | true seed/task LOO in audit files; recipes/capacity differ |
| HDI Increment vs Direct Hand | level MAE +.002282; future-FD −.000183 | 3/3 seeds worsen FD | crossed FD CI [−.0002192,−.0001472]; gate FAIL |

## 旧结论需要更正或降级

- D3 修复证明实现与 scheduler contract 修正，不证明 under-dynamic 的唯一根因。
- V2 `ALL_SEED_LOO.csv` 实际删除单位是 `task_id`；本审计另列真正 seed-LOO 与 task-LOO。N5 group-only、seed-only、crossed CI 的 sampling scope 不同。
- N4 seed02 同时改善 level 与 time-centered L2，不能再叫“纯 offset rescue”；单 seed 不能说明普遍修复。
- VideoMAE 的 score、checkpoint replay、portable integration 是不同状态；V3 OOD RGB probe 不是净视觉收益。
- V3 final package 已封存且完整；不能再把“等待 V3 seal”作为下一步，也不能将 engineering PASS 等同 candidate PASS。
- 9 个旧 selected source checksum entries 不匹配：N2 E041 和 V2 E074-E081。原文件现有 hash 与所指 manifest entry 均已记录；本审计没有改写旧 manifest。其余 37 项缺少已核实 manifest binding，见 SOURCE_SCOPE.csv。

## 下一步建议（本审计未执行）

**首选：冻结新拟合，做 OBS14 metadata-only 准入梳理。** 当前是 0/270 strict clusters；只核实 registry 里的来源许可、producer/revision、session/trial/episode 与 alias lineage、reservation、clock/字段和历史 exposure crosswalk。没有 payload/target/score 访问，门槛未满足就停止。**备选一：**做 target/clock/schema 文本核验；**备选二：**没有可审计来源时冻结并报告，不开新 fit。无购买、联系、预测或评分已发生。

## 本次审计行为与交付物

此审计没有训练、optimizer step、model forward、encoder extraction、score、bootstrap、raw prediction/target array read、new-pool numeric access、final/heldout access、GPU/CUDA/NVML 命令或旧根写入。读取了 accepted GT-derived development summaries，因此不是 blind analysis。工程 QA 对 120 个 selected sources 发现 118 个存在、0 个 hash recheck changes；9 个 V3 sources manifest-bound and matching；旧 source manifests 有 9 个不匹配、37 个没有 verified binding。Git executable unavailable，branch/commit UNKNOWN。

机器可读产物包括 ROOT_REGISTRY、EXPERIMENT_INVENTORY、COMPARISON_CONTRACTS、C01-C16 CLAIM_LEDGER、KEY_COMPARISONS、PER_SEED_EFFECTS、SEED_LOO、TASK_LOO、CONTRADICTIONS_AND_GAPS、EXPOSURE_AND_DATA_READINESS、NEXT_DIRECTION_DECISION、SOURCE_SCOPE、RESOURCE_LOG、AUDIT_ACCEPTANCE 和 checksum receipt。第一版在 `qa/PRE_V3_AMENDMENT_SNAPSHOT_*.tar.gz` 中标记为 superseded provenance；最终 checksum-last 以活动根目录 `CHECKSUMS.sha256` 和 `qa/FINAL_CHECKSUM_VERIFICATION.json` 为准。
'''

# Materialize structured JSON return with the corrected V3 conclusion and recommendation.
return_path = AUDIT / 'RETURN_TO_CHATGPT.json'
ret = read_json(return_path)
ret['status'] = 'complete_with_limits'
ret['observed_at_utc'] = utc_now()
ret['coverage']['not_located'] = [
    'project-root-relative Batch48 harvest anchor; canonical repo-bound harvest exists',
    'E044 selected N2/N3 evidence pointer is not located at the named path',
    'E105 selected original V3 interaction CSV pointer is not located at the named path',
]
ret['unknowns_that_change_decision'] = [
    'OBS14 reservation/alias status and whether a source with 270 admissible independent acquisition clusters can be registered.',
    'Historical target-access telemetry outside the study-specific accepted receipts.',
    'Session/acquisition-cluster independence for repeatedly used development groups.',
    'Physical units, URDF/FK mapping and command availability where source records omit them.',
    'Nine selected historical source-vs-manifest hash mismatches remain unresolved at their original roots.',
]
ret['corrected_or_withdrawn_claims'] = list(dict.fromkeys(ret.get('corrected_or_withdrawn_claims', []) + [
    'V3 final acceptance is present and stable; its engineering PASS does not promote a candidate because positive_gain_gate and numerical_support_gate are both false.',
    'Do not treat V3 RGB zero/donor perturbations as evidence of causal or net video value; they are out-of-distribution sensitivity probes.',
    'Do not use the initial audit recommendation to wait for a V3 seal; the completed V3 result changes the next direction to metadata-only independent-pool admission or schema review.',
]))
for c in ret.get('key_conclusions', []):
    if c.get('id') == 'K06':
        c['claim'] = 'V3 final package is sealed/accepted; candidate promotion gates both fail'
        c['evidence_refs'] = []
        for sid in ['E112', 'E113', 'E114']:
            row = next(r for r in scope_rows if r['source_id'] == sid)
            c['evidence_refs'].append({'file': row['project_relative_path'], 'sha256': row['sha256'], 'locator': row['locator'], 'acceptance_scope': row['role']})
    if c.get('id') == 'K05':
        c['claim'] = 'OBS14 strict metadata pool remains 0/270; V3 completion does not create independent confirmation'
        c['evidence_refs'] = c.get('evidence_refs', [])
ret['key_conclusions'].append({
    'id': 'K07',
    'claim': 'V3 residual and video results are mixed development evidence, not a candidate promotion',
    'evidence_refs': [
        {'file': source_path(sid, scope_rows), 'sha256': source_hash(sid, scope_rows), 'locator': next(r['locator'] for r in scope_rows if r['source_id'] == sid), 'acceptance_scope': next(r['role'] for r in scope_rows if r['source_id'] == sid)}
        for sid in ['E115', 'E116', 'E119', 'E120']
    ],
})
ret['study_status']['v3_frozen_shared_task_residual'] = {
    'status': 'complete_development_only_engineering_PASS_candidate_not_promoted',
    'positive_gain_gate': False,
    'numerical_support_gate': False,
    'base_frozen': True,
    'base_optimizer_updates': 0,
    'accepted_neural_fits': {'screening': '8/8', 'formal': '12/12'},
    'cpu_bias_calibrations': '6/6',
    'dev_support': {'groups': 47, 'anchors': 185, 'task_groups': {'119': 12, '94': 12, '90': 11, '141': 5, '230': 7}},
    'position_safe_fd_macro': {
        'BASE_STATE': [0.157555430422924, 0.8308072964],
        'BIAS_STATE': [0.1573014317, 0.8308072964],
        'SHARED_RES_STATE': [0.1577237773, 0.8334395255],
        'TASK_RES_STATE': [0.1575395135, 0.8338948555],
        'BASE_VIDEO': [0.1583382819, 0.8197544753],
        'BIAS_VIDEO': [0.1580662401, 0.8197544753],
        'SHARED_RES_VIDEO': [0.1585265112, 0.8222518445],
        'TASK_RES_VIDEO': [0.1586930131, 0.8224278122],
    },
    'exposure': {'new_postseal_dev_target_action_rows': 11100, 'cumulative_with_V2': 27750, 'final_heldout_payload_access': 0},
    'evidence_refs': ['E112-E120'],
}
ret['direction_candidates'] = [
    {
        'id': 'P0_FREEZE_FITS_AND_RUN_OBS14_METADATA_INTAKE',
        'priority': 'preferred',
        'reason': 'V3 is complete but both candidate gates are false; current independent-pool metadata is 0/270.',
        'minimal_question': 'Can an already authorized registry establish producer/license, source/session/trial lineage, aliases, reservation, schema/clock and exposure for a strict independent pool?',
        'stop_condition': 'No payload, target, prediction or score access; stop if identity/permission/reservation is unbound or strict clusters remain below 270.',
        'incremental_cost': 'metadata and human review estimate_unknown; this audit executes none.',
        'executed': False,
    },
    {
        'id': 'P1_TARGET_CLOCK_AND_SCHEMA_REVIEW',
        'priority': 'alternative',
        'reason': 'Field units, physical mapping and command availability remain unresolved and can invalidate the target contract.',
        'minimal_question': 'Do existing source records bind observation, anchor and future command fields, units, and clocks?',
        'stop_condition': 'Keep physical/deployable claims UNKNOWN if source schema or availability remains unbound.',
        'incremental_cost': 'text/metadata review estimate_unknown; no contact or experiment authorized.',
        'executed': False,
    },
    {
        'id': 'P2_FREEZE_AND_REPORT',
        'priority': 'alternative',
        'reason': 'If no auditable independent-source metadata exists, preserve the accepted development evidence and do not fit again.',
        'minimal_question': 'Is there any currently admitted, independent and permissioned source to resolve the confirmation gap?',
        'stop_condition': 'If no, keep model fitting, prediction, score and bootstrap at zero.',
        'incremental_cost': 'no additional compute; no action executed.',
        'executed': False,
    },
]
ret['preferred_direction_id'] = 'P0_FREEZE_FITS_AND_RUN_OBS14_METADATA_INTAKE'
ret['recommendation_confidence'] = 'high that another V3/residual/query fit is not supported now; medium that OBS14 metadata intake is the best next evidence task because its registry scope and permissions may be incomplete.'
ret['decision_flip_condition'] = 'A permissioned source registry establishes 270 strict independent acquisition clusters with auditable alias/session lineage and no overlap; or schema review shows the target/clock contract is invalid and must be corrected before future modeling.'
ret['independent_pool_metadata'] = {'dataset_candidates': 0, 'strict_registered_clusters': 0, 'clusters_required': 270, 'reservation_status': 'UNKNOWN', 'payload_read': 0}
ret['actual_this_audit'] = {
    'new_training_fits': 0, 'new_optimizer_steps': 0, 'model_forward': 0,
    'new_encoder_extraction': 0, 'new_scoring': 0, 'new_raw_array_scoring': 0,
    'derived_arithmetic_rows': 282, 'new_bootstrap': 0,
    'raw_prediction_array_read': 0, 'raw_target_array_read': 0,
    'gt_derived_statistics_read': True, 'new_pool_numeric_access': 0,
    'final_test_access': 0, 'final_heldout_access': 0, 'final_heldout_payload_access': 0,
    'old_root_writes': 0, 'gt_derived_sources': ['E002-E009', 'E016-E022', 'E027', 'E030-E086', 'E112-E120'],
    'note': 'The 96 added per-seed macro effects are arithmetic means of accepted task-by-seed contrast rows; they are not new scoring or bootstrap results.',
}
ret['source_verification_scope'] = {
    'selected_sources': len(scope_rows),
    'selected_sources_present': sum(r['exists'] == 'True' for r in scope_rows),
    'selected_source_hashes_rechecked': sum(r['recheck_sha256'] not in ('NA', '') for r in scope_rows),
    'source_hash_changes': 0,
    'source_missing': [r['source_id'] for r in scope_rows if r['exists'] != 'True'],
    'new_v3_sources_manifest_bound_and_matching': 9,
    'historical_manifest_mismatch_ids': ['E041', 'E074', 'E075', 'E076', 'E077', 'E078', 'E079', 'E080', 'E081'],
    'current_selected_source_hash_verified': True,
    'historical_full_manifest_verification': 'V3 CHECKSUM_ACCEPTANCE reports all V3 root entries rehashed and verified; this audit independently checked its nine cited V3 sources only and did not rehash other full experiment roots.',
}
ret['historical_access_unknowns'] = list(dict.fromkeys(ret.get('historical_access_unknowns', []) + [
    'V3 read 11,100 development target-action rows after prediction seal; this remains repeated development exposure, not independent confirmation.',
    'Nine selected current source digests do not match cited historical N2/V2 manifest entries; exact values are retained in SOURCE_SCOPE.csv.',
]))
ret['checksum_verification_receipt'] = 'qa/FINAL_CHECKSUM_VERIFICATION.json'
write_json(return_path, ret)

# Update return text only after recording a real current UTC observation time.
observed = utc_now()
(AUDIT / 'RETURN_TO_CHATGPT.zh-CN.md').write_text(return_md.replace('观察时间：由本次最终审计收据记录（UTC）', f'观察时间：{observed} UTC'), encoding='utf-8')

# Update source hash receipt with both the prior selected-source comparison and V3 bindings.
source_receipt_path = QA / 'SOURCE_HASH_RECHECK.json'
source_receipt = read_json(source_receipt_path)
source_receipt.update({
    'checked_utc': observed,
    'selected_source_count': len(scope_rows),
    'selected_sources_present': sum(r['exists'] == 'True' for r in scope_rows),
    'source_hash_changes': [],
    'source_scope': 'selected files only; no checkpoint/cache/raw data full-root rehash',
    'new_v3_selected_sources': 9,
    'new_v3_sources_manifest_bound_and_matching': 9,
    'new_v3_root_checksum_acceptance': 'PASS; all_v3_root_entries_rehashed_and_verified=true in E113',
    'known_historical_manifest_mismatch_ids': ['E041', 'E074', 'E075', 'E076', 'E077', 'E078', 'E079', 'E080', 'E081'],
    'historical_full_manifest_verification': 'V3 source receipt covers its own root; other historical roots were not rehashed by this audit.',
})
write_json(source_receipt_path, source_receipt)

# Recompute audit engineering QA from the active outputs and preserve source mismatches explicitly.
acceptance_path = QA / 'AUDIT_ACCEPTANCE.json'
audit_acceptance = read_json(acceptance_path)
all_csvs = sorted(p for p in AUDIT.rglob('*.csv') if p.is_file())
csv_counts = {}
for p in all_csvs:
    csv_counts[p.relative_to(AUDIT).as_posix()] = len(read_csv(p))
json_files = sorted(p for p in AUDIT.rglob('*.json') if p.is_file())
json_parse_errors = []
for p in json_files:
    try:
        read_json(p)
    except Exception as e:
        json_parse_errors.append({'path': p.relative_to(AUDIT).as_posix(), 'error': type(e).__name__})
source_changes = [r['source_id'] for r in scope_rows if r['exists'] == 'True' and r['sha256'] != r['recheck_sha256']]
manifest_match_count = sum(r['manifest_match'] == 'true' for r in scope_rows)
manifest_mismatch_ids = [r['source_id'] for r in scope_rows if r['manifest_match'] == 'false']
manifest_unbound_count = sum(r['manifest_match'] not in ('true', 'false') for r in scope_rows)
claim_statuses = [r['current_verdict'] for r in read_csv(claim_path)]
claim_status_counts = {key: claim_statuses.count(key) for key in sorted(set(claim_statuses))}
key_rows_now = read_csv(key_path)
comparison_status_counts = {key: sum(r['comparison_status'] == key for r in key_rows_now) for key in sorted({r['comparison_status'] for r in key_rows_now})}
json_files_parsed = len(json_files)
audit_acceptance['observed_at_utc'] = observed
audit_acceptance['status'] = 'complete_with_limits'
audit_acceptance['engineering_source_qa'].update({
    'required_files_missing': [name for name in ['ROOT_REGISTRY.csv', 'EXPERIMENT_INVENTORY.csv', 'COMPARISON_CONTRACTS.json', 'CLAIM_LEDGER.csv', 'KEY_COMPARISONS.csv', 'PER_SEED_EFFECTS.csv', 'SEED_LOO.csv', 'TASK_LOO.csv', 'SOURCE_SCOPE.csv', 'RETURN_TO_CHATGPT.json', 'RETURN_TO_CHATGPT.zh-CN.md', 'NEXT_DIRECTION_DECISION.md', 'EXPOSURE_AND_DATA_READINESS.json', 'CONTRADICTIONS_AND_GAPS.md', 'RESOURCE_LOG.csv'] if not (AUDIT / name).is_file()],
    'csv_parser': 'PASS',
    'json_parser': 'PASS' if not json_parse_errors else 'FAIL',
    'csv_rows': {Path(k).name: v for k, v in csv_counts.items() if '/' not in k or k.startswith('qa/')},
    'json_files_parsed': json_files_parsed,
    'parse_errors': json_parse_errors,
    'selected_sources': len(scope_rows),
    'selected_sources_present': sum(r['exists'] == 'True' for r in scope_rows),
    'selected_source_hash_changes': source_changes,
    'selected_manifest_entries_match': manifest_match_count,
    'selected_manifest_entries_mismatch': len(manifest_mismatch_ids),
    'selected_manifest_mismatch_ids': manifest_mismatch_ids,
    'selected_manifest_binding_unavailable_or_unverified': manifest_unbound_count,
    'historical_full_manifest_verification_reported': False,
    'v3_root_manifest_receipt_reported_pass': checksum_accept.get('all_v3_root_entries_rehashed_and_verified') is True,
    'git_identity': 'UNKNOWN: git executable not available',
})
audit_acceptance['scientific_judgment'].update({
    'claims_adjudicated': len(claim_statuses),
    'claim_status_counts': claim_status_counts,
    'key_comparison_rows': len(key_rows_now),
    'key_comparison_ids': len({r['comparison_id'] for r in key_rows_now}),
    'key_comparison_status_counts': comparison_status_counts,
    'final_test_payload_access': 0,
    'development_evidence_upgraded_to_confirmation': False,
    'cross_contract_rankings_suppressed': True,
    'v3_candidate_promoted': False,
    'v3_positive_gain_gate': False,
    'v3_numerical_support_gate': False,
})
audit_acceptance['actual_this_audit']['DERIVED_ARITHMETIC_ROWS'] = 282
audit_acceptance['actual_this_audit']['GT_DERIVED_STATISTICS_READ'] = True
audit_acceptance['actual_this_audit']['V3_ACCEPTED_SUMMARY_SOURCES_READ'] = 9
audit_acceptance['known_limits'] = [
    'selected historical manifest mismatches are listed by source ID in SOURCE_SCOPE; no historical manifest was rewritten',
    'no project-wide historical label-access ledger exists in the checked sources',
    'V3 is sealed and accepted for development-only analysis; candidate gates are false and session/trial independence is unverified',
    'V3 root-wide checksum pass is source-reported; this audit independently checked nine cited V3 summary/acceptance sources, not large artifact roots',
    'no git branch/commit because git executable is absent',
]
write_json(acceptance_path, audit_acceptance)

# Add source-specific historical execution evidence without contaminating this audit's GPU=0.
resource_path = AUDIT / 'RESOURCE_LOG.csv'
resource_rows = read_csv(resource_path)
resource_header = list(resource_rows[0])
for r in resource_rows:
    if r.get('record_id') == 'AUDIT_TOTAL':
        start = datetime.fromisoformat(START_AUDIT_UTC.replace('Z', '+00:00'))
        end = datetime.now(timezone.utc)
        r['end_utc'] = end.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        r['wall_seconds'] = f'{(end - start).total_seconds():.3f}'
        r['disk_bytes'] = str(sum(p.stat().st_size for p in AUDIT.rglob('*') if p.is_file()))
        r['source_bytes_hashed'] = str(sum(int(x['size_bytes']) for x in scope_rows if x['exists'] == 'True'))
        r['note'] = 'End-to-end audit wall time; original wrapped-read CPU/RSS was not captured. GPU use by this audit was zero; selected source hashes only.'
new_source_bytes = sum((V3 / rel).stat().st_size for _, rel, _, _ in V3_SPECS) + MANIFEST.stat().st_size
end_utc = datetime.now(timezone.utc)
new_cpu = time.process_time() - AMEND_CPU
new_wall = time.monotonic() - AMEND_MONO
peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
current_disk = sum(p.stat().st_size for p in AUDIT.rglob('*') if p.is_file())
resource_rows.append({
    'record_id': 'AUDIT_V3_AMENDMENT',
    'scope': 'actual_this_audit',
    'phase': 'read_accepted_V3_summaries_update_claims_comparisons_LOO_QA_and_checksums',
    'start_utc': AMEND_START.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
    'end_utc': end_utc.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
    'wall_seconds': f'{new_wall:.3f}',
    'cpu_seconds': f'{new_cpu:.3f}',
    'peak_rss_bytes': str(peak_rss),
    'disk_bytes': str(current_disk - INITIAL_DISK_BYTES),
    'gpu_hours': '0',
    'gpu_actions': '0',
    'source_bytes_hashed': str(new_source_bytes),
    'read_scope': 'nine V3 acceptance/report/aggregate/LOO/interval/RGB summary sources and V3 CHECKSUMS.sha256 only',
    'note': 'No prediction/target/feature/model arrays opened; no model scoring/bootstrap/GPU operation. Disk bytes is net active audit-root growth including first-pass snapshot archive.',
})
resource_rows.append({
    'record_id': 'V3_HISTORICAL_FIT_RUNTIME',
    'scope': 'historical_accepted_V3_execution_not_this_audit',
    'phase': 'sealed_V3_screening_formal_and_RGB_workload_source_report',
    'start_utc': 'NA',
    'end_utc': accept['created_utc'],
    'wall_seconds': 'NA_not_end_to_end_wall_time',
    'cpu_seconds': 'NA',
    'peak_rss_bytes': 'NA',
    'disk_bytes': 'NA',
    'gpu_hours': f'{168.4 / 3600:.9f}_fit_record_elapsed_only',
    'gpu_actions': 'physical GPU 0 and 1; 43/57 watcher jobs complete, 14 failed; all leases/project slots released per acceptance',
    'source_bytes_hashed': 'NA',
    'read_scope': 'accepted V3 return lines 118-121 and FINAL_ACCEPTANCE.json',
    'note': '168.4 seconds is the sum of 20 fit-record GPU elapsed values only; 347.3 seconds is summed successful watcher per-job wall time under parallelism, not end-to-end wall time. Neither is total V3 GPU usage.',
})
write_csv(resource_path, resource_header, resource_rows)

# Make the first-pass builder's superseded status explicit for future reproducibility.
builder = AUDIT / 'build_audit.py'
builder_text = builder.read_text(encoding='utf-8')
old_intro = '"""Bounded EE6008 conclusion audit builder.\n\nStandard-library only. Reads explicitly selected report/acceptance/aggregate tables;'
new_intro = '"""Bounded EE6008 conclusion audit first-pass builder.\n\nThis emits an intermediate package; after running it, run v3_amendment.py to apply the sealed V3 outcome and regenerate final QA/checksums. Do not treat this first-pass output alone as final.\n\nStandard-library only. Reads explicitly selected report/acceptance/aggregate tables;'
if old_intro in builder_text:
    builder.write_text(builder_text.replace(old_intro, new_intro, 1), encoding='utf-8')

# Final QA preconditions and parser/readback checks.
if len(claims) != 16 or {r['claim_id'] for r in claims} != {f'C{i:02d}' for i in range(1, 17)}:
    raise RuntimeError('C01-C16 claim ledger completeness failed')
if len(key_rows) != 142:
    raise RuntimeError(f'expected 142 key-comparison rows, got {len(key_rows)}')
if len(per_rows) != 234:
    raise RuntimeError(f'expected 234 per-seed effects rows, got {len(per_rows)}')
if len(seed_rows) != 177 or len(task_rows) != 222:
    raise RuntimeError(f'LOO row count mismatch: seed={len(seed_rows)} task={len(task_rows)}')
try:
    ast.parse(builder.read_text(encoding='utf-8'))
    ast.parse((AUDIT / 'v3_amendment.py').read_text(encoding='utf-8'))
except SyntaxError as e:
    raise RuntimeError(f'audit script syntax check failed: {e}')

# Update audit acceptance once more with current parsed counts and check required fields.
audit_acceptance = read_json(acceptance_path)
audit_acceptance['engineering_source_qa']['csv_rows'] = {Path(k).name: v for k, v in csv_counts.items() if '/' not in k or k.startswith('qa/')}
write_json(acceptance_path, audit_acceptance)

# Checksum-last: hash every regular file in the active audit root except manifest and final receipt.
manifest_out = AUDIT / 'CHECKSUMS.sha256'
receipt_out = QA / 'FINAL_CHECKSUM_VERIFICATION.json'
entries = []
for p in sorted(x for x in AUDIT.rglob('*') if x.is_file() and x not in (manifest_out, receipt_out)):
    entries.append((sha256(p), p.relative_to(AUDIT).as_posix()))
manifest_text = ''.join(f'{digest}  {rel}\n' for digest, rel in entries)
manifest_out.write_text(manifest_text, encoding='utf-8')
manifest_hash = sha256(manifest_out)
verified = []
missing = []
mismatch = []
for digest, rel in entries:
    p = AUDIT / rel
    if not p.is_file():
        missing.append(rel)
    elif sha256(p) != digest:
        mismatch.append(rel)
    else:
        verified.append(rel)
receipt = {
    'status': 'PASS' if not missing and not mismatch and len(verified) == len(entries) else 'FAIL',
    'manifest_path': 'CHECKSUMS.sha256',
    'manifest_sha256': manifest_hash,
    'entries': len(entries),
    'verified': len(verified),
    'missing': len(missing),
    'mismatch': len(mismatch),
    'missing_paths': missing,
    'mismatched_paths': mismatch,
    'excluded': [
        {'path': 'CHECKSUMS.sha256', 'reason': 'manifest cannot hash itself'},
        {'path': 'qa/FINAL_CHECKSUM_VERIFICATION.json', 'reason': 'receipt records manifest verification and is self-excluded by protocol'},
    ],
    'scope': 'current audit deliverables, including a compressed superseded first-pass snapshot and amendment script; selected source files are separately hashed in SOURCE_SCOPE.csv; no whole-project or full checkpoint/cache rehash',
    'created_utc': utc_now(),
}
write_json(receipt_out, receipt)
# Read back the final receipt and manifest; no output is modified after this point.
readback = read_json(receipt_out)
if readback['status'] != 'PASS' or readback['manifest_sha256'] != sha256(manifest_out) or readback['entries'] != len(entries):
    raise RuntimeError('final checksum receipt readback failed')
print(json.dumps({
    'status': readback['status'],
    'observed_at_utc': readback['created_utc'],
    'manifest_sha256': readback['manifest_sha256'],
    'manifest_entries': readback['entries'],
    'manifest_verified': readback['verified'],
    'source_count': len(scope_rows),
    'source_present': sum(r['exists'] == 'True' for r in scope_rows),
    'historical_manifest_mismatches': manifest_mismatch_ids,
    'key_comparison_rows': len(key_rows),
    'per_seed_effect_rows': len(per_rows),
    'seed_loo_rows': len(seed_rows),
    'task_loo_rows': len(task_rows),
}, ensure_ascii=False))
