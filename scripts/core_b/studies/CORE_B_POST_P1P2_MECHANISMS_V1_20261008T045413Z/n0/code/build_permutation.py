#!/usr/bin/env python3
"""Build/validate the single pre-registered cross-source-group N0 donor bijection."""
from __future__ import annotations
import os
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
P1_ROOT = PROJECT / 'reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z'
V3_ROOT = PROJECT / 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
SEED = 20261008
FIELDS = (
    'permutation_position', 'recipient_anchor_id', 'recipient_episode_index', 'recipient_source_group_id', 'recipient_input_row',
    'donor_anchor_id', 'donor_episode_index', 'donor_source_group_id', 'donor_input_row', 'permutation_seed', 'final_test_access',
)


def build_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    # Only ID/split fields are indexed; target/future arrays are never opened.
    with np.load(V3_ROOT / 'data/V3_MAIN_INPUTS.npz', allow_pickle=False) as source:
        anchor_id = source['anchor_id'].copy()
        episode_index = source['episode_index'].copy()
        split_code = source['split_code'].copy()
    if set(np.unique(split_code).tolist()) != {0, 1}:
        raise RuntimeError('primary input contains a split outside canonical train/dev')
    dev_rows = np.flatnonzero(split_code == 1).astype(np.int64)
    if len(dev_rows) != 2322 or len(np.unique(anchor_id[dev_rows])) != len(dev_rows):
        raise RuntimeError('N0 permutation requires the exact 2322 unique P1 dev anchor IDs')
    source_by_episode: dict[int, str] = {}
    with (P1_ROOT / 'inventory/SOURCE_GROUP_MAP.csv').open(newline='', encoding='utf-8') as handle:
        for record in csv.DictReader(handle):
            if record['split'] == 'dev':
                source_by_episode[int(record['episode_index'])] = record['source_group_id']
    if len(source_by_episode) != 603:
        raise RuntimeError('P1 dev source-group map does not contain exactly 603 episodes')
    by_group: dict[str, list[tuple[str, int, int]]] = {}
    for row in dev_rows:
        episode = int(episode_index[row])
        group = source_by_episode[episode]
        by_group.setdefault(group, []).append((str(anchor_id[row]), episode, int(row)))
    for group in by_group:
        by_group[group].sort(key=lambda value: value[0])
    ordered_groups = sorted(by_group, key=lambda group: (hashlib.sha256(f'{SEED}|{group}'.encode()).hexdigest(), group))
    ordered = [(aid, episode, row, group) for group in ordered_groups for aid, episode, row in by_group[group]]
    count = len(ordered)
    max_group_rows = max(len(v) for v in by_group.values())
    if count != 2322 or len(ordered_groups) != 603 or 2 * max_group_rows > count:
        raise RuntimeError('the locked full-population cross-group cyclic bijection is infeasible')
    output = []
    for i, recipient in enumerate(ordered):
        donor = ordered[(i + max_group_rows) % count]
        if recipient[3] == donor[3]:
            raise RuntimeError(f'same-group recipient/donor: {recipient[0]}')
        output.append({
            'permutation_position': i,
            'recipient_anchor_id': recipient[0], 'recipient_episode_index': recipient[1], 'recipient_source_group_id': recipient[3], 'recipient_input_row': recipient[2],
            'donor_anchor_id': donor[0], 'donor_episode_index': donor[1], 'donor_source_group_id': donor[3], 'donor_input_row': donor[2],
            'permutation_seed': SEED, 'final_test_access': 0,
        })
    if len({r['donor_anchor_id'] for r in output}) != count or {r['recipient_anchor_id'] for r in output} != {r['donor_anchor_id'] for r in output}:
        raise RuntimeError('donor mapping is not a unique bijection')
    meta = {
        'status': 'PASS_PRELOCKED_BIJECTION', 'permutation_seed': SEED, 'population': 'P1 common primary development anchors',
        'rows': count, 'source_groups': len(ordered_groups), 'max_group_anchor_rows_m': max_group_rows,
        'condition_2m_le_N': 2 * max_group_rows <= count,
        'group_order': 'ascending SHA256(f"20261008|source_group_id"), source_group_id tie-break',
        'within_group_order': 'ascending anchor_id', 'donor_rule': 'donor(list[i]) = list[(i+m) mod N]',
        'donors_unique': True, 'recipient_donor_sets_equal': True, 'all_recipient_donor_source_groups_differ': True,
        'no_target_or_score_inputs_used': True, 'target_arrays_opened': False, 'final_test_access': 0,
        'source_group_map_sha256': hashlib.sha256((P1_ROOT / 'inventory/SOURCE_GROUP_MAP.csv').read_bytes()).hexdigest(),
        'input_v3_sha256': json.loads((P1_ROOT / 'protocol/LOCKED_INPUT_IDENTITIES.json').read_text(encoding='utf-8'))['v3_input_sha256'],
    }
    return output, meta


def validate_existing(path: Path) -> dict[str, object]:
    expected, meta = build_rows()
    with path.open(newline='', encoding='utf-8') as handle:
        actual = list(csv.DictReader(handle))
    if len(actual) != len(expected):
        raise RuntimeError('saved permutation row count differs')
    for i, (left, right) in enumerate(zip(actual, expected)):
        for key in FIELDS:
            if str(left[key]) != str(right[key]):
                raise RuntimeError(f'saved permutation mismatch at row {i}, field {key}')
    meta['pair_permutation_path'] = 'protocol/PAIR_PERMUTATION.csv'
    meta['pair_permutation_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-existing', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.validate_existing == args.write:
        parser.error('choose exactly one of --validate-existing or --write')
    path = ROOT / 'protocol/PAIR_PERMUTATION.csv'
    meta_path = ROOT / 'protocol/PERMUTATION_HASH.json'
    if args.validate_existing:
        if not path.is_file() or not meta_path.is_file():
            raise FileNotFoundError('locked permutation artifacts not present')
        actual = validate_existing(path)
        recorded = json.loads(meta_path.read_text(encoding='utf-8'))
        for key in ('permutation_seed', 'rows', 'source_groups', 'max_group_anchor_rows_m', 'pair_permutation_sha256', 'source_group_map_sha256', 'input_v3_sha256'):
            if str(recorded.get(key)) != str(actual.get(key)):
                raise RuntimeError(f'permutation metadata mismatch: {key}')
        print(json.dumps({'status': 'PASS', 'rows': len(list(csv.DictReader(path.open()))), 'sha256': actual['pair_permutation_sha256']}, sort_keys=True))
        return 0
    if path.exists() or meta_path.exists():
        raise FileExistsError('refusing to overwrite an existing permutation lock')
    rows, meta = build_rows()
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    meta['pair_permutation_path'] = 'protocol/PAIR_PERMUTATION.csv'
    meta['pair_permutation_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta['source_group_map_sha256'] = hashlib.sha256((P1_ROOT / 'inventory/SOURCE_GROUP_MAP.csv').read_bytes()).hexdigest()
    meta['input_v3_sha256'] = json.loads((P1_ROOT / 'protocol/LOCKED_INPUT_IDENTITIES.json').read_text())['v3_input_sha256']
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'rows': len(rows), 'sha256': meta['pair_permutation_sha256']}, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
