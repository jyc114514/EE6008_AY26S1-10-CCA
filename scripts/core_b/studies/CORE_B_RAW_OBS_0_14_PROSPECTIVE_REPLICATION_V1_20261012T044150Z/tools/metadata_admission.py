#!/usr/bin/env python3
# Metadata-only gate. It never opens frame, prefix, feature, or target files.
import argparse
import csv
import hashlib
import json
from pathlib import Path

TASK = 'RAW_OBS_0_14_PROSPECTIVE_RECORDED_DOMAIN_REPLICATION'
EXPOSURE_FIELDS = ('exposure_fit', 'exposure_normalizer', 'exposure_cv',
                   'exposure_dev', 'related_dev_selection', 'final_test', 'heldout')
EXPECTED_FIELDS = [
    'observation.arm_joints[14]',
    'observation.hand_joints[14]',
    'observation.leg_joints[15]',
    'action[28]',
]


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def sha(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def _flag_clear(value):
    return value is False or value == 0 or value == 'false'


def _valid_sha256(value):
    value = str(value or '')
    return len(value) == 64 and all(ch in '0123456789abcdef' for ch in value.lower())


def admit_records(records, freeze, prior_groups):
    if not isinstance(records, list):
        raise ValueError('manifest records must be a list')
    keys = []
    key_to_record = {}
    per_record_reasons = {}
    for i, r in enumerate(records):
        key = str(r.get('candidate_record_id', ''))
        if not key or key in key_to_record:
            raise ValueError('candidate_record_id must be present and unique')
        keys.append(key)
        key_to_record[key] = r
        reasons = []
        if r.get('synthetic_fixture') is True:
            reasons.append('synthetic_fixture')
        if r.get('dataset_role') != 'research_nonfinal':
            reasons.append('dataset_role_not_research_nonfinal')
        if r.get('proposed_split','confirmation')!='confirmation':
            reasons.append('proposed_cross_split_or_nonconfirmation_role')
        allowed = r.get('allowed_use')
        if not (allowed == TASK or isinstance(allowed, list) and TASK in allowed):
            reasons.append('prospective_use_not_authorized')
        registry_id = str(r.get('dataset_registry_entry_id', '')).strip()
        registry_hash = str(r.get('registry_entry_sha256', '')).strip()
        explicit_pool_id = str(r.get('explicit_nonfinal_pool_id', '')).strip()
        registration_ref = str(r.get('registration_evidence_ref', '')).strip()
        registry_registration = bool(registry_id) and _valid_sha256(registry_hash)
        explicit_registration = bool(explicit_pool_id) and bool(registration_ref)
        if not (registry_registration or explicit_registration):
            reasons.append('formal_nonfinal_registration_evidence_missing_or_invalid')
        if r.get('producer_id') != freeze.get('producer_id'):
            reasons.append('producer_mismatch_or_unknown')
        if freeze.get('hardware_config_reference_status') != 'verified':
            reasons.append('baseline_hardware_configuration_unknown_in_frozen_source')
        if r.get('producer_code_sha256') != freeze.get('producer_code_sha256'):
            reasons.append('producer_code_version_mismatch_or_unknown')
        if r.get('source_schema_sha256') != freeze.get('source_schema_sha256'):
            reasons.append('source_schema_mismatch_or_unknown')
        if r.get('conversion_sha256') != freeze.get('conversion_sha256'):
            reasons.append('conversion_mismatch_or_unknown')
        if r.get('field_order') != EXPECTED_FIELDS:
            reasons.append('field_order_mismatch_or_unknown')
        if r.get('unit_status') not in ('verified', 'unverified_same_native_numeric'):
            reasons.append('unit_status_missing_or_incompatible')
        if r.get('unit_status') == 'unverified_same_native_numeric' and r.get('numeric_space_equivalence_evidence') != 'same_producer_schema_conversion':
            reasons.append('numeric_space_equivalence_unproven')
        if not r.get('hardware_config_id') or r.get('hardware_config_relation') != 'same_as_reference_with_evidence' or not r.get('hardware_config_evidence_ref'):
            reasons.append('hardware_config_compatibility_unverified')
        for hash_field in ('source_manifest_sha256','episode_metadata_sha256','frame_index_sha256','prefix_capture_sha256'):
            value=str(r.get(hash_field,''))
            if not _valid_sha256(value):
                reasons.append(hash_field+'_missing_or_invalid')
        if not r.get('session_id') or not r.get('source_group_id') or not r.get('episode_id'):
            reasons.append('independent_unit_ids_missing')
        if r.get('session_independence_verified') is not True or not r.get('independence_evidence_ref'):
            reasons.append('session_or_trial_independence_unverified')
        for f in EXPOSURE_FIELDS:
            if not _flag_clear(r.get(f)):
                reasons.append('exposure_not_cleared:' + f)
        old_identity_refs = {str(r.get('source_group_id', '')), str(r.get('canonical_cluster_id', ''))}
        for field in ('source_group_aliases','session_id','session_aliases','duplicate_record_ids','duplicate_of_record_id'):
            value = r.get(field, []) if field.endswith('_aliases') or field.endswith('_ids') else [r.get(field, '')]
            if isinstance(value, (str, bytes)):
                value = [value]
            old_identity_refs.update(str(x) for x in (value or []) if x)
        if old_identity_refs.intersection(prior_groups):
            reasons.append('previous_OBS43_group_or_identity_alias_overlap')
        if int(r.get('hard_qa_anchor_count', 0) or 0) < 1:
            reasons.append('no_metadata_eligible_anchor')
        per_record_reasons[key] = reasons

    # Connected components over typed source-group/session aliases, canonical IDs, and duplicate links.
    uf = UnionFind(keys)
    maps = {}
    for key, r in key_to_record.items():
        for value in [r.get('source_group_id', ''), *(r.get('source_group_aliases', []) or [])]:
            if value:
                maps.setdefault(('source_group', str(value)), []).append(key)
        for value in [r.get('session_id', ''), *(r.get('session_aliases', []) or [])]:
            if value:
                maps.setdefault(('session', str(value)), []).append(key)
        canonical = r.get('canonical_cluster_id')
        if canonical:
            maps.setdefault(('canonical_cluster', str(canonical)), []).append(key)
        maps.setdefault(('record', key), []).append(key)
        for value in r.get('duplicate_record_ids', []) or []:
            maps.setdefault(('record', str(value)), []).append(key)
        other = r.get('duplicate_of_record_id')
        if other:
            maps.setdefault(('record', str(other)), []).append(key)
    for grouped in maps.values():
        for other in grouped[1:]:
            uf.union(grouped[0], other)

    components = {}
    for key in keys:
        components.setdefault(uf.find(key), []).append(key)
    component_rows = []
    strict = []
    for members in components.values():
        component_reasons = sorted({reason for k in members for reason in per_record_reasons[k]})
        component_groups = sorted({str(key_to_record[k].get('source_group_id', '')) for k in members if key_to_record[k].get('source_group_id')})
        aliases = sorted({str(a) for k in members for a in (key_to_record[k].get('source_group_aliases', []) or [])})
        if set(component_groups).intersection(prior_groups) or set(aliases).intersection(prior_groups):
            component_reasons.append('connected_component_overlaps_previous_OBS43_group')
            component_reasons = sorted(set(component_reasons))
        identity_tokens = set()
        for k in members:
            r = key_to_record[k]
            for value in [r.get('source_group_id', ''), *(r.get('source_group_aliases', []) or [])]:
                if value:
                    identity_tokens.add('source_group:' + str(value))
            for value in [r.get('session_id', ''), *(r.get('session_aliases', []) or [])]:
                if value:
                    identity_tokens.add('session:' + str(value))
            if r.get('canonical_cluster_id'):
                identity_tokens.add('canonical_cluster:' + str(r['canonical_cluster_id']))
        component_id = 'cc_' + sha('|'.join(sorted(identity_tokens)))[:20]
        episode_map={}
        if not component_reasons:
            for k in members:
                r=key_to_record[k]
                ekey=str(r['episode_id'])
                raw_hash=r.get('episode_metadata_sha256') or r.get('source_manifest_sha256') or sha(k)
                episode_map[ekey]={'episode_id':ekey,'episode_metadata_hash':raw_hash,
                    'eligible_anchor_count':int(r.get('hard_qa_anchor_count',0) or 0),
                    'source_group_id':str(r.get('source_group_id',''))}
        cluster = {'canonical_cluster_id': component_id, 'record_ids': sorted(members),
                   'source_group_ids': component_groups, 'record_count': len(members),
                   'eligible_episodes':list(episode_map.values()),
                   'strict_admitted': not component_reasons, 'reasons': component_reasons}
        component_rows.append(cluster)
        if not component_reasons:
            strict.append(cluster)
        else:
            for k in members:
                per_record_reasons[k] = sorted(set(per_record_reasons[k] + component_reasons))

    out_rows = []
    for key, r in key_to_record.items():
        out_rows.append({'candidate_record_id': key,
                         'dataset_id': r.get('dataset_id', ''),
                         'source_group_id': r.get('source_group_id', ''),
                         'session_id': r.get('session_id', ''),
                         'episode_id': r.get('episode_id', ''),
                         'strict_admitted': not per_record_reasons[key],
                         'reason_codes': ';'.join(per_record_reasons[key])})
    return {'status': 'PASS_METADATA_ONLY',
            'candidate_records': len(records),
            'connected_components': len(components),
            'strict_pool_clusters': len(strict),
            'strict_pool_status': 'AVAILABLE' if strict else 'NO_STRICT_POOL',
            'new_outcomes_opened': False,
            'raw_frame_payloads_opened': False,
            'protected_split_access': 0,
            'components': component_rows,
            'records': out_rows}


def main():
    ap = argparse.ArgumentParser(description='Metadata-only admission gate; does not open numeric payloads.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--prior-groups', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    freeze = json.loads(Path(args.freeze).read_text(encoding='utf-8'))
    with Path(args.prior_groups).open(encoding='utf-8', newline='') as f:
        prior = {r['source_group_id'] for r in csv.DictReader(f)}
    result = admit_records(manifest.get('records', []), freeze, prior)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'DATA_ADMISSION_RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    with (out/'DATA_ADMISSION_RECORDS.csv').open('w', encoding='utf-8', newline='') as f:
        rows = result['records']
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['candidate_record_id','strict_admitted','reason_codes'])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: result[k] for k in ('status','candidate_records','connected_components','strict_pool_clusters','strict_pool_status')}))


if __name__ == '__main__':
    main()
