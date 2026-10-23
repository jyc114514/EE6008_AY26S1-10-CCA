#!/usr/bin/env python3
"""Write a supplemental scope-partitioned receipt without changing the original partial audit."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp=path.with_name('.'+path.name+'.part')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    tmp=path.with_name('.'+path.name+'.part')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def main() -> int:
    root=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V2_20261018T044202Z'))
    final_acceptance_path=root/'FINAL_ACCEPTANCE.json'
    regression_path=root/'RUNNER_METADATA_CPU_REGRESSION.json'
    runner_path=root/'source/replay_one_cell.py'
    test_path=root/'source/verify_runner_metadata_cpu_fixture.py'
    acceptance=json.loads(final_acceptance_path.read_text(encoding='utf-8'))
    regression=json.loads(regression_path.read_text(encoding='utf-8'))
    if acceptance.get('status')!='partial' or acceptance.get('AUDIT_WORK_COMPLETE') is not True or acceptance.get('HANDOFF_ACCEPTANCE_COMPLETE') is not False:
        raise AssertionError('original audit status is not the expected partial/audit-complete/handoff-incomplete state')
    runner_sha=digest(runner_path)
    test_sha=digest(test_path)
    if regression.get('status')!='PASS_POST_FIX_RUNNER_CPU_REGRESSION' or regression.get('runner_sha256')!=runner_sha or regression.get('test_source_sha256')!=test_sha:
        raise AssertionError('CPU regression receipt is not bound to the current final runner and fixture source')
    if regression.get('train_rows')!=5 or regression.get('dev_rows')!=3 or regression.get('train_dev_ids_disjoint') is not True or regression.get('prediction_arrays_bitwise_unchanged') is not True or regression.get('prediction_arrays_changed')!=0:
        raise AssertionError('CPU fixture receipt does not satisfy requested train/dev metadata regression')
    for split in regression['splits']:
        bundle=Path(split['bundle_path'])
        if digest(bundle)!=split['bundle_sha256']:
            raise AssertionError(f"fixture bundle hash changed: {split['split']}")
        if not all(split['metadata_checks'].values()) or split['prediction_array_sha256_before']!=split['prediction_array_sha256_after']:
            raise AssertionError(f"fixture split failed: {split['split']}")
    final_acceptance_sha=digest(final_acceptance_path)
    scope={
        'schema_version':'1.0',
        'receipt_id':'VIDEOMAE_V2_SCOPE_PARTITIONED_ACCEPTANCE_20261018',
        'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
        'overall_status':'partial',
        'original_audit_status':'partial',
        'original_audit_status_preserved':True,
        'original_final_acceptance_path':str(final_acceptance_path),
        'original_final_acceptance_sha256':final_acceptance_sha,
        'audit_work_complete':True,
        'handoff_acceptance_complete':False,
        'scope_acceptance':{
            'core_a_reuse':{
                'status':'accepted_reused',
                'basis':'V1 accepted Core A evidence was reused; this supplemental receipt does not re-run it.'
            },
            'videomae_safe_weights_and_structure':{
                'status':'accepted_with_caveats',
                'basis':'12/12 safetensors strict reload and structure checks passed; provenance grade remains hash_bound_exporter_attested.'
            },
            'videomae_cached_feature_predictor':{
                'status':'accepted_with_caveats',
                'basis':'12/12 cells and 24/24 train/dev outputs passed the predeclared engineering gate and were bitwise equal to historical GPU references; independent scorer QA passed.',
                'profile_scope':'current_profile_only; historical GPU UUID/name/architecture/cuDNN are unavailable, so original hardware identity remains unverified.',
                'portability_scope':'frozen_feature_predictor_replay',
                'runner_metadata_note':'initial train metadata packing failure was preserved; the later metadata-only correction changed zero prediction arrays.'
            },
            'post_fix_runner_metadata_cpu_regression':{
                'status':'accepted',
                'receipt_path':str(regression_path),
                'receipt_sha256':digest(regression_path),
                'runner_path':str(runner_path),
                'final_runner_sha256':runner_sha,
                'fixture_source_path':str(test_path),
                'fixture_source_sha256':test_sha,
                'train_rows':5,'dev_rows':3,
                'train_dev_lengths_differ':True,'train_dev_ids_disjoint':True,
                'empty_output_directory':True,
                'metadata_fields_exact':['original_task_index','anchor_frame','anchor_timestamp'],
                'prediction_arrays_bitwise_unchanged':True,
                'gpu_accessed':False,'targets_loaded':False
            },
            'historical_gpu_hardware_identity':{
                'status':'not_tested',
                'reason':'historical run records contain physical index only and omit UUID, model, architecture and cuDNN.'
            },
            'exporter_source_evidence_grade':{
                'status':'accepted_with_caveats',
                'grade':'hash_bound_exporter_attested',
                'reason':'no independently bound pre-export tensor catalogue or separate exporter execution receipt.'
            },
            'raw_video_end_to_end':{
                'status':'not_tested',
                'reason':'this audit scope uses archived cached features; no video decode or encoder extraction was run.'
            },
            'team_reference_erratum':{
                'status':'needs_fix',
                'owner':'team report owner',
                'pending':'provide the erratum link for the prior sealed TEAM STATE-CTRL H30 summary; old result remains untouched.'
            },
            'v_jepa_reference_provenance':{
                'status':'not_tested',
                'owner':'V-JEPA cache/extractor owner',
                'pending':'provide the historical extractor source SHA and intended 64-row cache bridge reference; separate from VideoMAE acceptance.'
            }
        },
        'runner_regression_evidence':{
            'status':regression['status'],
            'runner_sha256':runner_sha,
            'fixture_source_sha256':test_sha,
            'fixture_receipt_sha256':digest(regression_path),
            'split_bundle_sha256':{x['split']:x['bundle_sha256'] for x in regression['splits']},
            'prediction_array_sha256':{x['split']:x['prediction_array_sha256_after'] for x in regression['splits']}
        },
        'non_goals_and_safety':{
            'new_training':0,'optimizer_updates':0,'new_encoder_extraction':0,
            'unrestricted_pickle_loads':0,'final_test_access':0,'old_source_root_writes':0
        },
        'owner_followups':[
            {'owner':'team report owner','action':'Provide the prior sealed TEAM STATE-CTRL H30 erratum link.'},
            {'owner':'V-JEPA cache/extractor owner','action':'Provide the historical extractor SHA and 64-row bridge reference.'},
            {'owner':'safe-weight exporter owner','action':'If provenance grade must be upgraded, provide an independently bound pre-export tensor catalogue and exporter execution receipt.'}
        ]
    }
    scope_path=root/'SCOPE_ACCEPTANCE_RECEIPT.json'
    write_json(scope_path,scope)

    registry_path=root/'SOURCE_REGISTRY.csv'
    with registry_path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f));fields=list(rows[0])
    def upsert(role: str,path: Path,note: str) -> None:
        path=path.resolve()
        row=next((r for r in rows if r['role']==role),None)
        if row is None:
            row={k:'' for k in fields};rows.append(row)
        row.update({'role':role,'path':str(path),'exists':str(path.is_file()),
                    'bytes':str(path.stat().st_size) if path.is_file() else '',
                    'sha256':digest(path) if path.is_file() else '',
                    'status':'PASS' if path.is_file() else 'MISSING',
                    'expected_bytes':'','expected_sha256':'','note':note})
    upsert('audit_prediction_child_corrected',runner_path,'final runner SHA bound by post-fix 5-train/3-dev CPU metadata regression')
    upsert('runner_metadata_cpu_fixture_source',test_path,'unequal train/dev lengths and disjoint IDs; validates per-split task/frame/timestamp and unchanged predictions')
    upsert('runner_metadata_cpu_fixture_receipt',regression_path,'PASS; bound to final runner SHA and retained output bundles')
    upsert('scope_partitioned_acceptance_receipt',scope_path,'supplemental scope decisions; original FINAL_ACCEPTANCE status remains partial')
    upsert('scope_acceptance_finalizer',Path(__file__),'generates supplemental receipt without changing original acceptance status')
    for row in rows:
        local=Path(row.get('path',''))
        if local.is_absolute() and local.is_file() and local.resolve().is_relative_to(root.resolve()):
            row['exists']='True';row['bytes']=str(local.stat().st_size);row['sha256']=digest(local)
    tmp=registry_path.with_name('.'+registry_path.name+'.part')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    os.replace(tmp,registry_path)

    checksum_path=root/'CHECKSUMS.sha256'
    files=sorted(p for p in root.rglob('*') if p.is_file() and p!=checksum_path)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise AssertionError('symlink found in audit root')
    lines=[f'{digest(p)}  {p.relative_to(root).as_posix()}\n' for p in files]
    checksum_path.write_text(''.join(lines),encoding='utf-8')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=checksum_path}
    listed={line[66:].rstrip('\n') for line in lines}
    if listed!=actual:
        raise AssertionError('CHECKSUMS file coverage differs from actual files')
    subprocess.run(['sha256sum','--check','--quiet','CHECKSUMS.sha256'],cwd=root,check=True)
    if json.loads(final_acceptance_path.read_text(encoding='utf-8')).get('status')!='partial':
        raise AssertionError('original FINAL_ACCEPTANCE status changed')
    print(json.dumps({'status':'PASS_SCOPE_PARTITIONED_ACCEPTANCE_RECEIPT',
                      'original_audit_status':'partial','final_runner_sha256':runner_sha,
                      'fixture':'5 train / 3 dev; metadata PASS; prediction arrays unchanged',
                      'checksums_verified':len(lines)},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
