import os
#!/usr/bin/env python3
# Readiness-package QA only. It never reads new future labels or protected splits.
import csv
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OBS=PROJECT/'reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    wall0,cpu0=time.perf_counter(),time.process_time()
    disk0=shutil.disk_usage(ROOT).free
    required=['DATA_GAP.md','VERIFIED_DATA_GAP.md','DATA_COLLECTION_INTAKE.md',
      'SCOPE_AMENDMENT.md','INDEPENDENT_CONFIRMATION_SPEC.md','README_VALIDATION_READY.md',
      'CURRENT_BENCHMARK_CLOSEOUT.md','frozen_model/MANIFEST.json','frozen_model/ARX_OBS43_FULL_TRAIN.npz',
      'frozen_model/obs43_diagnostic.py','frozen_model/prepare_v3_inputs.py',
      'schemas/IMPORT_SCHEMA.json','schemas/examples/synthetic_candidate_rejected.json',
      'data/REQUIRED_DATA_MANIFEST.csv','data/DATA_ADMISSION.csv','data/DATA_ADMISSION.json',
      'data/PRIOR_EXPOSURE_GROUPS.csv','data/CLUSTER_GRAPH.csv','data/N_PLAN.json',
      'data/N_SENSITIVITY.csv','data/VERIFIED_DATA_GAP.csv','protocol/DATA_SPACE_LOCK.json',
      'protocol/MODEL_FREEZE.json','protocol/PROSPECTIVE_PROTOCOL.json','protocol/SAMPLE_LOCK.json',
      'protocol/FORECAST_LOCK_TEMPLATE.json','protocol/REGISTRY_SEARCH.json',
      'protocol/HARDWARE_CONFIG_REFERENCE_GAP.json','protocol/SOURCE_HASH_LOCK.json','protocol/PREFIT_LOCK_HASHES.json',
      'tools/metadata_admission.py','tools/sample_lock.py','tools/prefix_export.py',
      'tools/frozen_predict.py','tools/independent_score.py','tools/run_study.py',
      'tools/seal_sample.py','tools/replay_frozen_fixture.py','tools/validate_package.py',
      'qa/SYNTHETIC_TOOL_TESTS.json','qa/FROZEN_MODEL_FIXTURE_REPLAY_R002.json',
      'qa/FROZEN_FIXTURE_PREFIX.npz','qa/FINAL_ACCEPTANCE.json','qa/FINAL_DELIVERY_QA.json',
      'returns/RETURN_TO_CHATGPT.zh-CN.md','returns/RETURN_TO_CHATGPT.zh-CN.json']
    missing=[x for x in required if not (ROOT/x).is_file()]
    if missing: raise AssertionError('required readiness artifacts missing: '+repr(missing))
    json_files=list(ROOT.rglob('*.json'))
    for p in json_files: json.loads(p.read_text(encoding='utf-8'))
    csv_counts={}
    for p in ROOT.rglob('*.csv'):
        with p.open(encoding='utf-8',newline='') as f:
            rd=csv.reader(f); header=next(rd,None)
            if not header: raise AssertionError('empty CSV: '+str(p))
            count=0
            for row in rd:
                if len(row)!=len(header): raise AssertionError(f'CSV width error: {p}:{count+2}')
                count+=1
            csv_counts[str(p.relative_to(ROOT))]=count
    registry=json.loads((ROOT/'source/SOURCE_REGISTRY.json').read_text(encoding='utf-8'))
    mismatches=[]
    for x in registry['sources']:
        p=Path(x['path']); got=sha(p) if p.is_file() else 'MISSING'
        if got!=x['sha256']: mismatches.append({'name':x['name'],'expected':x['sha256'],'actual':got})
    if mismatches: raise AssertionError('source hash mismatches: '+repr(mismatches[:3]))
    freeze_path=ROOT/'protocol/MODEL_FREEZE.json'
    freeze=json.loads(freeze_path.read_text())
    space_path=ROOT/'protocol/DATA_SPACE_LOCK.json'
    space_hash=sha(space_path)
    model_manifest=json.loads((ROOT/'frozen_model/MANIFEST.json').read_text())
    if freeze.get('data_space_lock_sha256')!=space_hash or model_manifest.get('data_space_lock_sha256')!=space_hash:
        raise AssertionError('data-space lock hash is stale in frozen model manifests')
    if model_manifest.get('packaged_model_freeze_sha256')!=sha(freeze_path):
        raise AssertionError('frozen package manifest does not bind the model freeze')
    source_lock=json.loads((ROOT/'protocol/SOURCE_HASH_LOCK.json').read_text())
    for item in source_lock['files']:
        path=ROOT/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:
            raise AssertionError('source hash lock mismatch: '+item['path'])
    prefit_lock=json.loads((ROOT/'protocol/PREFIT_LOCK_HASHES.json').read_text())
    if prefit_lock.get('source_hash_lock_sha256')!=sha(ROOT/'protocol/SOURCE_HASH_LOCK.json'):
        raise AssertionError('prefit lock does not bind source hash lock')
    for item in prefit_lock['locked_objects']:
        path=ROOT/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:
            raise AssertionError('prefit object hash mismatch: '+item['path'])
    model_copy=ROOT/'frozen_model/ARX_OBS43_FULL_TRAIN.npz'
    if sha(model_copy)!=freeze['packaged_model_sha256']: raise AssertionError('frozen model copy hash mismatch')
    if sha(OBS/'models/ARX_OBS43_FULL_TRAIN.npz')!=freeze['source_model_sha256']: raise AssertionError('original model changed')
    with np.load(model_copy,allow_pickle=False) as z:
        if any(z[k].dtype.hasobject for k in z.files): raise AssertionError('object array in frozen model')
        model_keys=list(z.files)
    admit=json.loads((ROOT/'data/DATA_ADMISSION.json').read_text())
    nplan=json.loads((ROOT/'data/N_PLAN.json').read_text())
    sample=json.loads((ROOT/'protocol/SAMPLE_LOCK.json').read_text())
    run=json.loads((ROOT/'RUN_STATUS.json').read_text())
    owner=json.loads((ROOT/'protocol/OWNER_STATE.json').read_text())
    final_acceptance=json.loads((ROOT/'qa/FINAL_ACCEPTANCE.json').read_text())
    delivery_qa=json.loads((ROOT/'qa/FINAL_DELIVERY_QA.json').read_text())
    qa=json.loads((ROOT/'qa/SYNTHETIC_TOOL_TESTS.json').read_text())
    fixture=json.loads((ROOT/'qa/FROZEN_MODEL_FIXTURE_REPLAY_R002.json').read_text())
    if admit['strict_pool_clusters']!=0 or nplan['n_required_independent_clusters']!=270:
        raise AssertionError('prepared status/N changed')
    if admit.get('frozen_hardware_config_reference_status')!='UNKNOWN' or len(admit.get('strict_admission_blockers',[]))<2:
        raise AssertionError('strict admission blockers are not fully disclosed')
    if sample.get('sample_sealed') or run.get('confirmation_executed') or run.get('new_scores_read')!=0:
        raise AssertionError('current prepared status contains sample/score access')
    if run.get('status')!='prepared_not_executed' or run.get('strict_pool_clusters')!=0:
        raise AssertionError('run status is inconsistent with no strict pool')
    if owner.get('status')!='FINISHED' or owner.get('old_roots_read_only') is not True:
        raise AssertionError('owner state is not sealed or old-root boundary changed')
    if final_acceptance.get('status') not in ('PENDING_FINAL_PACKAGE_QA','PASS_PREPARED_NOT_EXECUTED') or final_acceptance.get('confirmation_executed'):
        raise AssertionError('final acceptance misstates readiness as confirmation')
    if delivery_qa.get('status') not in ('PENDING','PASS') or delivery_qa.get('human_review_status')!='pending':
        raise AssertionError('final delivery QA state is inconsistent')
    if qa.get('status')!='PASS' or qa.get('test_count')!=22 or qa.get('confirmation_result_written'):
        raise AssertionError('synthetic QA did not pass or was mislabeled')
    if fixture.get('status')!='PASS' or fixture.get('new_labels_opened') or fixture.get('confirmation_result'):
        raise AssertionError('frozen training fixture QA failed or was mislabeled')
    if (ROOT/'results/PROSPECTIVE_SCORE.json').exists() or (ROOT/'protocol/FORECAST_LOCK.json').exists() or (ROOT/'logs/OUTCOME_READ_RECEIPT.json').exists():
        raise AssertionError('new score/forecast/outcome receipt was written despite absent pool')
    if (ROOT/'predictions/FORECASTS_H30_NPZ.npz').exists():
        raise AssertionError('prospective forecasts were written despite absent pool')
    if list((ROOT/'data').glob('*labels*')): raise AssertionError('new target/label artifact exists')
    run_help=subprocess.run([sys.executable,str(ROOT/'tools/run_study.py'),'--help'],
        capture_output=True,text=True,timeout=20,env={'PYTHONDONTWRITEBYTECODE':'1','PATH':__import__('os').environ.get('PATH','')})
    if run_help.returncode!=0 or 'admit' not in run_help.stdout or 'score' not in run_help.stdout:
        raise AssertionError('runner --help failed')
    usage=resource.getrusage(resource.RUSAGE_SELF)
    report={'status':'PASS','readiness_status':'prepared_not_executed','required_artifacts':len(required),
      'json_files_parsed':len(json_files),'csv_files_parsed':len(csv_counts),
      'csv_row_counts':csv_counts,'source_registry_entries':len(registry['sources']),
      'source_hash_mismatches':0,'frozen_model_sha256':freeze['packaged_model_sha256'],
      'data_space_lock_sha256':space_hash,'source_hash_lock_files':len(source_lock['files']),
      'prefit_locked_objects':len(prefit_lock['locked_objects']),
      'frozen_model_member_keys':model_keys,'strict_pool_clusters':0,
      'n_required_independent_clusters':270,'sample_sealed':False,
      'new_labels_read':0,'new_scores_read':0,'new_model_fits':0,'gpu_actions':0,
      'final_heldout_access':0,'runner_help':'PASS','human_review_status':'pending',
      'submission_ready':False,'checksum_manifest':'generated_after_package_QA_and_verified_separately',
      'wall_seconds':time.perf_counter()-wall0,
      'process_cpu_seconds':time.process_time()-cpu0,'peak_rss_bytes':int(usage.ru_maxrss*1024),
      'disk_free_bytes_before':disk0,'disk_free_bytes_after':shutil.disk_usage(ROOT).free}
    out=ROOT/'qa/PACKAGE_QA.json'
    out.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    with (ROOT/'logs/STAGE_RESOURCES.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'stage':'readiness_package_qa','wall_seconds':report['wall_seconds'],
            'process_cpu_seconds':report['process_cpu_seconds'],'peak_rss_bytes':report['peak_rss_bytes'],
            'disk_free_bytes_before':disk0,'disk_free_bytes_after':report['disk_free_bytes_after'],
            'new_fits':0,'gpu_actions':0,'new_labels_read':0},sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':main()
