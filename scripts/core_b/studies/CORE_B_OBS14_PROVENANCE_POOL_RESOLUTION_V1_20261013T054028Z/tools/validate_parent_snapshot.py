#!/usr/bin/env python3
"""Read-only validation of the sealed 2026-10-12 parent package."""
import os
from pathlib import Path
import argparse, hashlib, json, time, resource, shutil
READY=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_RAW_OBS_0_14_PROSPECTIVE_REPLICATION_V1_20261012T044150Z'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description='Read-only validation of the immutable parent readiness package.')
 ap.add_argument('--out',required=True)
 a=ap.parse_args(); t0=time.perf_counter(); c0=time.process_time(); errors=[]
 manifest=READY/'CHECKSUMS.sha256'; entries=[]
 for n,line in enumerate(manifest.read_text(encoding='utf-8').splitlines(),1):
  parts=line.split('  ',1)
  if len(parts)!=2: errors.append(f'checksum_line_{n}_format'); continue
  expected,rel=parts; p=(READY/rel).resolve()
  try: p.relative_to(READY.resolve())
  except ValueError: errors.append(f'checksum_line_{n}_escapes_parent'); continue
  if not p.is_file() or sha(p)!=expected: errors.append(f'checksum_mismatch:{rel}')
  entries.append(rel)
 freeze=json.loads((READY/'protocol/MODEL_FREEZE.json').read_text())
 sample=json.loads((READY/'protocol/SAMPLE_LOCK.json').read_text())
 run=json.loads((READY/'RUN_STATUS.json').read_text())
 owner=json.loads((READY/'protocol/OWNER_STATE.json').read_text())
 admission=json.loads((READY/'data/DATA_ADMISSION.json').read_text())
 accept=json.loads((READY/'qa/FINAL_ACCEPTANCE.json').read_text())
 source_lock=json.loads((READY/'protocol/SOURCE_HASH_LOCK.json').read_text())
 prefit=json.loads((READY/'protocol/PREFIT_LOCK_HASHES.json').read_text())
 source_errors=[]
 for x in source_lock['files']:
  q=READY/x['path']
  if not q.is_file() or sha(q)!=x['sha256']: source_errors.append(x['path'])
 prefit_errors=[]
 for x in prefit['locked_objects']:
  q=Path(x['path'])
  if not q.is_file() or sha(q)!=x['sha256']: prefit_errors.append(x['path'])
 if len(entries)!=286: errors.append('final_checksum_entry_count_not_286')
 if source_errors: errors.append('source_hash_lock_mismatch')
 if prefit_errors: errors.append('prefit_hash_lock_mismatch')
 model=READY/'frozen_model/ARX_OBS43_FULL_TRAIN.npz'
 if sha(model)!=freeze.get('packaged_model_sha256'): errors.append('frozen_model_hash_mismatch')
 if owner.get('status')!='FINISHED' or run.get('status')!='prepared_not_executed': errors.append('parent_owner_or_run_state_mismatch')
 if run.get('confirmation_executed') is not False or sample.get('sample_sealed') is not False: errors.append('parent_is_not_unexecuted_prepared_state')
 if admission.get('strict_pool_clusters')!=0 or run.get('strict_pool_clusters')!=0: errors.append('parent_strict_pool_not_zero')
 if (READY/'protocol/FORECAST_LOCK.json').exists() or (READY/'results/PROSPECTIVE_SCORE.json').exists() or (READY/'logs/OUTCOME_READ_RECEIPT.json').exists(): errors.append('unexpected_parent_forecast_or_score_artifact')
 report={'status':'PASS' if not errors else 'FAIL','parent_root':str(READY),'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'final_checksum_entries':len(entries),'final_checksum_mismatches':0 if not errors else len([e for e in errors if e.startswith('checksum_mismatch:')]),'source_hash_lock_entries':len(source_lock['files']),'source_hash_lock_mismatches':len(source_errors),'prefit_locked_objects':len(prefit['locked_objects']),'prefit_hash_mismatches':len(prefit_errors),'model_sha256':sha(model),'parent_status':run.get('status'),'parent_scope_status':accept.get('readiness_status'),'strict_pool_clusters':admission.get('strict_pool_clusters'),'n_required':accept.get('n_required_independent_clusters'),'sample_sealed':sample.get('sample_sealed'),'parent_modified':False,'old_validator_executed':False,'gpu_actions':0,'new_labels_read':0,'errors':errors,'wall_seconds':time.perf_counter()-t0,'process_cpu_seconds':time.process_time()-c0,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),'disk_free_bytes':shutil.disk_usage(READY).free}
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2,sort_keys=True))
 if errors: raise SystemExit(1)
if __name__=='__main__': main()
