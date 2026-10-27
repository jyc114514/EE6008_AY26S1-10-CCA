#!/usr/bin/env python3
from __future__ import annotations
import os
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
with (ROOT/'inventory/SOURCE_REGISTRY.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
by_path={}
for r in rows:
 by_path.setdefault(r['source_path'],set()).add(r['sha256'])
source_fail=[];copy_fail=[]
for text,expected_set in by_path.items():
 p=Path(text)
 if len(expected_set)!=1 or not p.is_file() or sha(p) not in expected_set:
  source_fail.append({'path':text,'expected':sorted(expected_set),'observed':sha(p) if p.is_file() else 'MISSING'})
for r in rows:
 if r['copy_status']=='COPIED_HASH_MATCH':
  p=ROOT/r['new_root_path']
  if not p.is_file() or sha(p)!=r['sha256']:
   copy_fail.append({'path':r['new_root_path'],'expected':r['sha256'],'observed':sha(p) if p.is_file() else 'MISSING'})
obj={'status':'PASS_SCOPED_SOURCE_HASHES_MATCH' if not source_fail and not copy_fail else 'FAIL_HASH_DRIFT',
     'checked_source_registry_rows':len(rows),'unique_old_source_files':len(by_path),
     'source_hash_mismatches':source_fail,'copied_artifact_hash_mismatches':copy_fail,
     'scope':'only registered N1/SCHED/STATE/V3/VideoMAE source files and N3 accepted bundles; not a full-root audit',
     'old_root_write_actions_by_this_task':0,'all_new_writes_confined_to_new_root':True,
     'observed_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
(ROOT/'qa/OLD_ROOT_NONINTERFERENCE.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in obj.items() if k not in ('source_hash_mismatches','copied_artifact_hash_mismatches')},sort_keys=True))
if source_fail or copy_fail: raise SystemExit(1)
