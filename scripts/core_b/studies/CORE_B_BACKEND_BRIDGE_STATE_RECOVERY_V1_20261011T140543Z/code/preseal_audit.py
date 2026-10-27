#!/usr/bin/env python3
from __future__ import annotations
import os
import csv,json,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'))
json_count=csv_count=jsonl_count=0
errors=[]
for p in ROOT.rglob('*.json'):
 try:json.loads(p.read_text(encoding='utf-8'));json_count+=1
 except Exception as e:errors.append({'path':str(p.relative_to(ROOT)),'error':type(e).__name__})
for p in ROOT.rglob('*.jsonl'):
 try:
  for line in p.read_text(encoding='utf-8').splitlines():
   if line.strip():json.loads(line)
  jsonl_count+=1
 except Exception as e:errors.append({'path':str(p.relative_to(ROOT)),'error':type(e).__name__})
for p in ROOT.rglob('*.csv'):
 try:
  with p.open(newline='',encoding='utf-8') as f:list(csv.reader(f))
  csv_count+=1
 except Exception as e:errors.append({'path':str(p.relative_to(ROOT)),'error':type(e).__name__})
def rows(rel):
 with (ROOT/rel).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
n2=rows('protocol/N2_MATRIX24.csv');n3=rows('protocol/N3_MATRIX18.csv');accept=rows('qa/PREDICTION_ACCEPTANCE.csv')
if len(n2)!=24 or len(n3)!=18 or len(accept)!=18:errors.append({'path':'matrix_count','error':'unexpected N2/N3 view counts'})
if any(x['status']!='SOURCE_BLOCKED_SAFE_MODEL_EXPORT_UNAVAILABLE' for x in n2):errors.append({'path':'N2_MATRIX24.csv','error':'unexpected N2 status'})
if any(x['status'] not in ('PASS','PASS_HASH_AND_RECEIPT') for x in accept):errors.append({'path':'PREDICTION_ACCEPTANCE.csv','error':'unexpected N3 prediction status'})
size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
obj={'status':'PASS_PRESEAL_STRUCTURE' if not errors else 'FAIL_PRESEAL_STRUCTURE','json_files_parsed':json_count,
     'jsonl_files_parsed':jsonl_count,'csv_files_parsed':csv_count,'n2_matrix_rows':len(n2),'n2_accepted':0,
     'n3_matrix_rows':len(n3),'n3_accepted':len(accept),'parse_and_scope_errors':errors,
     'estimated_new_root_bytes_before_checksum':size,'observed_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
(ROOT/'qa/ARTIFACT_INVENTORY.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
usage=rows('coordination/RESOURCE_USAGE.csv')
usage[0]['new_root_disk_bytes_before_checksums']=str(size)
usage[0]['n3_accepted_prediction_copy_bytes']=str(sum(p.stat().st_size for p in (ROOT/'n3_refs/predictions').rglob('*.npz')))
usage[0]['resource_sample_utc']=obj['observed_utc']
with (ROOT/'coordination/RESOURCE_USAGE.csv').open('w',newline='',encoding='utf-8') as f:
 fields=[]
 for r in usage:
  for k in r:
   if k not in fields:fields.append(k)
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(usage)
print(json.dumps({k:v for k,v in obj.items() if k!='parse_and_scope_errors'},sort_keys=True))
if errors:raise SystemExit(1)
