#!/usr/bin/env python3
"""Recheck V8's recorded read-only source asset hashes; no target archives opened."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 sm=json.loads((ROOT/'inventory/SOURCE_MAP.json').read_text());checks=[]
 for name,item in sm['source_assets'].items():
  actual=sha(item['path']);checks.append({'name':name,'path':item['path'],'expected_sha256':item['sha256'],'actual_sha256':actual,'match':actual==item['sha256']})
 v7=Path(sm['source_roots']['V7']);v4=Path(sm['source_roots']['V4'])
 checks.extend([
  {'name':'V7_v7_train_current_source','path':str(v7/'code/v7_train.py'),'expected_sha256':sm['v7_runner_source_sha256'],'actual_sha256':sha(v7/'code/v7_train.py'),'match':sha(v7/'code/v7_train.py')==sm['v7_runner_source_sha256']},
  {'name':'V4_loss_helper','path':str(v4/'code/locked_source/d3_eq_contract.py'),'expected_sha256':sm['v4_loss_source_sha256'],'actual_sha256':sha(v4/'code/locked_source/d3_eq_contract.py'),'match':sha(v4/'code/locked_source/d3_eq_contract.py')==sm['v4_loss_source_sha256']},
 ])
 if not all(x['match'] for x in checks):raise RuntimeError('read-only source asset hash drift detected')
 released=json.loads((v7/'coordination/OWNER_RELEASED.json').read_text());state=json.loads((v7/'coordination/OWNER_STATE.json').read_text())
 if released.get('status')!='released' or state.get('status')!='complete' or state.get('released') is not True:raise RuntimeError('sealed V7 source root is not complete/released')
 out={'schema_version':1,'status':'PASS','checked_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'source_hash_checks':checks,
  'V7_owner_status':state.get('status'),'V7_owner_released':released.get('status'),'old_roots_written':False,'dev_target_archive_opened':False,'final_heldout_payload_access':0}
 (ROOT/'qa/SOURCE_INTEGRITY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','source_assets_checked':len(checks),'old_roots_written':False,'dev_target_read':False},sort_keys=True))
if __name__=='__main__':main()
