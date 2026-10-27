#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=('code/v9_score.py','code/v9_seal_predictions.py','code/v9_scorer_preflight.py','code/v9_common.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 path=ROOT/'protocol/SCORING_CODE_LOCK.json'
 previous=None
 if path.exists():
  previous=sha(path)
  stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  shutil.copy2(path,path.with_name('SCORING_CODE_LOCK.json.pre-relock-'+stamp+'.bak'))
 if (ROOT/'qa/DEV_TARGET_READ_STARTED.json').exists() or (ROOT/'qa/DEV_TARGET_SINGLE_READ.json').exists():
  raise RuntimeError('scoring code lock is immutable after dev target read starts')
 files={}
 for rel in FILES:
  p=ROOT/rel
  if not p.is_file():raise FileNotFoundError(p)
  files[rel]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
 out={'schema_version':1,'status':'LOCKED_BEFORE_DEV_TARGET_ACCESS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'files':files,'prediction_seal_required':True,'dev_target_archive_content_reads_allowed':1,
  'target_read_gate':'v9_score.gate_sealed verifies the V9 prediction seal and this scoring lock before opening the archive',
  'final_heldout_payload_access':0,'supersedes_previous_scoring_lock_sha256':previous}
 tmp=path.with_name('.'+path.name+'.part');tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
 print(json.dumps({'status':out['status'],'lock_sha256':sha(path),'files':len(files)},sort_keys=True))
if __name__=='__main__':main()

