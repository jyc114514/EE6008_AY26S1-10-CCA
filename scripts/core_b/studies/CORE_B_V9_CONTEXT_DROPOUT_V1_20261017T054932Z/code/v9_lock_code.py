#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=(
'code/v9_lock_study.py','code/v9_lock_code.py','code/v9_common.py','code/v9_train.py','code/v9_infer.py',
'code/v9_build_jobs.py','code/v9_run_queue.py','code/v2_model.py','code/prepare_v2_data.py','code/v2_scheduler.py',
'code/locked_source/D3_model.py','code/locked_source/d3_eq_contract.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 path=ROOT/'protocol/CODE_LOCK.json'
 previous=None
 if path.exists():
  previous=sha(path)
  stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  backup=path.with_name('CODE_LOCK.json.pre-relock-'+stamp+'.bak')
  shutil.copy2(path,backup)
 study=json.loads((ROOT/'protocol/STUDY_LOCK.json').read_text())
 recipe=json.loads((ROOT/'protocol/RECIPE_LOCK.json').read_text())
 if study.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV' or recipe.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV':
  raise RuntimeError('study and recipe locks must precede code lock')
 sources={}
 for rel in FILES:
  p=ROOT/rel
  if not p.is_file(): raise FileNotFoundError(p)
  sources[rel]={'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
 out={'schema_version':1,'status':'LOCKED_BEFORE_STAGE0_AND_FORMAL_FITS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'files':sources,'study_lock_sha256':sha(ROOT/'protocol/STUDY_LOCK.json'),'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json'),
  'mask_field_map_sha256':sha(ROOT/'protocol/MASK_FIELD_MAP.json'),'mask_schedule_sha256':sha(ROOT/'protocol/MASK_SCHEDULE.json'),
  'old_v8_runner_imported':False,'code_changes_after_lock_require_new_stage0_and_relock':True,
  'supersedes_previous_code_lock_sha256':previous}
 tmp=path.with_name('.'+path.name+'.part');tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,path)
 print(json.dumps({'status':out['status'],'code_lock_sha256':sha(path),'files':len(sources)},sort_keys=True))
if __name__=='__main__':main()

