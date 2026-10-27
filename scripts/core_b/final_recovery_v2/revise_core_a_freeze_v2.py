import os
#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
O=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj):
 tmp=path.with_name('.'+path.name+'.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 tmp.replace(path)
now=datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
freeze_path=O/'core_a/FREEZE_RECEIPT.json';amend_path=O/'core_a/PROTOCOL_AMENDMENT_V2.json'
if not freeze_path.is_file() or not amend_path.is_file():raise FileNotFoundError('initial pre-score freeze/amendment absent')
exposure=O/'core_a/SCORE_EXPOSURE.json'
if exposure.exists():raise RuntimeError('score exposure has started; freeze revision is no longer allowed')
for n in ['FULL602_PER_SEED.csv','FULL602_SUMMARY.csv','PAIRED_CONTRASTS_AND_CI.csv','CORE_A_SCORE_RECEIPT.json']:
 if (O/'core_a'/n).exists():raise RuntimeError('score output exists; refusing pre-score freeze revision')
old= json.loads(freeze_path.read_text());amend=json.loads(amend_path.read_text())
if old.get('status')!='LOCKED_BEFORE_V2_SCORE' or old.get('first_v2_metric_computed_utc') is not None:
 raise RuntimeError('prior freeze is not a pre-score lock')
old_sha=sha(freeze_path)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
for path in [freeze_path,amend_path]:
 backup=path.with_name(path.name+f'.pre-rank-cache-{stamp}.bak')
 if backup.exists():raise FileExistsError(backup)
 shutil.copy2(path,backup)
score=O/'code/score_core_a_v2.py';rev=O/'code/revise_core_a_freeze_v2.py'
new_code_sha=sha(score)
amend['engineering_revision']={
 'revision':2,'updated_utc':now,
 'reason':'Pre-score runtime improvement: stable per-row output rankings are computed once and reused across the paired bootstrap draws.',
 'scientific_contract_changed':False,'population_changed':False,'metric_changed':False,'tie_rule_changed':False,
 'scoring_code_sha256':new_code_sha,'final_score_exposure_at_revision':False
}
write(amend_path,amend)
new=json.loads(json.dumps(old))
new['revision']=2;new['supersedes_freeze_receipt_sha256']=old_sha;new['created_utc']=now
new['protocol_amendment_sha256']=sha(amend_path)
new['code_sha256']['score_core_a_v2.py']=new_code_sha
new['code_sha256']['revise_core_a_freeze_v2.py']=sha(rev)
new['revision_reason']='Pre-score rank-cache optimization; same IDs, predictions, classes, metrics, seeds, bootstrap draws and tie rule.'
new['first_v2_label_read_utc']=None;new['first_v2_metric_computed_utc']=None;new['metrics_computed']=False
write(freeze_path,new)
logpath=O/'core_a/FREEZE_REVISION_LOG.json'
if logpath.exists():raise FileExistsError(logpath)
write(logpath,{'schema_version':1,'protocol_id':O.name,'revision':2,'created_utc':now,
 'revisions':[{'revision':1,'freeze_sha256':old_sha,'status':'LOCKED_BEFORE_V2_SCORE'},
 {'revision':2,'freeze_sha256':sha(freeze_path),'status':'LOCKED_BEFORE_V2_SCORE'}],
 'reason':'Before any V2 final score, cache each row ranking once and reuse it across 5,000 paired bootstrap draws. Scientific contract and frozen inputs are unchanged.',
 'metrics_computed_before_revision':False,'label_manifest_read_for_scoring_before_revision':False,
 'backups':[str(freeze_path.with_name(freeze_path.name+f'.pre-rank-cache-{stamp}.bak')),
 str(amend_path.with_name(amend_path.name+f'.pre-rank-cache-{stamp}.bak'))]})
print(json.dumps({'status':new['status'],'revision':2,'freeze_sha256':sha(freeze_path),'score_code_sha256':new_code_sha,'scientific_contract_changed':False},ensure_ascii=False))
