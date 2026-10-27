import os
#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
O=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,obj):
 tmp=path.with_name('.'+path.name+'.'+str(__import__('os').getpid())+'.part')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 __import__('os').replace(tmp,path)
now=datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
freeze_path=O/'core_a/FREEZE_RECEIPT.json';amend_path=O/'core_a/PROTOCOL_AMENDMENT_V2.json';log_path=O/'core_a/FREEZE_REVISION_LOG.json'
exposure=json.loads((O/'core_a/SCORE_EXPOSURE.json').read_text())
attempt1=json.loads((O/'core_a/CORE_A_SCORE_ATTEMPT_01.json').read_text())
freeze=json.loads(freeze_path.read_text());amend=json.loads(amend_path.read_text());log=json.loads(log_path.read_text())
if freeze.get('revision')!=2 or freeze.get('status')!='LOCKED_BEFORE_V2_SCORE':raise RuntimeError('Expected the pre-score V2 revision-2 freeze')
if exposure.get('metrics_computed') is not False or exposure.get('first_v2_metric_computed_utc') is not None:raise RuntimeError('A metric was already exposed')
if attempt1.get('status')!='FAILED_AFTER_602_LABELS_BEFORE_ANY_METRIC' or attempt1.get('primary_label_rows_materialized')!=602:raise RuntimeError('Attempt 01 failure receipt is incomplete')
for name in ['FULL602_PER_SEED.csv','FULL602_SUMMARY.csv','PAIRED_CONTRASTS_AND_CI.csv','CORE_A_SCORE_RECEIPT.json']:
 if (O/'core_a'/name).exists():raise RuntimeError('Score artifacts exist; cannot perform a metric-free freeze revision')
score=O/'code/score_core_a_v2.py';test=O/'code/test_core_a_v2_scorer.py';rev=O/'code/revise_core_a_freeze_after_attempt01.py'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
old_freeze_sha=sha(freeze_path)
for path in (freeze_path,amend_path,log_path):
 backup=path.with_name(path.name+f'.pre-attempt02-relock-{stamp}.bak')
 if backup.exists():raise FileExistsError(backup)
 shutil.copy2(path,backup)
engineering={
 'revision':3,'updated_utc':now,
 'trigger':'Score attempt 01 materialized the exact 602 frozen labels and stopped on a KeyError before any metric because the scorer read C_final at the freeze top level while the receipt stores it under core_a.',
 'fix':'Read C_model/C_final and final_label_manifest_path/hash from their actual frozen receipt locations; create a separate attempt-02 receipt and preserve attempt-01 exposure.',
 'scientific_contract_changed':False,'population_changed':False,'metric_changed':False,'prediction_or_checkpoint_changed':False,
 'v2_attempt01_labels_materialized':602,'v2_attempt01_metrics_computed':0,
 'scoring_code_sha256':sha(score),'pytest_tests_passed':6,'prediction_preflight':'PASS'
}
prev=amend.get('engineering_revision_history',[])
if not isinstance(prev,list):prev=[]
if amend.get('engineering_revision'):prev.append(amend['engineering_revision'])
prev.append(engineering)
amend['engineering_revision_history']=prev
amend['engineering_revision']=engineering
amend['scoring_code_sha256']=sha(score)
atomic(amend_path,amend)
freeze['revision']=3;freeze['supersedes_freeze_receipt_sha256']=old_freeze_sha;freeze['created_utc']=now
freeze['protocol_amendment_sha256']=sha(amend_path)
freeze['code_sha256']['score_core_a_v2.py']=sha(score)
freeze['code_sha256']['test_core_a_v2_scorer.py']=sha(test)
freeze['code_sha256']['revise_core_a_freeze_after_attempt01.py']=sha(rev)
freeze['revision_reason']='Metric-free attempt 01 read the frozen 602 labels then stopped at a freeze-schema accessor. Revision 03 fixes the accessor and preserves attempt 01; it does not change any scientific choice.'
freeze['first_v2_label_read_utc']=attempt1['started_utc']
freeze['first_v2_metric_computed_utc']=None
freeze['metrics_computed']=False
freeze['v2_metric_free_failed_attempt']={'attempt_path':str(O/'core_a/CORE_A_SCORE_ATTEMPT_01.json'),
 'attempt_status':attempt1['status'],'labels_materialized':602,'metric_values_computed':0}
atomic(freeze_path,freeze)
log.setdefault('revisions',[]).append({'revision':3,'freeze_sha256':sha(freeze_path),'status':freeze['status'],
 'created_utc':now,'after_metric_free_label_read':True,'labels_materialized':602,'metrics_computed':0,
 'reason':engineering['fix']})
log['current_revision']=3;log['current_freeze_sha256']=sha(freeze_path)
atomic(log_path,log)
ledger_path=O/'qa/EXPOSURE_AND_RETRY_LEDGER.json'
ledger=json.loads(ledger_path.read_text())
ledger['freeze_revision_after_attempt01']={'revision':3,'freeze_sha256':sha(freeze_path),'metrics_computed_before_revision':0}
ledger['v2_score_attempts'][0]['freeze_revision']=2
ledger['v2_score_attempts'][0]['status']='FAILED_AFTER_602_LABELS_BEFORE_ANY_METRIC'
ledger['v2_score_attempts'][0]['metric_values_computed']=0
ledger['v2_score_attempts'][0]['failure']='freeze schema accessor KeyError C_final; no metric code ran'
atomic(ledger_path,ledger)
print(json.dumps({'status':freeze['status'],'revision':3,'freeze_sha256':sha(freeze_path),
 'score_code_sha256':sha(score),'attempt01_labels_materialized':602,'metrics_before_retry':0},ensure_ascii=False))
