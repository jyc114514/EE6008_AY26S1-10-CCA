#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
import v9_score as score
ROOT=Path(__file__).resolve().parents[1]
def utc():return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe_write(path,data):
 path=Path(path)
 if path.exists():
  current=score.read_csv(path)
  fields=list(current[0].keys()) if current else []
  expected_fields=[]
  for row in data:
   for key in row:
    if key not in expected_fields:expected_fields.append(key)
  if len(current)==len(data) and set(fields)==set(expected_fields):
   expected=[{k:('' if row.get(k) is None else str(row.get(k,''))) for k in fields} for row in data]
   if current==expected:return 'verified_existing'
  backup=path.with_name(path.name+'.pre-post-target-rebuild-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.bak')
  shutil.copy2(path,backup)
 score.write_csv(path,data)
 return 'written'
def numeric_contrast(name,metric,seed,aid,by):
 if name=='INTERACTION':
  return (float(by[('AD',seed,aid)][metric])-float(by[('VD',seed,aid)][metric]))-(float(by[('A0',seed,aid)][metric])-float(by[('V0',seed,aid)][metric]))
 reference,candidate=score.CONTRASTS[name]
 return float(by[(reference,seed,aid)][metric])-float(by[(candidate,seed,aid)][metric])
def main():
 seal=score.gate_sealed()
 single=ROOT/'qa/DEV_TARGET_SINGLE_READ.json';per=ROOT/'results/PER_WINDOW_NEW_DEV_LOSSES.csv'
 if not single.is_file() or not per.is_file():raise RuntimeError('post-read recovery requires completed single-read receipt and raw window losses')
 read=score.rj(single);rows=score.read_csv(per)
 if read.get('status')!='PASS' or int(read.get('archive_content_reads',0))!=1 or len(rows)!=5550:
  raise RuntimeError('saved single-read/raw-loss evidence incomplete')
 if read.get('prediction_seal_sha256')!=sha(ROOT/'qa/PREDICTION_SEAL.json'):raise RuntimeError('raw score table is not bound to current prediction seal')
 source=score.rj(ROOT/'inventory/SOURCE_MAP.json')['source_assets']
 dev_ids,meta=score.load_prefix(Path(source['dev_prefix_npz']['path']),185)
 historical,hist_sha=score.load_historical(dev_ids,meta)
 all_rows=historical+rows
 primary,task_seed,episode,per_episode,macro=score.primary_tables(all_rows)
 score.contrast_delta=numeric_contrast
 effects,seedloo,taskloo,ci,boot=score.paired_tables(all_rows,meta)
 masked=score.masked_diagnostic(rows)
 expected_primary=(len(score.PRIMARY)+len(score.HIST))*len(score.SEEDS)*(len(score.TASKS)+1)
 if len(primary)!=expected_primary:raise RuntimeError(f'primary table row count mismatch: {len(primary)} vs {expected_primary}')
 if len(ci)!=len(score.CONTRASTS)*len(score.METRICS) or len(masked)!=len(score.PROGRAMS)*len(score.SEEDS)*(len(score.TASKS)+1):
  raise RuntimeError('post-read table row count mismatch')
 outputs={
  ROOT/'results/PER_WINDOW_ALL_METHODS.csv':all_rows,
  ROOT/'results/PRIMARY_TASK_SEED.csv':primary,
  ROOT/'results/METHOD_MACRO.csv':macro,
  ROOT/'results/MASKED_DIAGNOSTIC.csv':masked,
  ROOT/'results/PAIRED_EFFECTS.csv':effects,
  ROOT/'results/SEED_LOO.csv':seedloo,
  ROOT/'results/TASK_LOO.csv':taskloo,
  ROOT/'results/PAIRED_GROUP_CI.csv':ci}
 table_actions={}
 for p,data in outputs.items():table_actions[p.name]=safe_write(p,data)
 boot_path=ROOT/'figure_data/BOOTSTRAP_DRAW_MANIFEST.json'
 if boot_path.exists():
  existing=score.rj(boot_path)
  if existing!=boot:raise RuntimeError('existing bootstrap draw manifest differs from deterministic recomputation')
 else:score.wj(boot_path,boot)
 safe_write(ROOT/'figure_data/PRIMARY_PLOT_DATA.csv',[r for r in macro if r['method'] in score.PRIMARY])
 safe_write(ROOT/'figure_data/PAIRED_EFFECTS_PLOT_DATA.csv',[r for r in effects if r['grain']=='five_task_three_seed_macro'])
 safe_write(ROOT/'figure_data/MASK_DIAGNOSTIC_PLOT_DATA.csv',[r for r in masked if r['grain']=='five_task_seed_macro'])
 safe_write(ROOT/'figure_data/MIXTURE_PLOT_DATA.csv',[r for r in macro if r['method'] in ('S','V0','VD','SV50','MIX0','MIXD')])
 recovery={'schema_version':1,'status':'PASS','table_actions':table_actions,'created_utc':utc(),'recovery_type':'post-target aggregation from saved per-window losses',
  'recovery_script_sha256':sha(Path(__file__)),'source_per_window_path':str(per),'source_per_window_sha256':sha(per),'per_window_rows':len(rows),
  'source_dev_target_read_receipt':str(single),'source_dev_target_read_receipt_sha256':sha(single),
  'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),'dev_target_archive_reopened':False,
  'dev_target_archive_content_reads_total':1,'historical_V7_rows_reused':len(historical),'historical_V7_sha256':hist_sha,
  'paired_effect_rows':len(effects),'paired_ci_rows':len(ci),'seed_loo_rows':len(seedloo),'task_loo_rows':len(taskloo),
  'bootstrap_draws':5000,'bootstrap_seed':20261016,'final_heldout_payload_access':0}
 score.wj(ROOT/'results/POST_TARGET_AGGREGATION_RECOVERY.json',recovery)
 final={'schema_version':1,'status':'PASS','created_utc':utc(),'prediction_seal_sha256':sha(ROOT/'qa/PREDICTION_SEAL.json'),
  'train_target_archive_reads':1,'train_full_context_evaluations':15360,'dev_target_archive_content_reads':1,
  'new_dev_logic_bundles':30,'new_dev_loss_evaluations':5550,'primary_bundles':18,'masked_diagnostic_bundles':12,
  'new_dev_task_views':150,'final_heldout_payload_access':0,'final_task_level_heldout_aggregate_access':0,
  'bootstrap_draws':5000,'bootstrap_seed':20261016,'p_values':False,'figures':'pending'}
 score.wj(ROOT/'qa/FINAL_ACCEPTANCE.json',final)
 print(json.dumps({'status':'PASS','per_window_rows':len(rows),'paired_ci_rows':len(ci),'dev_target_reads':1,
  'dev_target_reopened':False,'final_access':0},sort_keys=True))
if __name__=='__main__':main()

