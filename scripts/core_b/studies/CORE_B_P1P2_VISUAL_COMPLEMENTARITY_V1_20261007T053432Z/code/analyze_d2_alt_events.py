from __future__ import annotations
import os
import csv,hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

EXP=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
RESULT=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z'
LAB=PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z'
SEEDS=(20260950,20260951,20260952)
BOOTSTRAPS=10000;BOOTSTRAP_SEED=20260924
sys.path.insert(0,str(EXP/'code'))
import d2_v3_evaluator_reference as ev

def write_csv(path,rows):
 if path.exists():raise FileExistsError(path)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def main()->int:
 # The D2 secondary requires all ALT H30 dense bundles accepted; no model score is read before this gate.
 manifest=EXP/'predictions/DENSE_MANIFEST.csv'
 if not manifest.exists():raise RuntimeError('D2 onset secondary waits for accepted ALT dense bundles')
 with manifest.open(newline='',encoding='utf-8') as f:accepted=[r for r in csv.DictReader(f) if r.get('condition')=='ALT-POOL-R3D18' and r.get('horizon_steps')=='30' and r.get('status')=='accepted']
 if {int(r['seed']) for r in accepted}!=set(SEEDS) or len(accepted)!=3:raise RuntimeError('D2 onset secondary requires 3 accepted ALT H30 dense bundles')
 d2root=RESULT/'d2_v3_measurement';d2lock_path=d2root/'D2_V3_PROTOCOL_LOCKED.json'
 d2lock=json.loads(d2lock_path.read_text())
 if d2lock.get('status')!='locked_train_thresholds_ready':raise RuntimeError('D2 V3 train threshold lock is not accepted')
 accepted_ref=json.loads((d2root/'D2_V3_ACCEPTANCE_FINAL.json').read_text())
 if accepted_ref.get('status')!='PASS_D2_V3_EXPLORATORY_COMPLETE' or accepted_ref.get('final_test_access')!=0:raise RuntimeError('accepted D2 V3 reference is unavailable')
 matcher=json.loads((d2root/'D2_V3_MATCHER_LOCKED.json').read_text())
 if matcher.get('evaluator_code_sha256')!=sha_file(EXP/'code/d2_v3_evaluator_reference.py') or matcher.get('threshold_protocol_sha256')!=sha_file(d2lock_path):raise RuntimeError('D2 V3 evaluator/threshold lock hash mismatch')
 on={b:float(d2lock['thresholds'][b]['theta_on']) for b in ev.GROUPS};off={b:float(d2lock['thresholds'][b]['theta_off']) for b in ev.GROUPS}
 v2=json.loads((PROJECT/'reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z/protocol/D2_EVENT_REANALYSIS_V2.json').read_text())
 v2_on={'Hand':float(v2['theta_on'][0]),'Arm':float(v2['theta_on'][1])};v2_off={'Hand':float(v2['theta_off'][0]),'Arm':float(v2['theta_off'][1])}
 ev.SIGMA=np.asarray(json.loads((V3/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float64)
 ev.D2_V2_ON=v2_on;ev.D2_V2_OFF=v2_off
 # The accepted D2 secondary state machine uses V3 train thresholds for truth/state and the accepted V2 state thresholds only for its separate support audit.
 panel_path=LAB/'d2/D2_DENSE_INPUTS.npz'
 with np.load(panel_path,allow_pickle=False) as z:
  if any('target' in k.lower() or 'future' in k.lower() for k in z.files):raise RuntimeError('D2 dense prediction input unexpectedly contains target/future arrays')
  panel={k:z[k].copy() for k in ('dense_id','episode_index','original_task_index','anchor_frame','anchor_timestamp')}
 ids=panel['dense_id'].astype('U96');panel_count=len(ids);id_index={x:i for i,x in enumerate(ids)}
 if panel_count!=3718 or len(id_index)!=panel_count:raise RuntimeError('fixed D2 panel schema changed')
 # Load only primary development episode metadata and the 118 episodes in the fixed dense panel.
 metadata={}
 with (V3/'ANCHOR_MANIFEST.csv').open(newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):
   if row['population']=='primary' and row['split']=='dev' and row['final_test_access']=='0':metadata.setdefault(int(row['episode_index']),row)
 split_path=Path(json.loads((V3/'SOURCE_REGISTRY.json').read_text())['split_manifest'])
 import pyarrow.parquet as pq
 table=pq.read_table(split_path,columns=['episode_index','original_task_index','split','video_sha256_fingerprint','v3_benchmark_inclusion'],filters=[('split','=','validation'),('v3_benchmark_inclusion','=','primary')])
 source={int(r['episode_index']):{'task':int(r['original_task_index']),'group':str(r['video_sha256_fingerprint'])} for r in table.to_pylist()}
 panel_eps=sorted(set(int(e) for e in panel['episode_index']))
 if len(panel_eps)!=118 or any(e not in metadata or e not in source for e in panel_eps):raise RuntimeError('D2 fixed panel episode/source-group mismatch')
 group_task={source[e]['group']:source[e]['task'] for e in panel_eps};groups=sorted(group_task)
 if len(groups)!=118:raise RuntimeError('D2 panel source-group support differs from locked 118')
 episode_data={}
 for e in panel_eps:
  raw=ev.load_episode(metadata[e],source[e],ev.SIGMA)
  raw['episode_index']=e
  for body in ev.GROUPS:
   raw.setdefault('states_v3',{})[body]=ev.state_trace(raw['m'][body],8,len(raw['timestamp']),on[body],off[body])
  episode_data[e]=raw
 # ID/frame/time validation for dense query metadata.
 queries=[];queries_by_group=defaultdict(list)
 for i,did in enumerate(ids):
  e=int(panel['episode_index'][i]);frame=int(panel['anchor_frame'][i]);raw=episode_data[e];ts=raw['timestamp']
  if frame<8 or frame+30>=len(ts):raise RuntimeError(f'D2 fixed H30 panel boundary changed: {did}')
  if abs(float(ts[frame])-float(panel['anchor_timestamp'][i]))>1e-5:raise RuntimeError(f'D2 panel anchor timestamp mismatch: {did}')
  q={'dense_id':str(did),'episode_index':e,'task':int(panel['original_task_index'][i]),'source_group_id':source[e]['group'],'anchor_frame':frame,'query_time':float(ts[frame]),'full_h30':frame+30<len(ts),'effective_end_h30':float(ts[frame+28])}
  for body in ev.GROUPS:q[f'known_{body}']=bool(raw['states_v3'][body]['known'][frame])
  queries.append(q);queries_by_group[q['source_group_id']].append(q)
 # Ground-truth V3 onset events from the locked raw-command state machine; no model outcomes enter this stage.
 truth_by_group_body=defaultdict(dict)
 for e in panel_eps:
  raw=episode_data[e]
  for body in ev.GROUPS:
   for event in raw['states_v3'][body]['events']:
    onset=int(event['onset_frame']);gid=raw['group'];eid=f'{gid}|{body}|{onset}'
    truth_by_group_body[(gid,body)][eid]={'event_id':eid,'group':gid,'episode_index':e,'task':raw['task'],'body':body,'onset_frame':onset,'onset_time':float(raw['timestamp'][onset]),'confirmation_frame':int(event['confirmation_frame'])}
 opportunities={};lead_opps={}
 for gid in groups:
  for body in ev.GROUPS:
   events=truth_by_group_body[(gid,body)]
   for eid,event in events.items():
    leads=[]
    for q in queries_by_group[gid]:
     lead=event['onset_time']-q['query_time']
     if q[f'known_{body}'] and q['full_h30'] and 0<lead<=q['effective_end_h30']-q['query_time']+1e-8:leads.append(lead)
    if leads:
     opportunities[(gid,body)]=opportunities.get((gid,body),set());opportunities[(gid,body)].add(eid)
     for t in ev.LEAD_THRESHOLDS:
      if any(x>=t for x in leads):lead_opps[(gid,body,t)]=lead_opps.get((gid,body,t),set());lead_opps[(gid,body,t)].add(eid)
 exposure={g:max(q['query_time'] for q in queries_by_group[g])-min(q['query_time'] for q in queries_by_group[g]) for g in groups}
 alt_results=[];cluster_audit=[];group_results=[]
 # Historical D2 V3 H30 rows are copied by reference for aligned context; the new ALT is independently evaluated below.
 baseline_rows=[]
 with (RESULT/'d2_v3_measurement/D2_V3_RESULTS.csv').open(newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f):
   if row['horizon_steps']=='30' and row['model'] in ('S-HIST','F-POOL-D') and int(row['seed']) in SEEDS:
    baseline_rows.append({'model':row['model'],'seed':int(row['seed']),'horizon_steps':30,'event_group':row['event_group'],'recall':row['recall'],'recall_ci95_low':row['recall_ci95_low'],'recall_ci95_high':row['recall_ci95_high'],'false_alarms_per_hour':row['false_alarms_per_hour'],'false_alarms_per_hour_ci95_low':row['false_alarms_per_hour_ci95_low'],'false_alarms_per_hour_ci95_high':row['false_alarms_per_hour_ci95_high'],'full_opportunity_events':row['full_opportunity_events'],'hits':row['hits'],'source_groups':row['source_groups'],'tasks_with_complete_opportunity':row['tasks_with_complete_opportunity'],'status':'accepted_D2_V3_development_reference','final_test_access':0})
 for accepted_row in accepted:
  seed=int(accepted_row['seed']);bundle=EXP/accepted_row['bundle_path']
  sidecar=json.loads(bundle.with_suffix('.json').read_text())
  if sidecar.get('target_arrays_opened') is not False or int(sidecar.get('final_test_access',-1))!=0:raise RuntimeError(f'ALT target-blind receipt failed: {bundle.name}')
  with np.load(bundle,allow_pickle=False) as z:
   bid=z['source_id'].astype('U96');pred=z['predictions_native'].copy();be=z['episode_index'].copy();bf=z['anchor_frame'].copy();bt=z['anchor_timestamp'].astype(np.float64)
  if set(bid.tolist())!=set(ids.tolist()) or pred.shape!=(panel_count,30,28) or not np.isfinite(pred).all():raise RuntimeError(f'ALT event bundle ID/shape mismatch seed {seed}')
  ix={x:i for i,x in enumerate(bid)};aligned=pred[np.asarray([ix[x] for x in ids])]
  for i,did in enumerate(ids):
   e,frame,ts=int(panel['episode_index'][i]),int(panel['anchor_frame'][i]),float(panel['anchor_timestamp'][i])
   j=ix[did]
   if int(be[j])!=e or int(bf[j])!=frame or abs(float(bt[j])-ts)>1e-5:raise RuntimeError(f'ALT dense row metadata mismatch {did}')
  for body in ev.GROUPS:
   alerts_by_group=defaultdict(list);query_counts=Counter()
   for i,q in enumerate(queries):
    e=q['episode_index'];frame=q['anchor_frame'];raw=episode_data[e]
    known,armed,forecasts=ev.predict_events(aligned[i],raw['action'],raw['timestamp'],raw['q'][body],frame,raw['states_v3'][body],on[body],off[body])
    query_counts[q['source_group_id']]+=1
    if not known or not q['full_h30']:continue
    for f in forecasts:
     pt=float(f['predicted_onset_time'])
     if q['query_time']<pt<=q['effective_end_h30']+1e-8:
      alerts_by_group[q['source_group_id']].append({'source_group_id':q['source_group_id'],'episode_index':e,'task':q['task'],'body':body,'dense_id':q['dense_id'],'query_time':q['query_time'],'predicted_onset_time':pt,'predicted_onset_frame':int(f['predicted_onset_frame'])})
   for gid in groups:
    clusters=ev.cluster_alerts(alerts_by_group[gid]);events=truth_by_group_body[(gid,body)];opp=opportunities.get((gid,body),set());unmatched=set(opp);hits=0;lead_hit_counts={t:0 for t in ev.LEAD_THRESHOLDS};timing_abs=[];timing_signed=[]
    for cl in sorted(clusters,key=lambda x:(x['first_query_time'],x['center_predicted_onset'],x['first_dense_id'])):
     q=next(x for x in queries_by_group[gid] if x['dense_id']==cl['first_dense_id']);eff=q['effective_end_h30']-q['query_time'];candidates=[]
     for eid in unmatched:
      event=events[eid];lead=event['onset_time']-cl['first_query_time'];dt=abs(cl['center_predicted_onset']-event['onset_time'])
      if 0<lead<=eff+1e-8 and dt<=ev.TOLERANCE:candidates.append((dt,event['onset_time'],eid,event,lead))
     if candidates:
      _,_,eid,event,lead=min(candidates,key=lambda x:(x[0],x[1],x[2]));unmatched.remove(eid);hits+=1;timing_abs.append(abs(cl['center_predicted_onset']-event['onset_time']));timing_signed.append(cl['center_predicted_onset']-event['onset_time'])
      for t in ev.LEAD_THRESHOLDS:
       if lead>=t:lead_hit_counts[t]+=1
      cluster_audit.append({'model':'ALT-POOL-R3D18','seed':seed,'horizon_steps':30,'event_group':body,'source_group_id':gid,'task':cl['task'],'cluster_id':cl['cluster_id'],'first_dense_id':cl['first_dense_id'],'predicted_onset_time':cl['center_predicted_onset'],'matched_event_id':eid,'ground_truth_onset_time':event['onset_time'],'lead_seconds':lead,'match_status':'hit','final_test_access':0})
     else:cluster_audit.append({'model':'ALT-POOL-R3D18','seed':seed,'horizon_steps':30,'event_group':body,'source_group_id':gid,'task':cl['task'],'cluster_id':cl['cluster_id'],'first_dense_id':cl['first_dense_id'],'predicted_onset_time':cl['center_predicted_onset'],'matched_event_id':'','ground_truth_onset_time':'','lead_seconds':'','match_status':'false_alarm_cluster','final_test_access':0})
    for eid in unmatched:cluster_audit.append({'model':'ALT-POOL-R3D18','seed':seed,'horizon_steps':30,'event_group':body,'source_group_id':gid,'task':group_task[gid],'cluster_id':'','first_dense_id':'','predicted_onset_time':'','matched_event_id':eid,'ground_truth_onset_time':events[eid]['onset_time'],'lead_seconds':'','match_status':'missed_opportunity','final_test_access':0})
    group_results.append({'model':'ALT-POOL-R3D18','seed':seed,'horizon_steps':30,'event_group':body,'source_group_id':gid,'task':group_task[gid],'panel_queries':query_counts[gid],'opportunities':len(opp),'hits':hits,'misses':len(opp)-hits,'predicted_forecast_clusters':len(clusters),'false_alarms':len(clusters)-hits,'exposure_seconds':exposure[gid],'timing_abs_sum':sum(timing_abs),'timing_signed_sum':sum(timing_signed),'timing_count':len(timing_abs),'eligible_known_queries':sum(bool(q[f'known_{body}']) for q in queries_by_group[gid]),'final_test_access':0,**{f'lead_opportunities_{t:g}':len(lead_opps.get((gid,body,t),set())) for t in ev.LEAD_THRESHOLDS},**{f'lead_hits_{t:g}':lead_hit_counts[t] for t in ev.LEAD_THRESHOLDS}})
 # Source-group bootstrap for ALT H30 event metrics.
 rng=np.random.default_rng(BOOTSTRAP_SEED);boot=rng.integers(0,len(groups),size=(BOOTSTRAPS,len(groups)),dtype=np.int16);metrics=[]
 for seed in SEEDS:
  for body in ev.GROUPS:
   rr=[x for x in group_results if x['seed']==seed and x['event_group']==body]
   opp=np.asarray([x['opportunities'] for x in rr],dtype=np.float64);hits=np.asarray([x['hits'] for x in rr],dtype=np.float64);fa=np.asarray([x['false_alarms'] for x in rr],dtype=np.float64);exp=np.asarray([x['exposure_seconds'] for x in rr],dtype=np.float64)
   recall=hits.sum()/opp.sum() if opp.sum() else float('nan');recall_boot=np.divide(hits[boot].sum(axis=1),opp[boot].sum(axis=1),out=np.full(BOOTSTRAPS,np.nan),where=opp[boot].sum(axis=1)>0)
   fa_rate=fa.sum()/exp.sum()*3600 if exp.sum() else float('nan');fa_boot=np.divide(fa[boot].sum(axis=1),exp[boot].sum(axis=1),out=np.full(BOOTSTRAPS,np.nan),where=exp[boot].sum(axis=1)>0)*3600
   row={'model':'ALT-POOL-R3D18','seed':seed,'horizon_steps':30,'event_group':body,'episodes':118,'source_groups':118,'queries':panel_count,'full_opportunity_events':int(opp.sum()),'hits':int(hits.sum()),'misses':int(opp.sum()-hits.sum()),'predicted_forecast_clusters':int(sum(x['predicted_forecast_clusters'] for x in rr)),'false_alarms':int(fa.sum()),'recall':float(recall),'recall_ci95_low':float(np.nanquantile(recall_boot,.025)),'recall_ci95_high':float(np.nanquantile(recall_boot,.975)),'false_alarms_per_hour':float(fa_rate),'false_alarms_per_hour_ci95_low':float(np.nanquantile(fa_boot,.025)),'false_alarms_per_hour_ci95_high':float(np.nanquantile(fa_boot,.975)),'exposure_hours':float(exp.sum()/3600),'timing_abs_error_seconds':float(sum(x['timing_abs_sum'] for x in rr)/max(sum(x['timing_count'] for x in rr),1)),'bootstrap_unit':'video_sha256_fingerprint source group','bootstrap_replicates':BOOTSTRAPS,'bootstrap_seed':BOOTSTRAP_SEED,'status':'exploratory_secondary','final_test_access':0}
   for t in ev.LEAD_THRESHOLDS:
    num=np.asarray([x[f'lead_hits_{t:g}'] for x in rr],dtype=np.float64);den=np.asarray([x[f'lead_opportunities_{t:g}'] for x in rr],dtype=np.float64);vb=np.divide(num[boot].sum(axis=1),den[boot].sum(axis=1),out=np.full(BOOTSTRAPS,np.nan),where=den[boot].sum(axis=1)>0)
    row[f'opportunities_lead_ge_{t:g}']=int(den.sum());row[f'hits_lead_ge_{t:g}']=int(num.sum());row[f'recall_lead_ge_{t:g}']=float(num.sum()/den.sum()) if den.sum() else None;row[f'recall_lead_ge_{t:g}_ci95_low']=float(np.nanquantile(vb,.025)) if np.isfinite(vb).any() else None;row[f'recall_lead_ge_{t:g}_ci95_high']=float(np.nanquantile(vb,.975)) if np.isfinite(vb).any() else None
   metrics.append(row)
 # Add accepted historical H30 D2 reference rows by pointer; keep them labeled as old accepted outputs.
 d2rows=[]
 with (RESULT/'d2_v3_measurement/D2_V3_RESULTS.csv').open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   if r['horizon_steps']=='30' and r['model'] in ('S-HIST','F-POOL-D') and int(r['seed']) in SEEDS:d2rows.append({**r,'status':'accepted_D2_V3_reference','final_test_access':0})
 combined=d2rows+metrics
 for rel,rows in [('analysis/P2_EVENT_ONSET.csv',combined),('analysis/P2_EVENT_ONSET_ALT_GROUPS.csv',group_results),('analysis/P2_EVENT_ONSET_ALT_CLUSTERS.csv',cluster_audit)]:
  out=EXP/rel
  if out.exists():raise FileExistsError(out)
  with out.open('w',newline='',encoding='utf-8') as f:
   fields=list(dict.fromkeys(k for x in rows for k in x))
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 status={'status':'PASS_EXPLORATORY_SECONDARY','protocol_id':'CORE_B_D2_EVENT_MEASUREMENT_SENSITIVITY_V3','truth_rule':'accepted D2 V3 train-threshold 8-step smoothed command-change onset; at least 6 quiet steps to arm, 3 high steps to onset','prediction_rule':'accepted D2 V3 H30 predicted command-change state machine; tolerance 0.2 s; same 118-group/3718-query fixed development panel','thresholds_v3':d2lock['thresholds'],'thresholds_v2_state_support_only':{'on':v2_on,'off':v2_off},'alt_bundles_accepted':len(accepted),'old_reference_rows':len(d2rows),'alt_rows':len(metrics),'source_groups':len(groups),'panel_queries':panel_count,'final_test_access':0}
 q=EXP/'analysis/P2_EVENT_ONSET_STATUS.json'
 if q.exists():raise FileExistsError(q)
 q.write_text(json.dumps(status,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','alt_rows':len(metrics),'old_reference_rows':len(d2rows),'source_groups':len(groups),'queries':panel_count,'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
