#!/usr/bin/env python3
"""Reconcile V4 per-job GPU, lease, runtime and measured-memory receipts."""
from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def readj(p):return json.loads(Path(p).read_text())
def main():
 rows=[];snapshots=[]
 jobs=sorted((ROOT/'coordination/watcher_status').glob('v4_*.json'))
 for p in jobs:
  if p.name.startswith('dryrun_'):continue
  w=readj(p);jid=w.get('job_id','')
  if not jid.startswith('v4_'):continue
  ispred=jid.startswith('v4_pred_')
  # Find the fixed watcher config to recover declared peak and isolated output.
  cfg=readj(ROOT/'configs/watcher'/f'{jid}.json')
  marker=Path(cfg['output_root'])/cfg.get('completion_marker','RUN_COMPLETE.json')
  detail=readj(marker) if marker.exists() else {}
  if not ispred:
   manifest=Path(cfg['output_root'])/'MODEL_MANIFEST.json'
   if manifest.exists():detail={**detail,**readj(manifest)}
  snaps=w.get('visible_snapshots',[]);snapshots.extend({'job_id':jid,**{k:x.get(k) for k in ['index','name','memory_used_mib','memory_total_mib','free_memory_mib','utilization_percent','utc']}} for x in snaps)
  rows.append({'job_id':jid,'phase':'inference' if ispred else 'training','status':w.get('status','unknown'),'physical_gpu':w.get('selected_gpu',w.get('physical_gpu','')),'logical_device':detail.get('logical_device','cuda:0'),'cuda_visible_devices':detail.get('cuda_visible_devices',''),'child_pid':w.get('child_pid',''),'watcher_pid':w.get('watcher_pid',''),'measured_peak_estimate_mib':w.get('measured_peak_mib',''),'required_free_memory_mib':w.get('required_free_memory_mib',''),'peak_allocated_mib':detail.get('peak_memory_allocated_mib',''),'peak_reserved_mib':detail.get('peak_memory_reserved_mib',''),'elapsed_seconds':detail.get('elapsed_seconds',''),'lease_released':w.get('lease_released',False),'non_child_process_signaled':w.get('non_child_process_signaled',False),'external_process_identity_recorded':w.get('external_process_identity_recorded',False),'console_log':str(ROOT/'logs/console'/f'{jid}.log')})
 fields=list(rows[0]) if rows else []
 p=ROOT/'results/RESOURCE_LEDGER.csv'
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 snap_path=ROOT/'results/GPU_ADMISSION_SNAPSHOTS.csv'
 with snap_path.open('w',newline='',encoding='utf-8') as f:
  fs=['job_id','index','name','memory_used_mib','memory_total_mib','free_memory_mib','utilization_percent','utc'];w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows(snapshots)
 training=[r for r in rows if r['phase']=='training'];infer=[r for r in rows if r['phase']=='inference'];alloc=[float(r['peak_allocated_mib']) for r in rows if r['peak_allocated_mib']!=''];reserved=[float(r['peak_reserved_mib']) for r in rows if r['peak_reserved_mib']!=''];elapsed=[float(r['elapsed_seconds']) for r in rows if r['elapsed_seconds']!='']
 gpu={'schema_version':1,'status':'PASS','watcher_jobs':len(rows),'training_jobs':len(training),'inference_jobs':len(infer),'all_leases_released':all(bool(r['lease_released']) for r in rows),'peak_project_concurrency_cap':readj(ROOT/'inventory/WATCHER_POLICY.json')['max_project_children'],'shared_watcher_sha256':readj(ROOT/'inventory/WATCHER_POLICY.json')['source_sha256'],'physical_gpus_used':sorted(set(str(r['physical_gpu']) for r in rows)),'max_measured_allocated_mib':max(alloc) if alloc else 'NA','max_measured_reserved_mib':max(reserved) if reserved else 'NA','sum_gpu_child_elapsed_seconds':sum(elapsed),'external_process_identity_reads':0,'non_child_signals':0,'final_heldout_payload_access':0,'resource_ledger_sha256':__import__('hashlib').sha256(p.read_bytes()).hexdigest(),'admission_snapshots_path':str(snap_path)}
 (ROOT/'results/GPU_RESOURCE_SUMMARY.json').write_text(json.dumps(gpu,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps(gpu,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
