#!/usr/bin/env python3
"""Record N2 blocker, N3 visual/background evidence, and scoped coordination state."""
from __future__ import annotations
import os
import csv,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'))
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
VM=PROJECT/'reports/benchmarks/VIDEOMAE_HANDOFF_ACCEPTANCE_V1_20261009T025510Z'
VM_INDEX=PROJECT/'tmp/project-team/videomae_handoff_20261009T020906Z/HANDOFF_INDEX.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rd(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def wr(p,rows):
 fields=[]
 for x in rows:
  for k in x:
   if k not in fields:fields.append(k)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def add_source(src,rel,purpose):
 dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 h=sha(src);shutil.copy2(src,dest)
 if sha(dest)!=h:raise SystemExit('copy hash mismatch '+str(src))
 return {'source_path':str(src),'new_root_path':str(dest.relative_to(ROOT)),'sha256':h,'copy_status':'COPIED_HASH_MATCH','purpose':purpose}

# Keep the accepted VideoMAE handoff as a labeled, non-comparable background reference.
selected=[]
for x in rd(VM/'metrics/core_b/MODEL_SEED_SUMMARIES.csv'):
 if x['split']=='dev' and x['horizon'] in ('30','60') and x['model'] in ('VideoMAE','VM-STATE-CTRL') and x['metric']=='standardized_mae' and x['region_raw']=='action_full28':
  selected.append(x)
if len(selected)!=4:raise SystemExit(f'expected 4 selected VideoMAE background rows, got {len(selected)}')
extra=[
 add_source(VM/'metrics/core_b/MODEL_SEED_SUMMARIES.csv','n3_refs/background/MODEL_SEED_SUMMARIES.csv','accepted VideoMAE full28 H30/H60 background source table'),
 add_source(VM/'FINAL_ACCEPTANCE.json','n3_refs/background/FINAL_ACCEPTANCE.json','VideoMAE partial acceptance status and blockers'),
 add_source(VM/'AUDIT_REPORT.zh-CN.md','n3_refs/background/AUDIT_REPORT.zh-CN.md','accepted VideoMAE comparability/context report'),
]
source_path=ROOT/'inventory/SOURCE_REGISTRY.csv';sources=rd(source_path);sources.extend(extra);wr(source_path,sources)
vm_accept=json.loads((VM/'FINAL_ACCEPTANCE.json').read_text())
background=[]
for x in selected:
 background.append({'model':x['model'],'horizon_steps':x['horizon'],'metric':x['metric'],'region':x['region_raw'],
                    'mean':x['mean'],'sample_sd':x['sample_sd_ddof1'],'seed_20261001':x['seed_20261001'],
                    'seed_20261002':x['seed_20261002'],'seed_20261003':x['seed_20261003'],
                    'n_source_groups':x['n_paired_groups'],'acceptance_status':vm_accept.get('status'),
                    'comparability':'background_only_different_video_feature_layout_projection_and_pretraining',
                    'checkpoint_unpickle_attempted':False,'final_test_access':0})
wr(ROOT/'inventory/BACKGROUND_BASELINE_STATUS.csv',background)

# Append the exact prior D3EQ owner/session inspection; no session or lease was changed.
resource=rd(ROOT/'coordination/RESOURCE_LOG.csv')
resource.append({'observed_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                 'host':os.environ.get("EE6008_HOST_LABEL", "shared-host"),'host_type':'shared GPU host; no Slurm allocation detected',
                 'gpu_physical':'','uuid':'','model':'D3EQ remaining24 controller status',
                 'used_mib':'','free_mib':'','utilization_pct':'','runtime_usable':'',
                 'note':'Exact tmux session not found; log completed 2026-09-25T13:37:35Z with formal_phase_complete_or_partial, formal_complete_cells=36; no gpu_lease directory in that root; no process signaled.'})
resource.append({'observed_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                 'host':os.environ.get("EE6008_HOST_LABEL", "shared-host"),'host_type':'shared GPU host; no Slurm allocation detected',
                 'gpu_physical':'','uuid':'','model':'STATE root stale lease owner check',
                 'used_mib':'','free_mib':'','utilization_pct':'','runtime_usable':'',
                 'note':'Lease files referenced PIDs 3797417, 3687322, 3797426 from 2026-09-20; none existed at check; files left untouched.'})
wr(ROOT/'coordination/RESOURCE_LOG.csv',resource)

# Explicit branch-level QA: N2 could not safely load, while N3 reused accepted predictions.
for name,obj in (
 ('AS_RUN_REPLAY.json',{'status':'NOT_RUN_N2_SAFE_CHECKPOINT_SOURCE_BLOCKED','reason':'No safe model-only export; no checkpoint was loaded or replayed. Historical N1 sidecars remain the accepted as-run reference.','N3_reuse':'N3 copied predictions reconciled to accepted original scorer; this is not a checkpoint replay.','final_test_access':0}),
 ('COMMON_PROFILE_RELOAD.json',{'status':'NOT_RUN_SOURCE_BLOCKED','expected_views':24,'accepted_views':0,'reason':'N2 weights_only-safe checkpoint loading failed for all 12; no CUDA/CPU common-profile inference was launched.','final_test_access':0}),
 ('PARAM_BUFFER_INVARIANCE.json',{'status':'NOT_ASSESSED_NO_SAFE_MODEL_LOAD','reason':'Cannot compare model parameters/buffers without a safe tensor-only export; no unsafe pickle load or allowlist was used.','final_test_access':0}),
 ('VISUAL_QA.json',{'status':'PASS_AGENT_FINAL_PNG_PREVIEW','reviewer':'Codex full-preview check','figures':['FIG01_N3_VIDEO_GAIN_BY_STATE.png','FIG02_N3_INFORMATION_INTERACTIONS.png'],'observations':['final previews show no clipping or overlap','initial constrained-layout failure retained under figures/attempt_001','second render retained under figures/attempt_002 after OBS orange contrast review','final palette has contrast above 3:1 against white and uses marker/label redundancy'],'human_review_status':'pending','journal_requirements':'pending; target journal not specified','final_test_access':0}),
):
 (ROOT/'qa'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')

# JSON Lines ledgers for the two branches and visualization.
ledger=[
 {'module':'N2','status':'source_blocked_before_child_spawn','new_gpu_children':0,'new_inference_views':0,'new_training':0,'reason':'12/12 checkpoints fail torch.load(weights_only=True); no tensor-only export; unsafe load/allowlist not used','final_test_access':0},
 {'module':'N3','status':'complete','accepted_prediction_views':18,'new_inference_views':0,'new_training':0,'new_encoder_extraction':0,'score_reconciliation':'162/162 PASS; max absolute difference 5.551115123125783e-17','final_test_access':0},
 {'module':'figures','status':'complete_agent_visual_review','final_pngs':2,'final_pdfs':2,'superseded_attempts_preserved':2,'human_review_status':'pending','final_test_access':0},
]
(ROOT/'coordination/JOB_LEDGER.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in ledger),encoding='utf-8')

# Deviation ledger is append-only and remains in the new root.
dev=ROOT/'protocol/DEVIATIONS.jsonl'
with dev.open('a',encoding='utf-8') as f:
 for x in (
  {'time_utc':datetime.now(timezone.utc).isoformat(),'deviation':'N1 seed summary initially included an ALL aggregate label; seed selector was corrected to the three fixed seeds before output acceptance','impact':'no scientific result changed; only fixed-seed rows retained'},
  {'time_utc':datetime.now(timezone.utc).isoformat(),'deviation':'first figure render had constrained-layout overlap and was archived; second render passed layout but palette contrast review changed OBS orange; final render rechecked','impact':'final figures use the corrected third render'},
 ):
  f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
print({'status':'PASS_EVIDENCE_FINALIZED','videomae_background_rows':len(background),'source_registry_rows':len(sources),'resource_rows':len(resource)})
