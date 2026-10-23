#!/usr/bin/env python3
"""Final N4 parser, scope, source and checksum-last seal."""
from __future__ import annotations
import os
import argparse,csv,hashlib,json,os,subprocess
from pathlib import Path

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
R=P/'reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z'
N2=P/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
N3=P/'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'
EXPECTED_N2='21bfdf8baf1cabe863eb4aa184320e5e79d29ee6c1a589bf9e25ec944fb4472f'
EXPECTED_N3='60701b7b357e5282222c60b15994a1408fcc1398618025dbebe5388fc69f1a75'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def precheck():
 problems=[]
 if (R/'CHECKSUMS.sha256').exists():problems.append('checksum file exists before seal')
 if sha(N2/'CHECKSUMS.sha256')!=EXPECTED_N2:problems.append('N2 manifest identity changed')
 if sha(N3/'CHECKSUMS.sha256')!=EXPECTED_N3:problems.append('N3 manifest identity changed')
 # Validate source receipts and every selected N2 artifact without rescanning sealed roots.
 source=J(R/'inventory/SOURCE_ACCEPTANCE.json')
 if source.get('n2_checksum_manifest_sha256')!=EXPECTED_N2:problems.append('N4 source acceptance N2 manifest hash mismatch')
 for rec in source.get('n2_selected_file_inventory',[]):
  if sha(Path(rec['path']))!=rec['sha256']:problems.append('selected N2 hash mismatch: '+rec['path'])
 for rec in source.get('source_hashes',{}).values():
  if sha(Path(rec['path']))!=rec['sha256']:problems.append('locked source hash mismatch: '+rec['path'])
 for rec in source.get('additional_locked_sources',{}).values():
  if isinstance(rec,dict) and 'path' in rec and sha(Path(rec['path']))!=rec['sha256']:
   problems.append('additional locked source mismatch: '+rec['path'])
 # Final gates.
 pred=J(R/'qa/PREDICTION_ACCEPTANCE.json');analysis=J(R/'qa/ANALYSIS_ACCEPTANCE.json')
 final=J(R/'qa/FINAL_ACCEPTANCE.json');fig=J(R/'figures/FIGURE_MANIFEST.json')
 figqa=J(R/'qa/FINAL_FIGURE_FILE_QA.json');visual=J(R/'qa/FIGURE_AGENT_VISUAL_QA.json')
 exposure=J(R/'protocol/OUTCOME_EXPOSURE.json');owner=J(R/'coordination/OWNER_STATE.json')
 if pred.get('status')!='PASS_12_OF_12_TRAIN_AND_DEV_BUNDLES' or pred.get('bundles_accepted')!=24:
  problems.append('24-bundle prediction acceptance failed')
 if analysis.get('status')!='PASS_N4_ERROR_BUDGET_ANALYSIS' or analysis.get('output_counts',{}).get('bootstrap_ci_rows')!=513:
  problems.append('analysis acceptance or bootstrap count mismatch')
 if final.get('status')!='COMPLETE_N4_COMMAND_ERROR_BUDGET':problems.append('final acceptance status mismatch')
 if final.get('checksum_acceptance')!='PASS_CHECKSUMS_SHA256_GENERATED_LAST_AND_VERIFIED':
  problems.append('final acceptance checksum policy not set')
 if final.get('final_test_access')!=0 or final.get('heldout_access')!=0:problems.append('final/heldout access is not zero')
 if exposure.get('final_test_access')!=0 or exposure.get('heldout_access')!=0:problems.append('exposure final/heldout access is not zero')
 if exposure.get('n4_target_access_attempts')!=5 or exposure.get('failed_analysis_attempts_after_target_read')!=4:
  problems.append('target access attempt accounting mismatch')
 if fig.get('status')!='PASS_EXPORTS_CREATED' or len(fig.get('figures',[]))!=20:
  problems.append('figure manifest status/count mismatch')
 if figqa.get('status')!='PASS_PNG_PDF_CAPTION_DATA_METADATA':problems.append('figure file QA mismatch')
 if visual.get('status')!='PASS_AGENT_VISUAL_QA':problems.append('agent visual QA mismatch')
 if owner.get('owner_active') is not False or owner.get('status')!='COMPLETE':
  problems.append('N4 owner is not closed')
 watcher=J(R/'coordination/N4_TRAIN_WATCHER_STATUS.json')
 if watcher.get('status')!='complete' or watcher.get('exit_code')!=0 or watcher.get('lease_released') is not True:
  problems.append('N4 watcher/lease not complete/released')
 if sha(P/'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py')!=J(N2/'coordination/N2_CUDA_LAUNCH.json')['watcher_sha256']:
  problems.append('shared watcher bytes changed')
 if len(list((R/'predictions/train').glob('*/PREDICTION_COMPLETE.json')))!=12:
  problems.append('train prediction receipt count mismatch')
 if len(list((R/'results/PER_WINDOW_MOMENTS').glob('*.npz')))!=24:
  problems.append('per-window moment bundle count mismatch')
 if len(list((R/'figures/cases').glob('*.png')))!=14 or len(list((R/'figures/cases').glob('*.pdf')))!=14:
  problems.append('case figure count mismatch')
 if not (R/'figures/N4_CASE_CONTACT_SHEET.png').is_file():problems.append('case contact sheet missing')
 if len(list((R/'logs').glob('SCORING_ATTEMPT_*.json')))!=5:
  problems.append('scoring attempt records count mismatch')
 if len(list((R/'logs').glob('DELIVERABLE_BUILD_ATTEMPT_*.json')))<1:
  problems.append('final report build attempts are not preserved')
 if any(p.is_symlink() for p in R.rglob('*')):problems.append('symlink found in isolated root')
 if list(R.rglob('*.part')):problems.append('partial temporary files remain')
 # Parse JSON/JSONL/CSV and Python source without loading raw targets.
 counts={'json':0,'jsonl':0,'csv':0,'python_ast':0,'files':0,'bytes':0}
 import ast
 for path in sorted(R.rglob('*')):
  if not path.is_file():continue
  if path.name=='CHECKSUMS.sha256':continue
  counts['files']+=1;counts['bytes']+=path.stat().st_size
  try:
   if path.suffix=='.json':json.loads(path.read_text(encoding='utf-8'));counts['json']+=1
   elif path.suffix=='.jsonl':
    for line in path.read_text(encoding='utf-8').splitlines():
     if line.strip():json.loads(line)
    counts['jsonl']+=1
   elif path.suffix=='.csv':
    with path.open(newline='',encoding='utf-8') as f:
     rd=csv.reader(f);header=next(rd,None)
     if header is None:problems.append('empty csv: '+str(path.relative_to(R)))
     else:
      for i,row in enumerate(rd,2):
       if len(row)!=len(header):problems.append('csv width mismatch: '+str(path.relative_to(R))+':'+str(i));break
    counts['csv']+=1
   elif path.suffix=='.py':
    ast.parse(path.read_text(encoding='utf-8'),filename=str(path));counts['python_ast']+=1
  except Exception as e:problems.append(str(path.relative_to(R))+': '+type(e).__name__+': '+str(e))
 if problems:raise RuntimeError('final seal precheck failed: '+problems[0])
 return counts
def make_manifest():
 files=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='CHECKSUMS.sha256')
 lines=[]
 for path in files:
  rel=path.relative_to(R).as_posix()
  lines.append(sha(path)+'  ./'+rel)
 manifest=R/'CHECKSUMS.sha256'
 with manifest.open('x',encoding='utf-8') as f:
  f.write('\n'.join(lines)+'\n');f.flush();os.fsync(f.fileno())
 # Read-only verification after the final file write.
 checked=0;bad=[]
 for line in manifest.read_text(encoding='utf-8').splitlines():
  digest,rel=line.split('  ',1)
  if sha(R/rel.removeprefix('./'))!=digest:bad.append(rel)
  checked+=1
 if bad:raise RuntimeError('checksum verification failed: '+bad[0])
 return {'manifest':str(manifest),'manifest_sha256':sha(manifest),'entries':checked,
  'bytes_covered':sum(p.stat().st_size for p in files),'status':'PASS_CHECKSUMS_SHA256_GENERATED_LAST_AND_VERIFIED'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seal',action='store_true');args=ap.parse_args()
 counts=precheck()
 if args.seal:seal=make_manifest()
 else:seal={'status':'PRECHECK_ONLY_CHECKSUM_NOT_WRITTEN'}
 print(json.dumps({'precheck':'PASS','parser_counts':counts,'seal':seal},ensure_ascii=False,indent=2))
