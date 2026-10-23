#!/usr/bin/env python3
"""Fresh host, scheduler, aggregate GPU, disk and project-lease preflight."""
import os
import json, os, platform, shutil, socket, subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch

P=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
R=P/'reports/benchmarks/CORE_B_N4_COMMAND_ERROR_BUDGET_V1_20261012T044137Z'
N2=P/'reports/benchmarks/CORE_B_N2_SAFE_EXPORT_BRIDGE_RESUME_V1_20261011T160003Z'
N3=P/'reports/benchmarks/CORE_B_BACKEND_BRIDGE_STATE_RECOVERY_V1_20261011T140543Z'
N1=P/'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
LEASE=P/'repo/ee6008-humanoid-vjepa/reports/locks/gpu_lease'
EXPECTED={'GPU-00465dc6-e96e-67f1-285a-84ed61565ac2'}
def run(argv):
 return subprocess.run(argv,capture_output=True,text=True,check=False)
def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.part')
 tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(path)
def owner_status(root):
 p=Path(root)/'coordination/OWNER_STATE.json'
 if not p.is_file():return {'path':str(p),'exists':False}
 try:
  d=json.loads(p.read_text(encoding='utf-8'))
  return {'path':str(p),'exists':True,'owner_active':d.get('owner_active'),
          'status':d.get('status'),'phase':d.get('phase')}
 except Exception as e:return {'path':str(p),'exists':True,'parse_error':str(e)}
now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
host=socket.gethostname();fqdn=socket.getfqdn()
user=os.environ.get('USER','unknown')
scheduler_env={k:os.environ[k] for k in ('SLURM_JOB_ID','SLURM_JOB_NODELIST','PBS_JOBID','LSB_JOBID','JOB_ID') if os.environ.get(k)}
sched={}
for cmd in ('squeue','qstat','bjobs'):
 path=shutil.which(cmd)
 item={'available':bool(path),'path':path}
 if path and cmd=='squeue':
  x=run([path,'-h','-u',user]);item.update({'exit_code':x.returncode,'user_jobs':x.stdout.strip(),'stderr':x.stderr.strip()})
 elif path and cmd=='qstat':
  x=run([path,'-u',user]);item.update({'exit_code':x.returncode,'user_jobs':x.stdout.strip(),'stderr':x.stderr.strip()})
 elif path and cmd=='bjobs':
  x=run([path,'-u',user]);item.update({'exit_code':x.returncode,'user_jobs':x.stdout.strip(),'stderr':x.stderr.strip()})
 sched[cmd]=item
allocated=bool(scheduler_env) or any(x.get('user_jobs') for x in sched.values())
if allocated:host_class='scheduler-managed host/allocation detected'
elif host.startswith('gpu'):host_class='shared GPU host by hostname; login-versus-interactive subtype not independently identified'
else:host_class='host role unknown'
smi=shutil.which('nvidia-smi')
if not smi:raise SystemExit('nvidia-smi executable missing in approved host context')
q=run([smi,'--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu',
       '--format=csv,noheader,nounits'])
if q.returncode:raise SystemExit('aggregate nvidia-smi query failed: '+q.stderr.strip())
gpus=[]
for line in q.stdout.splitlines():
 cols=[x.strip() for x in line.split(',')]
 if len(cols)!=7:raise SystemExit('aggregate GPU row parse failure: '+line)
 gpus.append({'physical_gpu':int(cols[0]),'uuid':cols[1],'name':cols[2],
  'used_mib':int(cols[3]),'free_mib':int(cols[4]),'total_mib':int(cols[5]),'utilization_pct':int(cols[6])})
torchinfo={'version':str(torch.__version__),'cuda_build':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),'visible_device_count':torch.cuda.device_count(),
 'visible_devices':[{'logical_device':i,'name':torch.cuda.get_device_name(i),
  'total_memory_bytes':int(torch.cuda.get_device_properties(i).total_memory)}
  for i in range(torch.cuda.device_count())]}
lease_records=[]
if LEASE.exists():
 for f in sorted(LEASE.iterdir()):
  if f.is_file() and (f.name.startswith('project_child_') or f.name.startswith('gpu_')) and f.suffix=='.lease':
   try:lease_records.append({'path':str(f),'record':json.loads(f.read_text(encoding='utf-8'))})
   except Exception as e:lease_records.append({'path':str(f),'parse_error':str(e)})
disk=shutil.disk_usage(R)
session='ee6008_n4_train_20261012T044137Z'
tmux=shutil.which('tmux')
tmux_state='unavailable'
if tmux:
 x=run([tmux,'has-session','-t',session]);tmux_state='exists' if x.returncode==0 else 'absent'
pre={'schema_version':1,'observed_utc':now,'hostname':host,'fqdn':fqdn,'user':user,
 'host_classification':host_class,'kernel':platform.release(),'scheduler_environment':scheduler_env,
 'scheduler_queries':sched,'scheduler_allocation_or_user_jobs_detected':allocated,
 'container':{'cgroup_v2':Path('/sys/fs/cgroup/cgroup.controllers').exists(),
              'nvidia_devices_visible':[str(p) for p in Path('/dev').glob('nvidia*')]},
 'project_python':torchinfo,'gpu_inventory_source':'aggregate nvidia-smi fields only; no process/PID query',
 'gpus':gpus,'minimum_free_mib_for_job':598,
 'eligible_by_memory':[g['physical_gpu'] for g in gpus if g['free_mib']>=598],
 'expected_source_cuda_gpu_uuid_present':any(g['uuid'] in EXPECTED for g in gpus),
 'lease_directory':str(LEASE),'project_lease_files':lease_records,
 'n4_tmux_session':tmux_state,'disk':{'mount_path':str(R),'total_bytes':disk.total,'used_bytes':disk.used,'free_bytes':disk.free},
 'user_quota':'not queried; no quota command/result provided',
 'project_owner_states':[owner_status(N1),owner_status(N2),owner_status(N3),owner_status(R)],
 'other_user_processes_inspected':False,'signals_sent':False,'preflight_only':True}
write(R/'coordination/HOST_PREFLIGHT.json',pre)
print(json.dumps(pre,ensure_ascii=False,indent=2))
