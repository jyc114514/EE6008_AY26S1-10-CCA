from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT / 'coordination'))


def test_task_watcher_uses_all_four_visible_child_slots_without_gpu_queries():
    tmp = ROOT / 'tmp' / 'watcher_dynamic_slots_test'
    tmp.mkdir(parents=True, exist_ok=True)
    child_code = r"""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1]); job_path=Path(sys.argv[2]); lease=Path(sys.argv[3]); status=Path(sys.argv[4])
sys.path.insert(0,str(root/'coordination'))
import gpu_lease_watcher_task as watcher
stamp=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
snapshots=[watcher.GpuSnapshot(i,f'test-uuid-{i}','test GPU',100,8192,0,stamp) for i in range(4)]
by_index={x.index:x for x in snapshots}
job=json.loads(job_path.read_text())
rc=watcher.run_registered_job(job,lease_dir=lease,status_path=status,
 visible_snapshots_fn=lambda:list(snapshots),snapshot_fn=lambda gpu:by_index[gpu],
 poll_seconds=0.02,max_wait_seconds=5,max_oom_requeues=0)
raise SystemExit(rc)
"""
    children = []
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for i in range(4):
        job = {
            'job_id': f'dynamic-slot-test-{i}',
            'protocol_id': 'CPU_ONLY_TASK_WATCHER_TEST',
            'output_root': str(tmp / f'job-{i}'),
            'log_path': str(tmp / f'job-{i}.child.log'),
            'command': [sys.executable, '-c', 'import time; time.sleep(2.0)'],
            'environment': {'PYTHONDONTWRITEBYTECODE': '1'},
            'resource_estimate': {'measured_peak_mib': 128},
            'max_oom_requeues': 0,
        }
        job_path = tmp / f'job-{i}.json'
        status = tmp / f'job-{i}.status.json'
        job_path.write_text(json.dumps(job))
        children.append(subprocess.Popen(
            [sys.executable, '-c', child_code, str(ROOT), str(job_path), str(tmp / 'shared_leases'), str(status)],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ))
    outputs = []
    for child in children:
        stdout, stderr = child.communicate(timeout=15)
        outputs.append((child.returncode, stdout, stderr))
    assert all(code == 0 for code, _, _ in outputs), outputs
    records = [json.loads((tmp / f'job-{i}.status.json').read_text()) for i in range(4)]
    assert len({r['selected_gpu'] for r in records}) == 4
    assert all(r['max_concurrent_project_gpu_children'] == 4 for r in records)
    assert all(r['lease_released'] and r['project_child_slot_released'] for r in records)
    assert all(r['non_child_process_signaled'] is False for r in records)


def test_task_watcher_fallback_clears_ranking_hint_from_child_assignment():
    from datetime import datetime,timezone
    sys.path.insert(0,str(ROOT/'coordination'))
    import gpu_lease_watcher_task as watcher
    tmp=ROOT/'tmp'/'watcher_preferred_fallback_test'
    tmp.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
    snapshots=[watcher.GpuSnapshot(i,f'fallback-uuid-{i}','test GPU',8000 if i!=1 else 100,8192,0,stamp) for i in range(4)]
    by_index={x.index:x for x in snapshots}
    job={'job_id':'fallback-ranking-only-test','protocol_id':'CPU_ONLY_TASK_WATCHER_TEST',
      'output_root':str(tmp/'output'),'log_path':str(tmp/'child.log'),
      'command':[sys.executable,'-c','import os; assert os.environ["GPU_WATCHER_PHYSICAL_GPU"] == "1"'],
      'environment':{'PYTHONDONTWRITEBYTECODE':'1'},'resource_estimate':{'measured_peak_mib':128},
      'preferred_gpu':3,'max_oom_requeues':0}
    status=tmp/'status.json'
    rc=watcher.run_registered_job(job,lease_dir=tmp/'shared_leases',status_path=status,
      visible_snapshots_fn=lambda:list(snapshots),snapshot_fn=lambda idx:by_index[idx],
      poll_seconds=0.01,max_wait_seconds=2,max_oom_requeues=0)
    record=json.loads(status.read_text())
    assert rc==0,record
    assert record['preferred_gpu']==3 and record['selected_gpu']==1
    assert record['physical_gpu']==1 and record['lease_released'] and record['project_child_slot_released']
