import os
#!/usr/bin/env python3
# Replays only a locked old primary-train prefix fixture; no targets or new scores.
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, os, resource, shutil, time, zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from frozen_predict import predict_views

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OBS=PROJECT/'reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read_rows(path,key,indices):
    want=sorted(set(map(int,indices))); out=[]; cursor=0
    with zipfile.ZipFile(path) as zf:
        with zf.open(key+'.npy') as f:
            ver=np.lib.format.read_magic(f)
            if ver==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
            else: shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
            if fortran or dtype.hasobject: raise ValueError('unsupported NPZ member')
            row_bytes=int(np.prod(shape[1:],dtype=np.int64))*dtype.itemsize
            for idx in want:
                skip=(idx-cursor)*row_bytes
                while skip:
                    b=f.read(min(skip,1<<20))
                    if not b: raise EOFError('truncated member')
                    skip-=len(b)
                raw=f.read(row_bytes)
                if len(raw)!=row_bytes: raise EOFError('truncated row')
                out.append(np.frombuffer(raw,dtype).reshape(shape[1:]).copy())
                cursor=idx+1
    return np.stack(out)


def main():
    start_wall,start_cpu=time.perf_counter(),time.process_time()
    disk_before=shutil.disk_usage(ROOT).free
    threadpool_limits(limits=8)
    freeze=json.loads((ROOT/'protocol/MODEL_FREEZE.json').read_text())
    fixture_prefix=ROOT/'qa/FROZEN_FIXTURE_PREFIX.npz'
    model=ROOT/'frozen_model/ARX_OBS43_FULL_TRAIN.npz'
    if sha(model)!=freeze['packaged_model_sha256']: raise AssertionError('frozen model hash mismatch')
    prefix={k:v.copy() for k,v in np.load(fixture_prefix,allow_pickle=False).items()}
    ids=prefix['anchor_id'].astype(str)
    input_path=OBS/'data/train/inputs_prefix.npz'
    with np.load(input_path,allow_pickle=False) as z:
        train_ids=z['anchor_id'].astype(str)
        old_groups=z['source_group_id'].astype(str)
    index={x:i for i,x in enumerate(train_ids.tolist())}
    if any(x not in index for x in ids): raise AssertionError('fixture ID not in historical primary train')
    ix=[index[x] for x in ids]
    old_group_by_id={train_ids[i]:old_groups[i] for i in ix}
    if not np.array_equal(np.asarray([old_group_by_id[x] for x in ids]),prefix['source_group_id'].astype(str)):
        raise AssertionError('fixture source-group identity mismatch')
    predicted=predict_views(model,fixture_prefix)
    forecast_path=OBS/'predictions/train/FORECASTS_H30.npz'
    checks={}
    for view in ('OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND'):
        accepted=read_rows(forecast_path,view,ix)
        delta=np.abs(predicted[view]-accepted)
        checks[view]={'max_abs':float(delta.max()),'p99_abs':float(np.quantile(delta,.99)),
            'exact_elements':int(np.count_nonzero(delta==0)),'elements':int(delta.size)}
        if checks[view]['max_abs']!=0: raise AssertionError('frozen fixture mismatch: '+view)
    elapsed=time.perf_counter()-start_wall
    cpu=time.process_time()-start_cpu
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    receipt={'status':'PASS','fixture_scope':'old primary-train engineering fixture; not prospective sample',
        'fixture_anchor_count':len(ids),'fixture_anchor_ids_sha256':hashlib.sha256('\\n'.join(ids).encode()).hexdigest(),
        'frozen_model_sha256':sha(model),'frozen_source_code_sha256':freeze['source_inference_code_sha256'],
        'accepted_forecast_archive_sha256':sha(forecast_path),'view_checks':checks,
        'new_labels_opened':False,'future_commands_opened':False,'oracle_or_teacher_opened':False,
        'new_forecast_file_written':False,'new_model_fits':0,'gpu_actions':0,'confirmation_result':False,
        'wall_seconds':elapsed,'process_cpu_seconds':cpu,'peak_rss_bytes':int(rss),
        'disk_free_bytes_before':disk_before,'disk_free_bytes_after':shutil.disk_usage(ROOT).free,
        'created_utc':datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')}
    out=ROOT/'qa/FROZEN_MODEL_FIXTURE_REPLAY_R002.json'
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    log=ROOT/'logs/STAGE_RESOURCES.jsonl'
    with log.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'stage':'frozen_model_old_train_fixture','wall_seconds':elapsed,
            'process_cpu_seconds':cpu,'peak_rss_bytes':int(rss),'disk_free_bytes_before':disk_before,
            'disk_free_bytes_after':shutil.disk_usage(ROOT).free,'new_fits':0,'gpu_actions':0},sort_keys=True)+'\n')
    print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__=='__main__':main()
