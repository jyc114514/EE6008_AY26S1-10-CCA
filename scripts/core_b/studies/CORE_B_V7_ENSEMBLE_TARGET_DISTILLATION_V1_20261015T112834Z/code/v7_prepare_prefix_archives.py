#!/usr/bin/env python3
"""Create a V7 train-prefix-only archive for standalone teacher/student inference."""
from __future__ import annotations
import os
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
TRAIN=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/TRAIN_SELECTED_H30.npz'))
FIELDS=('anchor_id','episode_id','task_id','task_rank','source_cluster','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native')
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main()->int:
    data_dir=ROOT/'data';data_dir.mkdir(parents=True,exist_ok=True)
    out=data_dir/'TRAIN_PREFIXES_NO_LABELS.npz'
    if out.exists():raise FileExistsError(f'preserving existing prefix archive for inspection: {out}')
    with np.load(TRAIN,allow_pickle=False) as z:
        missing=set(FIELDS)-set(z.files)
        if missing:raise RuntimeError(f'accepted train archive missing prefix fields: {sorted(missing)}')
        prefix={k:z[k].copy() for k in FIELDS}
        target_was_not_requested='target_native_h30' not in FIELDS
    n=len(prefix['anchor_id'])
    if n!=1280 or len(set(prefix['anchor_id'].astype(str)))!=n:raise RuntimeError('prefix archive source rows/IDs differ from 1280-row lock')
    tmp=out.with_name(f'.{out.name}.{os.getpid()}.part')
    with tmp.open('wb') as f:np.savez_compressed(f,**prefix)
    os.replace(tmp,out)
    with np.load(out,allow_pickle=False) as z:
        if set(z.files)!=set(FIELDS) or 'target_native_h30' in z.files:raise RuntimeError('V7 train prefix archive contains an unexpected field')
        if len(z['anchor_id'])!=1280:raise RuntimeError('V7 train prefix reload row count mismatch')
    receipt={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
             'source_train_archive':str(TRAIN),'source_train_sha256':sha(TRAIN),'prefix_archive':str(out),'prefix_archive_sha256':sha(out),
             'fields':list(FIELDS),'target_native_h30_requested':False,'target_labels_copied':False,'row_count':n,'final_heldout_payload_access':0}
    (ROOT/'inventory/TRAIN_PREFIX_ARCHIVE.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','prefix_archive':str(out),'sha256':receipt['prefix_archive_sha256'],'rows':n,'target_labels_copied':False},sort_keys=True))
    return 0
if __name__=='__main__':raise SystemExit(main())
