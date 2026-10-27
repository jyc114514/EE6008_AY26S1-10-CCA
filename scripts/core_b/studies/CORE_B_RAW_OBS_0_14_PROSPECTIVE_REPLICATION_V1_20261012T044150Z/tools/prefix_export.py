#!/usr/bin/env python3
# Reads only frame/timestamp metadata and s[-1], s[0], u[0] prefix captures.
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

DELTA=1.0/30.0
CLOCK_TOL=1e-5
OBS_N=43
ACTION_N=28


def _rows(path, expected_fields):
    with Path(path).open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != list(expected_fields):
            raise ValueError(f'CSV columns differ from locked schema: {path}')
        return list(reader)


def _vector(row,prefix,width):
    return np.asarray([float(row[f'{prefix}_{i:02d}']) for i in range(width)],dtype=np.float32)


def export_prefix(sample_lock, frame_index_csv, prefix_capture_csv, out_npz, receipt_json, seed=20261012):
    if sample_lock.get('status')!='SAMPLE_SELECTED_AWAITING_PREFIX_QA' or sample_lock.get('sample_sealed'):
        raise ValueError('prefix export requires a pre-outcome selected, not yet sealed sample')
    episodes={(str(x['episode_source_group_id']),str(x['episode_id'])):x for x in sample_lock['selected_episodes']}
    frame_fields=['source_group_id','session_id','episode_id','frame_index','timestamp_seconds']
    prefix_fields=['source_group_id','session_id','episode_id','anchor_frame','anchor_timestamp_seconds']
    prefix_fields += [f's_prev_{i:02d}' for i in range(43)]
    prefix_fields += [f's0_{i:02d}' for i in range(43)]
    prefix_fields += [f'u0_{i:02d}' for i in range(28)]
    frame_rows=_rows(frame_index_csv,frame_fields)
    capture_rows=_rows(prefix_capture_csv,prefix_fields)
    grouped={}
    for r in frame_rows:
        key=(str(r['source_group_id']),str(r['episode_id']))
        grouped.setdefault(key,[]).append((int(r['frame_index']),float(r['timestamp_seconds']),str(r.get('session_id',''))))
    captures={}
    for r in capture_rows:
        key=(str(r['source_group_id']),str(r['episode_id']),int(r['anchor_frame']))
        if key in captures: raise ValueError('duplicate prefix capture for anchor')
        captures[key]=r
    all_inputs=[]
    failures=[]
    for (group,episode),sample in episodes.items():
        frames=sorted(grouped.get((group,episode),[]))
        if len(frames)<32:
            failures.append({'source_group_id':group,'episode_id':episode,'reason':'frame_index_support_too_short'})
            continue
        fi=np.asarray([x[0] for x in frames],dtype=np.int64)
        ts=np.asarray([x[1] for x in frames],dtype=np.float64)
        session_ids={x[2] for x in frames if x[2]}
        if len(session_ids)!=1:
            failures.append({'source_group_id':group,'episode_id':episode,'reason':'session_id_missing_or_inconsistent'})
            continue
        session_id=next(iter(session_ids))
        if len(set(fi.tolist()))!=len(fi) or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):
            failures.append({'source_group_id':group,'episode_id':episode,'reason':'frame_index_duplicate_or_clock_invalid'})
            continue
        candidates=[]
        grid=ts[0]+4.0
        last_grid=ts[-1]-30*DELTA
        while grid<=last_grid+1e-12:
            j=int(np.searchsorted(ts,grid,side='right')-1)
            if j<1 or j+30>=len(ts):
                grid+=1.0
                continue
            anchor=int(fi[j])
            window_idx=np.arange(anchor-1,anchor+31,dtype=np.int64)
            pos=np.searchsorted(fi,window_idx)
            if np.any(pos>=len(fi)) or not np.array_equal(fi[pos],window_idx):
                grid+=1.0
                continue
            tw=ts[pos]
            if np.max(np.abs(np.diff(tw)-DELTA))>CLOCK_TOL or grid-ts[j]>DELTA+CLOCK_TOL:
                grid+=1.0
                continue
            row=captures.get((group,episode,anchor))
            if row is None:
                grid+=1.0
                continue
            try:
                if 'anchor_timestamp_seconds' not in row or abs(float(row['anchor_timestamp_seconds'])-float(ts[j]))>CLOCK_TOL:
                    grid+=1.0
                    continue
                if str(row.get('session_id',''))!=session_id:
                    grid+=1.0
                    continue
                sp=_vector(row,'s_prev',43); s0=_vector(row,'s0',43); u0=_vector(row,'u0',28)
            except (KeyError,ValueError):
                grid+=1.0
                continue
            if not (np.isfinite(sp).all() and np.isfinite(s0).all() and np.isfinite(u0).all()):
                grid+=1.0
                continue
            aid='a_'+hashlib.sha256(f'{group}|{episode}|{anchor}'.encode()).hexdigest()[:24]
            candidates.append({'anchor_id':aid,'canonical_cluster_id':sample['canonical_cluster_id'],
                'source_group_id':group,'episode_id':episode,'anchor_frame':anchor,
                'anchor_timestamp':float(tw[1]),'timestamp_window':tw,
                's_prev':sp,'s0':s0,'u0':u0,
                'selection_hash':hashlib.sha256(f'{seed}|{group}|{episode}|{anchor}'.encode()).hexdigest()})
            grid+=1.0
        chosen=sorted(candidates,key=lambda x:x['selection_hash'])[:8]
        if not chosen:
            failures.append({'source_group_id':group,'episode_id':episode,'reason':'no_prefix_only_anchor_passed_locked_clock_grid'})
        all_inputs.extend(chosen)
    if failures:
        receipt={'status':'PREFIX_HARD_QA_FAILED_BEFORE_FORECAST','failure_count':len(failures),
                 'failures':failures,'new_label_payloads_opened':False,
                 'retry_policy':'exclude failed episode before forecasting; re-run sample lock in its frozen hash order; do not inspect outcomes'}
        Path(receipt_json).write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        return receipt
    if len({x['canonical_cluster_id'] for x in all_inputs})!=sample_lock['selected_clusters']:
        raise AssertionError('prefix export failed to retain exactly the selected independent clusters')
    arrays={'anchor_id':np.asarray([x['anchor_id'] for x in all_inputs],dtype='U64'),
        'canonical_cluster_id':np.asarray([x['canonical_cluster_id'] for x in all_inputs],dtype='U64'),
        'source_group_id':np.asarray([x['source_group_id'] for x in all_inputs],dtype='U64'),
        'episode_id':np.asarray([x['episode_id'] for x in all_inputs],dtype='U64'),
        'anchor_frame':np.asarray([x['anchor_frame'] for x in all_inputs],dtype=np.int64),
        'anchor_timestamp':np.asarray([x['anchor_timestamp'] for x in all_inputs],dtype=np.float64),
        'timestamp_window':np.stack([x['timestamp_window'] for x in all_inputs]).astype(np.float64),
        's_prev':np.stack([x['s_prev'] for x in all_inputs]).astype(np.float32),
        's0':np.stack([x['s0'] for x in all_inputs]).astype(np.float32),
        'u0':np.stack([x['u0'] for x in all_inputs]).astype(np.float32)}
    out=Path(out_npz); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**arrays)
    receipt={'status':'PREFIX_QA_PASS','anchor_count':len(arrays['anchor_id']),
        'cluster_count':len(set(arrays['canonical_cluster_id'].tolist())),
        'episode_count':len(set(arrays['episode_id'].tolist())),
        'canonical_cluster_ids':sorted(set(arrays['canonical_cluster_id'].astype(str).tolist())),
        'episode_ids':sorted(set(arrays['episode_id'].astype(str).tolist())),
        'keys':list(arrays),'future_observation_values_read':False,
        'future_command_values_read':False,'future_label_file_opened':False,
        'clock_tolerance_seconds':CLOCK_TOL,'anchor_sampling_seed':seed}
    Path(receipt_json).write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return receipt


def main():
    ap=argparse.ArgumentParser(description='Prefix-only extraction; never opens the future-label file.')
    ap.add_argument('--sample-lock',required=True); ap.add_argument('--frame-index',required=True)
    ap.add_argument('--prefix-capture',required=True); ap.add_argument('--out-npz',required=True)
    ap.add_argument('--receipt',required=True); ap.add_argument('--seed',type=int,default=20261012)
    a=ap.parse_args()
    lock=json.loads(Path(a.sample_lock).read_text(encoding='utf-8'))
    result=export_prefix(lock,a.frame_index,a.prefix_capture,a.out_npz,a.receipt,a.seed)
    print(json.dumps({k:result.get(k) for k in ('status','anchor_count','cluster_count','failure_count')}))


if __name__=='__main__':
    main()
