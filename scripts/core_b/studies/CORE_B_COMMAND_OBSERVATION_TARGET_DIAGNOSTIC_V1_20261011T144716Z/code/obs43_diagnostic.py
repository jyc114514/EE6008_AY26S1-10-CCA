#!/usr/bin/env python3
"""CPU-only OBS43 one-step transition and fixed five-view diagnostic."""
from __future__ import annotations
import os
import csv, hashlib, json, math, os, resource, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT = Path(__file__).resolve().parents[1]
V3 = PROJECT / 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
RIDGE = PROJECT / 'reports/benchmarks/CORE_B_H30_HORIZON_LINEAR_PROBE_V1_20261011T063238Z'
QUERY = PROJECT / 'reports/benchmarks/CORE_B_QUERY_GEOMETRY_FULL_DATA_V1_20261008T040522Z'
RAW = PROJECT / 'data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d'
DELTA = 1.0 / 30.0
HORIZONS = (1, 3, 6, 10, 15, 30)
LAMBDAS = (0.001, 0.01, 0.1, 1.0)
EPS = 1e-6
BLOCKS = {'arm_joints_source_alias': (0, 14), 'hand_joints_source_alias': (14, 28), 'leg_joints_source_alias': (28, 43), 'Full43': (0, 43)}


def now(): return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
def sha_bytes(b: bytes): return hashlib.sha256(b).hexdigest()
def sha_file(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def jsonable(x):
    if isinstance(x,Path): return str(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)
def write_json(path: Path, obj, replace=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,default=jsonable,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)
def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    rows=list(rows)
    if fields is None: fields=list(rows[0]) if rows else []
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise'); w.writeheader(); w.writerows(rows)
def read_csv(path: Path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))
def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    tmp=path.with_name('.'+path.name+'.part')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    tmp.replace(path)
def load_npz(path: Path, keys):
    with np.load(path,allow_pickle=False) as z: return {k:z[k].copy() for k in keys}
def stage_receipt(stage, started_wall, started_cpu, extras=None):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt={'stage':stage,'started_wall_monotonic':started_wall,'ended_utc':now(),
      'wall_seconds':time.perf_counter()-started_wall,'process_cpu_seconds':time.process_time()-started_cpu,
      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'cpu_affinity':sorted(os.sched_getaffinity(0)),'threadpools':threadpool_info(),'gpu_api_or_command_invoked':False}
    if extras: receipt.update(extras)
    write_json(ROOT/'logs'/f'{stage}_{stamp}.json',receipt)
    return receipt

def check_input_sources():
    files={
      'V3_MAIN_INPUTS.npz':(V3/'data/V3_MAIN_INPUTS.npz','61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78'),
      'prepare_v3_inputs.py':(V3/'code/prepare_v3_inputs.py','4c3460e83b2b7725d22997808fc106f95e64fb3cf88a17d3e99b8bc33b86f385'),
      'Ridge_FOLD_ASSIGNMENT.csv':(RIDGE/'protocol/FOLD_ASSIGNMENT.csv','51fd23ac9f4d2041ecf7cc889610814bba5403b7a50da0c25cf7f4e63c5a2b3b'),
      'Query_ANCHOR_MANIFEST.csv':(QUERY/'inventory/ANCHOR_MANIFEST.csv','32c0164a64bfda8afd5d2230b26b1c9138ce95b467920c9ba26e87cd48c2c527'),
      'Query_CHECKSUMS.sha256':(QUERY/'CHECKSUMS.sha256','eeff89b297d8324b09c6a117245f325e2eb8012e6ae69cd7b02d1278fa91f9a6'),
    }
    out=[]
    for name,(p,expected) in files.items():
        actual=sha_file(p)
        if actual!=expected: raise AssertionError(f'source hash changed: {name} expected={expected} actual={actual}')
        out.append({'name':name,'path':str(p),'sha256':actual,'bytes':p.stat().st_size,'status':'PASS'})
    for name,p in {
      'V3_ANCHOR_MANIFEST.csv':V3/'ANCHOR_MANIFEST.csv',
      'V3_DATA_ALLOWLIST.csv':V3/'DATA_ALLOWLIST.csv',
      'Ridge_PROTOCOL_LOCKED.json':RIDGE/'protocol/PROTOCOL_LOCKED.json',
      'Ridge_FEATURE_MAP.json':RIDGE/'protocol/FEATURE_MAP.json',
      'Ridge_SOURCE_ACCEPTANCE.json':RIDGE/'inventory/SOURCE_ACCEPTANCE.json',
      'OBS43_DIAGNOSTIC_CODE':ROOT/'code/obs43_diagnostic.py'} .items():
        out.append({'name':name,'path':str(p),'sha256':sha_file(p),'bytes':p.stat().st_size,'status':'READ_ONLY_SOURCE' if name!='OBS43_DIAGNOSTIC_CODE' else 'RUN_CODE'})
    write_json(ROOT/'source/SOURCE_REGISTRY.json',{'locked_utc':now(),'sources':out,'raw_parquet_policy':'only frame_index,timestamp,observation43,action28 columns parsed; raw files read-only; no videos/features/final/heldout'},replace=True)


def verify_prefit_lock():
    lockpath=ROOT/'protocol/PREFIT_LOCK_HASHES.json'
    if not lockpath.is_file(): raise FileNotFoundError('prefit protocol hash lock missing')
    lock=json.loads(lockpath.read_text(encoding='utf-8'))
    mismatches=[]
    for rel,expected in lock['files'].items():
        p=ROOT/rel
        got=sha_file(p) if p.is_file() else 'MISSING'
        if got!=expected: mismatches.append({'path':rel,'expected':expected,'actual':got})
    if mismatches: raise AssertionError(f'prefit protocol/code changed after lock: {mismatches[:3]}')
    return lock

def extract():
    import pyarrow.parquet as pq
    verify_prefit_lock()
    check_input_sources()
    vrows=[r for r in read_csv(V3/'ANCHOR_MANIFEST.csv') if r['population']=='primary']
    qrows=read_csv(QUERY/'inventory/ANCHOR_MANIFEST.csv')
    qmap={r['anchor_id']:r for r in qrows}
    if len(vrows)!=20079 or len(qmap)!=20079 or len(vrows)!=len({r['anchor_id'] for r in vrows}): raise AssertionError('canonical manifest population/uniqueness mismatch')
    with np.load(V3/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
        ids=z['anchor_id'].astype('U64',copy=True)
        episodes=z['episode_index'].astype(np.int64,copy=True)
        split_code=z['split_code'].astype(np.int8,copy=True)
        frames=z['anchor_frame'].astype(np.int64,copy=True)
        anchor_ts=z['anchor_timestamp'].astype(np.float64,copy=True)
        # Deliberately load accepted prefix and known anchor only; targets_native (including H60 storage) is never accessed.
        hist=z['history_native'].astype(np.float32,copy=True)
        anchor_cmd=z['anchor_command_native'].astype(np.float32,copy=True)
    if hist.shape!=(20079,32,71) or anchor_cmd.shape!=(20079,28): raise AssertionError(f'canonical input shape mismatch {hist.shape}/{anchor_cmd.shape}')
    id_index={a:i for i,a in enumerate(ids.tolist())}
    if len(id_index)!=len(ids): raise AssertionError('duplicate canonical archive ids')
    fold_rows_all=read_csv(RIDGE/'protocol/FOLD_ASSIGNMENT.csv')
    fold_rows=[r for r in fold_rows_all if r['split']=='train' and r['fold'].strip().isdigit()]
    fold_by_id={r['anchor_id']:int(r['fold']) for r in fold_rows}
    group_fold={r['source_group_id']:int(r['fold']) for r in fold_rows}
    if len(fold_by_id)!=len(fold_rows): raise AssertionError('duplicate locked fold ids')
    if len(group_fold)!=len({r['source_group_id'] for r in fold_rows}): raise AssertionError('source group has multiple old folds')
    recs=[]; episode_rows=defaultdict(list); identity_error=Counter()
    for r in vrows:
        aid=r['anchor_id']; q=qmap.get(aid); idx=id_index.get(aid)
        if q is None or idx is None: raise AssertionError(f'canonical anchor missing from paired source: {aid}')
        split=r['split']; ep=int(r['episode_index']); a=int(r['anchor_frame'])
        if q['split']!=split or int(q['episode_index'])!=ep or int(q['anchor_frame'])!=a: raise AssertionError(f'manifest identity mismatch {aid}')
        if int(r['final_test_access'])!=0 or int(q['final_test_access'])!=0: raise AssertionError(f'final/heldout row in primary manifest {aid}')
        split_np='train' if int(split_code[idx])==0 else ('dev' if int(split_code[idx])==1 else 'unknown')
        if split_np!=split or int(episodes[idx])!=ep or int(frames[idx])!=a: raise AssertionError(f'canonical array identity mismatch {aid}')
        rec={'anchor_id':aid,'episode_index':ep,'split':split,'source_group_id':q['source_group_id'],'anchor_frame':a,'anchor_timestamp_manifest':float(r['anchor_timestamp']),'anchor_timestamp_query':float(q['anchor_timestamp_seconds']),'parquet_path':r['parquet_path'],'canonical_index':idx}
        recs.append(rec); episode_rows[ep].append(rec)
    train_groups={r['source_group_id'] for r in recs if r['split']=='train'}
    dev_groups={r['source_group_id'] for r in recs if r['split']=='dev'}
    if len(train_groups)!=2712 or len(dev_groups)!=603 or train_groups&dev_groups: raise AssertionError('source group split support mismatch/overlap')
    for ep,rows in episode_rows.items():
        if len({r['split'] for r in rows})!=1 or len({r['source_group_id'] for r in rows})!=1 or len({r['parquet_path'] for r in rows})!=1: raise AssertionError(f'episode identity not unique {ep}')
    eligible_records=[]; all_manifest=[]; raw_sources=[]; excluded=Counter(); time_errors=[]; first_dev_label_time=None
    input_by_split={s:[] for s in ('train','dev')}; label_by_split={s:[] for s in ('train','dev')}; command_by_split={s:[] for s in ('train','dev')}
    expected_columns=['frame_index','timestamp','action','observation.arm_joints','observation.hand_joints','observation.leg_joints']
    for split in ('train','dev'):
      for ep in sorted(e for e,rr in episode_rows.items() if rr[0]['split']==split):
        rows=episode_rows[ep]; rel=rows[0]['parquet_path']; p=(RAW/rel).resolve()
        if not p.is_file():
            for r in rows:
                excluded['missing_source_episode']+=1; all_manifest.append({**{k:r[k] for k in ('anchor_id','episode_index','split','source_group_id','anchor_frame')},'eligible':0,'exclusion_reason':'missing_source_episode','max_interval_error_seconds':''})
            continue
        stat0=p.stat()
        if split=='dev' and first_dev_label_time is None: first_dev_label_time=now()
        tbl=pq.read_table(p,columns=expected_columns,use_threads=False)
        fr=tbl['frame_index'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64,copy=False)
        ts=tbl['timestamp'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64,copy=False)
        def as_matrix(name,width):
            x=np.asarray(tbl[name].to_pylist(),dtype=np.float32)
            if x.shape!=(len(fr),width): return x
            return x
        act=as_matrix('action',28)
        arm=as_matrix('observation.arm_joints',14); hand=as_matrix('observation.hand_joints',14); leg=as_matrix('observation.leg_joints',15)
        if arm.shape==(len(fr),14) and hand.shape==(len(fr),14) and leg.shape==(len(fr),15): obs=np.concatenate([arm,hand,leg],axis=1)
        else: obs=np.empty((len(fr),0),np.float32)
        stat1=p.stat()
        raw_sources.append({'episode_index':ep,'split':split,'parquet_path':rel,'bytes':stat0.st_size,'mtime_ns_before':stat0.st_mtime_ns,'mtime_ns_after':stat1.st_mtime_ns,'size_after':stat1.st_size,'columns_parsed':';'.join(expected_columns),'raw_write_detected':bool(stat0.st_size!=stat1.st_size or stat0.st_mtime_ns!=stat1.st_mtime_ns)})
        for r in rows:
            a=r['anchor_frame']; idx=r['canonical_index']; reasons=[]; maxerr=''
            if act.shape!=(len(fr),28): reasons.append('action_shape_or_nonfinite')
            if obs.shape!=(len(fr),43): reasons.append('observation_shape_or_nonfinite')
            lo=a-1; hi=a+31
            if lo<0 or hi>len(fr): reasons.append('insufficient_past_or_future_frames')
            if not reasons:
                expected=np.arange(lo,hi,dtype=np.int64)
                if not np.array_equal(fr[lo:hi],expected): reasons.append('frame_grid_failure')
                tw=ts[lo:hi]
                if tw.shape!=(32,) or not np.isfinite(tw).all() or np.any(np.diff(tw)<=0): reasons.append('timestamp_nonfinite_or_nonmonotonic')
                else:
                    errs=np.abs(np.diff(tw)-DELTA); maxerr=float(np.max(errs)); time_errors.extend(errs.tolist())
                    if maxerr>1e-5: reasons.append('nominal_clock_tolerance_failure')
                sw=obs[lo:hi]
                uw=act[a:a+30]
                if sw.shape!=(32,43) or not np.isfinite(sw).all(): reasons.append('observation_shape_or_nonfinite')
                if uw.shape!=(30,28) or not np.isfinite(uw).all(): reasons.append('action_shape_or_nonfinite')
                if not np.isfinite(r['anchor_timestamp_manifest']) or abs(float(ts[a])-r['anchor_timestamp_manifest'])>1e-5 or abs(float(ts[a])-r['anchor_timestamp_query'])>1e-5: reasons.append('anchor_identity_mismatch')
                if not reasons:
                    s0=sw[1].astype(np.float32,copy=False); u0=uw[0].astype(np.float32,copy=False)
                    if not np.array_equal(hist[idx,-1,:43],s0): reasons.append('raw_vs_canonical_anchor_observation_mismatch')
                    if not np.array_equal(anchor_cmd[idx],u0): reasons.append('raw_vs_canonical_anchor_command_mismatch')
            if reasons:
                for reason in set(reasons): excluded[reason]+=1
                all_manifest.append({'anchor_id':r['anchor_id'],'episode_index':r['episode_index'],'split':r['split'],'source_group_id':r['source_group_id'],'anchor_frame':a,'eligible':0,'exclusion_reason':';'.join(sorted(set(reasons))),'max_interval_error_seconds':maxerr})
                continue
            s_future=sw[2:32].copy(); u_future=uw.copy()
            inp={'anchor_id':r['anchor_id'],'source_group_id':r['source_group_id'],'episode_index':r['episode_index'],'anchor_frame':a,'anchor_timestamp':float(tw[1]),'s_prev':sw[0].copy(),'s0':s0.copy(),'u0':u0.copy(),'timestamp_window':tw.copy()}
            input_by_split[r['split']].append(inp); label_by_split[r['split']].append(s_future); command_by_split[r['split']].append(u_future)
            eligible_records.append(r)
            all_manifest.append({'anchor_id':r['anchor_id'],'episode_index':r['episode_index'],'split':r['split'],'source_group_id':r['source_group_id'],'anchor_frame':a,'eligible':1,'exclusion_reason':'','max_interval_error_seconds':maxerr})
    for split in ('train','dev'):
        items=input_by_split[split]
        ids_out=np.asarray([x['anchor_id'] for x in items],dtype='U64')
        def arr(key,dtype=None):
            x=np.asarray([i[key] for i in items],dtype=dtype)
            return x
        save_npz(ROOT/f'data/{split}/inputs_prefix.npz',anchor_id=ids_out,source_group_id=np.asarray([x['source_group_id'] for x in items],dtype='U64'),episode_index=arr('episode_index',np.int64),anchor_frame=arr('anchor_frame',np.int64),anchor_timestamp=arr('anchor_timestamp',np.float64),s_prev=arr('s_prev',np.float32),s0=arr('s0',np.float32),u0=arr('u0',np.float32),timestamp_window=arr('timestamp_window',np.float64))
        save_npz(ROOT/f'data/{split}/labels_obs43.npz',anchor_id=ids_out,obs_future=np.asarray(label_by_split[split],dtype=np.float32))
        save_npz(ROOT/f'data/{split}/commands_privileged.npz',anchor_id=ids_out,u_future=np.asarray(command_by_split[split],dtype=np.float32))
    eligible_train=[r for r in eligible_records if r['split']=='train']
    old_ok=all(r['anchor_id'] in fold_by_id for r in eligible_train) and all(r['source_group_id'] in group_fold for r in eligible_train)
    folds={}
    if old_ok:
        for r in eligible_train:
            f=fold_by_id[r['anchor_id']]
            if group_fold.get(r['source_group_id'])!=f: old_ok=False; break
            folds[r['anchor_id']]=f
    if not old_ok:
        groups=sorted({r['source_group_id'] for r in eligible_train},key=lambda g:(hashlib.sha256((g+'|seed=20261011').encode()).hexdigest(),g))
        gf={g:i%3 for i,g in enumerate(groups)}; folds={r['anchor_id']:gf[r['source_group_id']] for r in eligible_train}
        fold_source='fallback_sha256_group_sort_seed_20261011_rank_mod_3'
    else: fold_source='reused_accepted_Ridge_source_group_assignment_restricted_to_eligible_rows'
    frows=[{'anchor_id':r['anchor_id'],'episode_index':r['episode_index'],'source_group_id':r['source_group_id'],'fold':folds[r['anchor_id']],'final_test_access':0} for r in eligible_train]
    group_folds=defaultdict(set)
    for r in frows: group_folds[r['source_group_id']].add(int(r['fold']))
    if any(len(fs)!=1 for fs in group_folds.values()): raise AssertionError('group isolation issue in new folds')
    write_csv(ROOT/'data/FOLD_ASSIGNMENT.csv',frows)
    train_counts=Counter(r['episode_index'] for r in eligible_train)
    n_train=len(eligible_train); n_episode=len(train_counts)
    write_csv(ROOT/'data/TRAIN_EPISODE_WEIGHTS.csv',[
        {'anchor_id':r['anchor_id'],'episode_index':r['episode_index'],'source_group_id':r['source_group_id'],
         'raw_w_i':n_train/(n_episode*train_counts[r['episode_index']]),
         'normalized_p_i':1.0/(n_episode*train_counts[r['episode_index']])}
        for r in eligible_train])
    fold_weight_rows=[]
    for f in range(3):
        fr=[r for r in eligible_train if folds[r['anchor_id']]==f]
        fc=Counter(r['episode_index'] for r in fr); nf=len(fr); ef=len(fc)
        for r in fr:
            fold_weight_rows.append({'fold':f,'anchor_id':r['anchor_id'],'episode_index':r['episode_index'],'source_group_id':r['source_group_id'],
              'raw_w_i':nf/(ef*fc[r['episode_index']]),'normalized_p_i':1.0/(ef*fc[r['episode_index']])})
    write_csv(ROOT/'data/FOLD_EPISODE_WEIGHTS.csv',fold_weight_rows)
    # Hash-selected cases: choose one anchor per group, then choose groups, using IDs only.
    dev_by_group=defaultdict(list)
    for r in (x for x in eligible_records if x['split']=='dev'): dev_by_group[r['source_group_id']].append(r)
    candidates=[]
    for g,rr in dev_by_group.items():
        ar=min(rr,key=lambda x:hashlib.sha256(('case-anchor|seed=20261011|'+x['anchor_id']).encode()).hexdigest())
        candidates.append((hashlib.sha256(('case-group|seed=20261011|'+g).encode()).hexdigest(),g,ar))
    selected=sorted(candidates)[:12]
    case_rows=[]
    for gh,g,r in selected:
        channels={}
        for name,(start,end) in list(BLOCKS.items())[:3]:
            hh=hashlib.sha256(('case-channel|seed=20261011|'+r['anchor_id']+'|'+name).encode()).hexdigest()
            channels[name]=start+(int(hh[:8],16)%(end-start))
        case_rows.append({'anchor_id':r['anchor_id'],'source_group_id':g,'episode_index':r['episode_index'],'group_hash':gh,'arm_joints_channel':channels['arm_joints_source_alias'],'hand_joints_channel':channels['hand_joints_source_alias'],'leg_joints_channel':channels['leg_joints_source_alias'],'selected_before_score_access':1})
    if len(case_rows)<12: raise AssertionError('fewer than 12 eligible dev source groups for fixed cases')
    write_csv(ROOT/'data/DEV_CASES_LOCKED.csv',case_rows)
    write_csv(ROOT/'data/ELIGIBILITY_MANIFEST.csv',all_manifest)
    write_csv(ROOT/'source/RAW_PARQUET_READ_MANIFEST.csv',raw_sources)
    time_rows=[]
    for split in ('train','dev'):
        for item in input_by_split[split]:
            tw=np.asarray(item['timestamp_window'],dtype=np.float64)
            for h in HORIZONS:
                time_rows.append({'anchor_id':item['anchor_id'],'episode_index':item['episode_index'],'split':split,'horizon':h,
                  'anchor_frame':item['anchor_frame'],'target_frame':item['anchor_frame']+h,'anchor_timestamp':tw[1],
                  'target_timestamp':tw[h+1],'actual_target_delta_seconds':tw[h+1]-tw[1],
                  'nominal_target_delta_seconds':h*DELTA,'absolute_delta_error_seconds':abs((tw[h+1]-tw[1])-h*DELTA),
                  'final_test_access':0})
    write_csv(ROOT/'data/TARGET_TIME_GRID.csv',time_rows)
    tr_in=input_by_split['train']; dv_in=input_by_split['dev']
    train_elig=[r for r in eligible_records if r['split']=='train']; dev_elig=[r for r in eligible_records if r['split']=='dev']
    overlap={ 'train_dev_anchor_overlap':len({r['anchor_id'] for r in train_elig}&{r['anchor_id'] for r in dev_elig}), 'train_dev_group_overlap':len({r['source_group_id'] for r in train_elig}&{r['source_group_id'] for r in dev_elig}) }
    summary={'created_utc':now(),'canonical_primary_rows':len(recs),'canonical_train_rows':sum(r['split']=='train' for r in recs),'canonical_dev_rows':sum(r['split']=='dev' for r in recs),'eligible_train_rows':len(train_elig),'eligible_train_groups':len({r['source_group_id'] for r in train_elig}),'eligible_train_episodes':len({r['episode_index'] for r in train_elig}),'eligible_dev_rows':len(dev_elig),'eligible_dev_groups':len({r['source_group_id'] for r in dev_elig}),'eligible_dev_episodes':len({r['episode_index'] for r in dev_elig}),'excluded_rows':len(recs)-len(eligible_records),'exclusion_reason_occurrences':dict(excluded),'train_dev_overlap':overlap,'fold_source':fold_source,'fold_group_counts':{str(f):len({r['source_group_id'] for r in frows if r['fold']==f}) for f in range(3)},'clock_grid':'nominal_30Hz_validated' if eligible_records and not excluded.get('nominal_clock_tolerance_failure') else 'nominal_30Hz_with_exclusions','clock_intervals_checked':len(time_errors),'clock_max_abs_error_seconds':float(max(time_errors)) if time_errors else None,'clock_p99_abs_error_seconds':float(np.quantile(time_errors,.99)) if time_errors else None,'raw_columns_parsed':expected_columns,'raw_video_or_features_read':False,'final_or_heldout_access':0,'dev_label_extraction_started_utc':first_dev_label_time,'dev_labels_constructed':bool(dv_in),'canonical_target_array_accessed':False}
    write_json(ROOT/'data/ELIGIBILITY_SUMMARY.json',summary)
    write_json(ROOT/'data/INPUT_TARGET_ACCESS_LOG.json',{'raw_future_observation_and_command_values_first_materialized':first_dev_label_time,'use':'construct designated dev observation labels and privileged command arrays; not used by fit, lambda selection, or four-forecast prediction','fit_dev_labels_read':False,'forecast_prediction_dev_labels_read':False,'teacher_state_reset_dev_label_read':'not yet; separate post-forecast-acceptance stage','final_test_access':0})
    write_json(ROOT/'qa/DATA_CONTRACT_ACCEPTANCE.json',{'status':'PASS' if len(eligible_records)>0 and overlap['train_dev_group_overlap']==0 else 'FAIL','observation_fields':'raw independent recorded Parquet fields arm14+hand14+leg15','command_field':'raw action28','anchor_obs_reproduction_rows':len(eligible_records),'anchor_command_reproduction_rows':len(eligible_records),'known_anchor_availability':'recorded-clock benchmark assumption; raw issue_time absent','eligible_train_rows':len(train_elig),'eligible_dev_rows':len(dev_elig),'clock_grid':summary['clock_grid'],'final_test_access':0})
    write_json(ROOT/'qa/ELIGIBILITY_ACCEPTANCE.json',{'status':'PASS' if len(eligible_records)>0 and overlap['train_dev_group_overlap']==0 else 'FAIL','summary_path':'data/ELIGIBILITY_SUMMARY.json','all_views_same_support':True,'input_groups_disjoint':overlap['train_dev_group_overlap']==0,'no_motion_based_filtering':True,'no_padding_or_interpolation':True})
    return summary


def episode_weights(episodes):
    episodes=np.asarray(episodes)
    _,inv,cnt=np.unique(episodes,return_inverse=True,return_counts=True)
    return 1.0/(len(cnt)*cnt[inv])
def episode_stats(values,episodes):
    values=np.asarray(values,dtype=np.float64); episodes=np.asarray(episodes)
    means=[]; seconds=[]
    for ep in np.unique(episodes):
        x=values[episodes==ep].reshape(-1,values.shape[-1])
        means.append(x.mean(0)); seconds.append(np.mean(x*x,axis=0))
    mean=np.mean(means,axis=0); var=np.maximum(np.mean(seconds,axis=0)-mean*mean,0.0); raw=np.sqrt(var)
    safe=raw.copy(); safe[safe<=EPS]=1.0
    return mean,raw,safe

def episode_fd_scale(y,episodes):
    means=[]
    for ep in np.unique(episodes):
        x=y[episodes==ep].astype(np.float64)
        means.append(np.mean(np.abs(np.diff(x,axis=1)),axis=(0,1)))
    raw=np.maximum(np.mean(means,axis=0),EPS)
    return raw

def fit_input_scaler(x,episodes):
    w=episode_weights(episodes); mean=np.sum(x*w[:,None],axis=0); second=np.sum((x*x)*w[:,None],axis=0); raw=np.sqrt(np.maximum(second-mean*mean,0.0)); safe=raw.copy(); const=raw<=EPS; safe[const]=1.0
    return mean,safe,const,raw

def ridge_fit(x,y,episodes,lmbda):
    w=episode_weights(episodes); xmu,xstd,const,xraw=fit_input_scaler(x,episodes)
    xs=(x-xmu)/xstd
    ybar=np.sum(y*w[:,None],axis=0); xbar=np.sum(xs*w[:,None],axis=0)
    xc=xs-xbar; yc=y-ybar
    gram=xc.T@(xc*w[:,None]); rhs=xc.T@(yc*w[:,None]); gram.flat[::gram.shape[0]+1]+=lmbda
    coef=np.linalg.solve(gram,rhs); intercept=ybar-xbar@coef
    resid=y-intercept-xs@coef
    mse=float(np.sum(w[:,None]*resid*resid)); penalty=float(lmbda*np.sum(coef*coef))
    model={'x_mean':xmu,'x_std':xstd,'x_constant':const,'x_raw_std':xraw,'coef':coef,'intercept':intercept,'objective_mse':mse,'objective_penalty':penalty,'objective':mse+penalty}
    return model

def position_error_per_anchor(pred,truth,sigma):
    return np.mean(np.abs((pred-truth)/sigma[None,:]),axis=1)
def episode_equal_mean(vals,episodes):
    vals=np.asarray(vals,dtype=np.float64); episodes=np.asarray(episodes); per=[]
    for ep in np.unique(episodes):
        x=vals[episodes==ep]; x=x[np.isfinite(x)]
        if len(x): per.append(float(np.mean(x)))
    return float(np.mean(per)) if per else float('nan')

def load_inputs(split):
    return load_npz(ROOT/f'data/{split}/inputs_prefix.npz',['anchor_id','source_group_id','episode_index','anchor_frame','anchor_timestamp','s_prev','s0','u0','timestamp_window'])
def load_commands(split): return load_npz(ROOT/f'data/{split}/commands_privileged.npz',['anchor_id','u_future'])
def load_labels(split): return load_npz(ROOT/f'data/{split}/labels_obs43.npz',['anchor_id','obs_future'])
def construct_x(inp):
    v0=(inp['s0'].astype(np.float64)-inp['s_prev'].astype(np.float64))/DELTA
    return np.concatenate([inp['s0'].astype(np.float64),v0,inp['u0'].astype(np.float64)],axis=1),v0

def fit_model():
    started=time.perf_counter(); cpu=time.process_time()
    verify_prefit_lock()
    if not (ROOT/'qa/DATA_CONTRACT_ACCEPTANCE.json').is_file(): raise FileNotFoundError('data contract not accepted')
    if json.loads((ROOT/'qa/DATA_CONTRACT_ACCEPTANCE.json').read_text())['status']!='PASS': raise AssertionError('data contract gate failed')
    inp=load_inputs('train'); lab=load_labels('train'); ids=inp['anchor_id']; episodes=inp['episode_index']; groups=inp['source_group_id']
    if not np.array_equal(ids,lab['anchor_id']): raise AssertionError('train label/input IDs mismatch')
    folds={r['anchor_id']:int(r['fold']) for r in read_csv(ROOT/'data/FOLD_ASSIGNMENT.csv')}
    fold=np.asarray([folds[a] for a in ids.tolist()],dtype=np.int8)
    x,_=construct_x(inp); yfuture=lab['obs_future'].astype(np.float64)
    s0=inp['s0'].astype(np.float64); s1=yfuture[:,0]
    oof={l:np.empty((len(ids),43),dtype=np.float32) for l in LAMBDAS}; cvrows=[]; fold_receipts=[]
    for lmbda in LAMBDAS:
      for f in range(3):
        fit=(fold!=f); val=(fold==f)
        yscale,raw_sigma,sigma=episode_stats(yfuture[fit],episodes[fit]); fd=episode_fd_scale(yfuture[fit],episodes[fit]); safe=np.maximum(fd,0.001*sigma)
        target=(s1[fit]-s0[fit])/(DELTA*sigma[None,:])
        model=ridge_fit(x[fit],target,episodes[fit],lmbda)
        xv=(x[val]-model['x_mean'])/model['x_std']; vnorm=model['intercept']+xv@model['coef']; p=s0[val]+DELTA*(vnorm*sigma[None,:])
        oof[lmbda][val]=p.astype(np.float32)
        err=position_error_per_anchor(p,s1[val],sigma)
        cvrows.append({'lambda':lmbda,'fold':f,'fit_rows':int(fit.sum()),'fit_groups':len(set(groups[fit].tolist())),'fit_episodes':len(set(episodes[fit].tolist())),'validation_rows':int(val.sum()),'validation_groups':len(set(groups[val].tolist())),'validation_episodes':len(set(episodes[val].tolist())),'selection_metric_episode_equal_one_step_position_mae_std':episode_equal_mean(err,episodes[val]),'alpha_equivalent_W_times_lambda':float(fit.sum()*lmbda),'fit_objective_mse':model['objective_mse'],'fit_objective_penalty':model['objective_penalty'],'fit_objective_total':model['objective'],'sigma_sha256':sha_bytes(sigma.tobytes()),'input_scaler_sha256':sha_bytes(model['x_mean'].tobytes()+model['x_std'].tobytes()),'coefficient_sha256':sha_bytes(model['coef'].tobytes()+model['intercept'].tobytes()),'final_test_access':0})
        fold_receipts.append({'lambda':lmbda,'fold':f,'fit_anchor_ids_sha256':sha_bytes('\n'.join(ids[fit].tolist()).encode()),'fit_groups_sha256':sha_bytes('\n'.join(sorted(set(groups[fit].tolist()))).encode()),'fit_sigma_obs':sigma.tolist(),'fit_sigma_raw':raw_sigma.tolist(),'fit_fd_scale':fd.tolist(),'fit_safe_fd_obs':safe.tolist(),'fit_x_mean':model['x_mean'].tolist(),'fit_x_std':model['x_std'].tolist(),'fit_x_raw_std':model['x_raw_std'].tolist(),'fit_constant_obs_channels':np.flatnonzero(raw_sigma<=EPS).tolist(),'fit_constant_x_columns':np.flatnonzero(model['x_constant']).tolist()})
    selection=[]
    for lmbda in LAMBDAS:
        # The selection metric uses each fold's fit-only sigma; aggregate fold-specific OOF metrics below.
        fold_metric=[r['selection_metric_episode_equal_one_step_position_mae_std'] for r in cvrows if r['lambda']==lmbda]
        # Recompute exactly episode-equal pooled OOF loss, carrying each held-out fold's fit-only scale.
        pooled=[]
        for f in range(3):
            val=(fold==f); fit=(fold!=f); _,_,sig=episode_stats(yfuture[fit],episodes[fit]); e=position_error_per_anchor(oof[lmbda][val].astype(np.float64),s1[val],sig)
            for j,v in zip(np.flatnonzero(val),e): pooled.append((episodes[j],float(v)))
        epmeans=[]
        for ep in sorted({p[0] for p in pooled}): epmeans.append(float(np.mean([v for e,v in pooled if e==ep])))
        pooled_score=float(np.mean(epmeans))
        selection.append({'lambda':lmbda,'pooled_oof_episode_equal_one_step_position_mae_std':pooled_score,'fold_score_mean_diagnostic_only':float(np.mean(fold_metric))})
    best=min(r['pooled_oof_episode_equal_one_step_position_mae_std'] for r in selection)
    chosen=max(r['lambda'] for r in selection if r['pooled_oof_episode_equal_one_step_position_mae_std']<=best+1e-10)
    yscale,raw_sigma,sigma=episode_stats(yfuture,episodes); fd=episode_fd_scale(yfuture,episodes); safe=np.maximum(fd,0.001*sigma)
    target=(s1-s0)/(DELTA*sigma[None,:]); final=ridge_fit(x,target,episodes,chosen)
    model_path=ROOT/'models/ARX_OBS43_FULL_TRAIN.npz'
    # Rule: 5th percentile of strictly positive absolute within-future increments, per channel, floor 1e-6.
    inc=np.abs(np.diff(yfuture,axis=1)); threshold=np.empty(43,np.float64)
    for k in range(43):
        v=inc[:,:,k]; v=v[v>0]; threshold[k]=max(float(np.quantile(v,.05)) if v.size else 0.0,EPS)
    save_npz(model_path,x_mean=final['x_mean'],x_std=final['x_std'],x_constant=final['x_constant'],x_raw_std=final['x_raw_std'],coef=final['coef'],intercept=final['intercept'],sigma_obs=sigma,sigma_obs_raw=raw_sigma,fd_scale_obs=fd,safe_fd_obs=safe,active_threshold_obs=threshold)
    # Save all CV candidates' OOF positions for audit; these are train-only and participated in lambda selection.
    save_npz(ROOT/'predictions/oof/ONE_STEP_CV_OOF.npz',anchor_id=ids,**{f'lambda_{str(l).replace(".","p")}':v for l,v in oof.items()})
    write_csv(ROOT/'results/CV_FIT_RECEIPTS.csv',cvrows)
    write_csv(ROOT/'results/LAMBDA_SELECTION.csv',selection)
    write_json(ROOT/'models/CV_SCALER_RECEIPTS.json',{'status':'PASS','fold_receipts':fold_receipts,'fit_only_scales':True})
    model_sha=sha_file(model_path)
    full_objective={'objective_mse':final['objective_mse'],'objective_penalty':final['objective_penalty'],'objective_total':final['objective'],'alpha_equivalent_W_times_lambda':float(len(ids)*chosen)}
    npnorm=np.linalg.norm(final['coef'],axis=0)
    # Train-only augmented homogeneous transition matrix.
    K=(final['coef']/final['x_std'][:,None]).T*sigma[:,None]
    Cs=K[:,:43]; Cv=K[:,43:86]
    A=np.block([[np.eye(43)+DELTA*Cs,DELTA*Cv],[Cs,Cv]])
    eig=np.linalg.eigvals(A); rho=float(np.max(np.abs(eig)))
    growth=[]; power=np.eye(86)
    for h in range(1,31):
        power=A@power
        if h in HORIZONS: growth.append({'horizon':h,'spectral_norm_A_power_h':float(np.linalg.norm(power,ord=2)),'frobenius_norm_A_power_h':float(np.linalg.norm(power,'fro'))})
    write_csv(ROOT/'results/TRAIN_TRANSITION_SPECTRUM.csv',[{'eigen_real':float(z.real),'eigen_imag':float(z.imag),'eigen_abs':float(abs(z)),'spectral_radius':rho} for z in eig])
    write_json(ROOT/'results/TRAIN_TRANSITION_GROWTH.json',{'source':'full-train refit coefficients only; no dev labels','state_order':['s[43]','v[43]'],'matrix_shape':list(A.shape),'spectral_radius':rho,'eigenvalues_abs_gt_1':int(np.sum(np.abs(eig)>1.0)),'growth':growth,'matrix_sha256':sha_bytes(A.tobytes()),'interpretation':'rho>1 is a recursion growth signal, not a claim about physical system stability'})
    train_weight=episode_weights(episodes)
    write_json(ROOT/'models/MODEL_LOCK.json',{'status':'LOCKED_BEFORE_DEV_SCORE','locked_utc':now(),'model_path':'models/ARX_OBS43_FULL_TRAIN.npz','model_sha256':model_sha,'input_dim':114,'output_dim':43,'parameter_count_including_intercept':int(final['coef'].size+final['intercept'].size),'selected_lambda':chosen,'selection_metric':'pooled episode-equal one-step standardized position MAE over source-group OOF predictions using fold-fit-only sigma_obs','lambda_grid':LAMBDAS,'grid_extended':False,'cv_fits':len(cvrows),'full_train_refits':1,'train_rows':len(ids),'train_groups':len(set(groups.tolist())),'train_episodes':len(set(episodes.tolist())),'training_anchor_ids_sha256':sha_bytes('\n'.join(ids.tolist()).encode()),'normalized_episode_weights_sha256':sha_bytes(train_weight.tobytes()),'fit_objective':full_objective,'sigma_obs':sigma.tolist(),'sigma_obs_raw':raw_sigma.tolist(),'safe_fd_obs':safe.tolist(),'fd_scale_obs':fd.tolist(),'active_threshold_obs':threshold.tolist(),'constant_or_nearzero_observation_channels':np.flatnonzero(raw_sigma<=EPS).tolist(),'constant_x_feature_columns':np.flatnonzero(final['x_constant']).tolist(),'feature_scaler_sha256':sha_bytes(final['x_mean'].tobytes()+final['x_std'].tobytes()),'coef_sha256':sha_bytes(final['coef'].tobytes()+final['intercept'].tobytes()),'dev_labels_or_scores_read':False,'final_test_access':0})
    write_json(ROOT/'qa/FIT_ACCEPTANCE.json',{'status':'PASS','cv_fits':len(cvrows),'full_train_refits':1,'selected_lambda':chosen,'grid_extended':False,'input_dim':114,'output_dim':43,'parameter_count':int(final['coef'].size+final['intercept'].size),'model_lock':'models/MODEL_LOCK.json','dev_labels_or_scores_read':False,'final_test_access':0})
    return {'selected_lambda':chosen,'model_sha256':model_sha,'spectral_radius':rho,'constant_obs_channels':np.flatnonzero(raw_sigma<=EPS).tolist()}


def load_model():
    lock=json.loads((ROOT/'models/MODEL_LOCK.json').read_text(encoding='utf-8')); p=ROOT/'models/ARX_OBS43_FULL_TRAIN.npz'
    if sha_file(p)!=lock['model_sha256']: raise AssertionError('model hash mismatch')
    d=load_npz(p,['x_mean','x_std','coef','intercept','sigma_obs','safe_fd_obs','active_threshold_obs'])
    return d,lock

def apply_one(model,s,v,u):
    x=np.concatenate([s,v,u],axis=1).astype(np.float64,copy=False)
    xs=(x-model['x_mean'])/model['x_std']
    vnext=(model['intercept']+xs@model['coef'])*model['sigma_obs']
    return s+DELTA*vnext

def rollout(model,inp,u_future,mode):
    if u_future.shape[1:]!=(30,28): raise ValueError(f'commands must be exactly u[0..29], got {u_future.shape}')
    n=len(inp['anchor_id']); s=inp['s0'].astype(np.float64); v=(s-inp['s_prev'].astype(np.float64))/DELTA
    out=np.empty((n,30,43),np.float32)
    for j in range(30):
        u=inp['u0'].astype(np.float64) if mode=='hold' else u_future[:,j].astype(np.float64)
        sn=apply_one(model,s,v,u); v=(sn-s)/DELTA; s=sn; out[:,j]=sn.astype(np.float32)
        if not np.isfinite(sn).all(): raise FloatingPointError(f'nonfinite rollout at step {j+1} mode={mode}')
    return out

def predict_forecasts():
    started=time.perf_counter(); cpu=time.process_time(); model,lock=load_model(); before=sha_file(ROOT/'models/ARX_OBS43_FULL_TRAIN.npz')
    for split in ('train','dev'):
        inp=load_inputs(split); cmd=load_commands(split)
        if not np.array_equal(inp['anchor_id'],cmd['anchor_id']) or not np.array_equal(inp['u0'],cmd['u_future'][:,0]): raise AssertionError(f'{split}: prefix/oracle command mismatch')
        s0=inp['s0'].astype(np.float32); v0=(s0-inp['s_prev'].astype(np.float32))/DELTA
        persistence=np.repeat(s0[:,None,:],30,axis=1)
        cv=s0[:,None,:]+np.arange(1,31,dtype=np.float32)[None,:,None]*DELTA*v0[:,None,:]
        hold=rollout(model,inp,cmd['u_future'],'hold'); oracle=rollout(model,inp,cmd['u_future'],'oracle')
        save_npz(ROOT/f'predictions/{split}/FORECASTS_H30.npz',anchor_id=inp['anchor_id'],OBS_PERSISTENCE=persistence,OBS_CONSTANT_VELOCITY=cv.astype(np.float32),ARX_HOLD_COMMAND=hold,ARX_ORACLE_COMMAND=oracle)
    after=sha_file(ROOT/'models/ARX_OBS43_FULL_TRAIN.npz')
    if before!=after: raise AssertionError('model/scaler changed during prediction')
    return stage_receipt('predict_forecasts',started,cpu,{'model_hash_before':before,'model_hash_after':after,'splits':['train','dev'],'dev_labels_loaded':False,'dev_scores_loaded':False,'forecast_views':4})

def synthetic_qa():
    rng=np.random.default_rng(713)
    model={'x_mean':np.zeros(114),'x_std':np.ones(114),'coef':rng.normal(0,.001,(114,43)),'intercept':rng.normal(0,.001,43),'sigma_obs':np.ones(43)}
    n=5; inp={'anchor_id':np.array([f'a{i}' for i in range(n)]),'s_prev':rng.normal(size=(n,43)).astype(np.float32),'s0':rng.normal(size=(n,43)).astype(np.float32),'u0':rng.normal(size=(n,28)).astype(np.float32)}
    u0=inp['u0']; u=np.repeat(u0[:,None,:],30,axis=1)
    hold=rollout(model,inp,u,'hold'); oracle=rollout(model,inp,u,'oracle')
    assert np.allclose(hold,oracle,rtol=1e-6,atol=1e-6)
    actual=rng.normal(size=(n,30,28)).astype(np.float32); actual[:,0]=u0
    h=rollout(model,inp,actual,'hold'); o=rollout(model,inp,actual,'oracle')
    assert np.allclose(h[:,0],o[:,0],rtol=1e-6,atol=1e-6)
    m0=dict(model); m0['coef']=model['coef'].copy(); m0['coef'][86:,:]=0
    o0=rollout(m0,inp,actual,'oracle'); h0=rollout(m0,inp,actual,'hold')
    assert np.allclose(o0,h0,rtol=1e-6,atol=1e-6)
    try: rollout(model,inp,np.zeros((n,31,28)),'oracle'); raise AssertionError('u[30] unexpectedly accepted')
    except ValueError: pass
    # Prediction interface has no OBS future-label argument; teacher step consumes only current s/v/u.
    def teacher_step(cur_s,cur_v,cur_u): return apply_one(model,cur_s,cur_v,cur_u)
    a=teacher_step(inp['s0'],(inp['s0']-inp['s_prev'])/DELTA,inp['u0'])
    assert np.allclose(a,o[:,0],rtol=1e-6,atol=1e-6)
    return {'status':'PASS','tests':{'first_step_hold_equals_oracle':True,'constant_future_command_hold_equals_oracle_all_30':True,'zero_command_coefficient_hold_equals_oracle':True,'u30_rejected_not_used':True,'forecast_interface_has_no_future_observation_argument':True,'teacher_step_consumes_only_current_state_velocity_command':True}}

def accept_forecasts():
    qa=synthetic_qa(); hashes={}; shapes={}; lock=json.loads((ROOT/'models/MODEL_LOCK.json').read_text())
    for split in ('train','dev'):
        inp=load_inputs(split); p=ROOT/f'predictions/{split}/FORECASTS_H30.npz'
        d=load_npz(p,['anchor_id','OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'])
        if not np.array_equal(d['anchor_id'],inp['anchor_id']): raise AssertionError(f'{split} prediction IDs mismatch')
        for name in ('OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'):
            if d[name].shape!=(len(inp['anchor_id']),30,43) or not np.isfinite(d[name]).all(): raise AssertionError(f'bad prediction shape/value {split}/{name}')
        if not np.allclose(d['ARX_HOLD_COMMAND'][:,0],d['ARX_ORACLE_COMMAND'][:,0],rtol=1e-6,atol=1e-6): raise AssertionError('HOLD/ORACLE first step differs')
        hashes[split]=sha_file(p); shapes[split]={k:list(v.shape) for k,v in d.items() if k!='anchor_id'}
    write_json(ROOT/'qa/FORECAST_ARRAY_ACCEPTANCE.json',{'status':'PASS','accepted_utc':now(),'model_sha256':lock['model_sha256'],'prediction_hashes':hashes,'shapes':shapes,'all_four_views_accepted':True,'dev_labels_read':False,'dev_score_read':False,'synthetic_qa':qa,'final_test_access':0})
    write_json(ROOT/'qa/FORECAST_PREDICTION_ACCEPTANCE.json',{'status':'PASS','model_lock_before_predictions':True,'forecast_views_accepted':4,'first_step_hold_equals_oracle':True,'dev_targets_read':False,'dev_scores_read':False})
    return {'status':'PASS','prediction_hashes':hashes,'synthetic_qa':qa}

def teacher_predictions():
    acc=json.loads((ROOT/'qa/FORECAST_ARRAY_ACCEPTANCE.json').read_text())
    if acc.get('status')!='PASS': raise AssertionError('four-view forecast arrays not accepted')
    model,lock=load_model(); dev_label_read_utc=None; before=time.perf_counter(); cpu=time.process_time()
    for split in ('train','dev'):
        inp=load_inputs(split); cmd=load_commands(split)
        if split=='dev': dev_label_read_utc=now()
        lab=load_labels(split)
        if not np.array_equal(inp['anchor_id'],lab['anchor_id']) or not np.array_equal(inp['anchor_id'],cmd['anchor_id']): raise AssertionError('teacher portal ID mismatch')
        y=lab['obs_future'].astype(np.float64); n=len(y); pred=np.empty((n,30,43),np.float32)
        s0=inp['s0'].astype(np.float64); sp=inp['s_prev'].astype(np.float64); u=cmd['u_future'].astype(np.float64)
        for j in range(30):
            if j==0: curr=s0; vel=(s0-sp)/DELTA
            else:
                curr=y[:,j-1]
                prev=s0 if j==1 else y[:,j-2]
                vel=(curr-prev)/DELTA
            pred[:,j]=apply_one(model,curr,vel,u[:,j]).astype(np.float32)
        save_npz(ROOT/f'predictions/{split}/TEACHER_FORCED_LOCAL.npz',anchor_id=inp['anchor_id'],teacher_next_native=pred)
    write_json(ROOT/'qa/TEACHER_INPUT_ACCESS.json',{'status':'PASS','first_privileged_dev_label_read_utc':dev_label_read_utc,'forecast_array_acceptance_preceded_teacher_label_read':True,'teacher_inputs':['recorded s[j]','recorded v[j] from s[j-1:j]','raw u[j]'],'teacher_targets':'recorded s[j+1] for local scoring only','teacher_path_used_as_forecast':False,'dev_labels_used_for_H30_forecast':False,'model_sha256':lock['model_sha256']})
    return stage_receipt('teacher_privileged_local',before,cpu,{'dev_state_reset_portal':'explicitly used','forecast_array_acceptance_preceded_read':True})


def save_metrics_rows(split,preds,truth,episodes,groups,sigma,safe,threshold):
    model_names=list(preds)
    per_rows=[]; summary=[]
    for h in HORIZONS:
      for block,(lo,hi) in BLOCKS.items():
       sl=slice(lo,hi); y=truth[:,:h,sl].astype(np.float64)
       for view in model_names:
        p=preds[view][:,:h,sl].astype(np.float64)
        pos_std=np.mean(np.abs((p-y)/sigma[None,None,sl]),axis=(1,2)); pos_native=np.mean(np.abs(p-y),axis=(1,2))
        end_std=np.mean(np.abs((p[:,-1]-y[:,-1])/sigma[None,sl]),axis=1); end_native=np.mean(np.abs(p[:,-1]-y[:,-1]),axis=1)
        if h==1:
            fd_safe=np.full(len(p),np.nan); fd_native=np.full(len(p),np.nan); tvp=np.zeros(len(p)); tvg=np.zeros(len(p)); dir_acc=np.full(len(p),np.nan); active=np.zeros(len(p),int); positive=np.zeros(len(p),int); negative=np.zeros(len(p),int); zero_tv=np.zeros(len(p),int)
        else:
            dp=np.diff(p,axis=1); dy=np.diff(y,axis=1); fd_safe=np.mean(np.abs(dp-dy)/safe[None,None,sl],axis=(1,2)); fd_native=np.mean(np.abs(dp-dy),axis=(1,2))
            tvp=np.sum(np.abs(dp),axis=(1,2)); tvg=np.sum(np.abs(dy),axis=(1,2)); zero_tv=(tvg==0).astype(int)
            active_mask=np.abs(dy)>threshold[None,None,sl]; active=active_mask.sum(axis=(1,2)); correct=(np.sign(dp)==np.sign(dy))&active_mask; dir_acc=np.divide(correct.sum(axis=(1,2)),active,out=np.full(len(p),np.nan),where=active>0)
            positive=((dy>threshold[None,None,sl])&active_mask).sum(axis=(1,2)); negative=((dy < -threshold[None,None,sl])&active_mask).sum(axis=(1,2))
        for ep in np.unique(episodes):
            ix=np.flatnonzero(episodes==ep)
            gidset=np.unique(groups[ix]);
            if len(gidset)!=1: raise AssertionError('episode maps multiple source groups')
            gid=str(gidset[0]); ix0=ix
            # episode-level metrics, retaining support needed for paired group bootstrap
            row={'split':split,'view':view,'horizon':h,'block':block,'episode_index':int(ep),'source_group_id':gid,'anchors':len(ix)}
            for key,vec in [('prefix_position_standardized_mae',pos_std),('prefix_position_native_mae',pos_native),('endpoint_position_standardized_mae',end_std),('endpoint_position_native_mae',end_native),('future_internal_fd_safe_normalized',fd_safe),('future_internal_fd_native_mae',fd_native),('tv_pred',tvp),('tv_gt',tvg)]:
                z=vec[ix]; z=z[np.isfinite(z)]; row[key]=float(np.mean(z)) if len(z) else float('nan')
            row['tv_ratio_episode']=row['tv_pred']/row['tv_gt'] if row['tv_gt']>0 else float('nan')
            if h>1:
                # pool active direction support within episode before calculating accuracy/recalls
                am=int(active[ix].sum()); posn=int(positive[ix].sum()); negn=int(negative[ix].sum()); cor=float(np.nansum(dir_acc[ix]*active[ix]))
                row['direction_active_components']=am; row['direction_positive_components']=posn; row['direction_negative_components']=negn
                row['direction_accuracy_active']=cor/am if am else float('nan')
                # Positive/negative prediction correctness reconstructed by channel-step sample aggregation.
                dyp=np.diff(y[ix],axis=1); dpp=np.diff(p[ix],axis=1); th=threshold[None,None,sl]
                pm=(dyp>th); nm=(dyp < -th)
                row['positive_recall']=float(np.sum((np.sign(dpp)==1)&pm)/np.sum(pm)) if np.sum(pm)>0 else float('nan')
                row['negative_recall']=float(np.sum((np.sign(dpp)==-1)&nm)/np.sum(nm)) if np.sum(nm)>0 else float('nan')
                row['balanced_direction_accuracy']=0.5*(row['positive_recall']+row['negative_recall']) if np.isfinite(row['positive_recall']) and np.isfinite(row['negative_recall']) else float('nan')
            else:
                row.update(direction_active_components=0,direction_positive_components=0,direction_negative_components=0,direction_accuracy_active=float('nan'),positive_recall=float('nan'),negative_recall=float('nan'),balanced_direction_accuracy=float('nan'))
            row['zero_gt_tv_windows']=int(zero_tv[ix].sum())
            per_rows.append(row)
        eps=[r for r in per_rows if r['split']==split and r['view']==view and r['horizon']==h and r['block']==block]
        def mean(key):
            x=np.asarray([r[key] for r in eps],float); x=x[np.isfinite(x)]; return float(np.mean(x)) if len(x) else float('nan')
        base=next((r for r in per_rows if r['split']==split and r['view']=='OBS_PERSISTENCE' and r['horizon']==h and r['block']==block),None)
        posp=mean('prefix_position_standardized_mae'); fdp=mean('future_internal_fd_safe_normalized')
        base_pos=mean_from_eps(per_rows,split,'OBS_PERSISTENCE',h,block,'prefix_position_standardized_mae'); base_fd=mean_from_eps(per_rows,split,'OBS_PERSISTENCE',h,block,'future_internal_fd_safe_normalized')
        summary.append({'split':split,'view':view,'horizon':h,'block':block,'anchors':len(episodes),'episodes':len(np.unique(episodes)),'source_groups':len(np.unique(groups)),'position_prefix_standardized_mae':posp,'position_endpoint_standardized_mae':mean('endpoint_position_standardized_mae'),'position_prefix_native_mae':mean('prefix_position_native_mae'),'position_endpoint_native_mae':mean('endpoint_position_native_mae'),'position_skill_vs_obs_persistence':1.0-posp/base_pos if base_pos>0 and np.isfinite(posp) else float('nan'),'safe_fd_error':fdp,'safe_fd_native_mae':mean('future_internal_fd_native_mae'),'safe_fd_skill_vs_obs_persistence':1.0-fdp/base_fd if base_fd>0 and np.isfinite(fdp) else float('nan'),'tv_pred_episode_equal_mean':mean('tv_pred'),'tv_gt_episode_equal_mean':mean('tv_gt'),'tv_ratio_aggregate':mean('tv_pred')/mean('tv_gt') if mean('tv_gt')>0 else float('nan'),'zero_gt_tv_episodes':sum(r['tv_gt']==0 for r in eps if np.isfinite(r['tv_gt'])),'direction_accuracy_active_episode_equal':mean('direction_accuracy_active'),'balanced_direction_accuracy_episode_equal':mean('balanced_direction_accuracy'),'direction_active_components':sum(r['direction_active_components'] for r in eps),'direction_positive_components':sum(r['direction_positive_components'] for r in eps),'direction_negative_components':sum(r['direction_negative_components'] for r in eps),'direction_episodes_active':sum(r['direction_active_components']>0 for r in eps),'direction_episodes_both_signs':sum(r['direction_positive_components']>0 and r['direction_negative_components']>0 for r in eps),'zero_gt_tv_windows':sum(r['zero_gt_tv_windows'] for r in eps)})
    return per_rows,summary

def mean_from_eps(rows,split,view,h,block,key):
    vals=[r[key] for r in rows if r['split']==split and r['view']==view and r['horizon']==h and r['block']==block and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float('nan')

def teacher_metrics(split,teacher,truth,inputs,commands,episodes,groups,sigma,safe):
    rows=[]; s0=inputs['s0'].astype(np.float64); sp=inputs['s_prev'].astype(np.float64); y=truth.astype(np.float64); cmd=commands.astype(np.float64)
    for step in range(30):
        cur=s0 if step==0 else y[:,step-1]
        prev=sp if step==0 else (s0 if step==1 else y[:,step-2])
        localcv=cur+DELTA*((cur-prev)/DELTA)
        localp=cur
        tp=teacher[:,step].astype(np.float64); target=y[:,step]
        for block,(lo,hi) in BLOCKS.items():
            sl=slice(lo,hi)
            for view,p in [('TEACHER_FORCED_LOCAL',tp),('LOCAL_PERSISTENCE',localp),('LOCAL_CONSTANT_VELOCITY',localcv)]:
                e=np.mean(np.abs((p[:,sl]-target[:,sl])/sigma[None,sl]),axis=1)
                en=np.mean(np.abs(p[:,sl]-target[:,sl]),axis=1)
                incp=p[:,sl]-cur[:,sl]; incy=target[:,sl]-cur[:,sl]
                ie=np.mean(np.abs((incp-incy)/safe[None,sl]),axis=1); ien=np.mean(np.abs(incp-incy),axis=1)
                rows.append({'split':split,'step':step+1,'block':block,'view':view,'episode_equal_position_mae_std':episode_equal_mean(e,episodes),'episode_equal_position_mae_native':episode_equal_mean(en,episodes),'episode_equal_local_increment_error_safe':episode_equal_mean(ie,episodes),'episode_equal_local_increment_error_native':episode_equal_mean(ien,episodes),'anchors':len(episodes),'episodes':len(np.unique(episodes)),'source_groups':len(np.unique(groups)),'teacher_is_forecast':False})
    return rows

def paired_bootstrap(dev_episode_rows, seed=20261011, nrep=10000):
    rng=np.random.default_rng(seed); out=[]
    for block in BLOCKS:
      for horizon in (30,):
       for metric in ('prefix_position_standardized_mae','future_internal_fd_safe_normalized'):
        if metric.startswith('future') and horizon==1: continue
        for contrast,left,right in [('G_HOLD','OBS_PERSISTENCE','ARX_HOLD_COMMAND'),('G_DRIVER','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'),('G_CV','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND')]:
            # Group-level sums/counts of episode losses; bootstrap groups and retain every episode in each draw.
            by_view=defaultdict(dict)
            for r in dev_episode_rows:
                if r['horizon']==horizon and r['block']==block and r['view'] in (left,right) and np.isfinite(r[metric]): by_view[r['view']][r['source_group_id']+'|'+str(r['episode_index'])]=r
            keys=sorted({r['source_group_id'] for view in (left,right) for r in by_view[view].values()})
            if not keys: continue
            sums_l=[]; sums_r=[]; counts=[]; gids=[]
            for gid in keys:
                vl=[r[metric] for r in by_view[left].values() if r['source_group_id']==gid]; vr=[r[metric] for r in by_view[right].values() if r['source_group_id']==gid]
                if len(vl)!=len(vr) or not vl: raise AssertionError('paired group episode support mismatch')
                sums_l.append(sum(vl)); sums_r.append(sum(vr)); counts.append(len(vl)); gids.append(gid)
            ix=rng.integers(0,len(gids),size=(nrep,len(gids)))
            c=np.asarray(counts)[ix]; den=c.sum(1)
            draw=(np.asarray(sums_l)[ix].sum(1)-np.asarray(sums_r)[ix].sum(1))/den
            point=(sum(sums_l)-sum(sums_r))/sum(counts)
            lo,hi=np.quantile(draw,[.025,.975])
            out.append({'contrast':contrast,'split':'dev','horizon':horizon,'block':block,'metric':metric,'left_minus_right':float(point),'ci95_low':float(lo),'ci95_high':float(hi),'source_groups':len(gids),'replicates':nrep,'seed':seed,'scope':'conditional on fixed model and repeatedly exposed dev; no p-values'})
    return out

def score_all():
    acc=json.loads((ROOT/'qa/FORECAST_ARRAY_ACCEPTANCE.json').read_text())
    if acc.get('status')!='PASS': raise AssertionError('forecast arrays not accepted before dev scoring')
    if not (ROOT/'qa/TEACHER_INPUT_ACCESS.json').is_file(): raise AssertionError('teacher local portal not audited')
    start=now(); t0=time.perf_counter(); c0=time.process_time(); model,lock=load_model(); sigma=model['sigma_obs']; safe=model['safe_fd_obs']; threshold=model['active_threshold_obs']
    allper=[]; allsummary=[]; teach_rows=[]; dev_first_score_read=None
    for split in ('train','dev'):
        pred_path=ROOT/f'predictions/{split}/FORECASTS_H30.npz'; p=load_npz(pred_path,['anchor_id','OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'])
        inp=load_inputs(split)
        if not np.array_equal(p['anchor_id'],inp['anchor_id']): raise AssertionError('prediction identity mismatch')
        if split=='dev' and dev_first_score_read is None: dev_first_score_read=now()
        lab=load_labels(split); cmd=load_commands(split); teacher=load_npz(ROOT/f'predictions/{split}/TEACHER_FORCED_LOCAL.npz',['anchor_id','teacher_next_native'])
        if not np.array_equal(inp['anchor_id'],lab['anchor_id']) or not np.array_equal(inp['anchor_id'],teacher['anchor_id']) or not np.array_equal(inp['anchor_id'],cmd['anchor_id']): raise AssertionError('scorer inputs ID mismatch')
        truth=lab['obs_future']; episodes=inp['episode_index']; groups=inp['source_group_id']
        pred={name:p[name] for name in ('OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND')}
        per,summ=save_metrics_rows(split,pred,truth,episodes,groups,sigma,safe,threshold); allper.extend(per); allsummary.extend(summ)
        teach_rows.extend(teacher_metrics(split,teacher['teacher_next_native'],truth,inp,cmd['u_future'],episodes,groups,sigma,safe))
    write_csv(ROOT/'results/EPISODE_FORECAST_METRICS.csv',allper)
    write_csv(ROOT/'results/FORECAST_METRICS_BY_HORIZON.csv',allsummary)
    write_csv(ROOT/'results/TEACHER_LOCAL_METRICS.csv',teach_rows)
    dev=[r for r in allper if r['split']=='dev']
    write_csv(ROOT/'results/PAIRED_GROUP_BOOTSTRAP.csv',paired_bootstrap(dev))
    write_json(ROOT/'qa/SCORE_ACCEPTANCE.json',{'status':'PASS','forecast_arrays_accepted_before_score_read':True,'first_dev_score_label_read_utc':dev_first_score_read,'scoring_started_utc':start,'model_sha256':lock['model_sha256'],'dev_labels_used_by_forecast_models':False,'dev_labels_used_for_teacher_state_resets':True,'teacher_forced_ranked_as_forecast':False,'fixed_horizons':HORIZONS,'final_test_access':0,'heldout_access':0,'p_values':False})
    return stage_receipt('score',t0,c0,{'first_dev_score_label_read_utc':dev_first_score_read,'metrics_rows':len(allper),'summary_rows':len(allsummary),'teacher_local_rows':len(teach_rows),'bootstrap_rows':len(read_csv(ROOT/'results/PAIRED_GROUP_BOOTSTRAP.csv'))})


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    t0=time.perf_counter(); c0=time.process_time(); sums=read_csv(ROOT/'results/FORECAST_METRICS_BY_HORIZON.csv'); teachers=read_csv(ROOT/'results/TEACHER_LOCAL_METRICS.csv'); growth=json.loads((ROOT/'results/TRAIN_TRANSITION_GROWTH.json').read_text()); cases=read_csv(ROOT/'data/DEV_CASES_LOCKED.csv')
    views=['OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND']; colors={'OBS_PERSISTENCE':'#444444','OBS_CONSTANT_VELOCITY':'#377eb8','ARX_HOLD_COMMAND':'#4daf4a','ARX_ORACLE_COMMAND':'#e41a1c'}
    # Figure 1: source separation and nominal clock / view roles.
    fig,ax=plt.subplots(figsize=(11,5.8)); ax.axis('off')
    ax.text(.04,.84,'Recorded source fields (different recorded variables)',fontsize=13,weight='bold')
    ax.add_patch(plt.Rectangle((.04,.56),.43,.18,facecolor='#e8eef7',edgecolor='#345'))
    ax.text(.06,.65,'s(t) = observation.arm[14] + hand[14] + leg[15]\n43 native numeric channels; unit/anatomy unresolved',va='center',fontsize=11)
    ax.add_patch(plt.Rectangle((.53,.56),.43,.18,facecolor='#f7eee4',edgecolor='#853'))
    ax.text(.55,.65,'u(t) = action[28]\nseparate recorded command field; no 28↔43 pairing',va='center',fontsize=11)
    ax.annotate('',xy=(.49,.48),xytext=(.25,.55),arrowprops=dict(arrowstyle='->',lw=2,color='#345'))
    ax.annotate('',xy=(.70,.48),xytext=(.74,.55),arrowprops=dict(arrowstyle='->',lw=2,color='#853'))
    ax.text(.04,.39,'At t0: prefix inputs s[-1], s[0], u[0] (known-anchor contract assumption)',fontsize=10.5)
    ax.plot([.06,.94],[.26,.26],color='black',lw=1.4); ax.scatter([.22,.36,.50,.64,.78,.92],[.26]*6,c='black')
    for x,label in zip([.22,.36,.50,.64,.78,.92],['t−δ','t0','t1','t2','…','t30']): ax.text(x,.18,label,ha='center',fontsize=9)
    ax.text(.04,.055,'One-step fit: s[0]→s[1] once per train anchor. HOLD uses u[0]; ORACLE uses u[0..29] (privileged); teacher resets to GT state each step (local only).',fontsize=9.3)
    fig.tight_layout(); fig.savefig(ROOT/'figures/FIG01_SOURCE_AND_CLOCK.png',dpi=180); plt.close(fig)
    # Figure 2: forecast horizon skill.
    fig,axs=plt.subplots(1,2,figsize=(12,4.7))
    for ax,metric,title in [(axs[0],'position_skill_vs_obs_persistence','Prefix position skill vs OBS persistence'),(axs[1],'safe_fd_skill_vs_obs_persistence','Internal safe-FD skill vs OBS persistence')]:
        for v in views:
            rr=sorted([r for r in sums if r['split']=='dev' and r['view']==v and r['block']=='Full43' and (r[metric] not in ('nan',''))],key=lambda r:int(r['horizon']))
            x=[int(r['horizon']) for r in rr]; y=[float(r[metric]) for r in rr]
            ax.plot(x,y,marker='o',label=v.replace('_',' '),color=colors[v])
        ax.axhline(0,color='black',lw=.8); ax.set_xticks(HORIZONS); ax.set_xlabel('Fixed horizon (steps)'); ax.set_ylabel('Skill (1 − loss / persistence loss)'); ax.set_title(title); ax.grid(alpha=.25)
    axs[1].text(.98,.03,'H=1 internal FD = NA',transform=axs[1].transAxes,ha='right',fontsize=8)
    axs[0].legend(fontsize=7,loc='best'); fig.suptitle('DEV (previously exposed; exploratory) — four forecast views only'); fig.tight_layout(); fig.savefig(ROOT/'figures/FIG02_DEV_HORIZON_SKILL.png',dpi=180); plt.close(fig)
    # Figure 3: H30 Full43 + source-block position/FD errors.
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    blocks=list(BLOCKS); xx=np.arange(len(blocks)); width=.19
    for j,v in enumerate(views):
        rr=[next(r for r in sums if r['split']=='dev' and r['view']==v and r['horizon']=='30' and r['block']==b) for b in blocks]
        axs[0].bar(xx+(j-1.5)*width,[float(r['position_prefix_standardized_mae']) for r in rr],width,label=v.replace('_',' '),color=colors[v])
        axs[1].bar(xx+(j-1.5)*width,[float(r['safe_fd_error']) if r['safe_fd_error']!='nan' else np.nan for r in rr],width,color=colors[v])
    for ax,title in zip(axs,['H30 prefix position MAE / OBS σ','H30 future-internal safe-FD error']): ax.set_xticks(xx,blocks,rotation=15); ax.set_title(title); ax.grid(axis='y',alpha=.25)
    axs[0].legend(fontsize=7); fig.suptitle('DEV raw source blocks; unit unknown; same eligible anchors'); fig.tight_layout(); fig.savefig(ROOT/'figures/FIG03_DEV_H30_RAW_BLOCKS.png',dpi=180); plt.close(fig)
    # Figure 4: teacher local diagnostic, explicitly separated from rollouts.
    fig,ax=plt.subplots(figsize=(9,5))
    for v,c in [('TEACHER_FORCED_LOCAL','#e41a1c'),('LOCAL_PERSISTENCE','#444444'),('LOCAL_CONSTANT_VELOCITY','#377eb8')]:
        rr=sorted([r for r in teachers if r['split']=='dev' and r['view']==v and r['block']=='Full43'],key=lambda r:int(r['step']))
        ax.plot([int(r['step']) for r in rr],[float(r['episode_equal_position_mae_std']) for r in rr],label=v.replace('_',' '),color=c,lw=1.5)
    ax.set_xlabel('Local transition index j+1'); ax.set_ylabel('One-step position MAE / OBS σ'); ax.set_title('Teacher forcing is a GT-reset local diagnostic, not an H30 forecast'); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/'figures/FIG04_TEACHER_LOCAL_ONLY.png',dpi=180); plt.close(fig)
    # Figure 5: full-train linear recursion spectrum/growth.
    eig=np.asarray([float(r['eigen_abs']) for r in read_csv(ROOT/'results/TRAIN_TRANSITION_SPECTRUM.csv')]); gr=growth['growth']
    fig,axs=plt.subplots(1,2,figsize=(10,4.2)); axs[0].hist(eig,bins=24,color='#4c78a8'); axs[0].axvline(1,color='red',ls='--'); axs[0].set_xlabel('|eigenvalue|'); axs[0].set_ylabel('count'); axs[0].set_title(f'Train-only 86×86 transition; ρ={growth["spectral_radius"]:.3g}')
    axs[1].plot([int(r['horizon']) for r in gr],[float(r['spectral_norm_A_power_h']) for r in gr],marker='o'); axs[1].set_yscale('log'); axs[1].set_xticks([int(r['horizon']) for r in gr]); axs[1].set_xlabel('h'); axs[1].set_ylabel('||A^h||₂ (log scale)'); axs[1].set_title('Homogeneous linear perturbation growth'); axs[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(ROOT/'figures/FIG05_TRAIN_TRANSITION_SPECTRUM.png',dpi=180); plt.close(fig)
    # Figure 6: outcome-independent fixed-case native paths; 12-page PDF.
    inp=load_inputs('dev'); idx={a:i for i,a in enumerate(inp['anchor_id'].tolist())}
    lab=load_labels('dev')['obs_future']; forecast=load_npz(ROOT/'predictions/dev/FORECASTS_H30.npz',['OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'])
    write_csv(ROOT/'figure_data/HORIZON_METRICS.csv',sums)
    path_rows=[]
    for case in cases:
        i=idx[case['anchor_id']]
        for block_name,channel_key in [('arm_joints_source_alias','arm_joints_channel'),('hand_joints_source_alias','hand_joints_channel'),('leg_joints_source_alias','leg_joints_channel')]:
            cidx=int(case[channel_key])
            for step in range(30):
                path_rows.append({'anchor_id':case['anchor_id'],'source_group_id':case['source_group_id'],'block':block_name,'channel_index':cidx,'step':step+1,'view':'GROUND_TRUTH','value_native':float(lab[i,step,cidx]),'units':'unresolved'})
                for v in views:
                    path_rows.append({'anchor_id':case['anchor_id'],'source_group_id':case['source_group_id'],'block':block_name,'channel_index':cidx,'step':step+1,'view':v,'value_native':float(forecast[v][i,step,cidx]),'units':'unresolved'})
    write_csv(ROOT/'figure_data/CASE_PATHS_NATIVE.csv',path_rows)
    with PdfPages(ROOT/'figures/FIG06_FIXED_DEV_NATIVE_CASE_PATHS.pdf') as pdf:
      for case in cases:
        i=idx[case['anchor_id']]; fig,axs=plt.subplots(1,3,figsize=(14,4.5),sharex=True);
        chan=[int(case['arm_joints_channel']),int(case['hand_joints_channel']),int(case['leg_joints_channel'])]
        for ax,cidx,bname in zip(axs,chan,list(BLOCKS)[:3]):
            ax.plot(np.arange(1,31),lab[i,:,cidx],label='GT',color='black',lw=2)
            for v in views:
                ax.plot(np.arange(1,31),forecast[v][i,:,cidx],label=v.replace('_',' '),color=colors[v],lw=1.2,alpha=.9)
            ax.set_title(f'{bname}: channel {cidx}'); ax.set_xlabel('step'); ax.set_ylabel('recorded numeric native value (unit unresolved)'); ax.grid(alpha=.2)
        axs[0].legend(fontsize=6,loc='best'); fig.suptitle(f"Case {case['anchor_id']} — source group {case['source_group_id'][:10]}\nDEV repeated exposure; fixed by ID hash before scores")
        fig.text(.5,.01,'ARX_ORACLE_COMMAND: PRIVILEGED FUTURE COMMAND / NON-DEPLOYABLE',ha='center',color='#b2182b',weight='bold',fontsize=11)
        fig.tight_layout(rect=(0,.06,1,.9)); pdf.savefig(fig); plt.close(fig)
    # Write captions/alt.
    cap='''# Figure captions and alt text\n\n1. **Source and clock.** Raw Parquet stores OBS43 and action28 in separate columns. The chart shows the nominal 30 Hz anchor-to-H30 grid. Anchor action availability follows the accepted known-anchor benchmark contract; there is no raw issue-time field. Numeric units and anatomical interpretation are unresolved.\n2. **Development horizon skill.** Four forecast views on the same eligible development anchors; skill is relative to OBS persistence. Development data were previously exposed. H=1 future-internal FD is NA. Oracle uses privileged future commands and is non-deployable.\n3. **Development H30 raw blocks.** Position and safe-FD absolute losses by the source aliases arm14/hand14/leg15 and Full43. Blocks are field slices, not verified anatomy.\n4. **Teacher local diagnostic.** Each one-step prediction starts from recorded ground-truth state and velocity and uses the recorded current command. This is not an open-loop rollout or H30 forecast.\n5. **Train-only transition spectrum.** Eigenvalue magnitudes and powers of the fitted homogeneous state transition matrix. Values above one are a recursion signal, not a physical stability claim.\n6. **Fixed development native cases.** Twelve cases selected from eligible IDs by a deterministic hash before score access. One channel per source block is hash-selected; native numeric values are plotted, units unresolved. Oracle curves are labeled **PRIVILEGED FUTURE COMMAND / NON-DEPLOYABLE**.\n'''
    (ROOT/'figures/FIGURE_CAPTIONS_AND_ALT.zh-CN.md').write_text(cap,encoding='utf-8')
    return stage_receipt('figures',t0,c0,{'figures':['FIG01_SOURCE_AND_CLOCK.png','FIG02_DEV_HORIZON_SKILL.png','FIG03_DEV_H30_RAW_BLOCKS.png','FIG04_TEACHER_LOCAL_ONLY.png','FIG05_TRAIN_TRANSITION_SPECTRUM.png','FIG06_FIXED_DEV_NATIVE_CASE_PATHS.pdf'],'case_pages':len(cases)})


def main():
    if len(sys.argv)!=2 or sys.argv[1] not in {'extract','fit','predict','accept','teacher','score','figures'}:
        raise SystemExit('usage: obs43_diagnostic.py {extract|fit|predict|accept|teacher|score|figures}')
    stage=sys.argv[1]
    with threadpool_limits(limits=8):
        if stage=='extract':
            t=time.perf_counter(); c=time.process_time(); result=extract(); rec=stage_receipt('extract',t,c,{'eligible_train_rows':result['eligible_train_rows'],'eligible_dev_rows':result['eligible_dev_rows'],'raw_parquets':len(read_csv(ROOT/'source/RAW_PARQUET_READ_MANIFEST.csv'))})
        elif stage=='fit':
            t=time.perf_counter(); c=time.process_time(); result=fit_model()
            stage_receipt('fit',t,c,{'selected_lambda':result['selected_lambda'],'model_sha256':result['model_sha256'],'spectral_radius':result['spectral_radius'],'cv_fits':12,'full_train_refits':1})
            print(json.dumps(result,sort_keys=True))
        elif stage=='predict': result=predict_forecasts(); print(json.dumps(result,sort_keys=True))
        elif stage=='accept': result=accept_forecasts(); print(json.dumps(result,sort_keys=True))
        elif stage=='teacher': result=teacher_predictions(); print(json.dumps(result,sort_keys=True))
        elif stage=='score': result=score_all(); print(json.dumps(result,sort_keys=True))
        else: result=figures(); print(json.dumps(result,sort_keys=True))

if __name__=='__main__': main()
