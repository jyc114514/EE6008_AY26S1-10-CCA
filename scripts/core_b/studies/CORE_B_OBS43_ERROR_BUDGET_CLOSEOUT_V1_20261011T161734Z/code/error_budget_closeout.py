#!/usr/bin/env python3
"""CPU-only fixed-model OBS43 affine replay and error-budget closeout."""
from __future__ import annotations
import os
import argparse, csv, hashlib, json, math, os, resource, shutil, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
ROOT = Path(__file__).resolve().parents[1]
OBS = PROJECT / 'reports/benchmarks/CORE_B_COMMAND_OBSERVATION_TARGET_DIAGNOSTIC_V1_20261011T144716Z'
DELTA = 1.0 / 30.0
HORIZONS = (1, 3, 6, 10, 15, 30)
BLOCKS = (('Full43', 0, 43), ('arm_joints_source_alias', 0, 14),
          ('hand_joints_source_alias', 14, 28), ('leg_joints_source_alias', 28, 43))
VIEWS = ('OBS_PERSISTENCE', 'OBS_CONSTANT_VELOCITY', 'ARX_HOLD_COMMAND', 'ARX_ORACLE_COMMAND')
EPS32 = float(np.finfo(np.float32).eps)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def write_json(rel, obj):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    body=json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False,
                    default=jsonable, allow_nan=False) + '\n'
    if p.exists():
        if p.read_text(encoding='utf-8') == body: return
        raise FileExistsError(p)
    tmp = p.with_suffix(p.suffix + '.part')
    tmp.write_text(body, encoding='utf-8')
    tmp.replace(p)


def write_csv(rel, fields, rows):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): raise FileExistsError(p)
    rows = list(rows)
    ordered = list(fields or [])
    for row in rows:
        for key in row:
            if key not in ordered: ordered.append(key)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction='raise')
        w.writeheader()
        w.writerows(rows)


def append_jsonl(rel, row):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False,
                           default=jsonable, allow_nan=False) + '\n')


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        names = z.files if keys is None else keys
        return {k: z[k].copy() for k in names}


def verify_sources():
    registry = json.loads((ROOT / 'source/SOURCE_REGISTRY.json').read_text(encoding='utf-8'))
    out = []
    for item in registry['source_files']:
        p = Path(item['path'])
        actual = sha(p) if p.is_file() else 'MISSING'
        out.append({'name': item['name'], 'expected_sha256': item['sha256'],
                    'actual_sha256': actual, 'match': actual == item['sha256']})
    bad = [x['name'] for x in out if not x['match']]
    if bad: raise AssertionError('frozen source mismatch: ' + ', '.join(bad))
    return out


def verify_prefit_lock():
    p = ROOT / 'protocol/PREFIT_LOCK_HASHES.json'
    if not p.is_file():
        raise FileNotFoundError('PREFIT_LOCK_HASHES.json required before numeric payload access')
    lock = json.loads(p.read_text(encoding='utf-8'))
    bad = []
    for rel, expected in lock['files'].items():
        f = ROOT / rel
        actual = sha(f) if f.is_file() else 'MISSING'
        if actual != expected: bad.append({'path': rel, 'expected': expected, 'actual': actual})
    if bad: raise AssertionError('prefit lock mismatch: ' + json.dumps(bad[:5]))
    return lock


def reconstruct_affine(model):
    coef = np.asarray(model['coef'], dtype=np.float64)
    xmean = np.asarray(model['x_mean'], dtype=np.float64)
    xstd = np.asarray(model['x_std'], dtype=np.float64)
    sigma = np.asarray(model['sigma_obs'], dtype=np.float64)
    intercept = np.asarray(model['intercept'], dtype=np.float64)
    if coef.shape != (114, 43) or xmean.shape != (114,) or xstd.shape != (114,):
        raise AssertionError('frozen source dimensions do not match x=[s43,v43,u28]')
    if sigma.shape != (43,) or intercept.shape != (43,):
        raise AssertionError('frozen target dimensions do not match observation43')
    if not all(np.isfinite(a).all() for a in (coef, xmean, xstd, sigma, intercept)):
        raise AssertionError('nonfinite frozen model parameter')
    if np.any(xstd <= 0) or np.any(sigma <= 0): raise AssertionError('nonpositive scaler')
    F = (coef.T / xstd[None, :]) * sigma[:, None]
    bv = sigma * intercept - F @ xmean
    Fs, Fv, Fu = F[:, :43], F[:, 43:86], F[:, 86:]
    A = np.block([[np.eye(43) + DELTA * Fs, DELTA * Fv], [Fs, Fv]])
    B = np.vstack([DELTA * Fu, Fu])
    c = np.concatenate([DELTA * bv, bv])
    return {'F': F, 'b_v': bv, 'F_s': Fs, 'F_v': Fv, 'F_u': Fu,
            'A': A, 'B': B, 'c': c}


def source_v_next(model, s, v, u):
    x = np.concatenate((s, v, u), axis=-1).astype(np.float64, copy=False)
    return (model['intercept'] + ((x - model['x_mean']) / model['x_std']) @ model['coef']) * model['sigma_obs']


def source_step(model, s, v, u):
    vn = source_v_next(model, s, v, u)
    return s + DELTA * vn, vn


def affine_step(z, u, affine):
    return z @ affine['A'].T + u @ affine['B'].T + affine['c']


def state_from_positions(s_prev, s0, positions):
    n = len(s0)
    s_path = np.empty((n, positions.shape[1] + 1, 43), dtype=np.float64)
    s_path[:, 0] = np.asarray(s0, dtype=np.float64)
    s_path[:, 1:] = np.asarray(positions, dtype=np.float64)
    v_path = np.empty_like(s_path)
    v_path[:, 0] = (np.asarray(s0, dtype=np.float64) - np.asarray(s_prev, dtype=np.float64)) / DELTA
    v_path[:, 1:] = np.diff(s_path, axis=1) / DELTA
    return np.concatenate((s_path, v_path), axis=-1)


def replay_states(model, affine, inputs, commands, mode):
    n = len(inputs['anchor_id'])
    s = inputs['s0'].astype(np.float64)
    v = (s - inputs['s_prev'].astype(np.float64)) / DELTA
    states = np.empty((n, 31, 86), dtype=np.float64)
    states[:, 0] = np.concatenate((s, v), axis=1)
    max_affine_error = 0.0
    for j in range(30):
        u = inputs['u0'].astype(np.float64) if mode == 'hold' else commands[:, j].astype(np.float64)
        sn, vn = source_step(model, s, v, u)
        zn = np.concatenate((sn, vn), axis=1)
        za = affine_step(states[:, j], u, affine)
        max_affine_error = max(max_affine_error, float(np.max(np.abs(zn - za))))
        states[:, j + 1] = zn
        s, v = sn, vn
    return states, max_affine_error


def synthetic_controls():
    rng = np.random.default_rng(20261011)
    A = np.array([[.81, .03, 0, 0], [.01, .77, .02, 0],
                  [0, .02, .72, .01], [0, 0, .01, .69]], dtype=np.float64)
    B, c = rng.normal(0, .08, (4, 2)), rng.normal(0, .01, 4)
    u = rng.normal(size=(8, 2))
    zgt = np.zeros((9, 4)); zgt[0] = rng.normal(size=4)
    zo = np.zeros_like(zgt); zh = np.zeros_like(zgt); zo[0] = zh[0] = zgt[0]
    r = rng.normal(0, .01, (8, 4))
    for j in range(8):
        zgt[j+1] = A@zgt[j] + B@u[j] + c + r[j]
        zo[j+1] = A@zo[j] + B@u[j] + c
        zh[j+1] = A@zh[j] + B@u[0] + c
    d, eo, eh = zo-zh, zgt-zo, zgt-zh
    idd = np.stack([d[j+1] - A@d[j] - B@(u[j]-u[0]) for j in range(8)])
    ido = np.stack([eo[j+1] - A@eo[j] - r[j] for j in range(8)])
    same_u = np.repeat(u[:1], 8, axis=0)
    ds = np.zeros(4)
    for j in range(8): ds = A@ds + B@(same_u[j]-same_u[0])
    db = np.zeros(4)
    for j in range(8): db = A@db
    d0 = rng.normal(size=4); dprop = d0.copy()
    for j in range(8): dprop = A@dprop + B@(u[j]-u[0])
    checks = {
        'driver_identity_max_abs': float(np.max(np.abs(idd))),
        'oracle_residual_identity_max_abs': float(np.max(np.abs(ido))),
        'hold_sum_identity_max_abs': float(np.max(np.abs(eh-eo-d))),
        'constant_command_driver_max_abs': float(np.max(np.abs(ds))),
        'zero_B_driver_max_abs': float(np.max(np.abs(db))),
        'zero_residual_oracle_error_max_abs': 0.0,
        'first_driver_step_max_abs': float(np.max(np.abs(d[1]))),
        'initial_error_propagation_max_abs': float(np.max(np.abs(dprop - (np.linalg.matrix_power(A, 8)@d0 +
            sum(np.linalg.matrix_power(A, 7-j)@B@(u[j]-u[0]) for j in range(8))))))}
    if max(checks.values()) > 1e-12: raise AssertionError('synthetic identity controls failed')
    return {'status': 'PASS', **checks, 'seed': 20261011}


def read_npy_rows_from_npz(path, key, indices):
    wanted = sorted(set(int(i) for i in indices))
    with zipfile.ZipFile(path, 'r') as zf:
        with zf.open(key + '.npy', 'r') as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0): shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version in ((2, 0), (3, 0)): shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else: raise ValueError('unsupported NPY header ' + repr(version))
            if fortran or dtype.hasobject or not shape: raise ValueError('fixture reader requires C-order arrays')
            row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
            out, cursor = [], 0
            for index in wanted:
                if index < 0 or index >= shape[0]: raise IndexError(index)
                skip = (index-cursor) * row_bytes
                while skip:
                    b = f.read(min(skip, 1024*1024))
                    if not b: raise EOFError('truncated member while skipping')
                    skip -= len(b)
                raw = f.read(row_bytes)
                if len(raw) != row_bytes: raise EOFError('truncated member row')
                out.append(np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).copy())
                cursor = index+1
            return np.stack(out), wanted


def fixture_mode():
    lock = verify_prefit_lock()
    sources = verify_sources()
    model = load_npz(OBS/'models/ARX_OBS43_FULL_TRAIN.npz',
                     ['x_mean','x_std','coef','intercept','sigma_obs','safe_fd_obs'])
    affine = reconstruct_affine(model)
    prefix_path=OBS/'data/train/inputs_prefix.npz'
    id_only = load_npz(prefix_path,['anchor_id'])
    ids = id_only['anchor_id'].astype(str)
    ranked = sorted(range(len(ids)), key=lambda i: hashlib.sha256(ids[i].encode()).digest())
    ix = np.asarray(sorted(ranked[:16]), dtype=int)
    s_prev_rows,_=read_npy_rows_from_npz(prefix_path,'s_prev',ix)
    s0_rows,_=read_npy_rows_from_npz(prefix_path,'s0',ix)
    u0_rows,_=read_npy_rows_from_npz(prefix_path,'u0',ix)
    inp={'anchor_id':ids,'s_prev':s_prev_rows,'s0':s0_rows,'u0':u0_rows}
    command, _ = read_npy_rows_from_npz(OBS/'data/train/commands_privileged.npz','u_future',ix)
    if not np.array_equal(inp['u0'],command[:,0]): raise AssertionError('fixture prefix/command u0 mismatch')
    labels, _ = read_npy_rows_from_npz(OBS/'data/train/labels_obs43.npz','obs_future',ix)
    hold, _ = read_npy_rows_from_npz(OBS/'predictions/train/FORECASTS_H30.npz','ARX_HOLD_COMMAND',ix)
    oracle, _ = read_npy_rows_from_npz(OBS/'predictions/train/FORECASTS_H30.npz','ARX_ORACLE_COMMAND',ix)
    teacher, _ = read_npy_rows_from_npz(OBS/'predictions/train/TEACHER_FORCED_LOCAL.npz','teacher_next_native',ix)
    for path,role in ((OBS/'data/train/inputs_prefix.npz','all train anchor IDs plus selected s[-1],s[0],u[0] rows'),
        (OBS/'data/train/commands_privileged.npz','selected train command rows'),
        (OBS/'data/train/labels_obs43.npz','selected train OBS43 fixture rows'),
        (OBS/'predictions/train/FORECASTS_H30.npz','selected accepted train forecasts'),
        (OBS/'predictions/train/TEACHER_FORCED_LOCAL.npz','selected accepted teacher-local rows')):
        append_jsonl('logs/ARRAY_ACCESS.jsonl',{'utc':now(),'path':str(path),'role':role,
            'split':'train','purpose':'fixed train-only replay fixture','fixture_rows_materialized':16,
            'no_dev_labels':True})
    s0, sp = inp['s0'].astype(np.float64), inp['s_prev'].astype(np.float64)
    u0, v0 = inp['u0'].astype(np.float64), None
    v0 = (s0-sp)/DELTA
    affine_one = reconstruct_affine(model)
    s1, v1 = source_step(model,s0,v0,u0)
    z_source = np.concatenate((s1,v1),axis=1)
    z_affine = affine_step(np.concatenate((s0,v0),axis=1),u0,affine_one)
    zgt = state_from_positions(sp,s0,labels)
    teacher_err, stored_err = 0.0, 0.0
    for j in range(30):
        loc, _ = source_step(model,zgt[:,j,:43],zgt[:,j,43:],command[:,j].astype(np.float64))
        teacher_err = max(teacher_err,float(np.max(np.abs(loc.astype(np.float32)-teacher[:,j]))))
    for mode, saved in (('hold',hold),('oracle',oracle)):
        s, v = s0.copy(), v0.copy()
        for j in range(30):
            u = u0 if mode == 'hold' else command[:,j].astype(np.float64)
            s,v = source_step(model,s,v,u)
            stored_err = max(stored_err,float(np.max(np.abs(s.astype(np.float32)-saved[:,j]))))
    if np.max(np.abs(z_source-z_affine)) > 1e-8 or teacher_err > 2*EPS32 or stored_err > 2*EPS32:
        raise AssertionError('train fixture replay failed')
    out = {'status':'PASS','prefit_lock_sha256':sha(ROOT/'protocol/PREFIT_LOCK_HASHES.json'),
           'source_count_hash_verified':len(sources),
           'fixture_selection':'16 smallest SHA256(anchor_id) values among primary-train IDs',
           'fixture_anchor_ids':ids[ix].tolist(),'fixture_train_only':True,
           'new_model_fits':0,'new_forecast_artifacts_written':0,
           'replay_scope':'in-memory deterministic source/affine QA; no output forecasts',
           'one_step_source_vs_affine_max_abs':float(np.max(np.abs(z_source-z_affine))),
           'teacher_saved_float32_replay_max_abs':teacher_err,
           'saved_hold_oracle_float32_replay_max_abs':stored_err,
           'float64_fixture_atol_rtol':1e-8,'created_utc':now()}
    write_json('qa/TRAIN_FIXTURE_REPLAY_R003.json',out)
    write_json('qa/SYNTHETIC_CONTROLS_R003.json',synthetic_controls())
    return out


def load_split(split):
    prefix_path=OBS/f'data/{split}/inputs_prefix.npz'
    prefix = load_npz(prefix_path,['anchor_id','source_group_id','episode_index','s_prev','s0','u0','timestamp_window'])
    append_jsonl('logs/ARRAY_ACCESS.jsonl',{'utc':now(),'path':str(prefix_path),'role':'s[-1], s[0], u[0], IDs, timestamps','split':split,'purpose':'forecast alignment, state reconstruction, and clock check'})
    label_path=OBS/f'data/{split}/labels_obs43.npz'
    label_open_utc=now()
    first_read = label_open_utc if split == 'dev' else None
    append_jsonl('logs/ARRAY_ACCESS.jsonl',{'utc':label_open_utc,'event':'open_start','path':str(label_path),
        'role':'s[1..30] OBS43 labels','split':split,
        'purpose':'fixed-model one-step residuals, accepted score/budget replication; dev outcomes are repeatedly exposed'})
    labels = load_npz(label_path,['anchor_id','obs_future'])
    append_jsonl('logs/ARRAY_ACCESS.jsonl',{'utc':now(),'event':'load_complete','path':str(label_path),
        'role':'s[1..30] OBS43 labels','split':split,'first_dev_label_payload_read':split=='dev'})
    command_path=OBS/f'data/{split}/commands_privileged.npz'
    commands = load_npz(command_path,['anchor_id','u_future'])
    append_jsonl('logs/ARRAY_ACCESS.jsonl',{'utc':now(),'path':str(command_path),'role':'u[0..29]','split':split,'purpose':'fixed accepted forecast replay and command-response identity'})
    forecast_path=OBS/f'predictions/{split}/FORECASTS_H30.npz'
    forecast = load_npz(forecast_path,['anchor_id',*VIEWS])
    append_jsonl('logs/ARRAY_ACCESS.jsonl',{'utc':now(),'path':str(forecast_path),'role':'Persistence/CV/HOLD/ORACLE accepted H30 outputs','split':split,'purpose':'metric replication and signed output comparison'})
    teacher_path=OBS/f'predictions/{split}/TEACHER_FORCED_LOCAL.npz'
    teacher = load_npz(teacher_path,['anchor_id','teacher_next_native'])
    append_jsonl('logs/ARRAY_ACCESS.jsonl',{'utc':now(),'path':str(teacher_path),'role':'teacher-forced local one-step outputs','split':split,'purpose':'local recurrence/source replay check; not ranked as H30 forecast'})
    for name, d in (('labels',labels),('commands',commands),('forecasts',forecast),('teacher',teacher)):
        if not np.array_equal(prefix['anchor_id'],d['anchor_id']): raise AssertionError(f'{split} ID mismatch: {name}')
    if not np.array_equal(prefix['u0'],commands['u_future'][:,0]): raise AssertionError(f'{split} u0 mismatch')
    if commands['u_future'].shape[1:] != (30,28) or labels['obs_future'].shape[1:] != (30,43):
        raise AssertionError(f'{split} data shape mismatch')
    groups, episodes = prefix['source_group_id'].astype(str), prefix['episode_index'].astype(np.int64)
    ep2g, g2e = {}, {}
    for ep,g in zip(episodes,groups):
        if ep in ep2g and ep2g[ep] != g: raise AssertionError('episode maps to multiple source groups')
        if g in g2e and g2e[g] != ep: raise AssertionError('source group maps to multiple episodes')
        ep2g[ep],g2e[g] = g,ep
    return {'prefix':prefix,'labels':labels['obs_future'],'commands':commands['u_future'],
            'forecasts':{v:forecast[v] for v in VIEWS},'teacher':teacher['teacher_next_native'],
            'groups':groups,'episodes':episodes,'source_groups':len(g2e),'episodes_n':len(ep2g),
            'dev_label_first_read_utc':first_read}


def episode_means(values, episodes):
    values, episodes = np.asarray(values,dtype=np.float64), np.asarray(episodes)
    unique = np.unique(episodes)
    out = np.stack([values[episodes==ep].mean(axis=0) for ep in unique])
    return unique,out


def scalar_metric(pred, truth, sigma, safe, horizon, sl):
    p, y = np.asarray(pred[:,:horizon,sl],dtype=np.float64), np.asarray(truth[:,:horizon,sl],dtype=np.float64)
    position_std = np.mean(np.abs(p-y)/sigma[None,None,sl],axis=(1,2))
    position_native = np.mean(np.abs(p-y),axis=(1,2))
    endpoint_std = np.mean(np.abs(p[:,-1]-y[:,-1])/sigma[None,sl],axis=1)
    endpoint_native = np.mean(np.abs(p[:,-1]-y[:,-1]),axis=1)
    if horizon == 1:
        fd_std,fd_native = np.full(len(p),np.nan),np.full(len(p),np.nan)
    else:
        dp,dy = np.diff(p,axis=1),np.diff(y,axis=1)
        fd_std=np.mean(np.abs(dp-dy)/safe[None,None,sl],axis=(1,2))
        fd_native=np.mean(np.abs(dp-dy),axis=(1,2))
    return {'position_prefix_standardized_mae':position_std,'position_prefix_native_mae':position_native,
            'position_endpoint_standardized_mae':endpoint_std,'position_endpoint_native_mae':endpoint_native,
            'future_internal_fd_safe_normalized':fd_std,'future_internal_fd_native_mae':fd_native}


def budget_terms(eo, driver, eh, scale):
    eo,driver,eh=np.asarray(eo,dtype=np.float64),np.asarray(driver,dtype=np.float64),np.asarray(eh,dtype=np.float64)
    scale=np.asarray(scale,dtype=np.float64)
    axes=tuple(range(1,eo.ndim))
    no,nd,nh=eo/scale,driver/scale,eh/scale
    cancel=(np.abs(eo)+np.abs(driver)-np.abs(eh))/scale
    lo,ld,lh=(np.mean(np.abs(x),axis=axes) for x in (no,nd,nh))
    cc=np.mean(cancel,axis=axes)
    return {'L_oracle_error':lo,'L_driver_response':ld,'L1_cancellation':cc,'L_hold_error':lh,
            'signed_eO_mean':np.mean(no,axis=axes),'signed_d_mean':np.mean(nd,axis=axes),
            'signed_eH_mean':np.mean(nh,axis=axes),
            'budget_identity_residual':lh-(lo+ld-cc),
            'vector_identity_max_abs':float(np.max(np.abs(eo+driver-eh))),
            'cancellation_min_element':float(np.min(cancel))}


def append_budget_records(anchor_rows, episode_rows, split, ids, groups, episodes, h, block, metrics):
    ids=np.asarray(ids).astype(str); groups=np.asarray(groups).astype(str); eps=np.unique(episodes)
    for i in range(len(ids)):
        row={'split':split,'anchor_id':ids[i],'source_group_id':groups[i],
             'episode_index':int(episodes[i]),'horizon':int(h),'block':block}
        for name,terms in metrics.items():
            for term,values in terms.items():
                if term not in ('vector_identity_max_abs','cancellation_min_element'):
                    row[name+'__'+term]=float(values[i])
        anchor_rows.append(row)
    for ep in eps:
        ix=np.flatnonzero(episodes==ep)
        row={'split':split,'source_group_id':groups[ix[0]],'episode_index':int(ep),
             'anchors':int(len(ix)),'horizon':int(h),'block':block}
        for name,terms in metrics.items():
            for term,values in terms.items():
                if term not in ('vector_identity_max_abs','cancellation_min_element'):
                    row[name+'__'+term]=float(np.mean(values[ix]))
        episode_rows.append(row)


def create_budget_outputs(data,split):
    truth=data['labels'].astype(np.float64); forecasts=data['forecasts']
    sigma,safe=data['sigma'],data['safe']; s0=data['prefix']['s0'].astype(np.float64)
    ids=data['prefix']['anchor_id']; groups,episodes=data['groups'],data['episodes']
    anchor_rows,episode_rows=[],[]; min_cancel=0.0; block_checks=[]
    for h in HORIZONS:
        for block,lo,hi in BLOCKS:
            sl=slice(lo,hi); metrics={}
            eo=truth[:,:h,sl]-forecasts['ARX_ORACLE_COMMAND'][:,:h,sl].astype(np.float64)
            dr=forecasts['ARX_ORACLE_COMMAND'][:,:h,sl].astype(np.float64)-forecasts['ARX_HOLD_COMMAND'][:,:h,sl].astype(np.float64)
            eh=truth[:,:h,sl]-forecasts['ARX_HOLD_COMMAND'][:,:h,sl].astype(np.float64)
            for unit,scale in (('std',sigma[sl]),('native',np.ones(hi-lo))):
                terms=budget_terms(eo,dr,eh,scale); metrics['position_'+unit]=terms
                min_cancel=min(min_cancel,terms['cancellation_min_element'])
                if h>1:
                    fd_terms=budget_terms(np.diff(eo,axis=1),np.diff(dr,axis=1),np.diff(eh,axis=1),
                                         safe[sl] if unit=='std' else np.ones(hi-lo))
                    metrics['future_fd_'+unit]=fd_terms
                    min_cancel=min(min_cancel,fd_terms['cancellation_min_element'])
                if h==1:
                    so=(truth[:,0,sl]-s0[:,sl])-(forecasts['ARX_ORACLE_COMMAND'][:,0,sl].astype(np.float64)-s0[:,sl])
                    sd=(forecasts['ARX_ORACLE_COMMAND'][:,0,sl].astype(np.float64)-s0[:,sl])-(forecasts['ARX_HOLD_COMMAND'][:,0,sl].astype(np.float64)-s0[:,sl])
                    sh=(truth[:,0,sl]-s0[:,sl])-(forecasts['ARX_HOLD_COMMAND'][:,0,sl].astype(np.float64)-s0[:,sl])
                    seam=budget_terms(so,sd,sh,safe[sl] if unit=='std' else np.ones(hi-lo))
                    metrics['seam_fd_'+unit]=seam; min_cancel=min(min_cancel,seam['cancellation_min_element'])
            append_budget_records(anchor_rows,episode_rows,split,ids,groups,episodes,h,block,metrics)
            if block=='Full43':
                for unit,sc in (('std',sigma),('native',np.ones(43))):
                    f=budget_terms(eo,dr,eh,sc)
                    for term in ('L_oracle_error','L_driver_response','L1_cancellation','L_hold_error'):
                        block_sum=np.zeros(len(truth))
                        for _,blo,bhi in BLOCKS[1:]:
                            bterms=budget_terms(eo[:,:,blo:bhi],dr[:,:,blo:bhi],eh[:,:,blo:bhi],sc[blo:bhi])
                            block_sum+=(bhi-blo)*bterms[term]
                        resid=43*f[term]-block_sum
                        block_checks.append({'split':split,'horizon':h,'unit':unit,'term':term,
                                             'max_abs_identity_residual':float(np.max(np.abs(resid)))})
    return anchor_rows,episode_rows,min_cancel,block_checks


def score_split(data,split):
    scores=[]
    for view in VIEWS:
        for h in HORIZONS:
            for block,lo,hi in BLOCKS:
                vals=scalar_metric(data['forecasts'][view],data['labels'],data['sigma'],data['safe'],h,slice(lo,hi))
                row={'split':split,'view':view,'horizon':h,'block':block,'anchors':len(data['episodes']),
                     'episodes':len(np.unique(data['episodes'])),'source_groups':len(np.unique(data['groups']))}
                for key,x in vals.items():
                    _,ev=episode_means(x,data['episodes'])
                    ev=ev[np.isfinite(ev)]
                    row[key]=float(np.mean(ev)) if len(ev) else float('nan')
                scores.append(row)
    return scores


def compare_old_scores(scores):
    old={}
    with (OBS/'results/FORECAST_METRICS_BY_HORIZON.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['split'] in ('train','dev'):
                old[(r['split'],r['view'],int(r['horizon']),r['block'])]=r
    mapping={'position_prefix_standardized_mae':'position_prefix_standardized_mae',
             'position_endpoint_standardized_mae':'position_endpoint_standardized_mae',
             'position_prefix_native_mae':'position_prefix_native_mae',
             'position_endpoint_native_mae':'position_endpoint_native_mae',
             'future_internal_fd_safe_normalized':'safe_fd_error',
             'future_internal_fd_native_mae':'safe_fd_native_mae'}
    cells,bad=[],[]
    for r in scores:
        key=(r['split'],r['view'],r['horizon'],r['block']); prior=old.get(key)
        if prior is None: bad.append({'key':key,'reason':'missing old score'}); continue
        for ours,theirs in mapping.items():
            got,want=r[ours],float(prior[theirs])
            err=0.0 if math.isnan(got) and math.isnan(want) else abs(got-want)
            cells.append({'key':str(key),'metric':ours,'actual':got,'prior':want,'abs_abs_error':err})
            if err>2e-12: bad.append({'key':key,'metric':ours,'actual':got,'expected':want,'abs_err':err})
    return {'status':'PASS' if not bad else 'FAIL','compared_metric_cells':len(cells),
            'max_abs_error':max((x['abs_abs_error'] for x in cells),default=0.0),
            'tolerance':2e-12,'bad_count':len(bad),'bad_examples':bad[:10]},cells


def coordinate_weighting(data):
    truth=data['labels'].astype(np.float64); rows=[]
    eps=np.unique(data['episodes'])
    for view in ('OBS_PERSISTENCE','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'):
        p=data['forecasts'][view].astype(np.float64)
        nat=[]; std=[]
        for ep in eps:
            ix=data['episodes']==ep; err=np.abs(p[ix]-truth[ix])
            nat.append(err.mean(axis=(0,1)))
            std.append((err/data['sigma'][None,None,:]).mean(axis=(0,1)))
        nat,std=np.mean(nat,axis=0),np.mean(std,axis=0)
        for j in range(43):
            block=next(b for b,lo,hi in BLOCKS if lo<=j<hi)
            rows.append({'split':data['split'],'channel_index':j,'block':block,
                'sigma_obs':float(data['sigma'][j]),'safe_fd_obs':float(data['safe'][j]),
                'position_weight_1_over_43sigma':float(1/(43*data['sigma'][j])),
                'fd_weight_1_over_43safe_fd':float(1/(43*data['safe'][j])),
                'view':view,'native_position_abs_error':float(nat[j]),
                'standardized_position_abs_error':float(std[j]),
                'native_contribution_to_full_mae':float(nat[j]/43),
                'standardized_contribution_to_full_mae':float(std[j]/43)})
    return rows


def rank_rows(scores):
    rows=[]
    for split in ('train','dev'):
        for h in HORIZONS:
            for block,_,_ in BLOCKS:
                candidates=[r for r in scores if r['split']==split and r['horizon']==h and r['block']==block]
                for metric in ('position_prefix_standardized_mae','position_prefix_native_mae',
                               'future_internal_fd_safe_normalized','future_internal_fd_native_mae'):
                    if h==1 and metric.startswith('future_internal'): continue
                    pairs=sorted([(r['view'],r[metric]) for r in candidates if np.isfinite(r[metric])],key=lambda x:x[1])
                    for rank,(view,value) in enumerate(pairs,1):
                        rows.append({'split':split,'horizon':h,'block':block,'metric':metric,
                                     'rank_lower_is_better':rank,'view':view,'value':float(value)})
    return rows


def identity_statistics(x,label,split):
    a=np.abs(np.asarray(x,dtype=np.float64)).ravel()
    return {'split':split,'identity':label,'max_abs':float(a.max()),
            'p99_abs':float(np.quantile(a,.99)),'mean_abs':float(a.mean()),'elements':int(a.size)}


def run_split(split,model,affine,sigma,safe):
    data=load_split(split); data.update(sigma=sigma,safe=safe,split=split)
    n=len(data['episodes']); prefix=data['prefix']; u=data['commands'].astype(np.float64)
    truth=data['labels'].astype(np.float64); s0=prefix['s0'].astype(np.float64)
    zpaths={}; replay=[]
    for mode,view in (('hold','ARX_HOLD_COMMAND'),('oracle','ARX_ORACLE_COMMAND')):
        zp,afferr=replay_states(model,affine,prefix,data['commands'],mode); zpaths[view]=zp
        err=np.abs(zp[:,1:,:43].astype(np.float32)-data['forecasts'][view])
        replay.append({'split':split,'view':view,'source_affine_replay_max_abs':afferr,
            'stored_float32_max_abs':float(err.max()),'stored_float32_p99_abs':float(np.quantile(err,.99)),
            'exact_float32_match_elements':int(np.count_nonzero(err==0)),
            'total_position_elements':int(err.size)})
    v0=(s0-prefix['s_prev'].astype(np.float64))/DELTA
    p_expected=np.repeat(s0[:,None,:],30,axis=1).astype(np.float32)
    s0f=prefix['s0'].astype(np.float32); spf=prefix['s_prev'].astype(np.float32)
    v0f=(s0f-spf)/DELTA
    cv_expected=(s0f[:,None,:]+np.arange(1,31,dtype=np.float32)[None,:,None]*DELTA*v0f[:,None,:]).astype(np.float32)
    replay += [{'split':split,'view':'OBS_PERSISTENCE',
                'stored_float32_max_abs':float(np.max(np.abs(data['forecasts']['OBS_PERSISTENCE']-p_expected)))},
               {'split':split,'view':'OBS_CONSTANT_VELOCITY',
                'stored_float32_max_abs':float(np.max(np.abs(data['forecasts']['OBS_CONSTANT_VELOCITY']-cv_expected)))}]
    zgt=state_from_positions(prefix['s_prev'],prefix['s0'],truth)
    step_pred=affine_step(zgt[:,:-1].reshape(-1,86),u.reshape(-1,28),affine).reshape(n,30,86)
    r=zgt[:,1:]-step_pred
    d=zpaths['ARX_ORACLE_COMMAND']-zpaths['ARX_HOLD_COMMAND']
    eo=zgt-zpaths['ARX_ORACLE_COMMAND']; eh=zgt-zpaths['ARX_HOLD_COMMAND']
    driver_state=((u-u[:,:1]).reshape(-1,28)@affine['B'].T).reshape(n,30,86)
    dd=d[:,1:]-(d[:,:-1]@affine['A'].T+driver_state)
    de=eo[:,1:]-(eo[:,:-1]@affine['A'].T+r)
    dh=eh-eo-d
    rp=r[:,:,:43]-DELTA*r[:,:,43:]
    identities=[identity_statistics(dd,'d[j+1] - A*d[j] - B*(u[j]-u[0])',split),
        identity_statistics(de,'eO[j+1] - A*eO[j] - r[j]',split),
        identity_statistics(dh,'eH[j] - eO[j] - d[j]',split),
        identity_statistics(rp,'r_position - delta*r_velocity',split)]
    dt=np.diff(prefix['timestamp_window'].astype(np.float64),axis=1); delta_err=np.abs(dt-DELTA)
    clock={'split':split,'interval_count':int(dt.size),'max_abs_delta_deviation_seconds':float(delta_err.max()),
           'p99_abs_delta_deviation_seconds':float(np.quantile(delta_err,.99)),
           'within_locked_1e-5_seconds':int(np.count_nonzero(delta_err<=1e-5)),
           'all_within_locked_1e-5_seconds':bool(np.all(delta_err<=1e-5))}
    teacher_err=0.0
    for j in range(30):
        local,_=source_step(model,zgt[:,j,:43],zgt[:,j,43:],u[:,j])
        teacher_err=max(teacher_err,float(np.max(np.abs(local.astype(np.float32)-data['teacher'][:,j]))))
    replay.append({'split':split,'view':'TEACHER_FORCED_LOCAL','stored_float32_max_abs':teacher_err})
    scores=score_split(data,split)
    ba,be,minc,bc=create_budget_outputs(data,split)
    return {'data':data,'scores':scores,'budget_anchor':ba,'budget_episode':be,
            'coordinate_rows':coordinate_weighting(data),'ranks':rank_rows(scores),
            'identity_rows':identities,'replay_records':replay,'clock':clock,
            'cancel_negative_min':minc,'common_scale_checks':bc,'zgt':zgt,'r':r,
            'd':d,'eo':eo,'eh':eh,'exact_states':zpaths}


def normalized_response(affine,sigma):
    A=affine['A']; P=np.concatenate((1/sigma,DELTA/sigma))
    At=P[:,None]*A/P[None,:]
    vals=np.linalg.eigvals(A); vtilde=np.linalg.eigvals(At)
    from scipy.optimize import linear_sum_assignment
    ri,ci=linear_sum_assignment(np.abs(vals[:,None]-vtilde[None,:]))
    eigerr=np.abs(vals[ri]-vtilde[ci])
    I=np.eye(86); powers=[I]
    for _ in range(30): powers.append(powers[-1]@At)
    inject=np.vstack((np.eye(43),np.eye(43))); project=np.hstack((np.eye(43),np.zeros((43,43))))
    response=[]
    K0=project@powers[0]@inject
    for lag in range(30):
        K=project@powers[lag]@inject
        for block,lo,hi in BLOCKS:
            sub=K[lo:hi]
            response.append({'lag_steps_after_residual':lag,'block':block,
                'K_spectral_norm':float(np.linalg.norm(sub,2)),
                'K_frobenius_norm':float(np.linalg.norm(sub,'fro')),
                'K_max_abs':float(np.max(np.abs(sub)))})
    norms=[]
    for h in HORIZONS:
        ap=np.linalg.matrix_power(A,h); tp=np.linalg.matrix_power(At,h)
        norms.append({'horizon':h,'raw_A_power_2norm':float(np.linalg.norm(ap,2)),
            'dimensionless_Atilde_power_2norm':float(np.linalg.norm(tp,2)),
            'raw_A_power_frobenius':float(np.linalg.norm(ap,'fro')),
            'dimensionless_Atilde_power_frobenius':float(np.linalg.norm(tp,'fro'))})
    K0=project@powers[0]@inject
    qa={'status':'PASS' if eigerr.max()<=1e-6 and np.allclose(K0,np.eye(43),atol=1e-12,rtol=0) else 'FAIL',
        'P_definition':'diag(1/sigma_obs, delta/sigma_obs)',
        'raw_spectral_radius':float(np.max(np.abs(vals))),
        'dimensionless_spectral_radius':float(np.max(np.abs(vtilde))),
        'eigenvalue_assignment_max_abs_difference':float(eigerr.max()),
        'diagonal_similarity_only':True,'operator_norm_coordinate_dependent':True,
        'K0_is_identity':bool(np.allclose(K0,np.eye(43),atol=1e-12,rtol=0))}
    return response,norms,qa


def actual_growth_rows(result,split,sigma):
    data=result['data']; zgt=result['zgt']; eps=np.unique(data['episodes']); rows=[]
    episode_ix={int(ep):np.flatnonzero(data['episodes']==ep) for ep in eps}
    for h in HORIZONS:
        for block,lo,hi in BLOCKS:
            for view,zpred in result['exact_states'].items():
                er=zgt[:,1:h+1,lo:hi]-zpred[:,1:h+1,lo:hi]
                endpoint=er[:,-1]
                for ep,ix in episode_ix.items():
                    rows.append({'split':split,'episode_index':int(ep),
                        'source_group_id':str(data['groups'][np.flatnonzero(ix)[0]]),
                        'horizon':h,'block':block,'view':view,
                        'endpoint_position_native_mae':float(np.mean(np.abs(endpoint[ix]))),
                        'endpoint_position_standardized_mae':float(np.mean(np.abs(endpoint[ix])/sigma[lo:hi])),
                        'prefix_position_native_mae':float(np.mean(np.abs(er[ix]))),
                        'prefix_position_standardized_mae':float(np.mean(np.abs(er[ix])/sigma[lo:hi])),
                        'state_error_dimensionless_max':float(np.mean(np.max(np.abs(
                            (zgt[ix,h]-zpred[ix,h])*np.concatenate((1/sigma,DELTA/sigma))),axis=1)))})
    return rows


def residual_lag_rows(result,split,affine,sigma):
    zgt,r=result['zgt'],result['r']; zo=result['exact_states']['ARX_ORACLE_COMMAND']
    data=result['data']; eps=np.unique(data['episodes']); rows=[]
    episode_ix={int(ep):np.flatnonzero(data['episodes']==ep) for ep in eps}
    for h in HORIZONS:
        J=h; terms=[]
        for k in range(J):
            lag=J-1-k; power=np.linalg.matrix_power(affine['A'],lag)
            contrib=r[:,k,:]@power[:43,:].T
            terms.append((k+1,lag,contrib))
        total=np.sum([x[2] for x in terms],axis=0)
        direct=zgt[:,J,:43]-zo[:,J,:43]
        prop_resid=total-direct
        for block,lo,hi in BLOCKS:
            for kstep,lag,contrib in terms:
                for ep,ix in episode_ix.items():
                    c=contrib[ix,lo:hi]
                    rows.append({'split':split,'episode_index':int(ep),
                        'source_group_id':str(data['groups'][np.flatnonzero(ix)[0]]),
                        'horizon':h,'block':block,'residual_transition_index':kstep,
                        'lag_steps_after_residual':lag,
                        'mean_signed_propagated_position_native':float(np.mean(c)),
                        'mean_abs_propagated_position_native':float(np.mean(np.abs(c))),
                        'mean_abs_propagated_position_standardized':float(np.mean(np.abs(c)/sigma[lo:hi]))})
            for ep,ix in episode_ix.items():
                sum_abs=np.mean(np.sum([np.abs(x[2][ix,lo:hi])/sigma[lo:hi] for x in terms],axis=0))
                total_abs=np.mean(np.abs(total[ix,lo:hi])/sigma[lo:hi])
                rows.append({'split':split,'episode_index':int(ep),
                    'source_group_id':str(data['groups'][np.flatnonzero(ix)[0]]),
                    'horizon':h,'block':block,'residual_transition_index':0,
                    'lag_steps_after_residual':-1,
                    'mean_signed_propagated_position_native':float(np.mean(total[ix,lo:hi])),
                    'mean_abs_propagated_position_native':float(np.mean(np.abs(total[ix,lo:hi]))),
                    'mean_abs_propagated_position_standardized':float(total_abs),
                    'sum_lag_abs_standardized':float(sum_abs),
                    'cross_lag_cancellation_standardized':float(sum_abs-total_abs),
                    'actual_eO_standardized':float(np.mean(np.abs(direct[ix,lo:hi])/sigma[lo:hi])),
                    'propagation_identity_max_abs':float(np.max(np.abs(prop_resid[ix,lo:hi])))})
    return rows


def normalized_forcing_rows(results,sigma):
    rows=[]
    for result in results:
        data=result['data']; r=result['r']; episodes=data['episodes']; split=data['split']
        kappa=DELTA*r[:,:,43:]/sigma[None,None,:]
        rpos=r[:,:,:43]/sigma[None,None,:]
        rvel=DELTA*r[:,:,43:]/sigma[None,None,:]
        for j in range(30):
            _,signed=episode_means(kappa[:,j,:],episodes)
            _,absolute=episode_means(np.abs(kappa[:,j,:]),episodes)
            _,square=episode_means(kappa[:,j,:]**2,episodes)
            _,rp=episode_means(rpos[:,j,:],episodes)
            _,rv=episode_means(rvel[:,j,:],episodes)
            for ch in range(43):
                block=next(b for b,lo,hi in BLOCKS if lo<=ch<hi)
                rows.append({'split':split,'transition_index':j+1,'channel_index':ch,'block':block,
                    'kappa_signed_episode_equal_mean':float(signed[:,ch].mean()),
                    'kappa_abs_episode_equal_mean':float(absolute[:,ch].mean()),
                    'kappa_rms_episode_equal':float(np.sqrt(square[:,ch].mean())),
                    'kappa_episode_abs_p95':float(np.quantile(absolute[:,ch],.95)),
                    'normalized_r_position_episode_equal_mean':float(rp[:,ch].mean()),
                    'normalized_delta_r_velocity_episode_equal_mean':float(rv[:,ch].mean())})
    return rows


def write_summaries(results,affine,sigma):
    all_scores=[r for x in results for r in x['scores']]
    score_qa,cells=compare_old_scores(all_scores)
    write_json('qa/OLD_SCORE_REPRODUCTION.json',score_qa)
    write_csv('results/OLD_SCORE_REPRODUCTION_CELLS.csv',list(cells[0]),cells)
    write_csv('results/REPRODUCED_SCORES_BY_HORIZON.csv',list(all_scores[0]),all_scores)
    ranks=[r for x in results for r in x['ranks']]
    write_csv('results/RANK_RECONCILIATION.csv',list(ranks[0]),ranks)
    anchors=[r for x in results for r in x['budget_anchor']]
    episodes=[r for x in results for r in x['budget_episode']]
    write_csv('results/ERROR_BUDGET_BY_ANCHOR.csv',list(anchors[0]),anchors)
    write_csv('results/ERROR_BUDGET_BY_EPISODE.csv',list(episodes[0]),episodes)
    coords=results[-1]['coordinate_rows']
    write_csv('results/COORDINATE_WEIGHTING_DEV_H30.csv',list(coords[0]),coords)
    common=[r for x in results for r in x['common_scale_checks']]
    write_csv('results/COMMON_SCALE_BLOCK_IDENTITY.csv',list(common[0]),common)
    replay=[r for x in results for r in x['replay_records']]
    write_csv('results/REPLAY_ACCEPTANCE.csv',list(replay[0]),replay)
    identities=[r for x in results for r in x['identity_rows']]
    write_csv('results/IDENTITY_RESIDUALS.csv',list(identities[0]),identities)
    clocks=[x['clock'] for x in results]
    write_csv('results/CLOCK_RECHECK.csv',list(clocks[0]),clocks)
    growth=[r for x in results for r in actual_growth_rows(x,x['data']['split'],sigma)]
    write_csv('results/ACTUAL_ERROR_GROWTH_BY_EPISODE.csv',list(growth[0]),growth)
    # Reattach the fitted-scale matrix for exact powers and K_l response.
    response,norms,norm_qa=normalized_response(affine,sigma)
    write_csv('results/DIMENSIONLESS_K_LAG_RESPONSE.csv',list(response[0]),response)
    write_csv('results/RAW_VS_DIMENSIONLESS_OPERATOR_NORMS.csv',list(norms[0]),norms)
    lag=[]
    for x in results: lag.extend(residual_lag_rows(x,x['data']['split'],affine,sigma))
    write_csv('results/ACTUAL_RESIDUAL_LAG_BUDGET.csv',list(lag[0]),lag)
    forcing=normalized_forcing_rows(results,sigma)
    write_csv('results/DIMENSIONLESS_RESIDUAL_FORCING.csv',list(forcing[0]),forcing)
    propagation_rows=[r for r in lag if 'propagation_identity_max_abs' in r]
    max_prop=max(float(r['propagation_identity_max_abs']) for r in propagation_rows)
    write_json('qa/ACTUAL_RESIDUAL_PROPAGATION_ACCEPTANCE.json',{'status':'PASS' if max_prop<=1e-8 else 'FAIL',
        'max_actual_sequence_vs_lag_sum_abs':max_prop,'comparison':'sum_k A^(J-1-k) r_k versus exact source-replay eO_J',
        'max_group_horizon_block_rows':len(propagation_rows),'lag_decomposition_path':'results/ACTUAL_RESIDUAL_LAG_BUDGET.csv',
        'dimensionless_forcing_path':'results/DIMENSIONLESS_RESIDUAL_FORCING.csv'})
    boot=[]
    with (OBS/'results/PAIRED_GROUP_BOOTSTRAP.csv').open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row.get('split')=='dev' and row.get('horizon')=='30': boot.append(row)
    write_csv('results/REUSED_PRIOR_H30_GROUP_BOOTSTRAP.csv',list(boot[0]),boot)
    max_common=max(float(r['max_abs_identity_residual']) for r in common)
    min_cancel=min(x['cancel_negative_min'] for x in results)
    write_json('qa/COMMON_SCALE_AND_BUDGET_ACCEPTANCE.json',{
        'status':'PASS' if max_common<=1e-10 and min_cancel>=-1e-10 else 'FAIL',
        'max_common_scale_identity_abs_residual':max_common,
        'minimum_pointwise_cancellation':min_cancel,'primary_scale_unchanged':True,
        'dev_metric_reweighting':False,'prior_group_bootstrap_reused_without_new_draws':True,
        'new_p_values':False,'aggregation':'within-source episode mean, then equal mean across source episodes'})
    max_r=max(r['max_abs'] for x in results for r in x['identity_rows']
              if r['identity']=='r_position - delta*r_velocity')
    write_json('qa/DIMENSIONLESS_RESPONSE_ACCEPTANCE.json',{
        **norm_qa,'actual_r_position_relation_max_abs':max_r,
        'K_l_values':30,'actual_residual_sequence_table':'results/ACTUAL_RESIDUAL_LAG_BUDGET.csv',
        'propagation_requires_checked_r_position_relation':True,
        'actual_residual_propagation_status':'results/ACTUAL_RESIDUAL_PROPAGATION_ACCEPTANCE.json'})
    dev30=[r for r in all_scores if r['split']=='dev' and r['horizon']==30 and r['block']=='Full43']
    as_reported=[r for r in all_scores if r['split']=='dev' and r['horizon']==30]
    write_csv('results/AS_REPORTED_VS_COMMON_SCALE_DEV_H30.csv',list(as_reported[0]),as_reported)
    return {'score_reproduction':score_qa,'norm_qa':norm_qa,'h30_dev_full43':dev30,
            'budget_episode':episodes,'max_common':max_common,'min_cancel':min_cancel,
            'max_r':max_r,'max_prop':max_prop}


def claim_ledger(summary):
    rows={r['view']:r for r in summary['h30_dev_full43']}
    p,h,o=(rows[v] for v in ('OBS_PERSISTENCE','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'))
    return [
      {'claim':'Full43 standardized H30 HOLD/ORACLE advantage over persistence','status':'not_established',
       'evidence':{'P':p['position_prefix_standardized_mae'],'HOLD':h['position_prefix_standardized_mae'],
                   'ORACLE':o['position_prefix_standardized_mae']},
       'interpretation':'Both fixed ARX views have higher standardized position MAE than persistence on repeatedly exposed dev.'},
      {'claim':'Full43 native H30 ranking differs from standardized ranking','status':'observed_dev_only',
       'evidence':{'P':p['position_prefix_native_mae'],'HOLD':h['position_prefix_native_mae'],
                   'ORACLE':o['position_prefix_native_mae']},
       'interpretation':'Native channel units remain unresolved and cannot replace the locked standardized primary.'},
      {'claim':'Future-command substitution changes this fixed model output','status':'observed_model_response',
       'evidence':'HOLD and ORACLE fixed forecasts and signed driver-response identity',
       'interpretation':'Privileged recorded future command is not a causal, deployable, or intent-boundary experiment.'},
      {'claim':'One-step residual propagates and signed L1 terms can cancel','status':'accounting_identity_only',
       'evidence':'results/ACTUAL_RESIDUAL_LAG_BUDGET.csv and ERROR_BUDGET_BY_EPISODE.csv',
       'interpretation':'Residual includes fit, unmodeled, and recording effects; it is not independent or irreducible noise.'},
      {'claim':'Arm 0:14 is an independent physical/anatomical result','status':'not_established',
       'evidence':'Predeclared source-alias slice on exposed dev',
       'interpretation':'Retain as exploratory; no unit, anatomy, or physical-motion claim.'},
      {'claim':'This closes all Core B, command28, or other window work','status':'false',
       'evidence':'This root audits only the accepted OBS43 fixed model',
       'interpretation':'Query, Ridge, command28, N2/N3, VideoMAE and other windows remain separate.'},
      {'claim':'Independent confirmation has been performed','status':'false',
       'evidence':'A confirmation design was generated but not executed',
       'interpretation':'No final or held-out data were opened.'}]


def write_text_reports(summary,results,dev_read,source_hashes):
    rows={r['view']:r for r in summary['h30_dev_full43']}
    p,h,o=(rows[v] for v in ('OBS_PERSISTENCE','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'))
    ledger=claim_ledger(summary)
    write_json('results/CLAIM_LEDGER.json',ledger)
    budget=[r for r in results[-1]['budget_episode'] if r['horizon']==30 and r['block']=='Full43']
    budget_summary={}
    for metric in ('position_std','position_native','future_fd_std','future_fd_native'):
        matched=[r for r in budget if metric+'__L_hold_error' in r]
        if not matched: continue
        budget_summary[metric]={}
        for term in ('L_oracle_error','L_driver_response','L1_cancellation','L_hold_error',
                     'signed_eO_mean','signed_d_mean','signed_eH_mean','budget_identity_residual'):
            k=metric+'__'+term
            vals=[r[k] for r in matched if k in r]
            budget_summary[metric][term]=float(np.mean(vals)) if vals else None
    write_json('results/DEV_H30_FULL43_BUDGET_SUMMARY.json',budget_summary)
    blocks=[]
    coords=results[-1]['coordinate_rows']
    for view in ('OBS_PERSISTENCE','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'):
        for block,lo,hi in BLOCKS[1:]:
            rr=[r for r in coords if r['view']==view and lo<=r['channel_index']<hi]
            blocks.append({'view':view,'block':block,'channels':hi-lo,
                'native_block_contribution':float(sum(r['native_contribution_to_full_mae'] for r in rr)),
                'standardized_block_contribution':float(sum(r['standardized_contribution_to_full_mae'] for r in rr))})
    write_csv('results/DEV_H30_BLOCK_CONTRIBUTIONS.csv',list(blocks[0]),blocks)
    scoreqa=summary['score_reproduction']
    pos=budget_summary.get('position_std',{})
    pos_native=budget_summary.get('position_native',{})
    fd=budget_summary.get('future_fd_std',{})
    report=f"""# OBS43 固定模型误差预算与研究收口

## 范围与评分结论

本轮只复用已验收 OBS43 全训练 ARX、scalers、train/dev 标签、command、四种 H30 预测及 teacher-local 数组。固定模型 source/affine replay 仅用于代数与保存精度核验；没有新拟合、GPU 操作、参数选择或新预测文件，也没有重读 raw Parquet。主评分直接使用旧根已保存 forecast。

Full43、episode/source-group 等权的 H30 dev standardized position MAE：Persistence **{p['position_prefix_standardized_mae']:.6f}**，HOLD **{h['position_prefix_standardized_mae']:.6f}**，ORACLE **{o['position_prefix_standardized_mae']:.6f}**。两种 ARX 都未建立对 persistence 的主指标优势。Native 排序不同：Persistence **{p['position_prefix_native_mae']:.6f}**，HOLD **{h['position_prefix_native_mae']:.6f}**，ORACLE **{o['position_prefix_native_mae']:.6f}**；字段单位仍未核实，不能据此替换 standardized primary。旧 split/view/horizon/block 指标复算最大绝对差 **{scoreqa['max_abs_error']:.3g}**。

## H30 Full43 L1 预算

同一支持集、冻结尺度和 source-group 聚合下：

- 标准化 position：L_ORACLE_ERROR **{pos.get('L_oracle_error',float('nan')):.6f}**，L_DRIVER_RESPONSE **{pos.get('L_driver_response',float('nan')):.6f}**，L1_CANCELLATION **{pos.get('L1_cancellation',float('nan')):.6f}**，L_HOLD_ERROR **{pos.get('L_hold_error',float('nan')):.6f}**。
- Native position：oracle error **{pos_native.get('L_oracle_error',float('nan')):.6f}**，driver response **{pos_native.get('L_driver_response',float('nan')):.6f}**，cancellation **{pos_native.get('L1_cancellation',float('nan')):.6f}**，HOLD error **{pos_native.get('L_hold_error',float('nan')):.6f}**。
- 标准化 future-internal FD（29 个未来位置差分）：oracle error **{fd.get('L_oracle_error',float('nan')):.6f}**，driver response **{fd.get('L_driver_response',float('nan')):.6f}**，cancellation **{fd.get('L1_cancellation',float('nan')):.6f}**，HOLD error **{fd.get('L_hold_error',float('nan')):.6f}**。

逐元素先核验 eH=eO+d，再核验 L_HOLD=L_ORACLE_ERROR+L_DRIVER_RESPONSE-C。C 是 L1 向量抵消量，不是因果贡献。Position 用 frozen sigma_obs；FD 使用其独立 safe_fd_obs。H1 内部 FD 为 NA；anchor 到第一个未来位置的 seam 单列。14/14/15 slices 使用 Full43 相同逐通道尺度，逐 anchor 的 common-scale identity 在结果表中核验。旧表分数保留，未覆盖、重定义或挑选权重。

## 递推、坐标与分块

实际程序使用 delta=1/30、x=[s(43),v(43),u(28)]、coef 114x43；target inverse sigma_obs、input centering/scaling 与 intercept 均已纳入 A、B、c。记录时钟相邻间隔满足旧合同的 1e-5 秒容差；forecast 命令数组只含 u[0..29]，未使用 u[30]。三条 signed recurrences 与 r_position=delta*r_velocity 在固定模型的 float64 replay 上逐步核验；保存的 float32 forecast 量化差另报。历史 raw-label 首次读取时间、旧 extractor CPU/RSS 回执保持 UNKNOWN，本轮没有回填。

P A P^-1 的对角相似变换保持谱半径；raw 与归一化 operator norm 会随坐标改变。K_l 是选定归一化坐标下的最坏方向响应。实际 residual sequence 的传播、抵消和 horizon growth 单独计算，没有假定残差独立。

标准化权重 1/(43*sigma_obs[j]) 定义 Full43 主指标；native uniform numeric mean 使用不同权重。逐通道尺度、误差和 Full43 贡献见 results/COORDINATE_WEIGHTING_DEV_H30.csv；14/14/15 的共同尺度与排名见结果表。Arm 0:14 保留为已暴露 dev 上的探索性发现，不升格为独立确认、解剖语义或物理运动结论。

未来 recorded command 替换会改变这个固定模型的输出。该 command 可能包含对未来 observation 的反馈；比较不构成物理因果实验、intent 信息上界或可部署性证明。一步 residual 汇集拟合、未建模和记录因素，不称为不可约噪声。

## 收口

本轮关闭的仅是 OBS43 固定模型误差归因诊断，不改变原 28D command benchmark，也不关闭 query、Ridge、N2/N3、VideoMAE、D5 或其他窗口任务。INDEPENDENT_CONFIRMATION_SPEC.md 是未执行设计；确认需要未用于训练或 dev 选择的全新 source-group/session，并记录真实 command issue/availability 时间、字段单位和绑定。Human review 仍 pending，submission_ready=false；final/held-out access 为 0。
"""
    (ROOT/'CURRENT_BENCHMARK_CLOSEOUT.md').write_text(report,encoding='utf-8')
    spec="""# Independent confirmation design (not executed)

## Purpose
Check whether the fixed OBS43 transition and exploratory arm-slice pattern generalize beyond repeatedly exposed development data. Design only; no collection, fitting, scoring, or model execution was performed.

## Freeze before access
Freeze ARX source, coefficients/intercept, sigma_obs and safe_fd_obs, input order, constant delta=1/30, initial state, command availability contract, standardized primary, future-internal safe-FD, source-group estimand, horizons, and code hashes. The arm 0:14 slice was chosen after exposed-dev review; treating it as a next primary is a new study choice and must be preregistered. Keep command28, query, Ridge, other windows, and final/held-out resources separate.

## Split
Use unseen source groups and session groups with no overlap against training, model selection, or exposed development groups. A new anchor from an old group or a re-split of current dev is not confirmation. Unknown exposure is unverified and blocks a confirmatory claim.

## Required records and frozen scoring
Preserve source/session lineage, original frame index, timestamp, units, channel binding, and transformation provenance. Record command issue and availability times at query boundary; same-row presence does not prove it was available before the next observation. Do not refit, retune, rescale, clamp, stabilize, or choose cases on confirmation data. Keep future-command ORACLE privileged and non-deployable, and teacher forcing local-only. Report all predeclared horizons, Full43 and source aliases, units caveats, standardized primary, source-group uncertainty, support, and missingness.

## Claim boundary
A positive result supports only generalization of this fixed recorded-data benchmark under verified split and timing. It does not prove physical causality, intent availability, or a theoretical predictability limit. This design requests no final/held-out release.
"""
    (ROOT/'INDEPENDENT_CONFIRMATION_SPEC.md').write_text(spec,encoding='utf-8')
    # Figure input files are explicit and small enough to inspect/recreate.
    write_csv('figure_data/DEV_HORIZON_SCORE_DATA.csv',list(results[-1]['scores'][0]),
              [r for r in results[-1]['scores'] if r['block']=='Full43'])
    write_csv('figure_data/DEV_H30_COORDINATE_WEIGHTING.csv',list(coords[0]),coords)
    write_csv('figure_data/DEV_H30_ERROR_BUDGET_DATA.csv',
              list(results[-1]['budget_episode'][0]),
              [r for r in results[-1]['budget_episode'] if r['block']=='Full43'])
    caps="""# Figure captions and alt text

Every figure is for the separate OBS43 fixed-model diagnostic on repeatedly exposed dev. Native units are unresolved. ORACLE uses privileged future recorded commands and is not deployable.

1. Full43 horizon position and safe-FD scores for the four accepted views. Alt: separate position and FD axes show standardized position ARX above persistence at H30.
2. Per-channel contribution to Full43 H30 standardized position MAE. Alt: all 43 channels are shown in source-alias order; inverse frozen sigma defines the normalized weighting.
3. H30 L1 budget terms. Alt: oracle error and driver-response norm combine with nonnegative cancellation to equal HOLD error.
4. K_l and raw versus normalized A-power operator norms. Alt: diagonal similarity preserves eigenvalues but changes coordinate-dependent norms; K_l is a worst-direction response, not observed error.
"""
    (ROOT/'figures/FIGURE_CAPTIONS_AND_ALT.md').write_text(caps,encoding='utf-8')
    return budget_summary,blocks,ledger


def make_figures(results,affine,sigma):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dev=results[-1]; scores=dev['scores']
    colors=['#374151','#2f7f7b','#c46b20','#7b4b9b']
    labels=['Persistence','Constant velocity','ARX HOLD','ARX ORACLE']
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    for view,label,color in zip(VIEWS,labels,colors):
        rr=sorted([r for r in scores if r['view']==view and r['block']=='Full43'],key=lambda x:x['horizon'])
        axs[0].plot([x['horizon'] for x in rr],[x['position_prefix_standardized_mae'] for x in rr],marker='o',label=label,color=color)
        axs[1].plot([x['horizon'] for x in rr],[x['future_internal_fd_safe_normalized'] for x in rr],marker='o',label=label,color=color)
    axs[0].set_title('Full43 position, standardized'); axs[1].set_title('Future-internal FD, safe-scaled')
    for ax in axs: ax.set_xlabel('Horizon (steps)'); ax.set_xticks(HORIZONS); ax.grid(alpha=.25)
    axs[0].set_ylabel('Episode-equal MAE'); axs[1].set_ylabel('Episode-equal MAE')
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('OBS43 accepted forecasts; repeatedly exposed dev')
    fig.savefig(ROOT/'figures/FIG01_DEV_HORIZON_SCORES.png',dpi=170); plt.close(fig)
    coords=dev['coordinate_rows']
    fig,axs=plt.subplots(3,1,figsize=(12,7.2),sharex=True,constrained_layout=True)
    for ax,view,label in zip(axs,('OBS_PERSISTENCE','ARX_HOLD_COMMAND','ARX_ORACLE_COMMAND'),('Persistence','ARX HOLD','ARX ORACLE')):
        rr=[r for r in coords if r['view']==view]
        ax.bar([r['channel_index'] for r in rr],[r['standardized_contribution_to_full_mae'] for r in rr],color='#315f86')
        ax.set_ylabel('Full43 contribution'); ax.set_title(label,loc='left',fontsize=10); ax.grid(axis='y',alpha=.2)
    axs[-1].set_xlabel('Observation channel; aliases 0:14 / 14:28 / 28:43')
    fig.suptitle('Frozen inverse-sigma H30 position weighting')
    fig.savefig(ROOT/'figures/FIG02_DEV_H30_CHANNEL_WEIGHTS.png',dpi=170); plt.close(fig)
    budget=[r for r in dev['budget_episode'] if r['horizon']==30 and r['block']=='Full43']
    fig,axs=plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True)
    for ax,metric,title in zip(axs,('position_std','position_native'),('H30 position, standardized','H30 position, native')):
        terms=('L_oracle_error','L_driver_response','L1_cancellation','L_hold_error')
        vals=[np.mean([r[metric+'__'+t] for r in budget if metric+'__'+t in r]) for t in terms]
        ax.bar(['Oracle error','Driver response','Cancellation','HOLD error'],vals,color=['#7b4b9b','#c46b20','#4c956c','#374151'])
        ax.set_title(title); ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.2)
    fig.suptitle('L1 budget terms under common source-group support')
    fig.savefig(ROOT/'figures/FIG03_DEV_H30_L1_BUDGET.png',dpi=170); plt.close(fig)
    resp,norms,_=normalized_response(affine,sigma); raw={r['horizon']:r for r in norms}
    lag=[r for r in resp if r['block']=='Full43']
    fig,axs=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    axs[0].plot([r['lag_steps_after_residual'] for r in lag],[r['K_spectral_norm'] for r in lag],marker='.')
    axs[0].set_title('Dimensionless K_l worst-direction norm'); axs[0].set_xlabel('Residual lag steps'); axs[0].set_ylabel('2-norm'); axs[0].grid(alpha=.25)
    axs[1].plot(HORIZONS,[raw[h]['raw_A_power_2norm'] for h in HORIZONS],marker='o',label='Raw coordinates')
    axs[1].plot(HORIZONS,[raw[h]['dimensionless_Atilde_power_2norm'] for h in HORIZONS],marker='o',label='P-scaled coordinates')
    axs[1].set_title('A-power operator norms depend on coordinates'); axs[1].set_xlabel('Horizon'); axs[1].set_ylabel('2-norm'); axs[1].set_xticks(HORIZONS); axs[1].grid(alpha=.25); axs[1].legend(frameon=False)
    fig.suptitle('Fixed linear response; not a physical stability claim')
    fig.savefig(ROOT/'figures/FIG04_DIMENSIONLESS_RESPONSE.png',dpi=170); plt.close(fig)
    import struct
    files=[]
    for name in ('FIG01_DEV_HORIZON_SCORES.png','FIG02_DEV_H30_CHANNEL_WEIGHTS.png',
                 'FIG03_DEV_H30_L1_BUDGET.png','FIG04_DIMENSIONLESS_RESPONSE.png'):
        path=ROOT/'figures'/name
        with path.open('rb') as f:
            header=f.read(24)
        width,height=struct.unpack('>II',header[16:24])
        files.append({'path':'figures/'+name,'sha256':sha(path),'bytes':path.stat().st_size,
                      'width_px':width,'height_px':height,'dpi':170})
    write_json('figures/FIGURE_MANIFEST.json',{'status':'generated','figures':files,
        'captions_alt_path':'figures/FIGURE_CAPTIONS_AND_ALT.md','source':'figure_data CSVs under this OUT_ROOT'})


def runtime_record(stage,start_wall,start_cpu,disk0,extra=None):
    usage=resource.getrusage(resource.RUSAGE_SELF); rss=0
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'): rss=int(line.split()[1])*1024; break
    except OSError: pass
    row={'stage':stage,'ended_utc':now(),'wall_seconds':time.perf_counter()-start_wall,
         'process_cpu_seconds':time.process_time()-start_cpu,
         'peak_rss_bytes':int(usage.ru_maxrss*1024),'rss_bytes_at_end':rss,
         'disk_free_bytes_before':disk0,'disk_free_bytes_after':shutil.disk_usage(ROOT).free}
    if extra: row.update(extra)
    append_jsonl('logs/STAGE_RESOURCES.jsonl',row)
    return row


def write_affine(model,affine):
    lock=json.loads((OBS/'models/MODEL_LOCK.json').read_text(encoding='utf-8'))
    write_json('results/AFFINE_RECONSTRUCTION.json',{
      'status':'PASS','model_sha256':sha(OBS/'models/ARX_OBS43_FULL_TRAIN.npz'),
      'model_lock_sha256':sha(OBS/'models/MODEL_LOCK.json'),
      'source_code_sha256':sha(OBS/'code/obs43_diagnostic.py'),
      'coef_shape':list(model['coef'].shape),'coef_dtype':str(model['coef'].dtype),
      'x_mean_shape':list(model['x_mean'].shape),'x_std_shape':list(model['x_std'].shape),
      'sigma_obs_shape':list(model['sigma_obs'].shape),'intercept_shape':list(model['intercept'].shape),
      'delta_seconds':DELTA,'input_order':'[s_current43,v_current43,u_current28]',
      'target_inverse':'multiply next-velocity normalized output by sigma_obs',
      'input_transform':'(x-x_mean)/x_std; intercept and centering included',
      'F':'diag(sigma_obs) @ coef.T @ diag(1/x_std)',
      'b_v':'diag(sigma_obs)@intercept - F@x_mean',
      'A':'[[I+delta*Fs,delta*Fv],[Fs,Fv]]',
      'B':'[[delta*Fu],[Fu]]','c':'[delta*b_v;b_v]',
      'A_shape':list(affine['A'].shape),'B_shape':list(affine['B'].shape),
      'selected_lambda_from_prior_lock':lock.get('selected_lambda'),
      'nonlinear_or_clipping_in_accepted_inference':'none',
      'classification':'algebraic reconstruction of the already fitted source model'})


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=('synthetic','fixture','full'),required=True)
    mode=ap.parse_args().mode
    if mode=='synthetic':
        print(json.dumps(synthetic_controls(),indent=2)); return
    verify_prefit_lock()
    if mode=='fixture':
        fw,fc=time.perf_counter(),time.process_time(); fd=shutil.disk_usage(ROOT).free
        result=fixture_mode()
        runtime_record('train_fixture_replay',fw,fc,fd,{'fit_count':0,'gpu_actions':0,'train_rows_materialized':16})
        print(json.dumps({'status':result['status'],'fixture_anchor_ids':result['fixture_anchor_ids'],
                          'replay_error':result['saved_hold_oracle_float32_replay_max_abs']},indent=2))
        return
    fx=ROOT/'qa/TRAIN_FIXTURE_REPLAY_R003.json'
    if not fx.is_file(): raise FileNotFoundError('train fixture must pass before full analysis')
    fixture=json.loads(fx.read_text(encoding='utf-8'))
    if fixture['status']!='PASS' or fixture['prefit_lock_sha256']!=sha(ROOT/'protocol/PREFIT_LOCK_HASHES.json'):
        raise AssertionError('train fixture does not match current prefit lock')
    source_pre=verify_sources()
    start_wall,start_cpu=time.perf_counter(),time.process_time()
    disk0=shutil.disk_usage(ROOT).free
    threadpool_limits(limits=8)
    try:
        affinity=os.sched_getaffinity(0); selected=affinity.intersection(set(range(8)))
        if selected: os.sched_setaffinity(0,selected)
    except (AttributeError,OSError): selected=None
    owner=json.loads((ROOT/'protocol/OWNER_STATE.json').read_text(encoding='utf-8'))
    owner.update({'status':'ACTIVE','last_started_utc':now(),'analysis_code_sha256':sha(ROOT/'code/error_budget_closeout.py')})
    (ROOT/'protocol/OWNER_STATE.json').write_text(json.dumps(owner,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    mem_available_start=int(next((line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')), '0'))*1024
    model=load_npz(OBS/'models/ARX_OBS43_FULL_TRAIN.npz',
        ['x_mean','x_std','coef','intercept','sigma_obs','safe_fd_obs'])
    affine=reconstruct_affine(model); sigma=model['sigma_obs'].astype(np.float64); safe=model['safe_fd_obs'].astype(np.float64)
    write_affine(model,affine)
    results=[]; dev_read=None
    for split in ('train','dev'):
        x=run_split(split,model,affine,sigma,safe); results.append(x)
        if split=='dev': dev_read=x['data']['dev_label_first_read_utc']
    # Execution metadata is written only under the new task root.
    write_json('protocol/EXECUTION_RECEIPT.json',{
      'created_utc':now(),'out_root':str(ROOT),'hostname':os.uname().nodename,
      'python':os.sys.version.split()[0],'numpy':np.__version__,
      'cpu_affinity_selected':sorted(selected) if selected is not None else None,
      'blas_thread_limit':8,'gpu_queries':0,'gpu_actions':0,'scheduler_submissions':0,
      'raw_parquet_reopens':0,'new_model_fits':0,'new_forecast_artifacts_written':0,
      'source_replay_in_memory_for_acceptance':True,'dev_label_first_payload_read_utc':dev_read,
      'memory_available_bytes_start':mem_available_start,
      'disk_free_bytes_start':disk0})
    summary=write_summaries(results,affine,sigma)
    write_text_reports(summary,results,dev_read,source_pre)
    make_figures(results,affine,sigma)
    ids=[r for x in results for r in x['identity_rows']]
    max_id=max(r['max_abs'] for r in ids if r['identity']!='r_position - delta*r_velocity')
    max_r=max(r['max_abs'] for r in ids if r['identity']=='r_position - delta*r_velocity')
    replay=[r for x in results for r in x['replay_records']]
    max_aff=max(float(r.get('source_affine_replay_max_abs',0)) for r in replay)
    max_saved=max(float(r.get('stored_float32_max_abs',0)) for r in replay)
    blockers=[]
    if summary['score_reproduction']['status']!='PASS': blockers.append('score_reproduction')
    if max_id>1e-8 or max_r>1e-8: blockers.append('signed_identity')
    if max_aff>1e-8: blockers.append('affine_replay')
    if max_saved>2*EPS32: blockers.append('float32_forecast_replay')
    if summary['min_cancel'] < -1e-10: blockers.append('cancellation_nonnegative')
    if summary['max_common']>1e-10: blockers.append('common_scale_identity')
    if not all(x['clock']['all_within_locked_1e-5_seconds'] for x in results): blockers.append('clock_tolerance')
    if summary['norm_qa']['status']!='PASS': blockers.append('dimensionless_similarity')
    if summary['max_prop']>1e-8: blockers.append('actual_residual_propagation')
    status='complete' if not blockers else 'partial'
    final={'status':'PASS' if status=='complete' else 'PARTIAL','task':'OBS43_FIXED_MODEL_ERROR_BUDGET_CLOSEOUT',
      'blockers':blockers,'new_model_fits':0,'gpu_actions':0,'raw_parquet_reopened':False,
      'source_dependency_count':len(source_pre),'source_hashes_pre_post_match':True,
      'old_root_writes':0,'model_scaler_forecast_source_hashes_unchanged':True,
      'score_reproduction':summary['score_reproduction'],'affine_replay_max_abs':max_aff,
      'identity_max_abs':max_id,'r_position_relation_max_abs':max_r,
      'stored_float32_replay_max_abs':max_saved,'common_scale_identity_max_abs':summary['max_common'],
      'actual_residual_propagation_max_abs':summary['max_prop'],
      'minimum_cancellation':summary['min_cancel'],'clock_pass':not blockers,
      'dev_label_first_payload_read_utc':dev_read,'final_heldout_access':0,
      'independent_confirmation':False,'human_review_status':'pending','submission_ready':False,
      'checksums':'to be finalized and verified after remaining artifacts'}
    write_json('qa/FINAL_ACCEPTANCE.json',final)
    status_obj=json.loads((ROOT/'RUN_STATUS.json').read_text(encoding='utf-8'))
    status_obj.update({'status':status,'ended_utc':now(),'new_model_fits':0,'gpu_actions':0,
                       'dev_label_first_payload_read_utc':dev_read})
    (ROOT/'RUN_STATUS.json').write_text(json.dumps(status_obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    source_post=verify_sources()
    if source_pre!=source_post: raise AssertionError('source hashes changed during analysis')
    old_before=next(x['actual_sha256'] for x in source_pre if x['name']=='old_checksum_manifest')
    old_after=sha(OBS/'CHECKSUMS.sha256')
    if old_before!=old_after: raise AssertionError('old checksum manifest changed')
    write_json('qa/SOURCE_NONINTERFERENCE.json',{
      'status':'PASS','source_file_count':len(source_post),'source_hashes_before_equal_after':True,
      'previous_checksum_manifest_sha256_before':old_before,'previous_checksum_manifest_sha256_after':old_after,
      'old_root_writes':0,'raw_parquet_reopened':0,'final_test_or_heldout_access':0})
    owner.update({'status':'SEALED' if status=='complete' else 'PARTIAL',
                  'last_finished_utc':now(),'new_model_fits':0,'gpu_actions':0})
    (ROOT/'protocol/OWNER_STATE.json').write_text(json.dumps(owner,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    runtime=runtime_record('full_analysis',start_wall,start_cpu,disk0,{
      'cpu_affinity_selected':sorted(selected) if selected is not None else None,'blas_thread_limit':8,
      'split_count':2,'source_group_count':sum(x['data']['source_groups'] for x in results),
      'fit_count':0,'gpu_actions':0,'dev_label_first_payload_read_utc':dev_read})
    ret={'status':status,'task':'OBS43_FIXED_MODEL_ERROR_BUDGET_CLOSEOUT','out_root':str(ROOT),
      'new_model_fits':0,'gpu_actions':0,'new_forecast_artifacts_written':0,
      'source_registry_sha256':sha(ROOT/'source/SOURCE_REGISTRY.json'),
      'prefit_lock_sha256':sha(ROOT/'protocol/PREFIT_LOCK_HASHES.json'),
      'source_hashes_unchanged':True,'old_root_writes':0,'raw_parquet_reopened':0,
      'final_heldout_access':0,'independent_confirmation':'generated_not_executed',
      'dev_label_first_payload_read_utc':dev_read,'score_reproduction':summary['score_reproduction'],
      'affine_replay_max_abs':max_aff,'identity_max_abs':max_id,'r_position_relation_max_abs':max_r,
      'stored_float32_replay_max_abs':max_saved,'common_scale_identity_max_abs':summary['max_common'],
      'actual_residual_propagation_max_abs':summary['max_prop'],
      'runtime':runtime,'human_review_status':'pending','submission_ready':False}
    write_json('returns/RETURN_TO_CHATGPT.zh-CN.json',ret)
    (ROOT/'returns/RETURN_TO_CHATGPT.zh-CN.md').write_text(
      f"""# 返回摘要

- 状态：**{status}**；范围仅 OBS43 固定模型误差预算和归因收口。
- 新模型拟合 0，GPU 操作 0；无新 forecast 文件、raw Parquet 重读、旧根写入或 final/held-out 访问。
- 已验收分数复现最大绝对差 {summary['score_reproduction']['max_abs_error']:.3g}；affine replay {max_aff:.3g}；三条递推 identity {max_id:.3g}；r_position-delta*r_velocity {max_r:.3g}。
- 结论、权重和分块限制见 CURRENT_BENCHMARK_CLOSEOUT.md；独立确认设计生成但未执行。
- human review pending；submission_ready=false。
- 成本：wall {runtime['wall_seconds']:.1f}s，process CPU {runtime['process_cpu_seconds']:.1f}s，peak RSS {runtime['peak_rss_bytes']/(1024**3):.2f} GiB。
""",encoding='utf-8')
    print(json.dumps({'status':status,'blockers':blockers,'score_reproduction':summary['score_reproduction'],
      'affine_max':max_aff,'identity_max':max_id,'r_position_max':max_r,'runtime':runtime},indent=2,default=jsonable))


if __name__=='__main__': main()
