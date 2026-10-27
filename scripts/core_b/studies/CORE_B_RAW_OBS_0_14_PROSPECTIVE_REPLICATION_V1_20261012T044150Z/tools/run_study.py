#!/usr/bin/env python3
# Single resumable staged runner. Each stage has an explicit read boundary.
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from metadata_admission import admit_records
from sample_lock import select_sample
from prefix_export import export_prefix
from frozen_predict import predict_views
from independent_score import score_once

ROOT=Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def status():
    admission_path=ROOT/'data/DATA_ADMISSION_RESULT.json'
    admission=json.loads(admission_path.read_text(encoding='utf-8')) if admission_path.is_file() else json.loads((ROOT/'data/DATA_ADMISSION.json').read_text(encoding='utf-8'))
    plan=json.loads((ROOT/'data/N_PLAN.json').read_text(encoding='utf-8'))
    return {'status':'prepared_not_executed' if admission['strict_pool_clusters']==0 else 'pool_requires_gates',
        'strict_pool_clusters':admission['strict_pool_clusters'],
        'n_required_clusters':plan['n_required_independent_clusters'],
        'model_freeze_sha256':sha(ROOT/'protocol/MODEL_FREEZE.json'),
        'new_model_fits':0,'gpu_actions':0,'new_labels_read':0,'confirmation_executed':False}


def do_admit(manifest):
    if any((ROOT/p).exists() for p in ('protocol/SAMPLE_SELECTION.json','protocol/FORECAST_LOCK.json','logs/OUTCOME_READ_RECEIPT.json','results/PROSPECTIVE_SCORE.json')):
        raise RuntimeError('admission is locked after sample selection or forecast/outcome stage begins')
    sample_lock_path=ROOT/'protocol/SAMPLE_LOCK.json'
    if sample_lock_path.is_file() and json.loads(sample_lock_path.read_text()).get('sample_sealed'):
        raise RuntimeError('admission is locked after sample sealing')
    with Path(manifest).open(encoding='utf-8') as f: d=json.load(f)
    freeze=json.loads((ROOT/'protocol/DATA_SPACE_LOCK.json').read_text(encoding='utf-8'))
    import csv
    with (ROOT/'data/PRIOR_EXPOSURE_GROUPS.csv').open(encoding='utf-8',newline='') as f:
        prior={r['source_group_id'] for r in csv.DictReader(f)}
    result=admit_records(d.get('records',[]),freeze,prior)
    (ROOT/'data/DATA_ADMISSION_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return result


def do_sample(exclude_file=None):
    if any((ROOT/p).exists() for p in ('protocol/SAMPLE_SELECTION.json','protocol/FORECAST_LOCK.json','logs/OUTCOME_READ_RECEIPT.json','results/PROSPECTIVE_SCORE.json')):
        raise RuntimeError('sample selection is single-use; do not replace the frozen sample')
    if json.loads((ROOT/'protocol/SAMPLE_LOCK.json').read_text(encoding='utf-8')).get('sample_sealed'):
        raise RuntimeError('sample is already sealed')
    admission=json.loads((ROOT/'data/DATA_ADMISSION_RESULT.json').read_text(encoding='utf-8'))
    plan=json.loads((ROOT/'data/N_PLAN.json').read_text(encoding='utf-8'))
    excluded=json.loads(Path(exclude_file).read_text(encoding='utf-8')) if exclude_file else []
    result=select_sample(admission,plan['n_required_independent_clusters'],20261012,excluded)
    (ROOT/'protocol/SAMPLE_SELECTION.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return result


def do_prefix(frame_index,prefix_capture):
    if any((ROOT/p).exists() for p in ('protocol/FORECAST_LOCK.json','logs/OUTCOME_READ_RECEIPT.json','results/PROSPECTIVE_SCORE.json')):
        raise RuntimeError('prefix export is locked after forecast/outcome stage begins')
    if json.loads((ROOT/'protocol/SAMPLE_LOCK.json').read_text(encoding='utf-8')).get('sample_sealed'):
        raise RuntimeError('prefix export is locked after sample sealing')
    selection=json.loads((ROOT/'protocol/SAMPLE_SELECTION.json').read_text(encoding='utf-8'))
    result=export_prefix(selection,frame_index,prefix_capture,ROOT/'data/prefix_inputs.npz',ROOT/'qa/PREFIX_QA.json',20261012)
    return result


def do_seal():
    if json.loads((ROOT/'protocol/SAMPLE_LOCK.json').read_text(encoding='utf-8')).get('sample_sealed'):
        raise RuntimeError('sample lock is single-use and already sealed')
    if any((ROOT/p).exists() for p in ('protocol/FORECAST_LOCK.json','logs/OUTCOME_READ_RECEIPT.json','results/PROSPECTIVE_SCORE.json')):
        raise RuntimeError('sample cannot be resealed after forecast/outcome stage begins')
    selection=json.loads((ROOT/'protocol/SAMPLE_SELECTION.json').read_text(encoding='utf-8'))
    qa=json.loads((ROOT/'qa/PREFIX_QA.json').read_text(encoding='utf-8'))
    if selection.get('status')!='SAMPLE_SELECTED_AWAITING_PREFIX_QA' or qa.get('status')!='PREFIX_QA_PASS':
        raise RuntimeError('sample/prefix QA gate failed; no forecast may be run')
    if selection['selected_clusters']!=json.loads((ROOT/'data/N_PLAN.json').read_text())['n_required_independent_clusters']:
        raise RuntimeError('selected cluster count does not equal frozen N')
    planned={str(x['canonical_cluster_id']) for x in selection['selected_episodes']}
    if set(qa.get('canonical_cluster_ids',[]))!=planned:
        raise RuntimeError('prefix QA cluster IDs do not match selected sample')
    freeze_path=ROOT/'protocol/MODEL_FREEZE.json'
    sealed={'status':'SAMPLE_SEALED','sample_sealed':True,'selection':selection,
        'selected_clusters':selection['selected_clusters'],'selected_cluster_ids':[x['canonical_cluster_id'] for x in selection['selected_episodes']],
        'n_required_clusters':selection['n_required_clusters'],
        'prefix_qa_sha256':sha(ROOT/'qa/PREFIX_QA.json'),
        'prefix_inputs_sha256':sha(ROOT/'data/prefix_inputs.npz'),
        'model_freeze_sha256':sha(freeze_path),'model_sha256':json.loads(freeze_path.read_text())['packaged_model_sha256'],
        'sealed_utc':now(),'new_outcomes_opened':False,'new_model_fits':0,'gpu_actions':0}
    (ROOT/'protocol/SAMPLE_LOCK.json').write_text(json.dumps(sealed,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return sealed


def do_predict():
    if any((ROOT/p).exists() for p in ('protocol/FORECAST_LOCK.json','predictions/FORECASTS_H30_NPZ.npz','logs/OUTCOME_READ_RECEIPT.json','results/PROSPECTIVE_SCORE.json')):
        raise RuntimeError('forecast stage is single-use; refuse to overwrite forecasts or locks')
    sample=json.loads((ROOT/'protocol/SAMPLE_LOCK.json').read_text(encoding='utf-8'))
    if sample.get('status')!='SAMPLE_SEALED' or not sample.get('sample_sealed'):
        raise RuntimeError('sample is not sealed; stop before forecast')
    prefix=ROOT/'data/prefix_inputs.npz'; model=ROOT/'frozen_model/ARX_OBS43_FULL_TRAIN.npz'
    freeze_path=ROOT/'protocol/MODEL_FREEZE.json'; freeze=json.loads(freeze_path.read_text(encoding='utf-8'))
    if sha(model)!=freeze.get('packaged_model_sha256') or sample.get('model_sha256')!=sha(model):
        raise RuntimeError('model bytes do not match the sealed sample and frozen model lock')
    if sample.get('model_freeze_sha256')!=sha(freeze_path):
        raise RuntimeError('model freeze lock changed after sample seal')
    pred=predict_views(model,prefix)
    forecast_path=ROOT/'predictions/FORECASTS_H30_NPZ.npz'
    forecast_path.parent.mkdir(parents=True,exist_ok=True)
    import numpy as np
    np.savez_compressed(forecast_path,**pred)
    lock={'status':'SEALED_BEFORE_LABEL_ACCESS','created_utc':now(),
        'model_sha256':sha(model),'prefix_sha256':sha(prefix),'forecast_sha256':sha(forecast_path),
        'sample_lock_sha256':sha(ROOT/'protocol/SAMPLE_LOCK.json'),
        'admission_result_sha256':sha(ROOT/'data/DATA_ADMISSION_RESULT.json'),
        'final_or_heldout_access':0,
        'views':['OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND'],
        'oracle_used':False,'teacher_forcing_used':False,'new_model_fits':0,'gpu_actions':0}
    (ROOT/'protocol/FORECAST_LOCK.json').write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return lock


def main():
    ap=argparse.ArgumentParser(description='Prepared single-pass prospective workflow; never fits or accesses GPU.')
    sp=ap.add_subparsers(dest='stage',required=True)
    sp.add_parser('status',help='show current readiness; no data opened')
    p=sp.add_parser('admit',help='metadata-only registry admission')
    p.add_argument('--manifest',required=True)
    p=sp.add_parser('sample',help='select fixed N from strict metadata-admitted clusters')
    p.add_argument('--exclude-episodes',default=None)
    p=sp.add_parser('prefix',help='export only s[-1], s[0], u[0] after sample selection')
    p.add_argument('--frame-index',required=True); p.add_argument('--prefix-capture',required=True)
    sp.add_parser('seal',help='seal sample after prefix-only QA passes')
    sp.add_parser('predict',help='write the three frozen CPU forecast views, no future inputs')
    p=sp.add_parser('score',help='open new labels once, only after forecast lock')
    p.add_argument('--labels',required=True)
    args=ap.parse_args()
    if args.stage=='status': result=status()
    elif args.stage=='admit': result=do_admit(args.manifest)
    elif args.stage=='sample': result=do_sample(args.exclude_episodes)
    elif args.stage=='prefix': result=do_prefix(args.frame_index,args.prefix_capture)
    elif args.stage=='seal': result=do_seal()
    elif args.stage=='predict': result=do_predict()
    elif args.stage=='score':
        result=score_once(ROOT,args.labels,ROOT/'predictions/FORECASTS_H30_NPZ.npz',
            ROOT/'data/prefix_inputs.npz',ROOT/'protocol/SAMPLE_LOCK.json')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
