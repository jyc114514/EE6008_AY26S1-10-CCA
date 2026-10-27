#!/usr/bin/env python3
# The sample becomes sealed only after prefix-only QA passes.
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description='Seal a pre-outcome sample after independent prefix-only QA.')
    ap.add_argument('--selection',required=True); ap.add_argument('--prefix-qa',required=True)
    ap.add_argument('--prefix-inputs',required=True); ap.add_argument('--model-freeze',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args()
    if Path(a.out).exists(): raise RuntimeError('sample lock output already exists; refuse to overwrite')
    sel=json.loads(Path(a.selection).read_text(encoding='utf-8'))
    qa=json.loads(Path(a.prefix_qa).read_text(encoding='utf-8'))
    if sel.get('status')!='SAMPLE_SELECTED_AWAITING_PREFIX_QA' or sel.get('sample_sealed'):
        raise ValueError('selection is not awaiting prefix QA')
    if qa.get('status')!='PREFIX_QA_PASS' or qa.get('future_observation_values_read') is not False or qa.get('future_label_file_opened') is not False:
        raise ValueError('prefix QA must pass without label access')
    if int(qa.get('cluster_count',-1))!=int(sel['selected_clusters']):
        raise ValueError('prefix QA cluster support does not equal selected sample')
    planned={str(x['canonical_cluster_id']) for x in sel['selected_episodes']}
    if set(qa.get('canonical_cluster_ids',[]))!=planned:
        raise ValueError('prefix QA cluster IDs do not match the selected sample')
    freeze=json.loads(Path(a.model_freeze).read_text(encoding='utf-8'))
    obj={'status':'SAMPLE_SEALED','sample_sealed':True,'selection':sel,
         'prefix_qa_sha256':sha(a.prefix_qa),'prefix_inputs_sha256':sha(a.prefix_inputs),
         'model_freeze_sha256':sha(a.model_freeze),
         'sealed_utc':datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z'),
         'selected_clusters':sel['selected_clusters'],'selected_cluster_ids':[x['canonical_cluster_id'] for x in sel['selected_episodes']],
         'n_required_clusters':sel['n_required_clusters'],
         'new_outcomes_opened':False,'new_model_fits':0,'gpu_actions':0,
         'primary':'RAW_OBS_0_14 H30 standardized position and safe-FD',
         'model_sha256':freeze['packaged_model_sha256']}
    Path(a.out).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')


if __name__=='__main__':main()
