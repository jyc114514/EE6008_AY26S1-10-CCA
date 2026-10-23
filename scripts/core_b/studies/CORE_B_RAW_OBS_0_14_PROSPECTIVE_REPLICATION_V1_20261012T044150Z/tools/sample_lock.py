#!/usr/bin/env python3
# Deterministic cluster/episode selection from metadata only; no labels are accepted.
import argparse
import hashlib
import json
from pathlib import Path


def order_hash(seed, *parts):
    return hashlib.sha256('|'.join([str(seed), *map(str, parts)]).encode('utf-8')).hexdigest()


def select_sample(admission, n_required, seed=20261012, excluded_episode_ids=None):
    if admission.get('new_outcomes_opened') or admission.get('raw_frame_payloads_opened'):
        raise ValueError('sample lock must precede numeric outcomes')
    excluded=set(map(str,excluded_episode_ids or []))
    eligible=[]
    for cluster in admission.get('components', []):
        if not cluster.get('strict_admitted'):
            continue
        episodes=[e for e in cluster.get('eligible_episodes', [])
                  if str(e.get('episode_id')) not in excluded and int(e.get('eligible_anchor_count',0))>0]
        if episodes:
            eligible.append((cluster,episodes))
    if len(eligible) < n_required:
        return {'status':'prepared_not_executed','sample_sealed':False,
                'strict_pool_clusters':len(eligible),'n_required_clusters':int(n_required),
                'cluster_gap':int(n_required-len(eligible)),'selected_clusters':0,
                'selected_episodes':[],'selection_seed':int(seed),
                'selection_rule':'sha256(seed|canonical_cluster_id), one episode per connected cluster',
                'excluded_episode_ids':sorted(excluded),
                'reason':'strict metadata and hard-QA eligible clusters are fewer than locked N; no pilot or partial scoring'}
    ordered=sorted(eligible,key=lambda ce:order_hash(seed,ce[0]['canonical_cluster_id']))
    selected=[]
    for cluster,episodes in ordered:
        if len(selected)>=n_required:
            break
        ep=sorted(episodes,key=lambda e:order_hash(seed,cluster['canonical_cluster_id'],e['episode_id']))[0]
        selected.append({'canonical_cluster_id':cluster['canonical_cluster_id'],
                         'source_group_ids':cluster.get('source_group_ids',[]),
                         'episode_source_group_id':str(ep.get('source_group_id','')),
                         'episode_id':str(ep['episode_id']),
                         'episode_metadata_hash':ep['episode_metadata_hash'],
                         'eligible_anchor_count':int(ep['eligible_anchor_count'])})
    if len(selected)!=n_required:
        raise AssertionError('fixed-N selection did not reach required cluster count')
    return {'status':'SAMPLE_SELECTED_AWAITING_PREFIX_QA','sample_sealed':False,
            'strict_pool_clusters':len(eligible),'n_required_clusters':int(n_required),
            'selected_clusters':len(selected),'selected_episodes':selected,
            'selection_seed':int(seed),
            'selection_rule':'sha256(seed|canonical_cluster_id); one episode per cluster; sha256(seed|cluster|episode) tie break',
            'no_outcome_or_model_error_selection':True,'excluded_episode_ids':sorted(excluded),
            'seal_gate':'requires independent prefix QA receipt; no forecasts or outcomes yet'}


def main():
    ap=argparse.ArgumentParser(description='Lock one episode per independent admitted cluster, before scoring.')
    ap.add_argument('--admission',required=True); ap.add_argument('--n-required',type=int,required=True)
    ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=20261012)
    ap.add_argument('--exclude-episodes',default=None,help='JSON list of pre-outcome hard-QA failed episode IDs')
    a=ap.parse_args()
    admission=json.loads(Path(a.admission).read_text(encoding='utf-8'))
    excluded=json.loads(Path(a.exclude_episodes).read_text()) if a.exclude_episodes else []
    result=select_sample(admission,a.n_required,a.seed,excluded)
    Path(a.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','strict_pool_clusters','n_required_clusters','selected_clusters')}))


if __name__=='__main__':
    main()
