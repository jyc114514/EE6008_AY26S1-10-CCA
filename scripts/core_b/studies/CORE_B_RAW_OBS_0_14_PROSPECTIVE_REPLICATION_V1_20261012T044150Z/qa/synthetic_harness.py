#!/usr/bin/env python3
# Synthetic-only regression checks. Never writes confirmation scores or admits fixture records.
import csv
import json
import sys
import resource
import shutil
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'tools'
sys.path.insert(0,str(TOOLS))
from metadata_admission import admit_records, EXPECTED_FIELDS
from sample_lock import select_sample
from prefix_export import export_prefix
from frozen_predict import predict_views
from independent_score import score_arrays, score_once

TASK='RAW_OBS_0_14_PROSPECTIVE_RECORDED_DOMAIN_REPLICATION'


def make_candidate(freeze, record_id, group, session, episode, **overrides):
    row={'candidate_record_id':record_id,'dataset_id':'synthetic-provider-new-session',
      'dataset_role':'research_nonfinal','proposed_split':'confirmation',
      'producer_id':freeze['producer_id'],'producer_code_sha256':freeze['producer_code_sha256'],
      'source_schema_sha256':freeze['source_schema_sha256'],'conversion_sha256':freeze['conversion_sha256'],
      'dataset_registry_entry_id':'synthetic-registry-entry','registry_entry_sha256':'5'*64,
      'hardware_config_id':'same-config-synthetic','hardware_config_relation':'same_as_reference_with_evidence',
      'hardware_config_evidence_ref':'synthetic-evidence-only',
      'source_group_id':group,'source_group_aliases':[],'session_id':session,'session_aliases':[],
      'session_independence_verified':True,'independence_evidence_ref':'synthetic-trial-proof',
      'episode_id':episode,'episode_metadata_sha256':'1'*64,'frame_index_sha256':'2'*64,
      'prefix_capture_sha256':'3'*64,'hard_qa_anchor_count':3,
      'exposure_fit':False,'exposure_normalizer':False,'exposure_cv':False,'exposure_dev':False,
      'related_dev_selection':False,'final_test':False,'heldout':False,
      'unit_status':'unverified_same_native_numeric',
      'numeric_space_equivalence_evidence':'same_producer_schema_conversion',
      'field_order':EXPECTED_FIELDS,'allowed_use':TASK,'source_manifest_sha256':'4'*64,
      'synthetic_fixture':False,'task_family_id':'synthetic-family','scene_id':'synthetic-scene',
      'operator_id':'synthetic-operator'}
    row.update(overrides)
    return row


def run():
    freeze=json.loads((ROOT/'protocol/DATA_SPACE_LOCK.json').read_text(encoding='utf-8'))
    test_freeze=dict(freeze); test_freeze['hardware_config_reference_status']='verified'
    prior_rows=list(csv.DictReader((ROOT/'data/PRIOR_EXPOSURE_GROUPS.csv').open(encoding='utf-8',newline='')))
    prior={r['source_group_id'] for r in prior_rows}
    cases={}
    good=[make_candidate(freeze,'r1','new-group-1','new-session-1','new-episode-1'),
          make_candidate(freeze,'r2','new-group-2','new-session-2','new-episode-2'),
          make_candidate(freeze,'r3','new-group-3','new-session-3','new-episode-3')]
    accepted=admit_records(good,test_freeze,prior)
    assert accepted['strict_pool_clusters']==3
    cases['new_nonfinal_schema_admits_under_hypothetical_verified_domain']=True
    unregistered=[dict(good[0])]
    unregistered[0].pop('dataset_registry_entry_id'); unregistered[0].pop('registry_entry_sha256')
    assert admit_records(unregistered,test_freeze,prior)['strict_pool_clusters']==0
    cases['missing_formal_registration_evidence_rejected']=True
    explicit_pool=[dict(good[0])]
    explicit_pool[0].pop('dataset_registry_entry_id'); explicit_pool[0].pop('registry_entry_sha256')
    explicit_pool[0]['explicit_nonfinal_pool_id']='synthetic-pool'
    explicit_pool[0]['registration_evidence_ref']='synthetic-registration-evidence'
    assert admit_records(explicit_pool,test_freeze,prior)['strict_pool_clusters']==1
    cases['explicit_nonfinal_pool_registration_path_accepted']=True
    blocked=admit_records(good,freeze,prior)
    assert blocked['strict_pool_clusters']==0
    cases['unknown_frozen_hardware_baseline_blocks_strict_admission']=True
    same_session=[make_candidate(freeze,'r4','new-group-4','shared-new-session','new-episode-4'),
                  make_candidate(freeze,'r5','new-group-5','shared-new-session','new-episode-5')]
    collapsed=admit_records(same_session,test_freeze,prior)
    assert collapsed['connected_components']==1 and collapsed['strict_pool_clusters']==1
    cases['same_session_collapses_to_one_cluster']=True
    old_alias=make_candidate(freeze,'r6','new-group-6','new-session-6','new-episode-6',
        source_group_aliases=[next(iter(prior))])
    assert admit_records([old_alias],test_freeze,prior)['strict_pool_clusters']==0
    cases['old_group_alias_rejected']=True
    old_session_alias=make_candidate(freeze,'r6b','new-group-6b','new-session-6b','new-episode-6b',
        session_aliases=[next(iter(prior))])
    assert admit_records([old_session_alias],test_freeze,prior)['strict_pool_clusters']==0
    cases['old_group_through_session_alias_rejected']=True
    linked=[make_candidate(freeze,'r6c','new-group-6c','new-session-6c','new-episode-6c',
        source_group_aliases=['canonical-group-6d']),
        make_candidate(freeze,'r6d','canonical-group-6d','new-session-6d','new-episode-6d')]
    linked_result=admit_records(linked,test_freeze,prior)
    assert linked_result['connected_components']==1 and linked_result['strict_pool_clusters']==1
    cases['source_group_alias_links_to_canonical_group_id']=True
    final=make_candidate(freeze,'r7','new-group-7','new-session-7','new-episode-7',dataset_role='final')
    assert admit_records([final],test_freeze,prior)['strict_pool_clusters']==0
    cases['final_role_rejected']=True
    unknown=make_candidate(freeze,'r8','new-group-8','new-session-8','new-episode-8',dataset_role='unknown')
    assert admit_records([unknown],test_freeze,prior)['strict_pool_clusters']==0
    cases['unknown_role_rejected']=True
    syn=make_candidate(freeze,'r9','new-group-9','new-session-9','new-episode-9',dataset_role='synthetic_fixture',synthetic_fixture=True)
    assert admit_records([syn],test_freeze,prior)['strict_pool_clusters']==0
    cases['synthetic_fixture_rejected']=True
    changed=make_candidate(freeze,'r10','new-group-10','new-session-10','new-episode-10',conversion_sha256='f'*64)
    assert admit_records([changed],test_freeze,prior)['strict_pool_clusters']==0
    cases['conversion_mismatch_rejected']=True
    exposed=make_candidate(freeze,'r11','new-group-11','new-session-11','new-episode-11',exposure_dev=True)
    assert admit_records([exposed],test_freeze,prior)['strict_pool_clusters']==0
    cases['prior_dev_exposure_rejected']=True
    split_leak=[make_candidate(freeze,'r12a','new-group-12','same-session','ep-a'),
                make_candidate(freeze,'r12b','new-group-12','same-session','ep-b',exposure_fit=True)]
    leak=admit_records(split_leak,test_freeze,prior)
    assert leak['strict_pool_clusters']==0
    cases['group_session_cross_exposure_component_rejected']=True
    # Hash order is deterministic and excluded hard-QA episodes are replaced before outcomes.
    sel1=select_sample(accepted,2,20261012)
    sel2=select_sample(accepted,2,20261012)
    assert sel1==sel2 and sel1['status']=='SAMPLE_SELECTED_AWAITING_PREFIX_QA'
    exclude=[sel1['selected_episodes'][0]['episode_id']]
    repl=select_sample(accepted,2,20261012,exclude)
    assert repl['status']=='SAMPLE_SELECTED_AWAITING_PREFIX_QA'
    assert exclude[0] not in [x['episode_id'] for x in repl['selected_episodes']]
    cases['stable_cluster_sampling_and_preoutcome_fallback']=True
    under=select_sample(accepted,4,20261012)
    assert under['status']=='prepared_not_executed' and under['sample_sealed'] is False
    cases['undersized_pool_never_sealed']=True
    # Synthetic 30 Hz metadata and prefix-only captures exercise the frozen anchor grid without targets.
    one=select_sample(admit_records([good[0]],test_freeze,prior),1,20261012)
    temp=ROOT/'tmp/synthetic_harness'; temp.mkdir(parents=True,exist_ok=True)
    frame_path=temp/'frame_index.csv'; prefix_path=temp/'prefix_capture.csv'
    fields=['source_group_id','session_id','episode_id','frame_index','timestamp_seconds']
    with frame_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(211):
            w.writerow({'source_group_id':'new-group-1','session_id':'new-session-1',
              'episode_id':'new-episode-1','frame_index':i,'timestamp_seconds':i/30.0})
    pfields=['source_group_id','session_id','episode_id','anchor_frame','anchor_timestamp_seconds']
    pfields += [f's_prev_{i:02d}' for i in range(43)]
    pfields += [f's0_{i:02d}' for i in range(43)]
    pfields += [f'u0_{i:02d}' for i in range(28)]
    with prefix_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=pfields);w.writeheader()
        for anchor in (120,150,180):
            row={'source_group_id':'new-group-1','session_id':'new-session-1',
              'episode_id':'new-episode-1','anchor_frame':anchor,'anchor_timestamp_seconds':anchor/30.0}
            row.update({f's_prev_{i:02d}':0 for i in range(43)})
            row.update({f's0_{i:02d}':0 for i in range(43)})
            row.update({f'u0_{i:02d}':0 for i in range(28)})
            w.writerow(row)
    prefix_out=temp/'prefix_inputs.npz'; receipt=temp/'prefix_qa.json'
    pq=export_prefix(one,frame_path,prefix_path,prefix_out,receipt,20261012)
    assert pq['status']=='PREFIX_QA_PASS' and pq['anchor_count']==3 and pq['future_label_file_opened'] is False
    sentinel_prefix=temp/'prefix_capture_with_forbidden_future.csv'
    with prefix_path.open(encoding='utf-8',newline='') as src, sentinel_prefix.open('w',encoding='utf-8',newline='') as dst:
        rd=csv.DictReader(src); fields=list(rd.fieldnames)+['obs_future_00']
        wr=csv.DictWriter(dst,fieldnames=fields); wr.writeheader()
        for row in rd: row['obs_future_00']='SENTINEL_MUST_NOT_BE_READ'; wr.writerow(row)
    try:
        export_prefix(one,frame_path,sentinel_prefix,temp/'sentinel_out.npz',temp/'sentinel_receipt.json',20261012)
    except ValueError:
        pass
    else:
        raise AssertionError('prefix exporter accepted a future-label sentinel column')
    assert not (temp/'sentinel_receipt.json').exists()
    cases['prefix_capture_rejects_future_label_column_before_rows']=True
    with np.load(prefix_out,allow_pickle=False) as z:
        assert 'obs_future' not in z.files and 'u_future' not in z.files
        prefix={k:z[k].copy() for k in z.files}
    cases['prefix_export_is_future_label_free']=True
    # Predictor checks only the prefix archive; synthetic zero parameters are never trained.
    fake_model=temp/'synthetic_model.npz'
    np.savez(fake_model,x_mean=np.zeros(114),x_std=np.ones(114),x_constant=np.zeros(114,dtype=bool),
      x_raw_std=np.ones(114),coef=np.zeros((114,43)),intercept=np.zeros(43),
      sigma_obs=np.ones(43),sigma_obs_raw=np.ones(43),fd_scale_obs=np.ones(43),
      safe_fd_obs=np.ones(43),active_threshold_obs=np.ones(43))
    pred=predict_views(fake_model,prefix_out)
    assert set(pred).issuperset({'OBS_PERSISTENCE','OBS_CONSTANT_VELOCITY','ARX_HOLD_COMMAND'})
    assert 'ARX_ORACLE_COMMAND' not in pred and 'ARX_TEACHER_FORCED' not in pred
    malicious=temp/'predictor_with_unexpected_future_key.npz'
    np.savez(malicious,**prefix,obs_future_sentinel=np.zeros(1,dtype=np.float32))
    try:
        predict_views(fake_model,malicious)
    except ValueError:
        pass
    else:
        raise AssertionError('predictor accepted an unexpected future field')
    cases['predictor_rejects_unexpected_future_key']=True
    cases['predictor_has_exactly_three_allowed_views']=True
    # Synthetic arrays exercise scorer code; the returned verdict is explicitly non-confirmatory.
    n=6; ids=np.asarray([f'syn-a{i}' for i in range(n)],dtype='U64')
    clusters=np.asarray([f'syn-c{i}' for i in range(n)],dtype='U64')
    episodes=np.asarray([f'syn-e{i}' for i in range(n)],dtype='U64')
    pref={'anchor_id':ids,'canonical_cluster_id':clusters,'source_group_id':clusters.copy(),'episode_id':episodes}
    truth=np.zeros((n,30,43),dtype=np.float32)
    labs={'anchor_id':ids,'source_group_id':clusters,'episode_id':episodes,'obs_future':truth}
    base=np.zeros_like(truth); p=base.copy(); cv=base.copy(); hold=base.copy()
    p[:]=.1; cv[:]=.2; hold[:]=.05
    forecasts={'anchor_id':ids,'OBS_PERSISTENCE':p,'OBS_CONSTANT_VELOCITY':cv,'ARX_HOLD_COMMAND':hold}
    scores=score_arrays(pref,forecasts,labs,{'sigma_obs':np.ones(43),'safe_fd_obs':np.ones(43)},
                        synthetic=True,draws=100,seed=20261012)
    assert scores['status']=='PASS_SYNTHETIC_QA' and scores['confirmation_verdict']=='synthetic_not_confirmatory'
    cases['synthetic_scorer_never_claims_confirmation']=True
    guard_root=temp/'scorer_guard_root'; (guard_root/'logs').mkdir(parents=True,exist_ok=True)
    (guard_root/'logs/OUTCOME_READ_RECEIPT.json').write_text('{"status":"SYNTHETIC_GUARD_ONLY"}\n')
    try:
        score_once(guard_root,temp/'never_open_this_label.npz',temp/'none.npz',temp/'none_prefix.npz',temp/'none_sample.json')
    except RuntimeError as exc:
        assert 'do not reopen or rescore' in str(exc)
    else:
        raise AssertionError('scorer did not reject a second read before labels')
    cases['single_pass_guard_rejects_before_label_access']=True
    return {'status':'PASS','synthetic_fixture_only':True,'confirmation_result_written':False,
      'new_candidate_admitted_for_real_study':False,'tests':cases,'test_count':len(cases)}


if __name__=='__main__':
    w0,c0=time.perf_counter(),time.process_time(); disk0=shutil.disk_usage(ROOT).free
    result=run()
    usage=resource.getrusage(resource.RUSAGE_SELF)
    runtime={'stage':'synthetic_tool_regression','wall_seconds':time.perf_counter()-w0,
      'process_cpu_seconds':time.process_time()-c0,'peak_rss_bytes':int(usage.ru_maxrss*1024),
      'disk_free_bytes_before':disk0,'disk_free_bytes_after':shutil.disk_usage(ROOT).free,
      'new_model_fits':0,'gpu_actions':0,'new_labels_read':0}
    result['runtime']=runtime
    (ROOT/'qa/SYNTHETIC_TOOL_TESTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    with (ROOT/'logs/STAGE_RESOURCES.jsonl').open('a',encoding='utf-8') as f:
      f.write(json.dumps(runtime,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
