"""Freeze the new research question before any formal fits or new evaluation."""
import argparse,csv,hashlib,json,math,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

LEGACY_PREFIX=os.environ.get("EE6008_LEGACY_SOURCE_PREFIX", "").rstrip("/")
P=Path(LEGACY_PREFIX)
V9=Path('reports/benchmarks/CORE_B_V9_CONTEXT_DROPOUT_V1_20261017T054932Z')
RUN=os.environ.get("EE6008_LEGACY_REPLAY_REL", "")
TASKS=[119,94,90,141,230]
SEEDS=[202610131,202610132,202610133]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--run-root',type=Path,required=True)
    a=ap.parse_args();s=a.source_root.resolve();r=a.run_root.resolve()
    if not LEGACY_PREFIX:ap.error("Set EE6008_LEGACY_SOURCE_PREFIX before reading external assets")
    if not RUN or Path(RUN).is_absolute() or ".." in Path(RUN).parts:ap.error("Set EE6008_LEGACY_REPLAY_REL to a source-relative recovery output directory")
    def relocate(path):return s/Path(path).relative_to(P)
    jobs=[json.loads((s/V9/'configs/fits'/f'V0_seed{seed}.json').read_text()) for seed in SEEDS]
    train=relocate(jobs[0]['train_npz_path'])
    assert sha(train)==jobs[0]['train_npz_sha256']
    with np.load(train,allow_pickle=False) as z:data={k:z[k] for k in ['anchor_id','episode_id','task_id','source_cluster']}
    assert len(data['anchor_id'])==1280 and len(set(data['anchor_id']))==1280
    assert len(np.unique(data['episode_id']))==220 and set(data['task_id'])==set(TASKS)
    populations={'train':{'anchors':1280,'episodes':220}}
    episode_sets={'train':set(data['episode_id'])};cluster_sets={'train':set(data['source_cluster'].astype(str))}
    for split,prefix in [('dev','DEV'),('final_test','FINAL')]:
        path=s/RUN/'replay/core_b'/f'{prefix}_PREFIXES_CURRENT_PROFILE_V2.npz'
        with np.load(path,allow_pickle=False) as z:
            eps=z['episode_id'];ids=z['anchor_id'];clusters=z['source_cluster']
        assert len(ids)==len(set(ids))
        episode_sets[split]=set(eps);cluster_sets[split]=set(clusters.astype(str))
        populations[split]={'anchors':len(ids),'episodes':len(set(eps)),'prefix_sha256':sha(path)}
    for i,x in enumerate(episode_sets):
        for y in list(episode_sets)[i+1:]:
            assert not episode_sets[x]&episode_sets[y],f'Episode overlap {x}/{y}'
            assert not cluster_sets[x]&cluster_sets[y],f'Cluster overlap {x}/{y}'
    support=[]
    for t in TASKS:
        mask=data['task_id']==t;n=int(mask.sum());e=len(np.unique(data['episode_id'][mask]));steps=math.ceil(n/64)
        support.append({'task_id':t,'train_anchors':n,'train_episodes':e,'steps_per_epoch':steps,'optimizer_updates':steps*160,'tail_batch_size':n%64 or 64})
    source_receipts=[]
    for j in jobs:
        for pathkey,hashkey in [('train_npz_path','train_npz_sha256'),('init_path','init_file_sha256'),('normalizer_path','normalizer_sha256'),('sampler_orders_path','sampler_orders_sha256'),('feature_cache_manifest_path','feature_cache_manifest_sha256')]:
            path=relocate(j[pathkey]);actual=sha(path);assert actual==j[hashkey],path
            source_receipts.append({'seed':j['seed'],'role':pathkey,'path':str(path),'sha256':actual})
    lock={'schema_version':1,'status':'LOCKED_BEFORE_FORMAL_FITS_AND_NEW_EVALUATION','created_utc':datetime.now(timezone.utc).isoformat(),
        'research_question':'Does a bank of five independently trained V0 task experts improve forecast error versus one shared V0 under identical fixed feature and data conditions?',
        'source_root':str(s),'run_root':str(r),'task_ids':TASKS,'seeds':SEEDS,'expert_fits':15,'shared_control_fits':3,
        'architecture':'V2TaskConditionedPredictor','video_mode':'TS11-D','numeric_context_drop_probability':0.0,'waypoint_mode':None,
        'epochs':160,'batch_size':64,'tail_batch_policy':'retain every row; divide weighted loss by actual batch size',
        'optimizer':{'name':'AdamW','lr':0.0005,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.0001,'gradient_clip_norm':1.0},
        'scheduler':{'name':'UpdateDrivenCosineAnnealingLR','T_max':'160 * ceil(number of selected train anchors / 64)','eta_min':1e-5,'step_location':'optimizer post-step'},
        'expert_weights':'N_class / (episodes_in_class * anchors_in_episode)','shared_weights':'1280 / (5 * episodes_in_task * anchors_in_episode)',
        'expert_sampler':'filter each original global locked epoch permutation to the selected true task, retaining relative order',
        'initialization':'same original untrained shared_5task_base file per seed for shared and all five experts; no trained weights reused',
        'task_token':'unchanged five-way token, constant within each expert; evaluation token equals routing task',
        'normalization':'original global train-only normalization and metric scales unchanged',
        'checkpoint_rule':'fixed epoch160; no validation/final checkpoint selection or hyperparameter tuning',
        'prediction_conditions':['oracle_task','original_dynamic_core_a_task'],
        'class_policy':'original unrestricted120-class argmax; unsupported/missing/noncausal inputs abstain; no truth fallback',
        'seed_pairs':[[20260825,202610131],[20260826,202610132],[20260827,202610133]],
        'populations':populations,'train_support':support,'episode_and_source_cluster_splits_disjoint':True,
        'shared_updates_per_seed':3200,'expert_bank_updates_per_seed':sum(x['optimizer_updates'] for x in support),
        'sample_presentations_per_seed_each_system':160*1280,
        'capacity_caveat':'expert bank stores five models vs one shared; each input activates one model; total parameter budgets differ',
        'feature_profile_caveat':'Both systems train on identical historical Q0 features and evaluate identical CURRENT_PROFILE_V2 cached features. Their numerical bridge is historically unresolved; this is a comparison under that fixed feature shift.',
        'final_test_caveat':'Previously used fixed final set; this follow-up is not a newly unseen holdout.',
        'metrics':['position_mae','safe_fd_l1','seam_safe_fd_l1'],'aggregation':'anchors within episode, equal episodes within true task, equal five tasks, then mean/sampleSD over three seed systems',
        'bootstrap':{'draws':5000,'seed':2026110702,'unit':'source cluster within true task; verify episode correspondence','models':'fixed paired seeds, no seed resampling'},
        'seal_policy':'all18model receipts fixed, both dev/final predictions sealed before opening either evaluation target archive',
        'source_receipts':source_receipts}
    dest=r/'protocol/STUDY_LOCK.json';dest.parent.mkdir(exist_ok=True,parents=True)
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(lock,indent=2)+'\n')
    with (r/'protocol/TRAIN_SUPPORT.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
    print(json.dumps({'status':'PASS','study_lock_sha256':sha(dest),'train_support':support,'populations':populations},indent=2))
if __name__=='__main__':main()
