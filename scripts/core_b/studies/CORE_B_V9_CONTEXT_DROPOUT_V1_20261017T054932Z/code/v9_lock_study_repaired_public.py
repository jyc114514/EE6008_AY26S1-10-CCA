#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-009; source SHA-256: afb51ec07930528c1fa71050c1fa275fbebce136fa88744f2badb341eda79dbf
from __future__ import annotations
import csv, hashlib, json, os, shutil, zipfile
from datetime import datetime, timezone
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("EE6008_V9_STUDY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT = Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
V2 = PROJECT/'reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z'
V4 = PROJECT/'reports/benchmarks/CORE_B_V4_RGB_ALIGNMENT_REASSIGNMENT_V1_20261014T083805Z'
V5 = PROJECT/'reports/benchmarks/CORE_B_V5_LEVEL_SHAPE_COMPOSITION_V1_20261015T053806Z'
V7 = PROJECT/'reports/benchmarks/CORE_B_V7_ENSEMBLE_TARGET_DISTILLATION_V1_20261015T112834Z'
V8 = PROJECT/'reports/benchmarks/CORE_B_V8_CONTROLLED_REPLAY_V1_20261016T045154Z'
TASKS = (119,94,90,141,230)
SEEDS = (202610131,202610132,202610133)
PROGRAMS = {'A0':('STATE-CTRL',0.0),'AD':('STATE-CTRL',0.5),'V0':('TS11-D',0.0),'VD':('TS11-D',0.5)}

def utc():
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def writej(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.part')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)
def csv_write(path, rows, fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise RuntimeError('empty CSV: '+str(path))
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
    tmp=path.with_name('.'+path.name+'.part')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def must_file(path, expected=None):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    actual=sha(path)
    if expected and actual!=expected: raise RuntimeError('source hash mismatch for '+str(path))
    return {'path':str(path),'sha256':actual,'bytes':path.stat().st_size,'read_only':True}
def maxrun(bits):
    best=run=0
    for bit in bits:
        if bit: run+=1;best=max(best,run)
        else: run=0
    return best

def main():
    if (ROOT/'protocol/STUDY_LOCK.json').exists() or (ROOT/'protocol/RECIPE_LOCK.json').exists():
        raise SystemExit('V9 locks already exist; refusing to rewrite')
    owner=readj(ROOT/'coordination/OWNER.json')
    if owner.get('status')!='ACTIVE' or owner.get('owner_count')!=1: raise RuntimeError('single V9 owner lock missing')
    token=(ROOT/'coordination/.owner_token').read_text().strip()
    if hashlib.sha256(token.encode()).hexdigest()!=owner.get('token_sha256'): raise RuntimeError('owner token mismatch')

    v8_recipe=readj(V8/'protocol/RECIPE_LOCK.json')
    v8_map=readj(V8/'inventory/SOURCE_MAP.json')
    v8_fold=readj(V8/'inventory/FOLD_REUSE.json')
    v8_exposure=readj(V8/'inventory/OUTCOME_EXPOSURE.json')
    if v8_recipe.get('status')!='LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV' or v8_fold.get('status')!='PASS':
        raise RuntimeError('accepted V8 locks unavailable')
    sampler_src=V8/'protocol/SAMPLER_ORDERS.npz'
    sampler_expected=v8_recipe['training']['sampler_archive_sha256']
    sampler_dst=ROOT/'protocol/SAMPLER_ORDERS.npz'
    shutil.copy2(sampler_src,sampler_dst)
    sampler_sha=sha(sampler_dst)
    if sampler_sha!=sampler_expected: raise RuntimeError('copied locked sampler hash differs')
    sampler_manifest=readj(V8/'protocol/SAMPLER_SEQUENCE_MANIFEST.json')
    init_map=v8_recipe['fresh_init_by_seed']

    train_path=V2/'data/TRAIN_SELECTED_H30.npz'
    with np.load(train_path,allow_pickle=False) as z:
        required={'anchor_id','episode_id','task_id','source_cluster','task_rank','cache_index','anchor_frame','history_native','history_mask','relative_delta','anchor_native','target_native_h30','inner_role'}
        if not required.issubset(z.files): raise RuntimeError('accepted train archive schema differs')
        ids=z['anchor_id'].astype(str).copy()
        episode=z['episode_id'].astype(str).copy()
        task=z['task_id'].astype(np.int64).copy()
        source=z['source_cluster'].astype(str).copy()
        history_shape=list(z['history_native'].shape)
        if history_shape!=[1280,32,71]: raise RuntimeError('accepted receiver state shape differs: '+str(history_shape))
    if len(set(ids.tolist()))!=1280: raise RuntimeError('train anchor IDs not unique')
    if set(task.tolist())!=set(TASKS): raise RuntimeError('train tasks differ from lock')
    receivers=[(str(source[i]),str(episode[i]),int(task[i]),str(ids[i])) for i in range(len(ids))]
    if len(set(receivers))!=1280: raise RuntimeError('canonical receiver key is not unique')
    source_episode_task={(k[0],k[1],k[2]) for k in receivers}
    if len(source_episode_task)!=220: raise RuntimeError('accepted train receiver episodes differ from 220')
    correction=readj(V2/'protocol/INNER_SPLIT_CORRECTION.json')
    if correction.get('status')!='APPLIED' or 'parent cluster identity equals this key' not in correction.get('canonical_source_key',''):
        raise RuntimeError('canonical source key mapping not confirmed by V2 correction')

    # Construct the fixed receiver bank with SHA only; no numpy/torch RNG is used.
    episode_rows={}
    for i,key in enumerate(receivers):
        episode_rows.setdefault((key[2],key[0],key[1]),[]).append(i)
    mask_rates=[];episode_rows_out=[];receiver_rows=[];bank_summaries=[]
    bank_hashes={}
    for seed in SEEDS:
        bank=np.empty((160,1280),dtype=np.bool_)
        for e0 in range(160):
            for i,key in enumerate(receivers):
                receiver_key=[key[0],key[1],int(key[2]),key[3]]
                payload=json.dumps(['v9','numctx',int(seed),int(e0),receiver_key],ensure_ascii=False,separators=(',',':')).encode('utf-8')
                u64=int(hashlib.sha256(payload).hexdigest()[:16],16)
                bank[e0,i]=(u64<2**63)
        schedule_sha=hashlib.sha256(np.ascontiguousarray(bank).tobytes(order='C')).hexdigest()
        bank_path=ROOT/'mask_banks'/f'mask_seed{seed}.npz'
        np.savez_compressed(bank_path,drop_bits=bank,anchor_ids=ids.astype('U64'),canonical_source_key=source.astype('U128'),canonical_episode_id=episode.astype('U32'),task_id=task.astype(np.int64))
        bank_file_sha=sha(bank_path)
        bank_hashes[str(seed)]={'path':str(bank_path),'file_sha256':bank_file_sha,'bits_sha256':schedule_sha,'shape':list(bank.shape),'dtype':'bool','same_bank_for_AD_and_VD':True}
        task_rates={}
        for e0 in range(160):
            for tid in TASKS:
                idx=np.flatnonzero(task==tid)
                episode_keys=sorted(k for k in episode_rows if k[0]==tid)
                dropped=int(bank[e0,idx].sum())
                covered=sum(bool(bank[e0,episode_rows[k]].any()) for k in episode_keys)
                mask_rates.append({'seed':seed,'epoch0':e0,'task_id':tid,'receiver_n':len(idx),'dropped_n':dropped,'drop_rate':dropped/len(idx),'episode_n':len(episode_keys),'episodes_with_any_drop':covered,'episode_coverage_rate':covered/len(episode_keys)})
        per_receiver_run=np.asarray([maxrun(bank[:,i].tolist()) for i in range(1280)],dtype=np.int16)
        for (tid,sk,ep),idxs in sorted(episode_rows.items()):
            arr=bank[:,idxs]
            episode_rows_out.append({'seed':seed,'task_id':tid,'canonical_source_key':sk,'canonical_episode_id':ep,'anchor_n':len(idxs),
                'schedule_drop_n':int(arr.sum()),'schedule_possible_n':int(arr.size),'schedule_drop_rate':float(arr.mean()),
                'epochs_with_any_receiver_dropped':int(arr.any(axis=1).sum()),'episode_epoch_coverage':float(arr.any(axis=1).mean()),
                'max_contiguous_drop_epochs_any_receiver':int(max(per_receiver_run[i] for i in idxs)),
                'mean_receiver_max_contiguous_drop_epochs':float(np.mean([per_receiver_run[i] for i in idxs]))})
        for i,key in enumerate(receivers):
            receiver_rows.append({'seed':seed,'task_id':key[2],'canonical_source_key':key[0],'canonical_episode_id':key[1],'anchor_id':key[3],
                'epochs':160,'dropped_epochs':int(bank[:,i].sum()),'drop_rate':float(bank[:,i].mean()),
                'max_contiguous_drop_epochs':int(per_receiver_run[i])})
        task_rates[str(seed)]={str(t):float(bank[:,task==t].mean()) for t in TASKS}
        bank_summaries.append({'seed':seed,'overall_drop_rate':float(bank.mean()),'drop_rate_by_task':task_rates,
            'receiver_n':1280,'episode_n':220,'max_contiguous_drop_epochs':int(per_receiver_run.max())})
    csv_write(ROOT/'mask_banks'/'MASK_TASK_EPOCH_STATS.csv',mask_rates)
    csv_write(ROOT/'mask_banks'/'MASK_EPISODE_STATS.csv',episode_rows_out)
    csv_write(ROOT/'mask_banks'/'MASK_RECEIVER_STATS.csv',receiver_rows)

    attachment=Path(os.environ["EE6008_V9_TASK_ATTACHMENT"]).expanduser().resolve()
    dev_archive=V2/'data/DEV_TARGETS_SCORER_ONLY.npz'
    v5_exposure=readj(V5/'inventory/OUTCOME_EXPOSURE.json')
    # The dev target payload is deliberately not opened or hashed before prediction seal.
    if dev_archive.stat().st_size!=int(v5_exposure['target_archive_bytes']): raise RuntimeError('dev target archive bytes differ from already accepted exposure')
    assets={
        'v9_user_instruction':{'path':str(attachment),'sha256':sha(attachment),'read_only':True},
        'train_npz':must_file(train_path,v8_map['source_assets']['train_npz']['sha256']),
        'train_prefix_npz':must_file(V7/'data/TRAIN_PREFIXES_NO_LABELS.npz',v8_map['source_assets']['train_prefix_npz']['sha256']),
        'dev_prefix_npz':must_file(V4/'data/DEV_PREFIXES_NO_LABELS.npz',v8_map['source_assets']['dev_prefix_npz']['sha256']),
        'feature_cache_manifest':must_file(v8_map['source_assets']['feature_cache_manifest']['path'],v8_map['source_assets']['feature_cache_manifest']['sha256']),
        'normalizer':must_file(v8_map['source_assets']['normalizer']['path'],v8_map['source_assets']['normalizer']['sha256']),
        'metric_scale_lock':must_file(v8_map['source_assets']['metric_scale_lock']['path'],v8_map['source_assets']['metric_scale_lock']['sha256']),
        'split':must_file(V2/'protocol/INNER_SPLIT.csv',v8_map['source_assets']['split']['sha256']),
        'split_correction':must_file(V2/'protocol/INNER_SPLIT_CORRECTION.json'),
        'sampler_orders':{'path':str(sampler_dst),'sha256':sampler_sha,'upstream_path':str(sampler_src),'upstream_sha256':sampler_expected,'read_only_upstream':True},
        'dev_target_archive':{'path':str(dev_archive),'sha256_expected_from_accepted_V5_exposure':v5_exposure['target_archive_sha256'],
            'bytes_expected_from_accepted_V5_exposure':int(v5_exposure['target_archive_bytes']),'opened_before_prediction_seal':False},
        'historical_per_window':must_file(V7/'results/PER_WINDOW_ALL_METHODS.csv'),
        'accepted_S_OPT_dev_predictions':{},
        'v8_hard_video_reference':{},
        'watcher':must_file(v8_map['source_assets']['watcher']['path'],v8_map['source_assets']['watcher']['sha256'])
    }
    for seed in SEEDS:
        sdir=V2/'predictions/dev'/f'v2_formal_G_S_lr0p001_s{seed}'
        pred=sdir/'PREDICTIONS.safetensors';rows=sdir/'PREDICTION_ROWS.csv';complete=sdir/'PREDICTION_COMPLETE.json'
        rec=readj(complete)
        if rec.get('status')!='PASS' or int(rec.get('seed',-1))!=seed or rec.get('dataset_kind')!='dev' or int(rec.get('rows',-1))!=185: raise RuntimeError('accepted S_OPT dev prediction receipt mismatch')
        pred=Path(rec['prediction_path']); rows=Path(rec['row_ids_path'])
        assets['accepted_S_OPT_dev_predictions'][str(seed)]={'prediction':must_file(pred,rec['prediction_sha256']),'rows':must_file(rows,rec['row_ids_sha256']),
            'complete':must_file(complete),'model_path':rec.get('model_path'),'model_sha256':rec.get('model_sha256'),'receipt_prediction_sha256':rec['prediction_sha256']}
        hdir=V8/'model_only/GT_RETRAIN'/f'seed{seed}'/'attempt_01'
        hman=readj(hdir/'MODEL_MANIFEST.json')
        assets['v8_hard_video_reference'][str(seed)]={'model':must_file(hdir/'model.safetensors',hman['model_file_sha256']),
            'manifest':must_file(hdir/'MODEL_MANIFEST.json'),'dev_prediction':must_file(V8/'inference'/f'v8_pred_dev_GT_RETRAIN_s{seed}.json') if (V8/'inference'/f'v8_pred_dev_GT_RETRAIN_s{seed}.json').is_file() else None}
    roots={'V2':str(V2),'V4':str(V4),'V5':str(V5),'V7':str(V7),'V8':str(V8)}
    source_map={'schema_version':1,'status':'LOCKED_READ_ONLY_SOURCES','created_utc':utc(),'source_roots':roots,'source_assets':assets,
        'v8_source_map_path':str(V8/'inventory/SOURCE_MAP.json'),'v8_source_map_sha256':sha(V8/'inventory/SOURCE_MAP.json'),
        'v8_checklist_path':str(V8/'CHECKSUMS.sha256'),'v8_checklist_sha256':sha(V8/'CHECKSUMS.sha256'),
        'repo_root':str(PROJECT/'repo/ee6008-humanoid-vjepa'),'repo_git_metadata':'absent; no commit/branch claim',
        'v9_code_sources':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'code').rglob('*.py'))},
        'old_roots_written':False}
    writej(ROOT/'inventory/SOURCE_MAP.json',source_map)

    fold=dict(v8_fold)
    fold.update({'v9_reuse_source':'V2 corrected literal membership and accepted ordered IDs; no repartition/re-hash',
        'v9_train_anchors':1280,'v9_train_episodes':220,'v9_dev_anchors':185,'v9_dev_episodes':47,
        'v9_formal_seeds':list(SEEDS),'final_heldout_payload_access':0})
    writej(ROOT/'inventory/FOLD_REUSE.json',fold)
    exposure={'schema_version':1,'status':'locked_pre_fit','study':'post_V8_training_only_numeric_context_dropout_development',
        'prior_v5_v6_v7_v8_known_grain_primary_loss_evaluations':9435,'prior_unique_dev_windows':185,
        'legacy_35523_count':'unit_unverified_excluded','v9_expected_dev_logic_bundles':30,
        'v9_expected_primary_dev_logic_bundles':18,'v9_expected_masked_dev_logic_bundles':12,
        'v9_expected_new_dev_loss_evaluations':5550,'v9_expected_dev_task_views':150,
        'v9_dev_target_archive_content_reads_before_seal':0,'v9_dev_target_archive_content_reads_expected_after_seal':1,
        'v9_final_heldout_payload_access':0,'v9_final_task_level_heldout_aggregate_access':0,
        'train_target_role':'TRUE_Y_ONLY','train_Y_archive_loaded_by_formal_fits_expected':12,
        'train_Y_presentations_expected':12*160*1280,'target_forward_or_mask_input_access':False}
    writej(ROOT/'inventory/OUTCOME_EXPOSURE.json',exposure)
    bank_json={'schema_version':1,'status':'LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV','rule_version':'V9_NUMCTX_SHA256_V1',
        'probability':0.5,'epoch_index':'0-based 0..159','payload':'json(["v9","numctx",seed,epoch0,[canonical_source_key,canonical_episode_id,task_id,anchor_id]], ensure_ascii=False, separators=(",", ":"))',
        'hash':'SHA256(payload UTF-8)','bit':'int(first_16_hex,16) < 2**63','programs':{
            'AD':{'bank':'same seed bank','apply_bank_bits':True},'VD':{'bank':'same seed bank','apply_bank_bits':True},
            'A0':{'bank':'same seed bank','apply_bank_bits':False},'V0':{'bank':'same seed bank','apply_bank_bits':False}},
        'canonical_key_fields':{'canonical_source_key':'accepted source_cluster/split_assignment_key','canonical_episode_id':'accepted episode_id stored as string',
            'task_id':'accepted integer task ID','anchor_id':'accepted ordered anchor ID'},
        'unique_receiver_keys':1280,'bank_hashes':bank_hashes,'summaries':bank_summaries,
        'numpy_rng_used':False,'torch_rng_used':False,'cuda_rng_used':False,'dev_receiver_bits_generated':False,
        'input_policy_at_evaluation':'primary full_context; secondary force all numeric context receiver blocks to zero'}
    writej(ROOT/'protocol/MASK_SCHEDULE.json',bank_json)
    writej(ROOT/'protocol/MASK_FIELD_MAP.json',{
        'schema_version':1,'status':'LOCKED_BEFORE_FORMAL_FITS','source_input_adapter':'V8 accepted input_batch path; V9 local copy',
        'numeric_context_source_field':'history_native','model_receiver_key':'prefix_inputs["state"]',
        'source_shape':[1280,32,71],'model_shape':'[B,32,71]',
        'numeric_transform':'(history_native - state_mean) / state_safe_std; V8 accepted normalizer; float32',
        'mask_location':'out-of-place after normalization, before V2 state_projection',
        'mask_operation':'state_after = torch.where(drop[:,None,None], torch.zeros_like(state_before), state_before)',
        'whole_receiver_block_masked':True,'observation_action_channel_anatomy_claimed':False,
        'retained_separate_inputs':['history_mask','relative_delta','anchor_command','task_id','dynamic','static'],
        'retained_independent_command_anchor':'anchor_command, shape [B,28], passed into anchor_projection unchanged',
        'retained_task':'task_id and task_projection unchanged',
        'retained_time_and_modality_encoding':'history_mask, relative_delta, sinusoidal positions, spatial encodings, modality embeddings unchanged',
        'video_values_masked':False,'target_Y_masked_or_forwarded':False,
        'other_numeric_value_bypass_found':False,'only_numeric_context_receiver_tensor':'state',
        'force_masked_evaluation':'all receiver state tensors zero; no train bank lookup; command/task/video remain unchanged',
        'state_only_rgb_invariance':'model mode STATE-CTRL replaces video tensors with fixed zeros; RGB cache is not read',
        'independent_full_RNG_mask_generator':True,'final_heldout_payload_access':0})
    study={'schema_version':1,'status':'LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV','study':'post_V8_training_only_numeric_context_dropout_development',
        'programs':{k:{'video_mode':v[0],'numeric_context_drop_probability':v[1],
            'primary_dev_input':'full_context','secondary_dev_input':'force_numeric_context_drop'} for k,v in PROGRAMS.items()},
        'seeds':list(SEEDS),'fresh_init_by_seed':init_map,'targets':'true_Y_only','teacher_targets_used':False,
        'task_ids':list(TASKS),'outer_train':{'anchors':1280,'episodes':220},'dev':{'anchors':185,'episodes':47},
        'horizon':30,'action_dim':28,'numeric_context_receiver_shape':[32,71],
        'known_command_anchor_and_task_always_retained':True,'mask_bank_AD_equals_VD_per_seed':True,
        'A0_V0_mask_rate':0.0,'AD_VD_planned_mask_rate':0.5,'same_sampler_and_initial_rng_within_seed':True,
        'fresh_start_for_all_four_programs':True,'checkpoint':'fixed last epoch 160',
        'formal_fit_count':12,'p_search':0,'lr_search':0,'encoder_retraining_or_feature_extraction':0,
        'final_heldout_payload_access':0,'positive_gain_or_minimum_support_completion_gate':False,
        'locked_utc':utc()}
    writej(ROOT/'protocol/STUDY_LOCK.json',study)
    recipe={'schema_version':1,'status':'LOCKED_BEFORE_FORMAL_FITS_AND_NEW_DEV','programs':study['programs'],'seeds':list(SEEDS),
        'source_recipe':'V8 accepted synchronous VIDEO hard-label recipe; hard target true Y in all four conditions',
        'training':{'epochs':160,'updates_per_epoch':20,'total_updates_per_fit':3200,'effective_batch':64,'microbatch_size':64,'accumulation_steps':1,
            'optimizer':{'name':'AdamW','lr':0.0005,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.0001},
            'gradient_clip_norm':1.0,'scheduler':{'name':'UpdateDrivenCosineAnnealingLR','T_max':3200,'eta_min':1e-5,'step':'optimizer post-step'},
            'dtype':'float32','amp':False,'tf32':False,'cudnn_deterministic':True,'deterministic_algorithms':True,
            'batch_loss':'sum(w_i * (position_L1 + 0.1 * 29_future_internal_difference_L1))/B_actual',
            'sample_weight':'W/(5*E_task*n_episode_windows)','targets':'true_Y_only',
            'checkpoint':'last epoch 160; no dev/inner-tune checkpoint selection',
            'sampler_archive_path':str(sampler_dst),'sampler_archive_sha256':sampler_sha,
            'sampler_source_manifest_sha256':sha(V8/'protocol/SAMPLER_SEQUENCE_MANIFEST.json'),
            'same_seed_rng_and_batch_order_across_four_programs':True,'mask_generator_consumes_training_rng':False},
        'normalizer':assets['normalizer'],'metric_scale_lock':assets['metric_scale_lock'],
        'model_source_hashes':{'v2_model.py':sha(ROOT/'code/v2_model.py'),'D3_model.py':sha(ROOT/'code/locked_source/D3_model.py'),
            'episode_equal_loss_contract.py':sha(ROOT/'code/locked_source/d3_eq_contract.py'),'scheduler.py':sha(ROOT/'code/v2_scheduler.py')},
        'init_by_seed':{str(s):{'path':init_map[str(s)]['path'],'file_sha256':init_map[str(s)]['file_sha256'],'state_hash':init_map[str(s)]['state_hash']} for s in SEEDS},
        'feature_cache_manifest':assets['feature_cache_manifest'],'train_npz':assets['train_npz'],
        'train_prefix_npz':assets['train_prefix_npz'],'dev_prefix_npz':assets['dev_prefix_npz'],
        'split_source':assets['split'],'watcher':assets['watcher'],'fresh_init_shared_within_seed':True,
        'no_teacher_or_soft_target_paths':True,'dev_target_path_not_passed_to_fit_or_inference':True,
        'oom_fallback_registered':{'microbatch_size':32,'accumulation_steps':2,'effective_batch':64},
        'locked_utc':utc()}
    writej(ROOT/'protocol/RECIPE_LOCK.json',recipe)
    print(json.dumps({'status':'LOCKED','root':str(ROOT),'receiver_keys':len(receivers),'mask_banks':bank_hashes,
        'mask_summaries':bank_summaries,'source_assets':len(assets),'study_lock_sha256':sha(ROOT/'protocol/STUDY_LOCK.json'),
        'recipe_lock_sha256':sha(ROOT/'protocol/RECIPE_LOCK.json')},ensure_ascii=False,sort_keys=True))

if __name__=='__main__': main()

