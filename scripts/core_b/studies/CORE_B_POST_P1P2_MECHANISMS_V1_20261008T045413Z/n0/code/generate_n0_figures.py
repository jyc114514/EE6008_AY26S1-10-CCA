#!/usr/bin/env python3
"""Generate N0 locked-case and summary figures after accepted branchwise scoring."""
from __future__ import annotations
import os
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
N0_ROOT=HERE.parent
NEXT_ROOT=N0_ROOT.parent
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
P1_ROOT=PROJECT/'reports/benchmarks/CORE_B_P1P2_VISUAL_COMPLEMENTARITY_V1_20261007T053432Z'
V3_ROOT=PROJECT/'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'
SEEDS=(20260950,20260951,20260952)
MODELS=('F-POOL-D','ALT-POOL-R3D18')
CONDITIONS=('REAL','ZERO','PERM','HIST')
COLORS={'REAL':'#2878B5','ZERO':'#E07A27','PERM':'#27966F','HIST':'#555555'}

def load_bundle(path:Path):
    with np.load(path,allow_pickle=False) as z:
        return z['anchor_id'].astype(str),z['predictions_native'].astype(np.float32,copy=True)

def align(ids,source_ids,values):
    lookup={x:i for i,x in enumerate(source_ids)}
    if len(lookup)!=len(source_ids) or set(ids.tolist())!=set(lookup):
        raise RuntimeError('figure input prediction IDs do not equal the locked P1 anchors')
    return values[np.asarray([lookup[x] for x in ids.tolist()],dtype=np.int64)]

def main()->int:
    accepted=json.loads((NEXT_ROOT/'qa/PREDICTION_ACCEPTANCE.json').read_text(encoding='utf-8'))
    if accepted.get('status') not in ('PASS_ALL_6_REAL_REPLAYS_AND_12_INTERVENTIONS','PARTIAL_N0_ACCEPTANCE'):
        raise RuntimeError('N0 figures require branchwise accepted target-blind predictions')
    score=json.loads((N0_ROOT/'results/N0_SCORE_QA.json').read_text(encoding='utf-8'))
    if score.get('status') not in ('PASS_EXPLORATORY_N0_SCORING','PARTIAL_EXPLORATORY_N0_SCORING'):
        raise RuntimeError('N0 figures require the separate development scorer')
    real_keys={(r['model'],int(r['seed'])) for r in accepted.get('real_replays',[])}
    probe_keys={(r['model'],int(r['seed']),r['condition']) for r in accepted.get('interventions',[])}
    valid_by_model={m:sorted(s for mm,s in real_keys if mm==m and any((mm,s,c) in probe_keys for c in ('ZERO','PERM'))) for m in MODELS}
    available_models=[m for m in MODELS if valid_by_model[m]]
    if not available_models:raise RuntimeError('no accepted N0 model branch available for figures')
    available_seeds=sorted({s for m in MODELS for s in valid_by_model[m]})
    with np.load(V3_ROOT/'data/V3_MAIN_INPUTS.npz',allow_pickle=False) as z:
        rows=np.flatnonzero(z['split_code']==1)
        ids=z['anchor_id'][rows].astype(str)
        episodes=z['episode_index'][rows].astype(np.int64)
        tasks=z['original_task_index'][rows].astype(np.int64)
        anchors=z['anchor_command_native'][rows].astype(np.float32)
        target=z['targets_native'][rows,:30].astype(np.float32)
    sigma=np.asarray(json.loads((V3_ROOT/'data/V3_NORMALIZERS.json').read_text())['target_sigma'],dtype=np.float32)
    group_map={}
    with (P1_ROOT/'inventory/SOURCE_GROUP_MAP.csv').open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split']=='dev':group_map[int(r['episode_index'])]=r['source_group_id']
    groups=np.asarray([group_map[int(e)] for e in episodes],dtype='U64')
    selection=list(csv.DictReader((P1_ROOT/'qualitative/CASE_MANIFEST.csv').open(newline='',encoding='utf-8')))
    if len(selection)!=14 or len({r['anchor_id'] for r in selection})!=14:
        raise RuntimeError('locked 14-case P1 selection schema failure')
    channels=json.loads((P1_ROOT/'qualitative/CHANNEL_SELECTION.json').read_text())['selected_indices']
    if channels != [14,17,2,9]:raise RuntimeError('locked channel order changed')
    id_to_row={x:i for i,x in enumerate(ids.tolist())}
    if len(id_to_row)!=len(ids):raise RuntimeError('dev IDs are not unique')
    hist_by_seed={}
    primary=list(csv.DictReader((P1_ROOT/'predictions/PRIMARY_MANIFEST.csv').open(newline='',encoding='utf-8')))
    for seed in available_seeds:
        r=next(x for x in primary if x['condition']=='S-HIST' and int(x['seed'])==seed)
        p=Path(r['bundle_path']);p=p if p.is_absolute() else P1_ROOT/p
        source_ids,values=load_bundle(p);hist_by_seed[seed]=align(ids,source_ids,values)
    predictions={}
    for model in available_models:
        for seed in valid_by_model[model]:
            path=N0_ROOT/f'predictions/REAL_REPLAY/{model}-H30-S{seed}-REAL.npz'
            source_ids,values=load_bundle(path);predictions[(model,seed,'REAL')]=align(ids,source_ids,values)
            for condition in ('ZERO','PERM'):
                if (model,seed,condition) not in probe_keys:continue
                path=N0_ROOT/f'predictions/{model}-H30-S{seed}-{condition}.npz'
                source_ids,values=load_bundle(path);predictions[(model,seed,condition)]=align(ids,source_ids,values)
    out_data=N0_ROOT/'figure_data/N0_CASE_FIGURE_DATA.csv'
    out_data.parent.mkdir(parents=True,exist_ok=True)
    perm_rows=list(csv.DictReader((NEXT_ROOT/'protocol/PAIR_PERMUTATION.csv').open(newline='',encoding='utf-8')))
    perm_by_id={r['recipient_anchor_id']:r for r in perm_rows}
    data_rows=[]
    for case in selection:
        aid=case['anchor_id'];i=id_to_row[aid];slot=int(case['render_slot']);permrow=perm_by_id[aid]
        for action in channels:
            for seed in available_seeds:
                for channel_model in ('S-HIST',)+MODELS:
                    if channel_model=='S-HIST':
                        pred=hist_by_seed[seed][i];condition='HIST'
                    else:
                        if seed not in valid_by_model[channel_model]:continue
                        for condition in ('REAL','ZERO','PERM'):
                            if (channel_model,seed,condition) not in predictions:continue
                            pred=predictions[(channel_model,seed,condition)][i]
                            for step in range(30):
                                data_rows.append({'render_slot':slot,'anchor_id':aid,'episode_id':case['episode_id'],'episode_index':int(episodes[i]),'task':int(tasks[i]),
                                    'source_group_id':str(groups[i]),'donor_anchor_id':permrow['donor_anchor_id'] if condition=='PERM' else '',
                                    'donor_source_group_id':permrow['donor_source_group_id'] if condition=='PERM' else '',
                                    'model':channel_model,'seed':seed,'condition':condition,'horizon_step':step+1,'action_index':action,
                                    'target_std_displacement':float((target[i,step,action]-anchors[i,action])/sigma[action]),
                                    'prediction_std_displacement':float((pred[step,action]-anchors[i,action])/sigma[action]),'final_test_access':0})
                        continue
                    for step in range(30):
                        data_rows.append({'render_slot':slot,'anchor_id':aid,'episode_id':case['episode_id'],'episode_index':int(episodes[i]),'task':int(tasks[i]),
                            'source_group_id':str(groups[i]),'donor_anchor_id':'','donor_source_group_id':'','model':'S-HIST','seed':seed,'condition':'HIST',
                            'horizon_step':step+1,'action_index':action,'target_std_displacement':float((target[i,step,action]-anchors[i,action])/sigma[action]),
                            'prediction_std_displacement':float((pred[step,action]-anchors[i,action])/sigma[action]),'final_test_access':0})
    expected_csv=io.StringIO(newline='')
    writer=csv.DictWriter(expected_csv,fieldnames=list(data_rows[0]));writer.writeheader();writer.writerows(data_rows)
    expected_csv_text=expected_csv.getvalue()
    if out_data.exists():
        with out_data.open('r',newline='',encoding='utf-8') as existing_handle:existing_csv_text=existing_handle.read()
        if existing_csv_text!=expected_csv_text:raise RuntimeError('existing figure_data differs from the locked accepted inputs; preserving it')
    else:
        out_data.write_text(expected_csv_text,encoding='utf-8',newline='')
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7})
    figdir=N0_ROOT/'figures';capdir=N0_ROOT/'captions'
    figdir.mkdir(parents=True,exist_ok=True);capdir.mkdir(parents=True,exist_ok=True)
    cap_records=[]
    for case in selection:
        aid=case['anchor_id'];i=id_to_row[aid];slot=int(case['render_slot']);permrow=perm_by_id[aid]
        out=figdir/f'N0_CASE_{slot:02d}_{aid}.png'
        cap_path=capdir/f'N0_CASE_{slot:02d}_{aid}.md'
        if out.exists() or cap_path.exists():
            if not (out.is_file() and cap_path.is_file()) or out.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':
                raise RuntimeError(f'incomplete or invalid preserved case artifact: {out}')
            if aid not in cap_path.read_text(encoding='utf-8'):
                raise RuntimeError(f'preserved case caption does not identify locked anchor: {cap_path}')
            cap_records.append({'render_slot':slot,'anchor_id':aid,'figure_path':str(out.relative_to(N0_ROOT)),'caption_path':str(cap_path.relative_to(N0_ROOT)),
                'accepted_models':';'.join(available_models),'accepted_seeds_by_model':json.dumps(valid_by_model,sort_keys=True),
                'donor_anchor_id':permrow['donor_anchor_id'],'donor_group_differs':permrow['donor_source_group_id']!=permrow['recipient_source_group_id'],
                'frames_decoded':0,'final_test_access':0})
            continue
        fig,axs=plt.subplots(4,2,figsize=(15,12),sharex=True)
        for ci,action in enumerate(channels):
            for mi,model in enumerate(MODELS):
                ax=axs[ci,mi]
                seeds=valid_by_model[model]
                if not seeds:
                    ax.text(.5,.5,'branch not accepted',ha='center',va='center',transform=ax.transAxes);ax.axis('off');continue
                gt=(target[i,:,action]-anchors[i,action])/sigma[action]
                ax.plot(np.arange(1,31),gt,color='black',linewidth=1.6,linestyle='--',label='Ground truth' if ci==0 and mi==0 else None)
                hist_lines=[(hist_by_seed[seed][i,:,action]-anchors[i,action])/sigma[action] for seed in seeds]
                ax.plot(np.arange(1,31),np.mean(hist_lines,axis=0),color=COLORS['HIST'],linewidth=1.4,linestyle=':',label='S-HIST mean' if ci==0 and mi==0 else None)
                for condition in ('REAL','ZERO','PERM'):
                    used=[seed for seed in seeds if (model,seed,condition) in predictions]
                    if not used:continue
                    lines=[]
                    for seed in used:
                        y=(predictions[(model,seed,condition)][i,:,action]-anchors[i,action])/sigma[action]
                        lines.append(y);ax.plot(np.arange(1,31),y,color=COLORS[condition],alpha=.18,linewidth=.7)
                    ax.plot(np.arange(1,31),np.mean(lines,axis=0),color=COLORS[condition],linewidth=1.5,label=f'{condition} mean (n={len(used)})' if ci==0 and mi==0 else None)
                ax.set_title(f'{model} · action index {action} · accepted seeds {seeds}')
                ax.set_ylabel('standardized displacement');ax.grid(True,alpha=.2)
                if ci==3:ax.set_xlabel('future step')
        donor_anchor_id = permrow['donor_anchor_id']
        fig.suptitle(f'N0 cross-group content intervention · case {slot:02d} · {aid}\nPERM donor {donor_anchor_id} from a different source group', fontsize=12)
        handles=[];labels=[]
        for ax in axs.flat:
            h,l=ax.get_legend_handles_labels()
            for hh,ll in zip(h,l):
                if ll not in labels:labels.append(ll);handles.append(hh)
        if handles:fig.legend(handles,labels,loc='upper center',ncol=5,bbox_to_anchor=(.5,.97),frameon=False)
        fig.tight_layout(rect=(0,.02,1,.93))
        fig.savefig(out,dpi=160,bbox_inches='tight');plt.close(fig)
        seed_note='; '.join(f'{m}: {valid_by_model[m]}' for m in MODELS)
        caption=f'''# N0 case {slot:02d}: {aid}\n\n**Description:** Locked development anchor with ground truth and fixed-checkpoint REAL/ZERO/PERM predicted standardized displacements for numeric action indices {channels}. Thin curves are fixed-seed predictions; thick curves are descriptive seed means. Accepted seeds: {seed_note}.\n\n**PERM donor:** {permrow['donor_anchor_id']} (episode {permrow['donor_episode_index']}), from a different source group than the recipient. No donor frame or new video was decoded.\n\n**Scope:** Development-only functional-sensitivity diagnostic. ZERO/PERM are off-manifold input changes; curves do not establish causal or independent information value. Channel anatomy/physical units remain unresolved. FINAL_HELDOUT_ACCESS=0.\n'''
        cap_path.write_text(caption,encoding='utf-8')
        cap_records.append({'render_slot':slot,'anchor_id':aid,'figure_path':str(out.relative_to(N0_ROOT)),'caption_path':str(cap_path.relative_to(N0_ROOT)),
            'accepted_models':';'.join(available_models),'accepted_seeds_by_model':json.dumps(valid_by_model,sort_keys=True),
            'donor_anchor_id':permrow['donor_anchor_id'],'donor_group_differs':permrow['donor_source_group_id']!=permrow['recipient_source_group_id'],'frames_decoded':0,'final_test_access':0})

    dep=list(csv.DictReader((N0_ROOT/'results/N0_OUTPUT_DEPENDENCE.csv').open(newline='',encoding='utf-8')))
    err=list(csv.DictReader((N0_ROOT/'results/N0_ERROR_SHIFTS.csv').open(newline='',encoding='utf-8')))
    fig,axs=plt.subplots(2,3,figsize=(14,7))
    for ri,model in enumerate(MODELS):
        for ci,metric in enumerate(('D_out','D_level','D_shape')):
            ax=axs[ri,ci]
            conds=[c for c in ('ZERO','PERM') if any(r['model']==model and r['condition']==c for r in dep)]
            vals=[[float(r[metric]) for r in dep if r['model']==model and r['condition']==cond] for cond in conds]
            if vals:ax.boxplot(vals,tick_labels=conds,showfliers=False,whis=(5,95))
            else:ax.text(.5,.5,'branch not accepted',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(f'{model}: {metric}');ax.set_ylabel('standardized output displacement');ax.grid(axis='y',alpha=.2)
    fig.suptitle('N0 functional sensitivity distributions across accepted dev anchors/seeds')
    fig.tight_layout()
    summary_paths=[];out=figdir/'N0_DOUT_LEVEL_SHAPE_DISTRIBUTIONS.png';fig.savefig(out,dpi=170,bbox_inches='tight');plt.close(fig);summary_paths.append(out)
    fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=False)
    for ax,model in zip(axs,MODELS):
        for cond,color in zip(('REAL','ZERO','PERM','HIST'),(COLORS['REAL'],COLORS['ZERO'],COLORS['PERM'],COLORS['HIST'])):
            rr=[r for r in err if r['model']==model and r['condition']==cond]
            if not rr:continue
            means=[float(r['episode_equal_mae_std']) for r in rr]
            ax.plot([cond]*len(means),means,marker='o',linestyle='none',color=color,alpha=.55)
            ax.plot([cond],[float(np.mean(means))],marker='D',markersize=7,color=color)
        ax.set_title(model);ax.set_ylabel('episode-equal Full28 standardized MAE');ax.grid(axis='y',alpha=.2)
    fig.suptitle('REAL / ZERO / PERM / HIST error on the same development population')
    fig.tight_layout()
    out=figdir/'N0_REAL_ZERO_PERM_HIST_ERRORS.png';fig.savefig(out,dpi=170,bbox_inches='tight');plt.close(fig);summary_paths.append(out)
    fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,model in zip(axs,MODELS):
        for cond,color in zip(('ZERO','PERM'),(COLORS['ZERO'],COLORS['PERM'])):
            rr=sorted([r for r in err if r['model']==model and r['condition']==cond],key=lambda r:int(r['seed']))
            if rr:ax.plot([int(r['seed']) for r in rr],[float(r['deltaL_episode_equal_vs_REAL']) for r in rr],marker='o',color=color,label=cond)
        ax.axhline(0,color='black',linewidth=.8)
        ax.axhline(.0022313516970703788,color='#777777',linewidth=.8,linestyle='--',label='P1 display delta (not margin)')
        ax.axhline(-.0022313516970703788,color='#777777',linewidth=.8,linestyle='--')
        ax.set_title(model);ax.set_xlabel('fixed seed');ax.set_ylabel('probe − REAL episode-equal MAE');ax.grid(alpha=.2)
        if ax.lines:ax.legend(frameon=False)
    fig.suptitle('N0 descriptive error shifts; positive values mean the probe is worse')
    fig.tight_layout()
    out=figdir/'N0_DELTA_MAE_BY_SEED.png';fig.savefig(out,dpi=170,bbox_inches='tight');plt.close(fig);summary_paths.append(out)
    descriptions=[
        'Anchor/seed distributions of D_out, D_level and D_shape under the locked zero-content and cross-group permutation probes; no confidence intervals or p-values.',
        'Episode-equal Full28 standardized MAE on the same development population; P1 S-HIST is a repeatedly exposed historical reference.',
        'Per-seed episode-equal probe-minus-REAL error; positive means worse. The dashed P1 display delta is not a functional equivalence threshold.'
    ]
    for path,description in zip(summary_paths,descriptions):
        (capdir/(path.stem+'.md')).write_text(f'# {path.name}\n\n{description}\n\nDevelopment-only functional sensitivity; no causal or independent-information claim. Numeric action indices only; units unresolved. FINAL_HELDOUT_ACCESS=0.\n',encoding='utf-8')
    qa={'status':'generated_pending_visual_inspection','case_figures':len(cap_records),'summary_figures':3,'figure_data_rows':len(data_rows),
        'figure_data_path':str(out_data.relative_to(N0_ROOT)),'case_manifest_source_sha256':hashlib.sha256((P1_ROOT/'qualitative/CASE_MANIFEST.csv').read_bytes()).hexdigest(),
        'channel_indices':channels,'frames_decoded':0,'final_test_access':0,'final_heldout_access':0,'models':list(MODELS),'seeds':list(SEEDS),
        'accepted_models':available_models,'accepted_seeds_by_model':valid_by_model,'conditions':['REAL','ZERO','PERM','S-HIST'],'human_review_status':'pending'}
    visual_path=NEXT_ROOT/'qa/VISUAL_QA.json'
    if visual_path.exists():raise FileExistsError(visual_path)
    visual_path.write_text(json.dumps(qa,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    manifest_path=NEXT_ROOT/'qa/CASE_FIGURE_MANIFEST.csv'
    if manifest_path.exists():raise FileExistsError(manifest_path)
    with manifest_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(cap_records[0]));w.writeheader();w.writerows(cap_records)
    print(json.dumps({'status':qa['status'],'case_figures':len(cap_records),'summary_figures':3,'figure_data_rows':len(data_rows),
        'accepted_models':available_models,'final_test_access':0},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
