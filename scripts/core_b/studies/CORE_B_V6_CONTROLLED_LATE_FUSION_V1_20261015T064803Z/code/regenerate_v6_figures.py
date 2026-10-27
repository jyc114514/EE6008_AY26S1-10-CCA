#!/usr/bin/env python3
"""Rebuild all V6 figures from sealed numerical tables and saved trajectory data only."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import v6_score as score

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not (ROOT / 'qa/PREDICTION_SEAL.json').is_file() or not (ROOT / 'results/PRIMARY_TASK_SEED.csv').is_file():
        raise RuntimeError('figure rebuild requires an accepted prediction seal and saved score tables')
    if not (ROOT / 'results/PER_WINDOW_NEW_DEV_LOSSES.csv').is_file():
        raise RuntimeError('figure rebuild refuses missing seal-bound new loss table')
    alpha=score.read_csv(ROOT/'results/INNER_TUNE_ALPHA_CURVES.csv')
    primary=score.read_csv(ROOT/'results/PRIMARY_TASK_SEED.csv')
    pairs=score.read_csv(ROOT/'results/PAIRED_EFFECTS.csv')
    cases=score.read_csv(ROOT/'figure_data/TRAJECTORY_CASES.csv')
    trajectory=score.read_csv(ROOT/'figure_data/TRAJECTORY_DATA.csv')
    score.make_figures(alpha,primary,pairs,cases+trajectory)
    names=['ALPHA_SELECTION_CURVES.png','DUAL_METRIC_AND_SEAM.png',
           'PRIMARY_CONTRASTS_BY_TASK.png','TRAJECTORY_CASES.png']
    files=[]
    for name in names:
        p=ROOT/'figures'/name
        if not p.is_file() or p.stat().st_size < 1000:
            raise RuntimeError(f'figure missing or unexpectedly small: {name}')
        files.append({'path':str(p),'sha256':score.sha_file(p),'bytes':p.stat().st_size})
    receipt={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
             'prediction_seal_sha256':score.sha_file(ROOT/'qa/PREDICTION_SEAL.json'),
             'primary_task_seed_sha256':score.sha_file(ROOT/'results/PRIMARY_TASK_SEED.csv'),
             'paired_effects_sha256':score.sha_file(ROOT/'results/PAIRED_EFFECTS.csv'),
             'alpha_curve_sha256':score.sha_file(ROOT/'results/INNER_TUNE_ALPHA_CURVES.csv'),
             'trajectory_data_sha256':score.sha_file(ROOT/'figure_data/TRAJECTORY_DATA.csv'),
             'target_archive_reads_during_figure_generation':0,'figures':files,
             'human_visual_review_status':'pending'}
    score.write_json(ROOT/'qa/FIGURE_BUILD_QA.json',receipt)
    print(json.dumps({'status':'PASS','figures':len(files),'target_archive_reads':0},sort_keys=True))


if __name__=='__main__':
    main()
