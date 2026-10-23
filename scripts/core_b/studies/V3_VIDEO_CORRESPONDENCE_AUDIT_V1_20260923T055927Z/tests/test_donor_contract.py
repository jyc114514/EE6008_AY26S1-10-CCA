#!/usr/bin/env python3
"""CPU-only verification for outcome-blind donor maps and locked matrix."""
from __future__ import annotations
import os
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa'))
V3=REPO/"reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"

def rd(path):
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))

def main():
    maps=rd(ROOT/"DONOR_MAPS.csv")
    windows=rd(ROOT/"DONOR_WINDOW_MAPS.csv")
    eligible=rd(ROOT/"RECIPIENT_ELIGIBILITY.csv")
    matrix=rd(ROOT/"EVAL_MATRIX_120.csv")
    lock=json.loads((ROOT/"DONOR_MAP_LOCK.json").read_text())
    assert len(matrix)==120 and len(maps)==3468 and len(windows)==13332
    included={int(r["episode_index"]) for r in eligible if r["eligible_common_subset"]=="True"}
    excluded={int(r["episode_index"]) for r in eligible if r["eligible_common_subset"]=="False"}
    assert len(included)==578 and len(excluded)==25
    pair_keys=set()
    for row in maps:
        rec=int(row["recipient_episode"]);don=int(row["donor_episode"])
        assert rec in included and don in included and rec!=don
        assert row["recipient_group_key"]!=row["donor_group_key"]
        if row["pairing_type"]=="same_task":assert row["recipient_task"]==row["donor_task"]
        else:assert row["recipient_task"]!=row["donor_task"]
        pair_keys.add((int(row["map_seed"]),row["pairing_type"],rec))
    assert len(pair_keys)==3*2*len(included)
    for seed in lock["map_seeds"]:
        for kind in ("same_task","cross_task"):
            subset=[r for r in maps if int(r["map_seed"])==seed and r["pairing_type"]==kind]
            assert len({int(r["recipient_episode"]) for r in subset})==len(included)
            assert len({int(r["donor_episode"]) for r in subset})==len(included)
    row_by_id={}
    with np.load(V3/"data/V3_MAIN_INPUTS.npz",allow_pickle=False) as z:
        ids=z["anchor_id"].copy(); eps=z["episode_index"].copy(); split=z["split_code"].copy()
    for i,aid in enumerate(ids):row_by_id[str(aid)]=(i,int(eps[i]),int(split[i]))
    for m in windows:
        ri,re,_=row_by_id[m["recipient_anchor_id"]];di,de,_=row_by_id[m["donor_anchor_id"]]
        assert ri==int(m["recipient_row"]) and di==int(m["donor_row"])
        assert re==int(m["recipient_episode"]) and de==int(m["donor_episode"])
        assert split[ri]==1 and split[di]==1
        assert re!=de and m["recipient_group_key"]!=m["donor_group_key"]
    timeline={r["anchor_id"]:r for r in rd(V3/"OBSERVATION_TARGET_TIMELINE.csv")}
    for m in windows:
        donor_timeline=timeline[m["donor_anchor_id"]]
        assert donor_timeline["split"]=="dev" and donor_timeline["population"]=="primary"
        assert donor_timeline["past_le_anchor"]=="True"
        assert int(donor_timeline["past_end_frame"])<=int(donor_timeline["anchor_frame"])
        assert int(m["donor_anchor_frame"])==int(donor_timeline["anchor_frame"])
    extraction_text=(V3/"code/extract_v3_features.py").read_text(encoding="utf-8")
    assert 'np.linspace(int(row["anchor_frame"]) - 119, int(row["anchor_frame"]), 32)' in extraction_text
    source_text=(ROOT/"code/lock_and_map.py").read_text(encoding="utf-8")
    forbidden=("targets_native","predictions_native","future_error","model_error")
    assert not any(token in source_text for token in forbidden)
    diversity=rd(ROOT/"MAP_DIVERSITY.csv")
    assert len(diversity)==6 and all(float(r["different_pair_fraction"])>0.75 for r in diversity)
    print({"status":"PASS","recipient_episodes":len(included),"recipient_windows":len(windows)//6,"same_task_and_cross_task_maps":"pass","cross_duplicate_group":"pass","same_episode_exclusion":"pass","target_outcome_independence":"static-source-pass","map_diversity":"pass","eval_bundles":len(matrix)})
    return 0

if __name__=="__main__":raise SystemExit(main())
