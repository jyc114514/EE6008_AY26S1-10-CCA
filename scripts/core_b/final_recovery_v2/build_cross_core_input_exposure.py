#!/usr/bin/env python3
"""Join frozen Core A and Core B final video metadata without opening media/targets."""
from __future__ import annotations
import os
import csv,hashlib,json,re
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z'))
V1=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
A=V1/"protocol/CORE_A_FINAL_VIDEO_INDEX.csv"
B=V1/"protocol/CORE_B_FINAL_VIDEO_MAP.csv"
EXPECTED_A="db7ca6350caf4de8eff8be56c4c39f64d15e8406f42c3a48140b10f0df79c4cc"
EXPECTED_B="0040bd8b20de1641029c39e0b745ee6b27e9d5d932c93dadb52161290b4ad6d2"
CSV_OUT=OUT/"core_b/CROSS_CORE_INPUT_EXPOSURE.csv"
JSON_OUT=OUT/"qa/CROSS_CORE_INPUT_EXPOSURE_SUMMARY.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def main():
 if sha(A)!=EXPECTED_A or sha(B)!=EXPECTED_B:raise RuntimeError("frozen V1 video-index/map hash mismatch")
 ar=read(A);br=read(B)
 if len(ar)!=602 or len(br)!=162:raise RuntimeError(f"metadata counts differ: CoreA={len(ar)}, CoreB={len(br)}")
 a_by_path={}
 for row in ar:
  path=row["video_path"]
  if path in a_by_path:raise RuntimeError(f"duplicate CoreA video path: {path}")
  a_by_path[path]=int(row["episode_index"])
 b_by_path=defaultdict(list)
 for row in br:
  if row["dataset_role"]!="final_test":raise RuntimeError("CoreB video map contains a non-final role")
  path=row["video_path"]
  match=re.search(r"episode_(\d+)\.mp4$",path)
  if not match or int(match.group(1))!=int(row["episode_id"]):raise RuntimeError(f"path/episode metadata mismatch: {path}")
  b_by_path[path].append(row)
 all_paths=sorted(set(a_by_path)|set(b_by_path))
 fields=["video_path","core_a_episode_index","core_a_episode_present","core_a_prior_input_role","core_a_frame_scope","core_b_episode_id","core_b_final_anchor_count","core_b_anchor_ids","core_b_anchor_frames","core_b_planned_history_frame_ranges","core_b_v2_rgb_extracted","core_b_v2_predictor_forward","core_b_future_targets_materialized","cross_core_same_episode_path"]
 rows=[]
 for path in all_paths:
  a_ep=a_by_path.get(path)
  anchors=sorted(b_by_path.get(path,[]),key=lambda x:int(x["anchor_frame"]))
  b_ep=int(anchors[0]["episode_id"]) if anchors else None
  rows.append({"video_path":path,"core_a_episode_index":a_ep if a_ep is not None else "","core_a_episode_present":a_ep is not None,
   "core_a_prior_input_role":"V1 sealed Core A logits consumed accepted dynamic/static feature-cache rows; Core A window is first 4.0 s, 32 frames at 8 Hz" if a_ep is not None else "",
   "core_a_frame_scope":"first 4.0 s; 32 sampled frames at 8 Hz (exact per-row decode list remains in accepted V1 feature provenance)" if a_ep is not None else "",
   "core_b_episode_id":b_ep if b_ep is not None else "","core_b_final_anchor_count":len(anchors),
   "core_b_anchor_ids":";".join(x["anchor_id"] for x in anchors),
   "core_b_anchor_frames":";".join(str(int(x["anchor_frame"])) for x in anchors),
   "core_b_planned_history_frame_ranges":";".join(f"{int(x['anchor_frame'])-119}:{int(x['anchor_frame'])}" for x in anchors),
   "core_b_v2_rgb_extracted":False if anchors else "",
   "core_b_v2_predictor_forward":False if anchors else "",
   "core_b_future_targets_materialized":False if anchors else "",
   "cross_core_same_episode_path":bool(a_ep is not None and b_ep is not None and a_ep==b_ep)})
 CSV_OUT.parent.mkdir(parents=True,exist_ok=True)
 with CSV_OUT.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 overlap=sum(bool(r["cross_core_same_episode_path"]) for r in rows)
 summary={"schema_version":1,"status":"PASS_METADATA_ONLY_CROSS_CORE_EXPOSURE_MAP","created_utc":datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z"),
  "source_v1":{"core_a_video_index_path":str(A),"rows":len(ar),"sha256":sha(A),"core_b_final_video_map_path":str(B),"anchor_rows":len(br),"sha256_b":sha(B)},
  "v2_metadata_reads":{"core_a_final_video_index_rows":len(ar),"core_b_final_video_map_anchor_rows":len(br),"future_target_arrays_read":0,"raw_video_files_opened":0},
  "joined":{"core_a_unique_episodes":len(a_by_path),"core_b_unique_episodes":len(b_by_path),"core_b_anchor_rows":len(br),"unique_video_paths_union":len(all_paths),"same_episode_path_overlap_count":overlap,"core_a_only_path_count":sum(p in a_by_path and p not in b_by_path for p in all_paths),"core_b_only_path_count":sum(p in b_by_path and p not in a_by_path for p in all_paths)},
  "v2_core_b_final_rgb_extracted":0,"v2_core_b_final_predictor_forwards":0,"v2_core_b_future_target_values_materialized":0,
  "table_path":str(CSV_OUT),"table_sha256":sha(CSV_OUT),"final_video_map_is_metadata_only":True}
 JSON_OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
