#!/usr/bin/env python3
"""Prepare the fixed final RGB/input manifest from already reviewed metadata only."""
from __future__ import annotations
import os
import csv, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT=PROJECT/"reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
V1=PROJECT/"reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z/protocol"
MAP=V1/"CORE_B_FINAL_VIDEO_MAP.csv"
POP=V1/"CORE_B_FINAL_POPULATION_METADATA.csv"
OUT_CSV=OUT/"core_b/FINAL_CURRENT_PROFILE_INPUTS.csv"
OUT_JSON=OUT/"core_b/FINAL_CURRENT_PROFILE_INPUTS_RECEIPT.json"
FIELDS=("dataset_role","canonical_id","source_key","literal_source_key","episode_id","task_id","anchor_id","anchor_frame","cache_index","video_path","parquet_path")
TASKS=(119,94,90,141,230)
EXPECTED={119:48,94:48,90:34,141:20,230:12}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
 return h.hexdigest()

def atomic_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(f".{p.name}.{os.getpid()}.part")
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 os.replace(t,p)

def main():
 if not MAP.is_file() or not POP.is_file(): raise FileNotFoundError("fixed V1 final metadata map/population is missing")
 map_rows=list(csv.DictReader(MAP.open(encoding="utf-8-sig",newline="")))
 pop_rows=list(csv.DictReader(POP.open(encoding="utf-8-sig",newline="")))
 if len(map_rows)!=162 or len(pop_rows)!=162: raise ValueError("fixed Core B final population must contain 162 map and metadata rows")
 by_id={r["anchor_id"]:r for r in map_rows}
 pp={r["anchor_id"]:r for r in pop_rows}
 if len(by_id)!=162 or len(pp)!=162 or set(by_id)!=set(pp): raise ValueError("final map/population anchor IDs differ")
 rows=[]
 for source in map_rows:
  a=source["anchor_id"];p=pp[a]
  if source["dataset_role"]!="final_test" or p["dataset_role"]!="final_test": raise ValueError(f"non-final row in final map: {a}")
  for f in ("canonical_id","source_key","literal_source_key","episode_id","task_id","anchor_frame"):
   if str(source[f])!=str(p[f] if f!="canonical_id" else source[f]): raise ValueError(f"final map/population identity mismatch on {f}: {a}")
  if int(source["anchor_frame"])<119 or int(source["cache_index"])<0: raise ValueError(f"final anchor cannot form fixed prefix: {a}")
  if int(p["target_h30_row_start_inclusive"])!=int(source["anchor_frame"])+1 or int(p["target_h30_row_end_inclusive"])!=int(source["anchor_frame"])+30:
   raise ValueError(f"locked H30 row interval is not anchor+1..anchor+30: {a}")
  rows.append({key:str(source[key]) for key in FIELDS})
 if len({r["canonical_id"] for r in rows})!=162 or len({r["cache_index"] for r in rows})!=162: raise ValueError("final canonical IDs/cache indexes are not unique")
 if {t:sum(int(r["task_id"])==t for r in rows) for t in TASKS}!=EXPECTED: raise ValueError("fixed final task support differs from V1 lock")
 if len({r["episode_id"] for r in rows})!=41: raise ValueError("fixed final episode support differs from 41")
 if OUT_CSV.exists() or OUT_JSON.exists(): raise FileExistsError("refusing to overwrite final current-profile metadata manifest")
 OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
 tmp=OUT_CSV.with_name(f".{OUT_CSV.name}.{os.getpid()}.part")
 with tmp.open("w",encoding="utf-8",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="raise");writer.writeheader();writer.writerows(rows);f.flush();os.fsync(f.fileno())
 os.replace(tmp,OUT_CSV)
 receipt={"schema_version":1,"status":"PASS_FINAL_INPUT_METADATA_ONLY","created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
  "rows":162,"unique_episodes":41,"task_support":{str(t):EXPECTED[t] for t in TASKS},
  "source_map_path":str(MAP),"source_map_sha256":sha(MAP),"source_population_path":str(POP),"source_population_sha256":sha(POP),
  "output_path":str(OUT_CSV),"output_sha256":sha(OUT_CSV),"canonical_id_order_sha256":hashlib.sha256("\n".join(r["canonical_id"] for r in rows).encode()).hexdigest(),
  "rgb_frames_read":0,"parquet_payload_rows_read":0,"future_target_values_materialized":0,"final_payload_access":False}
 atomic_json(OUT_JSON,receipt);print(json.dumps(receipt,indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
