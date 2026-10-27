#!/usr/bin/env python3
"""Validate score inputs after prediction seal without opening dev target values."""
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import v8_score as score

def main():
 seal=score.ensure_pre_target_gate();resolved=score.read_json(ROOT/'inventory/RESOLVED_PATHS.json')
 ids,meta=score.prefix_metadata(Path(resolved['dev_prefix_npz']))
 predictions,receipts=score.load_new_prediction_bundles(ids,meta)
 historical,old_sha=score.load_v7_historical_per_window(ids,meta)
 if len(ids)!=185 or len(predictions)!=9 or len(receipts)!=9 or len(historical)!=7770:raise RuntimeError('score input support count mismatch')
 if not all(tuple(x.shape)==(185,30,28) for x in predictions.values()):raise RuntimeError('new prediction shape mismatch')
 result={'schema_version':1,'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
  'prediction_seal_sha256':score.sha(ROOT/'qa/PREDICTION_SEAL.json'),'new_dev_prediction_bundles':len(receipts),'new_dev_prediction_rows':len(predictions)*185,
  'historical_v7_per_window_rows_reused':len(historical),'historical_source_sha256':old_sha,'unique_dev_receivers':len(ids),
  'numeric_seed_and_roles':'PASS','prediction_row_ids_source_groups':'PASS','safe_tensor_schema_and_finite':'PASS',
  'dev_target_archive_opened':False,'dev_target_values_read':False,'final_heldout_payload_access':0}
 p=ROOT/'qa/SCORE_INPUT_PREFLIGHT.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','new_bundles':len(receipts),'historical_rows':len(historical),'dev_target_opened':False},sort_keys=True))
if __name__=='__main__':main()
