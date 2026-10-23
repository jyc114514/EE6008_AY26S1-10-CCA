from __future__ import annotations
import os
import csv
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

ROOT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_FUSION_QUERY_MECHANISM_V1_20261007T110705Z'))
PROJECT=Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
SCHED_CODE=PROJECT/"reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code"
if str(SCHED_CODE) not in sys.path: sys.path.insert(0,str(SCHED_CODE))
from d3_model import sinusoidal
from orthogonal_query import canonical_sylvester_hadamard,hadamard_future_table,hash_float32_array

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def spectrum(matrix):
 q=np.ascontiguousarray(matrix,dtype=np.float32).astype(np.float64)
 s=np.linalg.svd(q,compute_uv=False)
 cut=float(s[0]*np.finfo(np.float32).eps*max(q.shape))
 return {"shape":list(q.shape),"singular_values":s.tolist(),"sigma_max":float(s[0]),
         "fp32_threshold":cut,"fp32_effective_rank":int(np.sum(s>cut)),
         "threshold_formula":"float32 epsilon * max(matrix shape) * sigma_max; SVD in float64 of FP32 values"}

def main():
 model_path=SCHED_CODE/"d3_model.py"
 expected="6d4c321c65bb101f5c756ba8f1186a98d0358e16dc49dfa39f1d677c35d9e589"
 if sha(model_path)!=expected: raise AssertionError("locked model source hash changed")
 h=canonical_sylvester_hadamard(128)
 q30,o30,r30=hadamard_future_table(30)
 q60,o60,r60=hadamard_future_table(60)
 if not torch.equal(o30,o60[:30]): raise AssertionError("old query H30 is not exact H60 prefix")
 if not torch.equal(q30,q60[:30]): raise AssertionError("candidate H30 is not exact H60 prefix")
 max_mean=max(float(q.mean(dim=1).abs().max().item()) for q in [q30,q60])
 max_row_rms_diff=max(float(torch.max(torch.abs(torch.sqrt(torch.mean(q*q,dim=1))-r)).item()) for q,r in [(q30,r30),(q60,r60)])
 normalized=q60/torch.linalg.vector_norm(q60,dim=1,keepdim=True)
 gram=normalized@normalized.T
 diag_error=float(torch.max(torch.abs(torch.diag(gram)-1)).item())
 off=gram-torch.diag(torch.diag(gram))
 off_error=float(torch.max(torch.abs(off)).item())
 if max_mean>1e-7 or max_row_rms_diff>1e-7 or diag_error>1e-6 or off_error>1e-6:
  raise AssertionError("candidate query geometry contract failed")
 npz=ROOT/"protocol/HADAMARD_QUERY_TABLES.npz"
 np.savez_compressed(npz,hadamard_128=h.numpy().astype(np.float32),
                     old_future_h30=o30.numpy().astype(np.float32),old_future_h60=o60.numpy().astype(np.float32),
                     candidate_h30=q30.numpy().astype(np.float32),candidate_h60=q60.numpy().astype(np.float32),
                     old_row_rms_h30=r30.numpy().astype(np.float32),old_row_rms_h60=r60.numpy().astype(np.float32),
                     candidate_hadamard_row_indices_h30=np.arange(1,31,dtype=np.int64),
                     candidate_hadamard_row_indices_h60=np.arange(1,61,dtype=np.int64))
 fields=["horizon","future_step_1based","time_seconds","hadamard_row_0based","old_row_rms","old_query_sha256","candidate_query_sha256"]+[f"old_q{j:03d}" for j in range(128)]+[f"candidate_q{j:03d}" for j in range(128)]
 rows=[]
 for horizon,old,candidate,rms in [(30,o30,q30,r30),(60,o60,q60,r60)]:
  for i in range(horizon):
   oldrow=old[i].numpy().astype(np.float32); candrow=candidate[i].numpy().astype(np.float32)
   rows.append({"horizon":horizon,"future_step_1based":i+1,"time_seconds":(i+1)/30.0,
    "hadamard_row_0based":i+1,"old_row_rms":float(rms[i].item()),
    "old_query_sha256":hash_float32_array(oldrow),"candidate_query_sha256":hash_float32_array(candrow),
    **{f"old_q{j:03d}":float(x) for j,x in enumerate(oldrow)},
    **{f"candidate_q{j:03d}":float(x) for j,x in enumerate(candrow)}})
 table_csv=ROOT/"protocol/HADAMARD_QUERY_TABLES.csv"
 with table_csv.open("w",newline="",encoding="utf-8") as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 meta={
  "schema_version":1,"status":"PASS_QUERY_TABLES_LOCKED","created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
  "model_source_path":str(model_path),"model_source_sha256":expected,
  "construction":"Canonical Sylvester H128 with recursion H_(2n)=[[H_n,H_n],[H_n,-H_n]]; no row/column permutation.",
  "dc_row_zero_excluded":True,"candidate_row_indices_zero_based":{"H30":list(range(1,31)),"H60":list(range(1,61))},
  "column_order":"canonical Sylvester order; not randomized",
  "step_semantics":"candidate row index for absolute future step t is Hadamard row t; old sinusoidal time is float32 t/30 seconds.",
  "row_scale":"multiply each Hadamard row by the RMS of the old fixed sinusoidal query at the same absolute future step; no target/model-derived scale.",
  "old_h30_is_old_h60_prefix":bool(torch.equal(o30,o60[:30])),
  "candidate_h30_is_candidate_h60_prefix":bool(torch.equal(q30,q60[:30])),
  "candidate_H30_fit_only":True,"candidate_H60_train_run":False,
  "max_abs_row_mean":max_mean,"max_abs_row_rms_match_error":max_row_rms_diff,
  "normalized_gram_diagonal_max_error":diag_error,"normalized_gram_off_diagonal_max_abs":off_error,
  "trainable_parameters_added":0,"candidate_table_buffer_key":"future_query_table on FixedFutureQueryOverride wrapper; not a base-model parameter",
  "old_table_hashes":{"H30":hash_float32_array(o30),"H60":hash_float32_array(o60)},
  "candidate_table_hashes":{"H30":hash_float32_array(q30),"H60":hash_float32_array(q60)},
  "query_spectra":{"old_H30":spectrum(o30.numpy()),"old_H60":spectrum(o60.numpy()),
                   "candidate_H30":spectrum(q30.numpy()),"candidate_H60":spectrum(q60.numpy())},
  "saved_artifacts":{"npz_path":str(npz),"npz_sha256":sha(npz),"csv_path":str(table_csv),"csv_sha256":sha(table_csv),
                     "npz_bytes":npz.stat().st_size,"csv_bytes":table_csv.stat().st_size},
  "scope_note":"Hadamard changes time distance, spectrum, smoothness and conditioning together. RMS and parameter parity do not isolate rank as a cause or establish forecasting suitability."
 }
 (ROOT/"protocol/QUERY_TABLES_AND_HASHES.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2,sort_keys=True)+chr(10),encoding="utf-8")
 print(json.dumps({"status":meta["status"],"H30_hash":meta["candidate_table_hashes"]["H30"],"H60_hash":meta["candidate_table_hashes"]["H60"],"max_row_rms_match_error":max_row_rms_diff,"gram_offdiag_max_abs":off_error,"npz_bytes":npz.stat().st_size,"csv_bytes":table_csv.stat().st_size},indent=2))

if __name__=="__main__":
 main()

