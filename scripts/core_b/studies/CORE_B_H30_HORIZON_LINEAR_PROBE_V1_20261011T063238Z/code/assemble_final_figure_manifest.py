from __future__ import annotations
import csv,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read_csv(p):
 with p.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def main():
 stems=["FIG01_H30_QUERY_GAINS_BY_SEED","FIG02_TRAIN_HORIZON_PROFILE","FIG03_DEV_HORIZON_PROFILE"]
 png=[ROOT/"figures"/(s+".png") for s in stems]
 pdf=[ROOT/"figures"/(s+".pdf") for s in stems]
 png += [ROOT/"figures/FIG04_H30_FOUR_METRICS_TRAIN_AND_DEV_AMEND001.png"]
 pdf += [ROOT/"figures/FIG04_H30_FOUR_METRICS_TRAIN_AND_DEV_AMEND001.pdf"]
 cases=read_csv(ROOT/"figure_data/LOCKED_CASE_IDS.csv")
 if len(cases)!=12:raise AssertionError("locked case count mismatch")
 ids=[r["case_id"] for r in cases]
 png += [ROOT/"figures"/("FIG05_CASE_"+c+"_AMEND002.png") for c in ids]
 pdf += [ROOT/"figures/FIG05_FIXED_H30_CASE_PATHS_AMEND002.pdf"]
 if any(not p.is_file() for p in png+pdf):raise FileNotFoundError("accepted figure missing")
 raster=[]
 for p in png:
  im=Image.open(p)
  raster.append({"path":str(p.relative_to(ROOT)),"format":im.format,"mode":im.mode,
   "dimensions_px":list(im.size),"bytes":p.stat().st_size,"sha256":sha(p)})
 pdfmeta=[]
 for p in pdf:
  out=subprocess.run(["/usr/bin/pdfinfo",str(p)],check=True,capture_output=True,text=True).stdout
  info={}
  for line in out.splitlines():
   if ":" in line:
    k,v=line.split(":",1)
    if k.strip() in ("Pages","Page size","PDF version"):info[k.strip()]=v.strip()
  pdfmeta.append({"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p),**info})
 manifest={"status":"PASS_FINAL_AMENDED_FIGURE_SET","created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
  "accepted_png_count":len(png),"accepted_pdf_count":len(pdf),"case_count":len(cases),
  "accepted_figures":[{"png":str(p.relative_to(ROOT)),"sha256":sha(p)} for p in png]+[{"pdf":str(p.relative_to(ROOT)),"sha256":sha(p)} for p in pdf],
  "raster_metadata":raster,"pdf_metadata":pdfmeta,
  "case_path_table":"figure_data/CASE_PATHS_H30_STANDARDIZED.csv","native_values_preserved":True,
  "human_review_status":"pending","submission_ready":False,"publisher_compliance":"not assessed; journal unspecified",
  "amendments":["qa/FIGURE_MANIFEST_AMENDMENT_001.json","qa/FIGURE_CASE_AMENDMENT_002.json"]}
 mp=ROOT/"figures/FIGURE_MANIFEST_FINAL.json"
 if mp.exists():raise FileExistsError(mp)
 mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 qa={"status":"PASS_FINAL_ACCEPTED_FIGURE_METADATA","png_count":len(png),"pdf_count":len(pdf),
  "case_pdf_pages":int(pdfmeta[-1]["Pages"]),"all_raster_dimensions_recorded":len(raster)==16,
  "pdfinfo_used":True,"pypdf_unavailable":True,"agent_visual_review":"PASS_CONTACT_SHEET_AND_DETAIL_PREVIEWS",
  "human_review":"pending","submission_ready":False}
 qp=ROOT/"qa/FIGURE_FINAL_EXPORT_QA_AMENDMENT_002.json"
 if qp.exists():raise FileExistsError(qp)
 qp.write_text(json.dumps(qa,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 print(json.dumps({"status":qa["status"],"png_count":qa["png_count"],"pdf_count":qa["pdf_count"],"case_pages":qa["case_pdf_pages"]},sort_keys=True))
if __name__=="__main__":main()

