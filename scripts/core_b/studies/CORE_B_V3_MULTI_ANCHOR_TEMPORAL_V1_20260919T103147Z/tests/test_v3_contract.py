from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def test_cache_acceptance():
    assert json.loads((ROOT/"cache/CACHE_ACCEPTED.json").read_text())["status"]=="PASS"
    a=np.load(ROOT/"cache/V3_TEMPORAL_FEATURES.npy",mmap_mode="r")
    assert a.shape==(20143,2,16,768) and np.isfinite(a).all()

def test_prediction_acceptance():
    value=json.loads((ROOT/"analysis/PREDICTION_ACCEPTANCE.json").read_text())
    assert value["status"]=="PASS" and value["accepted"]=="36/36"

def test_formal_matrix():
    rows=list(csv.DictReader((ROOT/"MATRIX_36.csv").open()))
    assert len(rows)==36

def test_final_lock():
    text=(ROOT/"FINAL_STATUS.ini").read_text()
    assert "FINAL_TEST_ACCESS=0" in text and "ENCODER_UPDATED=false" in text
