"""CPU-only checks for the changed observation and accepted config references."""
import ast
import csv
import importlib.util
from pathlib import Path
import sys
import numpy as np

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("followup", here / "run_core_a_followup.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

frames = np.arange(32, dtype=np.uint8)[:, None, None, None] * np.ones((32, 2, 3, 3), dtype=np.uint8)
out = mod.repeat_fourth_second(frames, np)
assert out.shape == frames.shape
assert out[:, 0, 0, 0].tolist() == list(range(24, 32)) * 4
assert out[:, 0, 0, 0].min() == 24 and out[:, 0, 0, 0].max() == 31
try:
    mod.repeat_fourth_second(frames[:31], np)
    raise AssertionError("Short input was accepted")
except ValueError:
    pass
if len(sys.argv) > 1:
    source = Path(sys.argv[1])
    with (source / mod.MATRIX_REL).open(newline="", encoding="utf-8-sig") as f:
        refs = [r for r in csv.DictReader(f) if r["core"] == "A" and r["program"] == "A_STATIC"]
    assert sorted(int(r["seed"]) for r in refs) == list(mod.SEEDS)
    for r in refs:
        path = mod.relocate(r["config_path"], source)
        assert mod.sha(path) == r["config_sha256"], path
        keys = {line.split(":", 1)[0] for line in path.read_text().splitlines() if line and not line.startswith(" ")}
        assert set(mod.TRAIN_FIELDS).issubset(keys), (path, set(mod.TRAIN_FIELDS) - keys)
    source_ast = ast.parse((source / mod.REPO_REL / "scripts/run_batch48_ablation_matrix_v3.py").read_text())
    experiment = next(n for n in source_ast.body if isinstance(n, ast.ClassDef) and n.name == "Experiment")
    fields = {n.target.id for n in experiment.body if isinstance(n, ast.AnnAssign)}
    wrapper_ast = ast.parse((here / "run_core_a_followup.py").read_text())
    call = next(n for n in ast.walk(wrapper_ast) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "Experiment")
    declared = {k.arg for k in call.keywords if k.arg is not None} | set(mod.TRAIN_FIELDS)
    assert fields == declared, {"missing": fields - declared, "extra": declared - fields}
print("PASS: synthetic fourth-second 8-frame cycle x4 and short-input rejection" +
      ("; supplied external source-config hashes and Experiment fields verified" if len(sys.argv) > 1 else "; no external assets read"))
