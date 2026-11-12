"""Read-only CPU verification of completed train/dev prediction exports."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    project, run = args.project_root.resolve(), args.run_root.resolve()
    require(run.is_relative_to((project / "tmp/s126mdg33_02").resolve()), "Personal run required")
    config = json.loads((run / "CONFIG.json").read_text())
    status = json.loads((run / "RUN_STATUS.json").read_text())
    require(status["status"] == "PASS_TRAIN_AND_EXPORT" and status["epochs"] == 40
            and status["optimizer_steps"] == 11120, "Incomplete formal run")
    require(config["resume_check"] is False and config["final_test_access"] == 0
            and status["final_test_access"] == 0, "Run scope mismatch")
    require(config["mode"] in {"TS11-D", "STATE-CTRL"} and config["horizon"] in {30, 60}, "Cell mismatch")
    require(digest(run / "checkpoint_last.pt") == status["checkpoint_sha256"], "Checkpoint hash mismatch")
    inputs = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    require(digest(inputs) == "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78", "Input identity mismatch")
    summary = {}
    with np.load(inputs, allow_pickle=False) as source:
        codes = source["split_code"]
        require(set(codes.tolist()) == {0, 1}, "Unexpected split codes")
        episodes_by_split = []
        for code, name, count, episode_count in ((0, "train", 17757, 2712), (1, "dev", 2322, 603)):
            rows = np.flatnonzero(codes == code)
            require(len(rows) == count, "Population mismatch")
            path = run / f"predictions_{name}.npz"
            require(digest(path) == status[f"{name}_prediction_sha256"], "Prediction hash mismatch")
            with np.load(path, allow_pickle=False) as saved:
                require(set(saved.files) == {"anchor_id", "episode_index", "predictions_native"}, "Prediction schema mismatch")
                for key in ("anchor_id", "episode_index"):
                    require(np.array_equal(saved[key], source[key][rows]), f"Prediction alignment mismatch: {key}")
                require(len(np.unique(saved["anchor_id"])) == count, "Duplicate anchors")
                episodes = set(saved["episode_index"].tolist())
                require(len(episodes) == episode_count, "Episode count mismatch")
                episodes_by_split.append(episodes)
                values = saved["predictions_native"]
                require(values.shape == (count, config["horizon"], 28) and values.dtype == np.float32,
                        "Prediction shape/dtype mismatch")
                require(bool(np.isfinite(values).all()), "Nonfinite predictions")
                summary[name] = {"anchors": count, "episodes": episode_count,
                                 "shape": list(values.shape), "hash_and_order_match": True}
        require(not episodes_by_split[0] & episodes_by_split[1], "Train/dev episode overlap")
    print(json.dumps({"status": "PASS_PREDICTION_EXPORT_AUDIT", "mode": config["mode"],
                      "horizon": config["horizon"], "seed": config["seed"], "splits": summary,
                      "checkpoint_hash_matches": True, "checkpoint_load_tested": False,
                      "targets_loaded": False, "gpu_used": False,
                      "metrics_evaluated": False, "final_test_access": 0}, indent=2))


if __name__ == "__main__":
    main()
