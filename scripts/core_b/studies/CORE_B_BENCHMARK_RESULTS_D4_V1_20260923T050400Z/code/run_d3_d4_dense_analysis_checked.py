"""Checked runner for the D3 dense rolling analysis module."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("analyze_d3_d4_dense.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")

    namespace = runpy.run_path(str(SCRIPT), run_name="d3_d4_dense_analysis_module")
    original_group_means = namespace["group_means"]
    context = namespace["build_context"]()
    if args.preflight:
        namespace["save_preflight"](context)
        return
    predictions = namespace["load_predictions"](context)

    def group_means(values, group_labels_or_indices, ordered_groups):
        labels = np.asarray(group_labels_or_indices)
        if np.issubdtype(labels.dtype, np.integer):
            result = np.full(len(ordered_groups), np.nan, dtype=np.float64)
            for index in range(len(ordered_groups)):
                local = labels == index
                if local.any():
                    result[index] = float(np.mean(values[local]))
            return result
        return original_group_means(values, labels, ordered_groups)

    namespace["context"] = context
    namespace["predictions"] = predictions
    namespace["group_means"] = group_means
    sys.argv = [str(SCRIPT)]
    namespace["main"]()


if __name__ == "__main__":
    main()
