"""Use the study-isolated prediction root as the watcher output root."""

from __future__ import annotations

import build_d3_eq_inference_jobs as builder


def main() -> int:
    original = builder.job_record

    def run_isolated_job(config, horizon, seed, *, stage0: bool, peak_mib: int):
        record = original(config, horizon, seed, stage0=stage0, peak_mib=peak_mib)
        # Prediction filenames include the unique run_id; their parent is the
        # single new-study output root, separate from historical D3 artifacts.
        record["job"]["output_root"] = record["output_root"]
        return record

    builder.job_record = run_isolated_job
    return builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
