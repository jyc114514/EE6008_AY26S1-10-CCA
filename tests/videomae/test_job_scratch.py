"""Standard-library tests for VideoMAE scratch propagation into watcher jobs."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "videomae"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_videomae_cache_job import main as prepare_cache_job
from queue_videomae_core_b_remaining import with_scratch_environment


class ScratchJobTests(unittest.TestCase):
    def test_prepared_cache_job_uses_and_propagates_configured_scratch(self):
        scratch_name = "ci_test"
        with tempfile.TemporaryDirectory() as temporary_root:
            project = Path(temporary_root)
            extractor = project / "extractor.py"
            extractor.write_text("# synthetic placeholder\n", encoding="utf-8")

            argv = [
                "prepare_videomae_cache_job.py",
                "--project-root", str(project),
                "--extractor", str(extractor),
                "--phase", "stage0",
            ]
            output = StringIO()
            with patch.dict(os.environ, {"EE6008_VIDEOMAE_SCRATCH_NAME": scratch_name}):
                with patch.object(sys, "argv", argv):
                    with patch("prepare_videomae_cache_job.subprocess.run") as watcher:
                        with redirect_stdout(output):
                            result = prepare_cache_job()

            self.assertEqual(result, 0)
            scratch_root = project / "tmp" / scratch_name
            self.assertTrue(scratch_root.is_dir())
            jobs = list(scratch_root.glob("videomae_cache_stage0_*/job.json"))
            self.assertEqual(len(jobs), 1)

            job = json.loads(jobs[0].read_text(encoding="utf-8"))
            self.assertEqual(job["environment"]["EE6008_VIDEOMAE_SCRATCH_NAME"], scratch_name)
            self.assertEqual(job["environment"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(job["environment"]["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(Path(job["output_root"]).parent, scratch_root)
            self.assertTrue(job["job_id"].startswith("videomae-cache-stage0-"))

            watcher.assert_called_once()
            self.assertIn("--dry-run", watcher.call_args.args[0])

    def test_scratch_environment_keeps_existing_keys_and_accepts_legacy_jobs(self):
        scratch_name = "ci_test"
        self.assertEqual(
            with_scratch_environment(None, scratch_name),
            {"EE6008_VIDEOMAE_SCRATCH_NAME": scratch_name},
        )
        self.assertEqual(
            with_scratch_environment({"HF_HUB_OFFLINE": "1"}, scratch_name),
            {
                "HF_HUB_OFFLINE": "1",
                "EE6008_VIDEOMAE_SCRATCH_NAME": scratch_name,
            },
        )

    def test_scratch_environment_rejects_a_different_configured_name(self):
        with self.assertRaises(ValueError):
            with_scratch_environment(
                {"EE6008_VIDEOMAE_SCRATCH_NAME": "other"}, "ci_test"
            )

    def test_scratch_environment_rejects_a_non_mapping(self):
        with self.assertRaises(ValueError):
            with_scratch_environment(["unexpected"], "ci_test")


if __name__ == "__main__":
    unittest.main()
