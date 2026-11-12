"""Dependency-light contract checks for explicit VideoMAE path arguments.

These subprocesses exercise argparse help/required-argument behavior only. Stub
modules keep them independent of PyTorch and the external VideoMAE runtime.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPOSITORY / "scripts" / "videomae"

REQUIRED_FLAGS = {
    "audit_videomae_task_a.py": (
        "--manifest",
        "--feature-dir",
        "--result-dir",
        "--archive-source",
    ),
    "compare_videomae_static_dynamic.py": ("--run-dir",),
    "extract_videomae_features_task_a.py": (
        "--manifest",
        "--g1-root",
        "--output-dir",
    ),
    "train_videomae_linear_probe_task_a.py": (
        "--feature-dir",
        "--output-dir",
    ),
}


class ExplicitPathCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="videomae-cli-stubs-")
        self.addCleanup(self.tempdir.cleanup)
        self.stub_root = Path(self.tempdir.name)
        self._write_stubs()

    def _write_stubs(self) -> None:
        (self.stub_root / "torch.py").write_text(
            '"""Import-only stub used by CLI argument tests."""\n',
            encoding="utf-8",
        )
        package = self.stub_root / "src"
        models = package / "models"
        models.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (models / "__init__.py").write_text("", encoding="utf-8")
        (models / "videomae_encoder.py").write_text(
            "class VideoMAEEncoder: pass\n"
            "def load_videomae_config(*args, **kwargs): pass\n"
            "def select_device(*args, **kwargs): pass\n",
            encoding="utf-8",
        )

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.stub_root)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.pop("EE6008_VIDEOMAE_SCRATCH_NAME", None)
        return environment

    def _run(self, script_name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT_ROOT / script_name), *arguments],
            cwd=REPOSITORY,
            env=self._environment(),
            capture_output=True,
            check=False,
            text=True,
        )

    def test_help_succeeds_without_scratch_configuration(self) -> None:
        for script_name, flags in REQUIRED_FLAGS.items():
            with self.subTest(script=script_name):
                result = self._run(script_name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                for flag in flags:
                    self.assertIn(flag, result.stdout)

    def test_missing_path_arguments_fail_before_data_access(self) -> None:
        for script_name, flags in REQUIRED_FLAGS.items():
            with self.subTest(script=script_name):
                result = self._run(script_name)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("required", result.stderr.lower())
                for flag in flags:
                    self.assertIn(flag, result.stderr)


if __name__ == "__main__":
    unittest.main()
