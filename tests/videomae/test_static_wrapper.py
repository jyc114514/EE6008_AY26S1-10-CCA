"""Behavior checks for the VideoMAE static wrapper's path contract.

All launcher calls are intercepted by a temporary stub. These tests never load
model code, datasets, checkpoints, or a GPU.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
WRAPPER = REPOSITORY / "scripts" / "videomae" / "run_videomae_static_task_a.sh"
RUN_RELATIVE = Path("videomae_baseline_20260907")


class StaticWrapperPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="videomae wrapper ")
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.cwd = self.root / "workspace with spaces"
        self.cwd.mkdir()

        self.project = self.cwd / "project root"
        self.project.mkdir()
        self.tmp_root = self.project / "tmp"
        self.tmp_root.mkdir()
        self.scratch = self.tmp_root / "safe_scratch"
        self.run = self.scratch / RUN_RELATIVE
        self.run.mkdir(parents=True)

        input_root = self.cwd / "authorized inputs"
        input_root.mkdir()
        self.manifest = input_root / "train validation.parquet"
        self.manifest.write_bytes(b"synthetic placeholder")
        self.g1_root = input_root / "G1 data"
        self.g1_root.mkdir()
        self.config = input_root / "profile.yaml"
        self.config.write_text("synthetic: true\n", encoding="utf-8")

        self.wrapper_dir = self.cwd / "tools"
        self.wrapper_dir.mkdir()
        self.launcher = self.wrapper_dir / "record_args.py"
        self._write_stub_launcher(self.launcher)
        self.launch_log = self.cwd / "launcher calls.jsonl"

    def _write_stub_launcher(self, path: Path) -> None:
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "record = {'argv': sys.argv[1:], 'hf_home': os.environ.get('HF_HOME')}\n"
            "with open(os.environ['VIDEOMAE_TEST_LAUNCH_LOG'], 'a', encoding='utf-8') as f:\n"
            "    f.write(json.dumps(record) + '\\n')\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | 0o111)

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "EE6008_PROJECT_ROOT": str(self.project),
                "EE6008_VIDEOMAE_SCRATCH_NAME": self.scratch.name,
                "VIDEOMAE_ENV_LAUNCHER": str(self.launcher),
                "VIDEOMAE_MANIFEST": str(self.manifest),
                "VIDEOMAE_G1_ROOT": str(self.g1_root),
                "VIDEOMAE_CONFIG": str(self.config),
                "VIDEOMAE_TEST_LAUNCH_LOG": str(self.launch_log),
            }
        )
        environment.pop("HF_HOME", None)
        return environment

    def _run(self, environment: dict[str, str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(WRAPPER), "0"],
            cwd=cwd or self.cwd,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def _calls(self) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in self.launch_log.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def _flag(self, command: list[str], name: str) -> str:
        self.assertIn(name, command)
        return command[command.index(name) + 1]

    def test_runs_only_expected_static_commands_with_explicit_paths(self) -> None:
        result = self._run(self._environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        calls = self._calls()
        self.assertEqual(len(calls), 7)
        commands = [record["argv"] for record in calls]
        self.assertTrue(all(isinstance(command, list) for command in commands))
        extract = [
            command
            for command in commands
            if Path(command[1]).name == "extract_videomae_features_task_a.py"
        ]
        train = [
            command
            for command in commands
            if Path(command[1]).name == "train_videomae_linear_probe_task_a.py"
        ]
        compare = [
            command
            for command in commands
            if Path(command[1]).name == "compare_videomae_static_dynamic.py"
        ]
        self.assertEqual((len(extract), len(train), len(compare)), (3, 3, 1))

        expected_script_dir = REPOSITORY / "scripts" / "videomae"
        for command in extract:
            self.assertEqual(Path(command[1]).resolve(), expected_script_dir / Path(command[1]).name)
            self.assertEqual(Path(self._flag(command, "--config")), self.config)
            self.assertEqual(Path(self._flag(command, "--manifest")), self.manifest)
            self.assertEqual(Path(self._flag(command, "--g1-root")), self.g1_root)
            self.assertEqual(Path(self._flag(command, "--output-dir")).parent, self.run / "outputs")
            self.assertIn("--local-files-only", command)
            self.assertIn("--variant", command)
            self.assertEqual(self._flag(command, "--variant"), "static")
        self.assertEqual(self._flag(extract[0], "--split"), "validation")
        self.assertEqual(self._flag(extract[0], "--max-episodes"), "8")
        self.assertEqual(Path(self._flag(extract[0], "--output-dir")), self.run / "outputs" / "smoke_static")
        self.assertEqual(
            [self._flag(command, "--split") for command in extract[1:]],
            ["validation", "train"],
        )
        self.assertTrue(
            all(Path(self._flag(command, "--output-dir")) == self.run / "outputs" / "features"
                for command in extract[1:])
        )

        self.assertEqual(
            [self._flag(command, "--seed") for command in train],
            ["6008", "7", "42"],
        )
        self.assertTrue(
            all(self._flag(command, "--variant") == "static" for command in train)
        )
        self.assertTrue(
            all("--cpu" in command for command in train)
        )
        self.assertTrue(
            all(Path(self._flag(command, "--feature-dir")) == self.run / "outputs" / "features"
                for command in train)
        )
        self.assertTrue(
            all(Path(self._flag(command, "--output-dir")) == self.run / "outputs" / "linear_probe_static"
                for command in train)
        )
        self.assertEqual(Path(compare[0][1]).resolve(), expected_script_dir / Path(compare[0][1]).name)
        self.assertEqual(Path(self._flag(compare[0], "--run-dir")), self.run)

        default_hf_home = str(self.scratch / "hf_cache")
        self.assertTrue(all(record["hf_home"] == default_hf_home for record in calls))
        output = result.stdout + result.stderr
        self.assertNotIn("/usr1/home/", output)
        self.assertNotRegex(output, r"/(?:s[0-9]{8,}|[A-Za-z][0-9]{8,}_[0-9]+)/")

    def test_missing_or_invalid_scratch_name_fails_before_launch(self) -> None:
        invalid_names = (None, "", ".", "..", "nested/name", r"nested\\name", "路径")
        for name in invalid_names:
            with self.subTest(name=("missing" if name is None else "<invalid>")):
                self.launch_log.unlink(missing_ok=True)
                environment = self._environment()
                if name is None:
                    environment.pop("EE6008_VIDEOMAE_SCRATCH_NAME", None)
                else:
                    environment["EE6008_VIDEOMAE_SCRATCH_NAME"] = name
                result = self._run(environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.launch_log.exists())
                output = result.stdout + result.stderr
                self.assertNotIn("/usr1/home/", output)

    def test_launcher_directory_is_rejected_before_launch(self) -> None:
        environment = self._environment()
        environment["VIDEOMAE_ENV_LAUNCHER"] = str(self.wrapper_dir)
        result = self._run(environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.launch_log.exists())

    def test_relative_inputs_are_resolved_before_wrapper_changes_directory(self) -> None:
        environment = self._environment()
        for name in (
            "EE6008_PROJECT_ROOT",
            "VIDEOMAE_ENV_LAUNCHER",
            "VIDEOMAE_MANIFEST",
            "VIDEOMAE_G1_ROOT",
            "VIDEOMAE_CONFIG",
        ):
            environment[name] = os.path.relpath(environment[name], self.cwd)
        result = self._run(environment, cwd=self.cwd)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        calls = self._calls()
        self.assertEqual(len(calls), 7)
        extract = [
            record["argv"]
            for record in calls
            if Path(record["argv"][1]).name == "extract_videomae_features_task_a.py"
        ]
        self.assertEqual(len(extract), 3)
        for record in calls:
            command = record["argv"]
            self.assertTrue(Path(command[1]).is_absolute())
        for command in extract:
            self.assertEqual(Path(self._flag(command, "--config")), self.config)
            self.assertEqual(Path(self._flag(command, "--manifest")), self.manifest)
            self.assertEqual(Path(self._flag(command, "--g1-root")), self.g1_root)
        self.assertTrue(
            all(record["hf_home"] == str(self.scratch / "hf_cache") for record in calls)
        )

    def test_relative_hf_home_is_rejected_before_launch(self) -> None:
        environment = self._environment()
        environment["HF_HOME"] = "relative-cache"
        result = self._run(environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.launch_log.exists())
        self.assertNotIn("/usr1/home/", result.stdout + result.stderr)

    def _make_symlink(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"directory symlinks unavailable: {error.__class__.__name__}")

    def test_scratch_root_and_escape_symlinks_are_rejected(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()

        alias_to_tmp = self.tmp_root / "alias_to_tmp"
        self._make_symlink(alias_to_tmp, self.tmp_root)
        environment = self._environment()
        environment["EE6008_VIDEOMAE_SCRATCH_NAME"] = alias_to_tmp.name
        self.launch_log.unlink(missing_ok=True)
        result = self._run(environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.launch_log.exists())

        alias_outside = self.tmp_root / "alias_outside"
        self._make_symlink(alias_outside, outside)
        environment = self._environment()
        environment["EE6008_VIDEOMAE_SCRATCH_NAME"] = alias_outside.name
        self.launch_log.unlink(missing_ok=True)
        result = self._run(environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.launch_log.exists())

    def test_run_symlink_to_scratch_or_outside_is_rejected(self) -> None:
        outside = self.root / "outside run"
        outside.mkdir()

        for target_name, target in (
            ("scratch", self.scratch),
            ("outside", outside),
        ):
            with self.subTest(target=target_name):
                real_run = self.run
                real_run.rmdir()
                self._make_symlink(real_run, target)
                self.launch_log.unlink(missing_ok=True)
                result = self._run(self._environment())
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.launch_log.exists())
                real_run.unlink()
                real_run.mkdir()

    def test_no_private_server_path_is_printed_or_logged(self) -> None:
        result = self._run(self._environment())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = result.stdout + result.stderr + self.launch_log.read_text(encoding="utf-8")
        self.assertNotIn("/usr1/home/", output)
        self.assertNotRegex(output, r"/(?:s[0-9]{8,}|[A-Za-z][0-9]{8,}_[0-9]+)/")


if __name__ == "__main__":
    unittest.main()
