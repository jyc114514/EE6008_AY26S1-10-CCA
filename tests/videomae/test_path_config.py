"""Dependency-free tests for the VideoMAE scratch path contract."""

import ast
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "videomae"))
import path_config
from close_videomae_core_b_handoff import CACHE, cache_root


class ScratchPathTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_import_is_lazy_when_environment_is_unset(self):
        self.assertIs(importlib.reload(path_config), path_config)
        with self.assertRaisesRegex(ValueError, "EE6008_VIDEOMAE_SCRATCH_NAME"):
            path_config.scratch_name()

    def test_valid_name_resolves_below_project_tmp_without_creating_it(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            project = Path(temporary_root)
            (project / "tmp").mkdir()
            with patch.dict(os.environ, {"EE6008_VIDEOMAE_SCRATCH_NAME": "ci_test"}):
                scratch = path_config.scratch_root(project)
            self.assertEqual(scratch, (project / "tmp" / "ci_test").resolve())
            self.assertFalse(scratch.exists())

    def test_invalid_names_are_rejected(self):
        for name in ("", ".", "..", "nested/name", r"nested\\name", "含中文", "x" * 65):
            with self.subTest(name="<invalid>"):
                with patch.dict(os.environ, {"EE6008_VIDEOMAE_SCRATCH_NAME": name}):
                    with self.assertRaises(ValueError):
                        path_config.scratch_name()

    def test_tmp_root_alias_and_escape_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            project = Path(temporary_root) / "project"
            tmp_root = project / "tmp"
            outside = Path(temporary_root) / "outside"
            tmp_root.mkdir(parents=True)
            outside.mkdir()
            aliases = (
                (tmp_root / "same", tmp_root),
                (tmp_root / "escape", outside),
            )
            for alias, target in aliases:
                try:
                    alias.symlink_to(target, target_is_directory=True)
                except (OSError, NotImplementedError) as error:
                    self.skipTest(f"Directory symlinks unavailable: {error.__class__.__name__}")
                with patch.dict(
                    os.environ,
                    {"EE6008_VIDEOMAE_SCRATCH_NAME": alias.name},
                ):
                    with self.assertRaises(ValueError):
                        path_config.scratch_root(project)

    def test_handoff_cache_root_matches_copied_project_layout(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            archive = Path(temporary_root) / "archive"
            expected = archive / "tmp" / "ci_test" / CACHE
            manifest = expected / "cache" / "CACHE_MANIFEST.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"status":"synthetic"}\\n', encoding="utf-8")
            with patch.dict(os.environ, {"EE6008_VIDEOMAE_SCRATCH_NAME": "ci_test"}):
                resolved = cache_root(archive)
            self.assertEqual(resolved, expected.resolve())
            self.assertTrue((resolved / "cache" / "CACHE_MANIFEST.json").is_file())

    def test_handoff_copy_lists_include_the_helper(self):
        root = Path(__file__).resolve().parents[2]
        package = ast.parse(
            (root / "scripts/videomae/package_videomae_handoff.py").read_text(encoding="utf-8")
        )
        names = None
        for node in ast.walk(package):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "code_names"
                for target in node.targets
            ):
                names = ast.literal_eval(node.value)
                break
        self.assertIsNotNone(names)
        self.assertIn("path_config.py", names)

        closer = ast.parse(
            (root / "scripts/videomae/close_videomae_core_b_handoff.py").read_text(encoding="utf-8")
        )
        copy_names = []
        for node in ast.walk(closer):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "name":
                if isinstance(node.iter, ast.Tuple):
                    try:
                        values = ast.literal_eval(node.iter)
                    except (TypeError, ValueError):
                        continue
                    if "close_videomae_core_b_handoff.py" in values:
                        copy_names = list(values)
                        break
        self.assertIn("path_config.py", copy_names)

        for copied_names, module_name in (
            (names, "package_videomae_handoff"),
            (copy_names, "close_videomae_core_b_handoff"),
        ):
            with self.subTest(module=module_name), tempfile.TemporaryDirectory() as temporary_root:
                code_dir = Path(temporary_root) / "code"
                code_dir.mkdir()
                for name in copied_names:
                    shutil.copy2(root / "scripts" / "videomae" / name, code_dir / name)
                environment = os.environ.copy()
                environment.pop("EE6008_VIDEOMAE_SCRATCH_NAME", None)
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module_name}"],
                    cwd=code_dir,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
