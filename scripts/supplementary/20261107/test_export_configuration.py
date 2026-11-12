"""Publication-only path tests; no external assets or network operations."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PREFIX = 'EE6008_LEGACY_SOURCE_PREFIX'
REPLAY = 'EE6008_LEGACY_REPLAY_REL'


def load(relative, environment):
    with patch.dict(os.environ, environment, clear=True):
        spec = importlib.util.spec_from_file_location('export_configuration_test_module', ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class ExplicitConfiguration(unittest.TestCase):
    def test_missing_configuration_rejects_before_relocation(self):
        for relative, name in [('core_a/run_core_a_followup.py', 'relocate'),
                               ('core_b/run_cascade.py', 'relocated'),
                               ('core_b_experts/train_experts.py', 'source_path')]:
            module = load(relative, {})
            with self.assertRaisesRegex(ValueError, PREFIX):
                if name == 'relocate':
                    getattr(module, name)('/synthetic/legacy/assets/a', Path('synthetic-source'))
                else:
                    getattr(module, name)(Path('synthetic-source'), '/synthetic/legacy/assets/a')

    def test_exact_configured_prefix_relocates_to_supplied_source(self):
        settings = {PREFIX: '/synthetic/legacy/', REPLAY: 'outputs/example/recovery'}
        source = Path('synthetic-source')
        for relative, name in [('core_a/run_core_a_followup.py', 'relocate'),
                               ('core_b/run_cascade.py', 'relocated'),
                               ('core_b_experts/train_experts.py', 'source_path')]:
            module = load(relative, settings)
            if name == 'relocate':
                actual = getattr(module, name)('/synthetic/legacy/assets/a', source)
            else:
                actual = getattr(module, name)(source, '/synthetic/legacy/assets/a')
            self.assertEqual(actual, source / 'assets/a')

    def test_replay_directory_must_be_explicit_and_source_relative(self):
        for replay in ['', '../escape', str(Path.cwd() / 'absolute')]:
            module = load('core_b/run_cascade.py', {PREFIX: '/synthetic/legacy', REPLAY: replay})
            with self.assertRaisesRegex(ValueError, REPLAY):
                module.require_external_paths()
        module = load('core_b/run_cascade.py', {PREFIX: '/synthetic/legacy', REPLAY: 'outputs/example/recovery'})
        module.require_external_paths()


if __name__ == '__main__':
    unittest.main(verbosity=2)
