"""CPU regression harness for the locked trainer's actual seed-registry AST expression."""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from types import SimpleNamespace


EQ_ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY_PATH = EQ_ROOT / "protocol/D3_EQ_SEED_REGISTRY.json"
SEEDS = (20261001, 20261002, 20261003)


def expected_init_statements(trainer: pathlib.Path) -> list[ast.stmt]:
    tree = ast.parse(trainer.read_text(encoding="utf-8"), filename=str(trainer))
    train_fn = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train"), None)
    if train_fn is None:
        raise AssertionError("trainer has no train() function")
    statements = []
    found_expected = False
    for node in train_fn.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if names & {"seed_registry", "expected_init"}:
            statements.append(node)
        found_expected = found_expected or "expected_init" in names
    if not found_expected:
        raise AssertionError("train() has no expected_init assignment")
    return statements


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_d3_eq_seed_registry_lookup_repair_v1.py TRAINER.py")
    trainer = pathlib.Path(sys.argv[1]).resolve()
    statements = expected_init_statements(trainer)
    compiled = compile(ast.Module(body=statements, type_ignores=[]), str(trainer), "exec")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    expected = registry["seeds"]
    for seed in SEEDS:
        namespace = {
            "json": json,
            "EQ_ROOT": EQ_ROOT,
            "args": SimpleNamespace(seed=seed),
        }
        try:
            exec(compiled, namespace, namespace)
        except KeyError as exc:
            raise AssertionError(f"actual expected_init expression failed for seed {seed}: {exc}") from exc
        observed = namespace.get("expected_init")
        want = expected[str(seed)]["init_state_hash"]
        if observed != want:
            raise AssertionError(f"seed {seed} init hash mismatch: observed={observed} expected={want}")
    print(json.dumps({"status": "PASS", "seeds_checked": list(SEEDS), "trainer": trainer.name, "cuda_used": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
