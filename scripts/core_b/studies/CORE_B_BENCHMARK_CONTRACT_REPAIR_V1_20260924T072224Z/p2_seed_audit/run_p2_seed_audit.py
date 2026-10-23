"""Run the P2 audit with explicit metric-table comparability and CSV schemas."""

from __future__ import annotations

from pathlib import Path

import p2_seed_audit as audit


_read_csv = audit.read_csv
_write_csv = audit.write_csv


def read_csv(path: Path):
    rows = _read_csv(path)
    if path.name != "D1_LEARNED_RESULTS.csv":
        return rows
    # Only `overall` has the same support definition as `natural_overall`.
    # Older low/mid/high rows are not the newer supported-only strata and must
    # not be mislabeled as exact-comparable observations.
    comparable = []
    for row in rows:
        copy = dict(row)
        if copy.get("stratum") == "overall":
            copy["stratum"] = "natural"
        else:
            copy["stratum"] = f"not_exact_comparable_{copy.get('stratum', '')}"
        comparable.append(copy)
    return comparable


def write_csv(path: Path, rows, columns):
    # The trajectory point table and aggregate diagnostics intentionally have
    # different fields; write their union explicitly with blank NA cells.
    union = list(dict.fromkeys(key for row in rows for key in row))
    _write_csv(path, rows, union or columns)


audit.read_csv = read_csv
audit.write_csv = write_csv
audit.main()
