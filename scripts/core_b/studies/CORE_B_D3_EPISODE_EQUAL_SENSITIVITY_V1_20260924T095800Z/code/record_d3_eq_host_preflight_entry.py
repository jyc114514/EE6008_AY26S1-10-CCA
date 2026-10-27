"""Use the watcher-exact 13,517 MiB Stage0 admission threshold."""

from __future__ import annotations

import record_d3_eq_host_preflight as gate


def main() -> int:
    gate.STAGE0_REQUIRED_FREE_MIB = 13517
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
