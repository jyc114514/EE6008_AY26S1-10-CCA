"""Report-only field aliases for the P0 regression JSON schema."""

from __future__ import annotations

import build_p0_final_report as builder


_loads = builder.json.loads


def loads_with_report_aliases(value, *args, **kwargs):
    result = _loads(value, *args, **kwargs)
    if isinstance(result, dict) and "toy_two_episode" in result:
        toy = result["toy_two_episode"]
        toy.setdefault("legacy_value", toy.get("toy_legacy_value"))
        toy.setdefault("episode_equal_value", toy.get("toy_episode_equal_value"))
        toy.setdefault("legacy_gradient", toy.get("toy_legacy_gradient"))
        toy.setdefault("episode_equal_gradient", toy.get("toy_episode_equal_gradient"))
    return result


builder.json.loads = loads_with_report_aliases
builder.main()
