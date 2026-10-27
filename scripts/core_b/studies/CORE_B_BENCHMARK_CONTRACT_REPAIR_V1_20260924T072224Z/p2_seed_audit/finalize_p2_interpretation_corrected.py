"""Read-only alias for the descriptive P2 narrative key."""

from __future__ import annotations

import finalize_p2_interpretation as finalizer


_loads = finalizer.json.loads


def loads_with_alias(value, *args, **kwargs):
    result = _loads(value, *args, **kwargs)
    if isinstance(result, dict) and "posthoc_worst_F_TOK_D_H30_seed51_anchor_id" in result:
        result["posthoc_worst_seed51_h30_anchor_id"] = result["posthoc_worst_F_TOK_D_H30_seed51_anchor_id"]
    return result


finalizer.json.loads = loads_with_alias
finalizer.main()
