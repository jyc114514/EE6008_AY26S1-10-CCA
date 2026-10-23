"""Compatibility alias for the final P0 report field; inputs stay immutable."""

from __future__ import annotations

import finalize_p0_audit as finalizer


_loads = finalizer.json.loads


def loads_with_field_alias(value, *args, **kwargs):
    result = _loads(value, *args, **kwargs)
    if isinstance(result, dict) and "partial_cycle" in result:
        partial = result["partial_cycle"]
        if "legacy_divide_by_2_vs_correct_max_abs_parameter_gradient_difference" not in partial:
            partial["legacy_divide_by_2_vs_correct_max_abs_parameter_gradient_difference"] = partial["legacy_mean_product_divide2_vs_correct_gradient_difference"]
    return result


finalizer.json.loads = loads_with_field_alias
finalizer.main()
