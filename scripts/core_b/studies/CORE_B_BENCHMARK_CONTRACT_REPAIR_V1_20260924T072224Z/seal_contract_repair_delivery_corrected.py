"""Run the delivery sealer with an explicit absent-V3-QA-file sentinel."""

from __future__ import annotations

import seal_contract_repair_delivery as sealer


_json_file = sealer.json_file
_absent_v3_acceptance = sealer.V3 / "qa/FINAL_ACCEPTANCE.json"


def json_file_with_v3_status(path):
    if path == _absent_v3_acceptance:
        return {"status": "not_present_under_this_name; source/checksum manifests and individual prediction markers are inventoried"}
    return _json_file(path)


sealer.json_file = json_file_with_v3_status
sealer.main()
