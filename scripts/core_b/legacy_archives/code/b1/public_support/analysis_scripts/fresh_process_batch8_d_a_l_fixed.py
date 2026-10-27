"""B1-only adapters extracted from the checkpoint-fix recovery source.

Source artifact: EE6008-B1-RECOVERY-20260904
Source SHA-256: 00b604181612359539fd55a6804e3a72130ae67feb03ce69fba1cc0011f2a754
Only the two functions called by the public B1 runner are included. The source
file is unchanged in its historical archive; training, inference, file IO, and
other experiment helpers are intentionally not imported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

def prepare_checkpoint_payload(
    *,
    protocol_id: str,
    run_identity: Mapping[str, object],
    epoch: int,
    global_optimizer_step: int,
    accumulation_state: Mapping[str, object],
    model_state_dict: Mapping[str, object],
    head_state_dict: Mapping[str, object],
    lora_adapter_state_dict: Mapping[str, object],
    optimizer_state_dict: Mapping[str, object],
    scheduler_state_dict: Mapping[str, object],
    scaler_state_dict: Mapping[str, object] | None,
    early_stopping_state: Mapping[str, object],
    data_provenance: Mapping[str, object],
    rng_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prepare the runner's complete checkpoint through one canonical seam."""

    from checkpoint_utils import build_checkpoint_payload

    return build_checkpoint_payload(
        protocol_id=protocol_id,
        run_identity=run_identity,
        epoch=epoch,
        global_optimizer_step=global_optimizer_step,
        accumulation_state=accumulation_state,
        model_state_dict=model_state_dict,
        head_state_dict=head_state_dict,
        lora_adapter_state_dict=lora_adapter_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        scaler_state_dict=scaler_state_dict,
        early_stopping_state=early_stopping_state,
        data_provenance=data_provenance,
        rng_state=rng_state,
    )
