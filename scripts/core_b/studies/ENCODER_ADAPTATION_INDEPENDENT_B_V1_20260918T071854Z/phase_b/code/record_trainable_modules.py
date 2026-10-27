#!/usr/bin/env python3
"""Write the pre-registered trainable-module map for the six B cells."""

from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
rows = [
    {"cell": "FT-D-FRZ", "adaptation": "FRZ", "video": "dynamic", "encoder_trainable": "none", "head_trainable": "all", "qkv_scope": "none"},
    {"cell": "FT-S-FRZ", "adaptation": "FRZ", "video": "static_last_frame", "encoder_trainable": "none", "head_trainable": "all", "qkv_scope": "none"},
    {"cell": "FT-D-LORA4", "adaptation": "LORA4", "video": "dynamic", "encoder_trainable": "blocks[8:12].attn.qkv.lora_A_q/lora_B_q/lora_A_v/lora_B_v", "head_trainable": "all", "qkv_scope": "Q/V only; K unchanged"},
    {"cell": "FT-S-LORA4", "adaptation": "LORA4", "video": "static_last_frame", "encoder_trainable": "blocks[8:12].attn.qkv.lora_A_q/lora_B_q/lora_A_v/lora_B_v", "head_trainable": "all", "qkv_scope": "Q/V only; K unchanged"},
    {"cell": "FT-D-DENSE4", "adaptation": "DENSE4", "video": "dynamic", "encoder_trainable": "blocks[8:12]", "head_trainable": "all", "qkv_scope": "Q/K/V plus attention/MLP/block norms inside top4"},
    {"cell": "FT-S-DENSE4", "adaptation": "DENSE4", "video": "static_last_frame", "encoder_trainable": "blocks[8:12]", "head_trainable": "all", "qkv_scope": "Q/K/V plus attention/MLP/block norms inside top4"},
]
path = ROOT / "configs/TRAINABLE_MODULES.csv"
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(path)
