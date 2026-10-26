# V-JEPA2 source subset provenance

This directory contains the exact B1-required source subset from the official Meta research repository [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2), pinned to commit [`204698b45b3712590f06245fbfba32d3be539812`](https://github.com/facebookresearch/vjepa2/commit/204698b45b3712590f06245fbfba32d3be539812) (`Fix figure (#143)`, 2026-03-23).

| File | Git blob SHA-1 | SHA-256 | Bytes |
|---|---|---|---:|
| `src/models/attentive_pooler.py` | `3bc2802c26e2f4a6af9d5d930deb8e488f887e25` | `9be7047d6bfce50575956a57e36d87a37bf63ae84ec92a9ba8649bf1ab7d5feb` | 4,372 |
| `src/models/utils/modules.py` | `4b36564f56bdfd6ac7e51f7fedc5ee6b8c498cd3` | `b93f6c7e0747deb216419c000c2878f11a9189024a9adeacfd437e172396dff0` | 23,001 |
| `src/utils/tensors.py` | `bf7bfafc67c69bfd10f663ce259c4fd7bfe6ab45` | `782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8` | 1,832 |
| `LICENSE` | `a45a376fb0fcd4a3de06b6c096e62028929a2dcb` | `cf9b17822d1fcd4ff32ccbe14183386fb3adf6f2ff92dc184130823f7fc28173` | 1,087 |

The source files above are byte-identical to the official commit. The included [`LICENSE`](LICENSE) is the upstream MIT license. The four `__init__.py` files below `src/` are local package markers added to make this subset an isolated package and prevent another project’s generic `src` namespace from being imported. They are not upstream files and do not alter model code.

`attentive_pooler.py` imports `Block`, `CrossAttention`, and `CrossAttentionBlock` from `src.models.utils.modules`, and `trunc_normal_` from `src.utils.tensors`. `modules.py` imports `drop_path` from `timm.models.layers`; the public B1 dependency set pins `timm==1.0.28`. No model weights or checkpoints are included. A small randomly initialized CPU instance is tested; B1’s original 768-dimensional, depth-4 construction is retained in the runner and is not instantiated in this check.
