> Archived study summary for `JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z` (source snapshot `20260912`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# JY-02 Persistence-Residual × Temporal Decoder

这是独立的 Core B JY-02 2×2×2 development experiment。输入为 cached
STATE 或 chronological dynamic Frozen feature + STATE；target 为 absolute
或 residual-to-persistence；decoder 为 direct chunk 或 fixed small temporal
GRU。Frozen encoder/cache 不变，禁止 LoRA、raw-video 和 final-test。

Stage1 先运行 8 格 single seed。只有全部 8 格通过工程/数据 gate，且候选
同时满足相对 Persistence、同 branch absolute-direct、first-difference、TV、
high-motion/late/endpoint、hand/full 和 Fusion-vs-STATE 条件，才允许 Stage2
4 configs × 3 fresh seeds。否则 Stage2 必须记录 `not_launched_by_gate`。
