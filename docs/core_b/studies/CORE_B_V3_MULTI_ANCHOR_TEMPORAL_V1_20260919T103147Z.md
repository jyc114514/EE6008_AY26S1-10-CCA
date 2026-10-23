> Archived study summary for `CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z` (source snapshot `20260919`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# Core B v3：多 anchor × 多 horizon × 时序视频表示回传

## 1. 完成状态

```ini
V3_STATUS=complete
STAGE0_ACCEPTED=12/12
PILOT_ACCEPTED=12/12
FULL_TRAINING_ACCEPTED=36/36
PRIMARY_PREDICTIONS_ACCEPTED=36/36
FINAL_TEST_ACCESS=0
ENCODER_UPDATED=false
```

本轮没有按 pilot 结果筛选配置。三 seed、六配置、两个 horizon 全部训练完成并完成固定 last-checkpoint development inference。

## 2. 数据与 cache

| population | episodes | windows | tasks |
|---|---:|---:|---:|
| primary train | 2712 | 17757 | 120 |
| primary dev | 603 | 2322 | 118 |
| supplementary 1s dev | 64 | 64 | supplementary only |

canonical train/dev episode allowlist 仍为 2741/614。40 个 episode 没有合法 2s primary anchor（29 train、11 dev），保留在 inherited split 但排除 primary 2s population；2 个 dev task 没有 primary 2s support。1s supplementary population 没有进入训练。

Frozen temporal cache：

- D/S temporal tokens：`(20143, 2, 16, 768)`；
- raw encoder tokens：`(9216,768)`，reshape 为 `16×24×24×768` 后 spatial mean；
- 全部 finite，D/S mean absolute difference=`0.169256`；
- cache 大小约 `1.844 GiB`；
- encoder hash=`848a77c33cc9e6649ed2119cbea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`；
- cache 使用 OpenCV deterministic frame sampling，所有 frame <= anchor。

## 3. 训练与推理验收

- Seeds：`20260950 / 20260951 / 20260952`；
- 每格：40 epochs、11120 optimizer steps、epoch40 last checkpoint；
- Stage0：12/12；pilot formal：12/12；完整 formal：36/36；
- fresh reload prediction difference：所有 run 为 `0.0`；
- Full-28 直接预测，没有旧 Arm 拼接；
- fixed-last development inference：36/36；primary dev 每格 2322 anchors，H30 supplementary 每格64 anchors；
- GPU：物理 GPU2/3，最多同时两个项目 child；
- final-test access：0。

## 4. Aggregate primary development results

数值为三 seed mean±SD；MAE 为 train-only sigma 下的 standardized MAE，FD 为 native adjacent-difference MAE。

| horizon | cell | Arm MAE | Hand MAE | Full MAE | Hand FD |
|---:|---|---:|---:|---:|---:|
| 1s | S-HIST | 0.2480±0.0009 | 0.1413±0.0018 | 0.1946±0.0011 | 0.00635±0.00003 |
| 1s | F-POOL-D | 0.2483±0.0002 | 0.1401±0.0015 | 0.1942±0.0007 | 0.00635±0.00003 |
| 1s | F-TOK-D | 0.2526±0.0065 | 0.1466±0.0138 | 0.1996±0.0101 | 0.00652±0.00032 |
| 1s | F-TOK-S | 0.2484±0.0011 | 0.1414±0.0014 | 0.1949±0.0001 | 0.00636±0.00001 |
| 1s | V-TOK-D-A | 0.2584±0.0033 | 0.1609±0.0093 | 0.2097±0.0062 | 0.00675±0.00024 |
| 1s | V-TOK-S-A | 0.2626±0.0030 | 0.1731±0.0017 | 0.2178±0.0022 | 0.00689±0.00000 |
| 2s | S-HIST | 0.3172±0.0230 | 0.2474±0.0259 | 0.2823±0.0245 | 0.00675±0.00019 |
| 2s | F-POOL-D | 0.3063±0.0027 | 0.2298±0.0008 | 0.2681±0.0010 | 0.00663±0.00000 |
| 2s | F-TOK-D | 0.3051±0.0022 | 0.2296±0.0006 | 0.2673±0.0013 | 0.00663±0.00001 |
| 2s | F-TOK-S | 0.3047±0.0025 | 0.2324±0.0017 | 0.2686±0.0004 | 0.00664±0.00001 |
| 2s | V-TOK-D-A | 0.3254±0.0016 | 0.2522±0.0049 | 0.2888±0.0023 | 0.00676±0.00003 |
| 2s | V-TOK-S-A | 0.3283±0.0009 | 0.2605±0.0071 | 0.2944±0.0037 | 0.00682±0.00004 |

Slope baseline was included as a prespecified command-value extrapolation control and was substantially worse than Persistence on the primary population; it is not interpreted as physical velocity.

## 5. Feature contrasts and horizon interactions

The primary family is the locked 16-item family: four contrasts × two horizons × Arm/Hand. Aggregate crossed seed×episode intervals are in `analysis/PAIRED_EFFECTS_AGGREGATE.csv`.

Main qualitative results:

- `F-TOK-D vs S-HIST`: no stable universal video-beyond-state result. The aggregate Arm effect is negative at 1s with a CI below zero, while the 2s effect is positive but broad; Hand is not stable across seeds/horizons.
- `F-TOK-D vs F-POOL-D`: temporal tokens do not provide a consistent 1s gain; the 2s Arm difference is small and does not survive the full 16-item Holm interpretation as a general temporal claim.
- `F-TOK-D vs F-TOK-S`: dynamic/static differences are horizon- and group-dependent; no universal video-dynamics claim is made.
- `V-TOK-D-A vs V-TOK-S-A`: dynamic tokens outperform static tokens descriptively in both groups, but these cells intentionally contain no full state history; they are not strict video-only models because the anchor command remains present.
- Horizon interactions are reported with paired 2s−1s effects and CIs; they do not establish a universal interaction across all contrasts.

The largest seed instability is in `F-TOK-D` at 1s: one seed is materially worse than the other two. This is retained as training-seed variation, not removed by selecting a best seed.

## 6. Scientific interpretation

Supported:

1. Multi-anchor Full-28 training and fixed inference were engineered successfully under the locked split and horizon contract.
2. Video features can affect development performance in a configuration- and horizon-dependent way.
3. Dynamic-vs-static information has a clearer signal in the no-history anchor-controlled cells than in the full-history cells.

Not supported:

1. A universal video increment beyond state history.
2. A universal advantage of temporal tokens over pooled video.
3. A physical-trajectory or physical-velocity claim.
4. Final or independent generalization.

Old V2/fixed-anchor scores are reference-only: V3 uses different anchor populations, time-invariant normalizers, direct Full-28 prediction and multiple horizons. They are not directly interchangeable standardized scores.

## 7. Cost and next step

Measured GPU cost is approximately `8.681 GPU-hours`:

- feature extraction: `6.716`;
- Stage0: `0.027`;
- formal training: `1.892`;
- inference: `0.045`.

New cache disk use is approximately `1.844 GiB`, below the 100 GiB ceiling. No external process was signaled or modified.

No new training Prompt was executed after this matrix. A possible follow-up is a preregistered analysis of the 2s `F-TOK-D` signal and dynamic/static contrast, but it is `generated_not_executed`; final/held-out evaluation remains untouched.
