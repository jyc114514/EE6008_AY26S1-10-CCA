> Archived study summary for `POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z` (source snapshot `20260913`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# JY-02 Post-freeze Persistence-gated residual factorial exploration

状态：`LOCKED_BEFORE_TRAINING`。这是隔离的 single-seed exploratory sandbox，不是 final evaluation，也不改变 freeze root、pending final protocol 或冻结的 `JY02-F-R-D`。

- 8 格：STATE/FUSION × shared/split Arm+Hand head × ungated/group-time gated residual。
- exploratory seed：`20260924`，在结果前锁定；不使用 Stage2/final seeds。
- input：development `JY02_TARGETS.npz`，train/dev=`2741/614`，cached features，no raw-video/LoRA/encoder update。
- GPU：遵循实际 project watcher allowlist GPU2/3；最多 2 个 child；GPU0 不使用。
- final-test：`FINAL_TEST_EXPLICITLY_AUTHORIZED=false`，target access=`0`。

先完成全部 8 格 acceptance，再进行 factorial/paired bootstrap 和 figures。即使出现新候选，也必须另做 fresh-seed confirmation 和新 freeze，不得自动晋级。
