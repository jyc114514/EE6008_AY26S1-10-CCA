# Core A limitations and remaining evidence gaps

- The current Full-120 Stage1 matrix has one training seed. Episode bootstrap
  intervals quantify paired uncertainty on 617 development/validation episodes;
  they do not establish cross-seed stability.
- The recovery effective batch (32) differs from the locked batch-48 protocol.
- The D-P-L source run failed the BF16 merged-export equivalence diagnostic and
  has no source run-level metrics file. The accepted scientific number is the
  independently recomputed unmerged derived path.
- Dynamic/static comparisons are mixed by head and are not evidence of causal
  dynamics understanding. Static repeated-first-frame is a shortcut control,
  not a complete counterfactual for all visual information.
- State-only scores demonstrate a strong shortcut signal, but historical
  state-only and current STATE-CTRL streams are different and cannot be paired.
- The 12-task frozen and LoRA pilots are engineering/exploratory evidence only;
  they cannot be extrapolated to 120 classes.
- Historical Batch-4/32/48, R3D-18, pooled, attentive, and nonparametric rows
  use distinct protocols and should not be collapsed into one leaderboard.
- No final-test IDs, labels, predictions, or metrics are included or used.
- Raw data, features, checkpoints, full logs, environment directories, and
  third-party source trees are intentionally absent, so end-to-end public
  retraining is not claimed.
- Data, model, and third-party source licenses remain separate governance
  decisions. This branch adds no blanket license for materials that do not
  already have confirmed redistribution rights.

## Remaining work

The next scientific decision is whether to authorize a locked-protocol,
multi-seed confirmation. Stage2 was not launched in this release, and no new
training should be inferred from the presence of its plan or scripts.
