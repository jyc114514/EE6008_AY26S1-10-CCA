> Archived study summary for `CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z` (source snapshot `20261013`). Its split and access statements describe this study only; current project access/status is summarized in the current release documentation.

# V2 实验结果回传

状态：PASS。这是开发实验；负收益照实保留，不构成独立确认。

## 样本与执行

| Task | Train ep/anchors | Fit ep/anchors | Tune ep/anchors | Dev ep/anchors |
|---:|---:|---:|---:|---:|
| 119 | 56/362 | 44/287 | 12/75 | 12/48 |
| 94 | 56/345 | 44/268 | 12/77 | 12/48 |
| 90 | 48/203 | 38/155 | 10/48 | 11/41 |
| 141 | 30/232 | 24/188 | 6/44 | 5/20 |
| 230 | 30/138 | 24/112 | 6/26 | 7/28 |

Fits accepted: screening 36/36; formal 36/36; waypoint 6/6.
Corrected-split accepted fits: screening 36/36, formal 36/36, waypoint 6/6; corrected-split attempts 78.
Superseded dataset-salted run archived 78 completed logical fits and 8 failed GPU attempts; total historical training attempts 164. Old dev scores were not used for corrected selection.
Recorded technical/QA failure events 35 in qa/FAILED_ATTEMPTS.jsonl.
Physical GPUs used across both protocol versions: 1, 2; watcher cap 2; corrected fit/inference runtime 3145.1/22.1s; superseded runtime 3161.8/19.1s; leases released=True.

## Inner-tune LR / epoch choices

| Model | LR | Epoch | Tune J | Valid candidates |
|---|---:|---:|---:|---:|
| G-S | 0.001 | 160 | 0.247475 | 9/9 |
| G-V | 0.0005 | 160 | 0.247568 | 9/9 |
| S-S-119 | 0.001 | 80 | 0.303320 | 9/9 |
| S-V-119 | 0.0005 | 80 | 0.292342 | 9/9 |
| S-S-94 | 0.0005 | 160 | 0.231738 | 9/9 |
| S-V-94 | 0.0005 | 160 | 0.231429 | 9/9 |
| S-S-90 | 0.001 | 160 | 0.209246 | 9/9 |
| S-V-90 | 0.0005 | 160 | 0.210763 | 9/9 |
| S-S-141 | 0.002 | 160 | 0.244814 | 9/9 |
| S-V-141 | 0.001 | 160 | 0.244967 | 9/9 |
| S-S-230 | 0.0005 | 80 | 0.256115 | 9/9 |
| S-V-230 | 0.0005 | 160 | 0.253794 | 9/9 |

## Absolute exposed-dev losses

Means over the three formal seeds, with window errors averaged within episode and episodes equally weighted. Native command units are unidentified. safe-FD uses accepted primary-train scales; the 29 future-internal differences and anchor-to-first-step seam are separate.

| Task | Program | Position MAE | safe-FD L1 | Seam safe-FD L1 |
|---:|---|---:|---:|---:|
| 119 | G-S | 0.2022 | 1.0710 | 1.0223 |
| 119 | G-V | 0.2051 | 1.0417 | 1.1319 |
| 119 | S-S-119 | 0.2129 | 1.0696 | 1.2308 |
| 119 | S-V-119 | 0.2140 | 1.0606 | 1.4273 |
| 119 | B-AUX_ONLY | 0.2117 | 1.0602 | 1.3999 |
| 119 | B-GOAL_COND | 0.2130 | 1.0605 | 1.3297 |
| 94 | G-S | 0.1440 | 0.7495 | 0.7395 |
| 94 | G-V | 0.1451 | 0.7416 | 0.7438 |
| 94 | S-S-94 | 0.1503 | 0.7421 | 0.8801 |
| 94 | S-V-94 | 0.1480 | 0.7447 | 0.8663 |
| 90 | G-S | 0.1626 | 0.5712 | 0.5943 |
| 90 | G-V | 0.1626 | 0.5712 | 0.6002 |
| 90 | S-S-90 | 0.1488 | 0.5970 | 0.6488 |
| 90 | S-V-90 | 0.1462 | 0.5845 | 0.6507 |
| 141 | G-S | 0.1260 | 0.8089 | 0.6220 |
| 141 | G-V | 0.1257 | 0.7908 | 0.6414 |
| 141 | S-S-141 | 0.1487 | 0.8494 | 0.7113 |
| 141 | S-V-141 | 0.1346 | 0.8076 | 0.6406 |
| 230 | G-S | 0.1531 | 0.9534 | 0.8566 |
| 230 | G-V | 0.1531 | 0.9534 | 0.8616 |
| 230 | S-S-230 | 0.1456 | 0.9413 | 1.0542 |
| 230 | S-V-230 | 0.1469 | 0.9417 | 0.9509 |

## Paired contrasts

Positive values mean benefit. ALL_SEED_D_V_I.csv has every task/seed; SUMMARY_STATISTICS.csv has task and macro mean/sample SD/median; ALL_SEED_LOO.csv has leave-one-task-out values; CONDITIONAL_CI.csv has the paired 5,000-draw source-group bootstrap.

| Scope | Contrast | Metric | Mean | Sample SD | Median |
|---|---|---|---:|---:|---:|
| five_task_macro / ALL5_OR_AVAILABLE | D_S | position_mae | -0.0037170894040788327 | 6.952628879503396e-05 | -0.0037491245032874377 |
| task / 119 | D_S | position_mae | -0.010756032322844769 | 0.0024575745082563924 | -0.011604585420753738 |
| task / 94 | D_S | position_mae | -0.006359968677649884 | 0.0013048537190183953 | -0.005770797879085382 |
| task / 90 | D_S | position_mae | 0.013717820693339164 | 0.0011196498152624536 | 0.013101768889517085 |
| task / 141 | D_S | position_mae | -0.022662091108731446 | 0.003289237986258316 | -0.02448975827469027 |
| task / 230 | D_S | position_mae | 0.0074748243954927684 | 0.0004900624798753256 | 0.0075316929338433425 |
| five_task_macro / ALL5_OR_AVAILABLE | D_V | position_mae | 0.0003920975672627225 | 0.0007551403454972766 | 1.978161713387383e-05 |
| task / 119 | D_V | position_mae | -0.008841573317375886 | 0.0047084995549890145 | -0.00967762950842499 |
| task / 94 | D_V | position_mae | -0.00285670104048145 | 0.00100895786343577 | -0.0030157919116837184 |
| task / 90 | D_V | position_mae | 0.0163369266631485 | 0.0009524498257524119 | 0.016387192539736833 |
| task / 141 | D_V | position_mae | -0.008893544105598625 | 0.0017989140392131184 | -0.008304722515044272 |
| task / 230 | D_V | position_mae | 0.006215379636621073 | 0.0009150581976869884 | 0.005955686186810971 |
| five_task_macro / ALL5_OR_AVAILABLE | V_G | position_mae | -0.0007828514981177186 | 0.0014556115947331294 | -0.00043741898266427826 |
| task / 119 | V_G | position_mae | -0.0029439483032665617 | 0.0021555625757116973 | -0.003176591522241784 |
| task / 94 | V_G | position_mae | -0.0011630504216741482 | 0.0036859746636656934 | -0.002040411683245591 |
| task / 90 | V_G | position_mae | -2.3333857674751046e-05 | 7.30424627949864e-05 | -1.4093795404263032e-05 |
| task / 141 | V_G | position_mae | 0.000292917321187867 | 0.00466263146370941 | 0.0001926430485776065 |
| task / 230 | V_G | position_mae | -7.684222916099863e-05 | 9.620910556526043e-05 | -9.337490363972578e-05 |
| five_task_macro / ALL5_OR_AVAILABLE | V_S | position_mae | 0.0033263354732238366 | 0.00177557412460775 | 0.004237877727905298 |
| task / 119 | V_S | position_mae | -0.0010294892977976793 | 0.0010153715183683692 | -0.0007581769574939723 |
| task / 94 | V_S | position_mae | 0.0023402172154942855 | 0.0027320504097931193 | 0.001746682714666159 |
| task / 90 | V_S | position_mae | 0.0025957721121345847 | 0.0005630109578187261 | 0.0022976848473631184 |
| task / 141 | V_S | position_mae | 0.014061464324320686 | 0.009442471471760193 | 0.016519546068766933 |
| task / 230 | V_S | position_mae | -0.001336286988032694 | 0.0006428558333380412 | -0.001166836108105307 |
| five_task_macro / ALL5_OR_AVAILABLE | I | position_mae | 0.004109186971341556 | 0.0006856501328166688 | 0.0037689061204213116 |
| task / 119 | I | position_mae | 0.0019144590054688826 | 0.002994691224373592 | 0.0029992575397553844 |
| task / 94 | I | position_mae | 0.0035032676371684335 | 0.00224067379403208 | 0.0024378035666176023 |
| task / 90 | I | position_mae | 0.0026191059698093357 | 0.0006292650610694877 | 0.002258570329428583 |
| task / 141 | I | position_mae | 0.01376854700313282 | 0.005049588133408832 | 0.016326903020189326 |
| task / 230 | I | position_mae | -0.0012594447588716955 | 0.0007208791582826655 | -0.0010031224476603207 |
| task_119 / 119 | B_GAIN | position_mae | -0.001293317638776011 | 0.0015656743540598253 | -0.002063759986007968 |
| task_119 / 119 | AUX_VS_SV | position_mae | 0.002200424705839162 | 0.006081439316355153 | 0.003455532804073569 |
| task_119 / 119 | GOAL_VS_SV | position_mae | 0.0009071070670631509 | 0.004701490236538499 | 0.0011310500479764785 |
| five_task_macro / ALL5_OR_AVAILABLE | D_S | safe_fd_l1 | -0.009080621452178325 | 0.0018524529064880013 | -0.009763747255233856 |
| task / 119 | D_S | safe_fd_l1 | 0.0013508917382077563 | 0.00990002239515528 | 0.0004596482928793666 |
| task / 94 | D_S | safe_fd_l1 | 0.007428469016230621 | 0.0038655187115685394 | 0.009109491475353226 |
| task / 90 | D_S | safe_fd_l1 | -0.02578630105242526 | 0.004185512560423343 | -0.02457964112419142 |
| task / 141 | D_S | safe_fd_l1 | -0.04048616838976086 | 0.01813745269042446 | -0.04381110935040733 |
| task / 230 | D_S | safe_fd_l1 | 0.012090001426856123 | 0.001252468353417187 | 0.012639146883541397 |
| five_task_macro / ALL5_OR_AVAILABLE | D_V | safe_fd_l1 | -0.00805860028726652 | 0.005583828144083959 | -0.00962762024737156 |
| task / 119 | D_V | safe_fd_l1 | -0.018868135804182673 | 0.00900207079246104 | -0.023376317764950638 |
| task / 94 | D_V | safe_fd_l1 | -0.003076392658590499 | 0.007265629194668943 | -0.005417117516063108 |
| task / 90 | D_V | safe_fd_l1 | -0.0132751556540777 | 0.002578710339194344 | -0.012082114822733359 |
| task / 141 | D_V | safe_fd_l1 | -0.016711173245128868 | 0.0114181190386957 | -0.018010284433279233 |
| task / 230 | D_V | safe_fd_l1 | 0.011637855925647145 | 0.0007790433807546161 | 0.011523726651552213 |
| five_task_macro / ALL5_OR_AVAILABLE | V_G | safe_fd_l1 | 0.011052821037148663 | 0.0024967716227828874 | 0.012418877845188248 |
| task / 119 | V_G | safe_fd_l1 | 0.029239717061865456 | 0.004875524820349492 | 0.029269092058036073 |
| task / 94 | V_G | safe_fd_l1 | 0.00791674160529062 | 0.006442522472922128 | 0.008257975884962354 |
| task / 90 | V_G | safe_fd_l1 | 2.5018446549599876e-05 | 9.124444301700985e-05 | 3.604432574288996e-05 |
| task / 141 | V_G | safe_fd_l1 | 0.018101153062869746 | 0.009362006856180249 | 0.019502788536028937 |
| task / 230 | V_G | safe_fd_l1 | -1.8524990832102795e-05 | 3.369664893849532e-05 | -3.5842655479179086e-05 |
| five_task_macro / ALL5_OR_AVAILABLE | V_S | safe_fd_l1 | 0.01207484220206047 | 0.00900733019093198 | 0.013435376427883306 |
| task / 119 | V_S | safe_fd_l1 | 0.009020689519475026 | 0.007396604843208403 | 0.012617041791712857 |
| task / 94 | V_S | safe_fd_l1 | -0.002588120069530501 | 0.001944369761886184 | -0.003632224583696164 |
| task / 90 | V_S | safe_fd_l1 | 0.01253616384489716 | 0.001781030166882247 | 0.012533570627200952 |
| task / 141 | V_S | safe_fd_l1 | 0.04187614820750174 | 0.038667056755943674 | 0.04530361345315703 |
| task / 230 | V_S | safe_fd_l1 | -0.0004706704920410809 | 0.0006926434467396903 | -0.0004861169683674227 |
| five_task_macro / ALL5_OR_AVAILABLE | I | safe_fd_l1 | 0.0010220211649118055 | 0.006807366176141724 | 0.0008668830115098336 |
| task / 119 | I | safe_fd_l1 | -0.020219027542390428 | 0.003592218481515966 | -0.02016906630301807 |
| task / 94 | I | safe_fd_l1 | -0.01050486167482112 | 0.004864678574062471 | -0.011890200468658518 |
| task / 90 | I | safe_fd_l1 | 0.01251114539834756 | 0.0016903264659548203 | 0.012497526301458062 |
| task / 141 | I | safe_fd_l1 | 0.023774995144631994 | 0.029322510071951524 | 0.025800824917128096 |
| task / 230 | I | safe_fd_l1 | -0.0004521455012089781 | 0.0006920134189863303 | -0.0005064259323617204 |
| task_119 / 119 | B_GAIN | safe_fd_l1 | -0.00031437624680608067 | 0.009518553045479732 | 0.002311899732796663 |
| task_119 / 119 | AUX_VS_SV | safe_fd_l1 | 0.00035383684466701776 | 0.00483389790570526 | 0.0030143218772626934 |
| task_119 / 119 | GOAL_VS_SV | safe_fd_l1 | 3.9460597860937106e-05 | 0.009714941938757049 | -0.0029139686703194556 |

## Predicted waypoint endpoint

Normalized endpoint errors use the accepted canonical primary-train target_sigma; native units remain unidentified.

| Seed | AUX_ONLY endpoint MAE (norm) | GOAL_COND endpoint MAE (norm) | B_GOAL_HEAD_GAIN | AUX_ONLY native MAE | GOAL_COND native MAE |
|---:|---:|---:|---:|---:|---:|
| 202610131 | 0.3273206660078974 | 0.32510882654799556 | 0.0022 | 0.11579975623340032 | 0.11505210462500837 |
| 202610132 | 0.335369964658053 | 0.33755544805007204 | -0.0022 | 0.11940537389807017 | 0.12042360733112002 |
| 202610133 | 0.33290433858052826 | 0.33229922872211753 | 0.0006 | 0.11799053305840278 | 0.11775689562088182 |

## Limitations and artifacts

- Waypoint goal-head endpoint MAE is reported separately in WAYPOINT_ENDPOINT.csv; action forecast comparisons remain in WAYPOINT.csv.
- Dev-label access audit: the current run read 5,550 future-action rows only after its valid 42/42 prediction seal; labels entered the scorer only, never model forward. Two earlier archived attempts each read 5,550 rows: the invalid dataset-salted split after a seal with 36 predictions and 6 missing, and a corrected-split zero-prediction scoring attempt after a seal with 0 predictions. The latter was invalid; both earlier score outputs were excluded from corrected selection and their numeric values were not opened. These are 16,650 cumulative row reads across three attempts (re-reads, not unique rows). Final/heldout payload access is 0.
- Zero-prediction scoring archive: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/archive/CORRECTED_SPLIT_ZERO_PREDICTION_SCORING_ATTEMPT_20261013T092800Z/ARCHIVE_SUMMARY.json
- Parent aggregate metadata exposure is preserved. Cache lineage retains partial historical extractor binding. Collection/session independence is unverified at episode/source level.
- The dataset-salted split run was invalidated because its tune groups differed from the literal V2 source-key hash. Its fits, predictions, and scores are preserved in the archive and excluded from corrected selection. The earlier sealed dev-label exposure is recorded; corrected LR/epoch selection uses inner-tune only.
- TRAINING_SOURCE_PROVENANCE.csv binds weights, normalizers, model/scheduler code, and the training-script version by archived source hash and manifest completion timestamp; older fit manifests did not embed their own script SHA.
- Bootstrap intervals are conditional development diagnostics; they do not establish independent confirmation. Human figure review is pending.
- The palette audit passed white-background graphical contrast; four pairwise grayscale distinctions were flagged heuristically and are paired with distinct markers/line styles/hatching. This is not a publisher-compliance certification.
- Absolute losses: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/ABSOLUTE_LOSSES.csv
- D/V/I: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/ALL_SEED_D_V_I.csv
- LOO: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/ALL_SEED_LOO.csv
- Conditional CI: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/CONDITIONAL_CI.csv
- Waypoint action contrasts: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/WAYPOINT.csv
- Waypoint endpoint errors: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/WAYPOINT_ENDPOINT.csv
- Training source provenance: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/TRAINING_SOURCE_PROVENANCE.csv
- Dynamic GPU leases/resources: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/results/GPU_LEASE_LEDGER.csv
- Figures: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/figures


Checksum manifest: server-only://project-artifacts/reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/CHECKSUMS.sha256
