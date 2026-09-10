# Core A public metadata

This directory contains only small metadata needed to identify the v3 task
population and configuration: task names, the label map, a sanitized split
summary, the train-only class-weight record, and the 12-task selection rule.

The raw G1 data, Parquet files, video files, final-test identifiers/labels,
feature caches, and model checkpoint are not included. The public split summary
records `final_test_accessed=0` and `final_test_metrics_computed=false`.
