# Expected Paper Metrics

These CSV files are immutable reference outputs reported in the paper:

- `controlled_seed7.csv`: controlled 5GC generator, chronological split;
- `telecomts_seed17.csv`: main open-data comparison;
- `telecomts_multiseed_raw.csv`: all models for seeds 7, 11, 17, 23, and 29;
- `telecomts_multiseed_summary.csv`: mean and sample standard deviation.

`python -m benchmark.verify_results` joins by model and seed, then compares
every metric with absolute tolerance `5e-4`. This tolerance verifies the four
decimal places displayed in the manuscript while allowing harmless
cross-platform BLAS rounding.
