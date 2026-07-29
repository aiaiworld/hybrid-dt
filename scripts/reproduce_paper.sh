#!/usr/bin/env bash
set -euo pipefail

python -m benchmark.run_benchmark \
  --timesteps 1800 \
  --window 12 \
  --horizon 3 \
  --seed 7 \
  --outdir outputs/controlled_seed7

python -m benchmark.run_telecomts_benchmark \
  --samples 800 \
  --input-len 96 \
  --seed 17 \
  --cache data/telecomts_800_even.jsonl.gz \
  --outdir outputs/telecomts_seed17

python -m benchmark.run_multiseed \
  --seeds 7 11 17 23 29 \
  --samples 800 \
  --input-len 96 \
  --cache data/telecomts_800_even.jsonl.gz \
  --outdir outputs/telecomts_multiseed

python -m benchmark.verify_results
