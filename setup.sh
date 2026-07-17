#!/usr/bin/env bash
set -euo pipefail

uv sync --frozen

uv run --frozen python \
  vamos_ws/src/vamos/src/vamos/dowload_hf_model.py \
  --model_id "mateoguaman/vamos_navigation_only"