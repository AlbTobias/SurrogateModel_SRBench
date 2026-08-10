#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

.venv/bin/python surrogate/generate_cantilever.py
.venv/bin/python -m experiment.evaluate_surrogate \
  --algorithm gplearn \
  --train data/cantilever/train.tsv.gz \
  --test data/cantilever/test.tsv.gz \
  --output results/cantilever/gplearn-local.json
