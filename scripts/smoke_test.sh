#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir/experiment"

python3 -m pytest -q test_algorithm.py --ml gplearn
python3 -m pytest -q test_evaluate_model.py --ml sklearn_linear

