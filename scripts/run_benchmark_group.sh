#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
algorithms=(gplearn operon pysr geneticengine itea eql)
seeds=(${BENCHMARK_SEEDS:-42 43 44})
failed=()

for seed in "${seeds[@]}"; do
  for algorithm in "${algorithms[@]}"; do
    echo "Running benchmark trial: algorithm=$algorithm seed=$seed"
    if ! "$project_dir/scripts/run_benchmark_trial.sh" "$algorithm" "$seed"; then
      failed+=("$algorithm:$seed")
    fi
  done
done

"$project_dir/.venv/bin/python" "$project_dir/scripts/summarize_benchmark.py"

if (( ${#failed[@]} > 0 )); then
  printf 'Failed trials: %s\n' "${failed[*]}" >&2
  exit 1
fi
