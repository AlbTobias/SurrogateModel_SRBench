#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
algorithms=(gplearn operon pysr geneticengine itea eql)
seeds=(${BENCHMARK_SEEDS:-42 43 44})
problems=(${BENCHMARK_PROBLEMS:-cantilever borehole piston})
failed=()

for problem in "${problems[@]}"; do
  for seed in "${seeds[@]}"; do
    for algorithm in "${algorithms[@]}"; do
      echo "Running benchmark trial: problem=$problem algorithm=$algorithm seed=$seed"
      if ! "$project_dir/scripts/run_benchmark_trial.sh" "$algorithm" "$seed" "$problem"; then
        failed+=("$problem:$algorithm:$seed")
      fi
    done
  done

  "$project_dir/.venv/bin/python" "$project_dir/scripts/summarize_benchmark.py" \
    --problem "$problem" \
    --config "$project_dir/${BENCHMARK_CONFIG:-configs/benchmark_suite_v2.json}"
done

if (( ${#failed[@]} > 0 )); then
  printf 'Failed trials: %s\n' "${failed[*]}" >&2
  exit 1
fi
