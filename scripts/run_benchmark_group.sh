#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
algorithms=(gplearn operon pysr geneticengine itea eql)
seeds=(${BENCHMARK_SEEDS:-42 43 44})
problems=(${BENCHMARK_PROBLEMS:-cantilever borehole piston})
input_scalings=(${BENCHMARK_SCALINGS:-raw domain_minmax})
failed=()

for problem in "${problems[@]}"; do
  for input_scaling in "${input_scalings[@]}"; do
    for seed in "${seeds[@]}"; do
      for algorithm in "${algorithms[@]}"; do
        echo "Running benchmark trial: problem=$problem scaling=$input_scaling algorithm=$algorithm seed=$seed"
        if ! "$project_dir/scripts/run_benchmark_trial.sh" \
          "$algorithm" "$seed" "$problem" "$input_scaling"; then
          failed+=("$problem:$input_scaling:$algorithm:$seed")
        fi
      done
    done

    "$project_dir/.venv/bin/python" "$project_dir/scripts/summarize_benchmark.py" \
      --problem "$problem" \
      --input-scaling "$input_scaling" \
      --config "$project_dir/${BENCHMARK_CONFIG:-configs/benchmark_suite_v3.json}"
  done
done

if (( ${#failed[@]} > 0 )); then
  printf 'Failed trials: %s\n' "${failed[*]}" >&2
  exit 1
fi
