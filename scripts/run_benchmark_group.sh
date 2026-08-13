#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${BENCHMARK_CONFIG:-configs/benchmark_suite_v8.json}"
config_path="$project_dir/$config_file"
if [[ ! -f "$config_path" ]]; then
  echo "Benchmark configuration not found: $config_path" >&2
  exit 2
fi
mapfile -t configured_algorithms < <("$project_dir/.venv/bin/python" -c \
  'import json, sys; print(*json.load(open(sys.argv[1]))["algorithms"], sep="\n")' "$config_path")
mapfile -t configured_seeds < <("$project_dir/.venv/bin/python" -c \
  'import json, sys; print(*json.load(open(sys.argv[1]))["seeds"], sep="\n")' "$config_path")
mapfile -t configured_problems < <("$project_dir/.venv/bin/python" -c \
  'import json, sys; print(*json.load(open(sys.argv[1]))["problems"], sep="\n")' "$config_path")
benchmark_name="$("$project_dir/.venv/bin/python" -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$config_path")"
analysis_timeout="$("$project_dir/.venv/bin/python" -c \
  'import json, sys; c=json.load(open(sys.argv[1])); print(c.get("execution_controls", {}).get("expression_analysis_timeout_seconds_per_stage", c.get("expression_analysis_timeout_seconds_per_stage", 60)))' "$config_path")"
algorithms=(${BENCHMARK_ALGORITHMS:-${configured_algorithms[*]}})
seeds=(${BENCHMARK_SEEDS:-${configured_seeds[*]}})
problems=(${BENCHMARK_PROBLEMS:-${configured_problems[*]}})
input_scalings=(${BENCHMARK_SCALINGS:-raw domain_minmax})
failed=()

for problem in "${problems[@]}"; do
  for input_scaling in "${input_scalings[@]}"; do
    for seed in "${seeds[@]}"; do
      for algorithm in "${algorithms[@]}"; do
        echo "Running benchmark trial: problem=$problem scaling=$input_scaling algorithm=$algorithm seed=$seed"
        if "$project_dir/scripts/run_benchmark_trial.sh" \
          "$algorithm" "$seed" "$problem" "$input_scaling"; then
          :
        else
          failed+=("$problem:$input_scaling:$algorithm:$seed")
        fi
      done
    done

    "$project_dir/.venv/bin/python" "$project_dir/scripts/analyze_benchmark_results.py" \
      --problem "$problem" \
      --benchmark "$benchmark_name" \
      --config "$config_path" \
      --input-scaling "$input_scaling" \
      --timeout "$analysis_timeout" \
      --jobs 2

    "$project_dir/.venv/bin/python" "$project_dir/scripts/summarize_benchmark.py" \
      --problem "$problem" \
      --input-scaling "$input_scaling" \
      --config "$config_path"
  done
done

if (( ${#failed[@]} > 0 )); then
  printf 'Failed trials: %s\n' "${failed[*]}" >&2
  exit 1
fi
