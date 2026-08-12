#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
algorithm="${1:?Usage: scripts/run_benchmark_trial.sh ALGORITHM SEED [PROBLEM] [INPUT_SCALING]}"
seed="${2:?Usage: scripts/run_benchmark_trial.sh ALGORITHM SEED [PROBLEM] [INPUT_SCALING]}"
problem="${3:-cantilever}"
input_scaling="${4:-raw}"
config_file="${BENCHMARK_CONFIG:-configs/benchmark_suite_v3.json}"
config_path="$project_dir/$config_file"
if [[ ! -f "$config_path" ]]; then
  echo "Benchmark configuration not found: $config_path" >&2
  exit 2
fi
benchmark_name="$("$project_dir/.venv/bin/python" -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["name"])' "$config_path")"
case "$problem" in
  cantilever|borehole|piston) ;;
  *)
    echo "Unknown problem: $problem (expected cantilever, borehole, or piston)" >&2
    exit 2
    ;;
esac
case "$input_scaling" in
  raw|domain_minmax) ;;
  *)
    echo "Unknown input scaling: $input_scaling (expected raw or domain_minmax)" >&2
    exit 2
    ;;
esac
locked_image="$(awk -F= -v algorithm="$algorithm" \
  '$1 == algorithm {print substr($0, index($0, "=") + 1)}' \
  "$project_dir/containers/images.lock")"

if [[ -z "$locked_image" ]]; then
  echo "No image is locked for algorithm: $algorithm" >&2
  exit 2
fi

mkdir -p \
  "$project_dir/data/$problem" \
  "$project_dir/results/$problem/$benchmark_name/$input_scaling/$algorithm" \
  "$project_dir/results/$problem/$benchmark_name/$input_scaling/failures/$algorithm" \
  "$project_dir/results/runtime/$benchmark_name/$problem/$input_scaling/$algorithm/seed-$seed"

failure_file="$project_dir/results/$problem/$benchmark_name/$input_scaling/failures/$algorithm/seed-$seed.json"
record_failure() {
  exit_code=$?
  "$project_dir/.venv/bin/python" -c \
    'import json, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"problem": sys.argv[2], "input_scaling": sys.argv[3], "algorithm": sys.argv[4], "seed": int(sys.argv[5]), "status": "failed", "exit_code": int(sys.argv[6])}, indent=2) + "\n")' \
    "$failure_file" "$problem" "$input_scaling" "$algorithm" "$seed" "$exit_code"
}
trap record_failure ERR

"$project_dir/.venv/bin/python" "$project_dir/surrogate/generate_$problem.py" \
  --output-dir "$project_dir/data/$problem"

docker run --rm \
  --volume "$project_dir:/workspace" \
  --workdir "/workspace/results/runtime/$benchmark_name/$problem/$input_scaling/$algorithm/seed-$seed" \
  --env PYTHONPATH=/workspace \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env JAX_ENABLE_X64=true \
  "$locked_image" \
    python -m experiment.evaluate_surrogate \
    --algorithm "$algorithm" \
    --problem "$problem" \
    --train "/workspace/data/$problem/train.tsv.gz" \
    --test "/workspace/data/$problem/test.tsv.gz" \
    --output "/workspace/results/$problem/$benchmark_name/$input_scaling/$algorithm/seed-$seed.json" \
    --seed "$seed" \
    --input-scaling "$input_scaling" \
    --profile benchmark \
    --config "/workspace/$config_file"

trap - ERR
rm -f "$failure_file"
