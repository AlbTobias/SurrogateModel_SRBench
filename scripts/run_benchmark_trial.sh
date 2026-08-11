#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
algorithm="${1:?Usage: scripts/run_benchmark_trial.sh ALGORITHM SEED}"
seed="${2:?Usage: scripts/run_benchmark_trial.sh ALGORITHM SEED}"
problem="${3:-cantilever}"
config_file="${BENCHMARK_CONFIG:-configs/benchmark_suite_v1.json}"
case "$problem" in
  cantilever|borehole|piston) ;;
  *)
    echo "Unknown problem: $problem (expected cantilever, borehole, or piston)" >&2
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
  "$project_dir/results/$problem/benchmark/$algorithm" \
  "$project_dir/results/runtime/$problem/$algorithm/seed-$seed"

"$project_dir/.venv/bin/python" "$project_dir/surrogate/generate_$problem.py" \
  --output-dir "$project_dir/data/$problem"

docker run --rm \
  --volume "$project_dir:/workspace" \
  --workdir "/workspace/results/runtime/$problem/$algorithm/seed-$seed" \
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
    --output "/workspace/results/$problem/benchmark/$algorithm/seed-$seed.json" \
    --seed "$seed" \
    --profile benchmark \
    --config "/workspace/$config_file"
