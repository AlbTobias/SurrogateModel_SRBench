#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
algorithm="${1:?Usage: scripts/run_benchmark_trial.sh ALGORITHM SEED}"
seed="${2:?Usage: scripts/run_benchmark_trial.sh ALGORITHM SEED}"
config_file="${BENCHMARK_CONFIG:-configs/benchmark_pilot_v1.json}"
locked_image="$(awk -F= -v algorithm="$algorithm" \
  '$1 == algorithm {print substr($0, index($0, "=") + 1)}' \
  "$project_dir/containers/images.lock")"

if [[ -z "$locked_image" ]]; then
  echo "No image is locked for algorithm: $algorithm" >&2
  exit 2
fi

mkdir -p \
  "$project_dir/data/cantilever" \
  "$project_dir/results/cantilever/benchmark/$algorithm" \
  "$project_dir/results/runtime/$algorithm/seed-$seed"

"$project_dir/.venv/bin/python" "$project_dir/surrogate/generate_cantilever.py" \
  --output-dir "$project_dir/data/cantilever"

docker run --rm \
  --volume "$project_dir:/workspace" \
  --workdir "/workspace/results/runtime/$algorithm/seed-$seed" \
  --env PYTHONPATH=/workspace \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env JAX_ENABLE_X64=true \
  "$locked_image" \
  python -m experiment.evaluate_surrogate \
    --algorithm "$algorithm" \
    --train /workspace/data/cantilever/train.tsv.gz \
    --test /workspace/data/cantilever/test.tsv.gz \
    --output "/workspace/results/cantilever/benchmark/$algorithm/seed-$seed.json" \
    --seed "$seed" \
    --profile benchmark \
    --config "/workspace/$config_file"
