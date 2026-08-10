#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
algorithm="${1:?Usage: scripts/run_algorithm_docker.sh ALGORITHM}"
locked_image="$(awk -F= -v algorithm="$algorithm" \
  '$1 == algorithm {print substr($0, index($0, "=") + 1)}' \
  "$project_dir/containers/images.lock")"

if [[ -z "$locked_image" ]]; then
  echo "No image is locked for algorithm: $algorithm" >&2
  exit 2
fi

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required but was not found on PATH." >&2
  exit 127
}

mkdir -p \
  "$project_dir/data/cantilever" \
  "$project_dir/results/cantilever" \
  "$project_dir/results/runtime/$algorithm"
"$project_dir/.venv/bin/python" "$project_dir/surrogate/generate_cantilever.py" \
  --output-dir "$project_dir/data/cantilever"

docker pull "$locked_image"
docker run --rm \
  --volume "$project_dir:/workspace" \
  --workdir "/workspace/results/runtime/$algorithm" \
  --env PYTHONPATH=/workspace \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  "$locked_image" \
  python -m experiment.evaluate_surrogate \
    --algorithm "$algorithm" \
    --train /workspace/data/cantilever/train.tsv.gz \
    --test /workspace/data/cantilever/test.tsv.gz \
    --output "/workspace/results/cantilever/${algorithm}-docker.json" \
    --profile smoke \
    --population-size 100 \
    --iterations 10 \
    --time-limit 60
