#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
locked_image="$(awk -F= '$1 == "gplearn" {print substr($0, index($0, "=") + 1)}' \
  "$project_dir/containers/images.lock")"
image_ref="${SRBENCH_IMAGE:-$locked_image}"

if [[ -z "$image_ref" ]]; then
  echo "No gplearn image was found in containers/images.lock." >&2
  exit 1
fi

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required but was not found on PATH." >&2
  exit 127
}

mkdir -p "$project_dir/data/cantilever" "$project_dir/results/cantilever"

# Dataset generation remains on the host and is deterministic.
"$project_dir/.venv/bin/python" "$project_dir/surrogate/generate_cantilever.py" \
  --output-dir "$project_dir/data/cantilever"

docker pull "$image_ref"
resolved_image="$(docker image inspect "$image_ref" --format '{{index .RepoDigests 0}}')"
printf '%s\n' "$resolved_image" > "$project_dir/results/cantilever/gplearn-image.txt"

docker run --rm \
  --volume "$project_dir:/workspace" \
  --workdir /workspace \
  "$resolved_image" \
  python -m experiment.evaluate_surrogate \
    --algorithm gplearn \
    --train data/cantilever/train.tsv.gz \
    --test data/cantilever/test.tsv.gz \
    --output results/cantilever/gplearn-docker.json
