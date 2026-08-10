#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
algorithms=(gplearn operon pysr geneticengine itea)
failed=()

for algorithm in "${algorithms[@]}"; do
  echo "Running $algorithm"
  if ! "$project_dir/scripts/run_algorithm_docker.sh" "$algorithm"; then
    failed+=("$algorithm")
  fi
done

if (( ${#failed[@]} > 0 )); then
  printf 'Failed algorithms: %s\n' "${failed[*]}" >&2
  exit 1
fi

