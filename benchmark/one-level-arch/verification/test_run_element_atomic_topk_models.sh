#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/run_element_atomic_topk.sh"

bash -n "$runner"
"$runner" --self-test-model-selection

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
true_bin=$(type -P true)
artifact_dir="$tmp_dir/existing-artifact"
mkdir -p "$artifact_dir"
printf 'preserve-me\n' > "$artifact_dir/sentinel"
if env -u MODELS \
    COMPILER_DIR="$tmp_dir/compiler" \
    SSM="$tmp_dir/model" \
    LINX_RUNTIME_ROOT="$tmp_dir/runtime" \
    API_INCLUDE="$tmp_dir/api" \
    GFRUN="$true_bin" \
    ARTIFACT_DIR="$artifact_dir" \
    "$runner" > "$tmp_dir/existing.log" 2>&1; then
    echo "pre-existing artifact directory was unexpectedly accepted" >&2
    exit 1
fi
grep -F "artifact directory already exists: $artifact_dir" \
    "$tmp_dir/existing.log" >/dev/null
[[ $(cat "$artifact_dir/sentinel") == preserve-me ]]
[[ $(find "$artifact_dir" -mindepth 1 -maxdepth 1 | wc -l) -eq 1 ]]
echo "element_atomic_topk existing artifact rejection: PASS"

