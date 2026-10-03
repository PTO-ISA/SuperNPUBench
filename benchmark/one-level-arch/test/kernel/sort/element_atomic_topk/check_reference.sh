#!/usr/bin/env bash
set -euo pipefail

case_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/element-atomic-topk.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

${CXX:-c++} -std=c++20 -O2 -Wall -Wextra -Werror \
    "$case_dir/reference_test.cpp" -o "$tmp_dir/reference_test"
"$tmp_dir/reference_test"
