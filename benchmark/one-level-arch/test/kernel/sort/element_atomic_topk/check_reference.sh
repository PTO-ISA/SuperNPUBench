#!/usr/bin/env bash
set -euo pipefail

case_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
one_level_dir=$(cd "$case_dir/../../../.." && pwd)
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/element-atomic-topk.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

host_cxx=${CXX:-c++}
common_flags=(-std=c++20 -Wall -Wextra -Werror -Wno-unknown-pragmas \
    -I "$one_level_dir")

"$host_cxx" "${common_flags[@]}" -O2 \
    "$case_dir/reference_test.cpp" -o "$tmp_dir/reference_test"
"$tmp_dir/reference_test"

# The actual kernel is executed with the real TileOp API on gfrun/gfsim.
# Native boundary cases live in tile_element_suite/topk_boundaries.cpp; this
# host test supplies the independent radix/reference oracle only.
