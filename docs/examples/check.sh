#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CASE="${1:-all}"

if [[ -z "${COMPILER_DIR:-}" ]]; then
  echo "error: COMPILER_DIR must point to the Linx LLVM bin directory" >&2
  exit 2
fi

if [[ ! -x "$COMPILER_DIR/clang++" ]]; then
  echo "error: Linx clang++ not found at $COMPILER_DIR/clang++" >&2
  exit 2
fi

if [[ "${KEEP_DOC_OBJECTS:-0}" == "1" ]]; then
  OUT_DIR="$SCRIPT_DIR/output"
  mkdir -p "$OUT_DIR"
else
  OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/supernpu-doc-examples.XXXXXX")"
  trap 'rm -rf "$OUT_DIR"' EXIT
fi

COMMON_FLAGS=(
  -target linx64-linx-none-elf
  -std=c++20
  -O2
  -c
)

CANONICAL_PTO_ROOT="${PTO_KERNELS_ROOT:-$REPO_ROOT/../pto_kernels}"
CANONICAL_PTO_FLAGS=(
  -fenable-matrix
  -I"$CANONICAL_PTO_ROOT/include"
)

if [[ -n "${LINX_SYSROOT:-}" ]]; then
  COMMON_FLAGS+=(
    --sysroot="$LINX_SYSROOT"
    -idirafter "$LINX_SYSROOT/usr/include"
  )
fi

compile_cpp_language() {
  "$COMPILER_DIR/clang++" "${COMMON_FLAGS[@]}" \
    "$SCRIPT_DIR/cpp_language.cpp" -o "$OUT_DIR/cpp_language.o"
}

compile_tile_add() {
  "$COMPILER_DIR/clang++" "${COMMON_FLAGS[@]}" "${CANONICAL_PTO_FLAGS[@]}" \
    "$SCRIPT_DIR/tile_add.cpp" -o "$OUT_DIR/tile_add.o"
}

compile_gemm_tile() {
  if [[ ! -f "$CANONICAL_PTO_ROOT/include/pto/linx/TileOps.hpp" ]]; then
    echo "error: PTO_KERNELS_ROOT does not contain include/pto/linx/TileOps.hpp" >&2
    exit 2
  fi
  "$COMPILER_DIR/clang++" "${COMMON_FLAGS[@]}" "${CANONICAL_PTO_FLAGS[@]}" \
    "$SCRIPT_DIR/gemm_tile.cpp" -o "$OUT_DIR/gemm_tile.o"
}

compile_flash_attention() {
  if [[ ! -f "$CANONICAL_PTO_ROOT/include/pto/linx/AutoModeKernels.hpp" ]]; then
    echo "error: PTO_KERNELS_ROOT does not contain include/pto/linx/AutoModeKernels.hpp" >&2
    exit 2
  fi
  "$COMPILER_DIR/clang++" "${COMMON_FLAGS[@]}" "${CANONICAL_PTO_FLAGS[@]}" \
    "$SCRIPT_DIR/flash_attention.cpp" -o "$OUT_DIR/flash_attention.o"
}

case "$CASE" in
  all)
    compile_cpp_language
    compile_tile_add
    compile_gemm_tile
    compile_flash_attention
    ;;
  cpp-language) compile_cpp_language ;;
  tile-add) compile_tile_add ;;
  gemm-tile) compile_gemm_tile ;;
  flash-attention) compile_flash_attention ;;
  *)
    echo "error: unknown case '$CASE' (expected all, cpp-language, tile-add, gemm-tile, or flash-attention)" >&2
    exit 2
    ;;
esac

echo "compiled documentation example(s) to $OUT_DIR"
