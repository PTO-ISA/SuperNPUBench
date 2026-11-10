#!/bin/bash
# Build the two half-fixture ELFs for the dual-stream experiment.
#   htA: queries [0,3072),   no load bias      (stream A, -f)
#   htB: queries [3072,6144), +0x50000000 bias (stream B, --elf2)
set -e
ROOT=/Users/lfm/project/benchmark/REPRO_hashtable_1104
CLANG=$ROOT/llvm-project/build-hash/bin/clang++
LD=$ROOT/llvm-project/build-hash/bin/ld.lld
SRC=$ROOT/out/src_dual
REF=$ROOT/SuperNPUBench/eval/perf_data/hashtable_lookup/hash_6144_PE4.elf
OUT=$ROOT/out

build() {
  local name=$1; shift
  "$CLANG" --target=linx64v5-unknown-linux-musl -mlxbc -fenable-matrix -O3 -std=c++20 \
    -D__linx -DENABLE_TENSOR_INSTR -ffreestanding -fno-exceptions -fno-rtti \
    -fno-vectorize -fno-slp-vectorize \
    -mllvm -linxv5-enable-generic-element=true \
    -mllvm -enable-all-vector-as-tilereg=true \
    --sysroot=$ROOT/linx-toolchain-build/linx_blockisa_llvm_musl/sysroot \
    -I$ROOT/Linx-TileOP-API/include -I$ROOT/SuperNPUBench \
    -DHASH_DATA_ELF_PATH="\"$REF\"" -DHASH_QUERY_REPEATS=1 -DHASH_WIDE_LANES=256 \
    -mllvm -linxv5-force-element-tile-mask=true \
    "$@" -c "$SRC/fixture.cpp" -o "$OUT/$name.o"
  "$LD" -T "$SRC/link.ld" "$OUT/$name.o" -o "$OUT/$name.elf"
  shasum -a 256 "$OUT/$name.elf"
}

build htA -DHASH_QUERY_OFFSET=0 -DHASH_LOAD_BIAS=0 \
          -DHASH_QUERY_FILE_OFF=0x26ea000 -DHASH_GOLDEN_FILE_OFF=0x26f6000
build htB -DHASH_QUERY_OFFSET=3072 -DHASH_LOAD_BIAS=0x50000000 \
          -DHASH_QUERY_FILE_OFF=0x26f0000 -DHASH_GOLDEN_FILE_OFF=0x26f9000

# Standalone (no load bias) copy of the second half, for the single-stream check.
build htB0 -DHASH_QUERY_OFFSET=3072 -DHASH_LOAD_BIAS=0 \
           -DHASH_QUERY_FILE_OFF=0x26f0000 -DHASH_GOLDEN_FILE_OFF=0x26f9000
