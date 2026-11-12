#!/bin/zsh
# 构建 hashtable_lookup 结果 ELF。
#   ./build_kernel.sh matrix <64|128|256> <2_2|2_4|2_6> <outdir>
#   ./build_kernel.sh soa_operator <outdir>        # 256-lane 2+6，走 basic_op 的 soa_wide
#
# 必填环境变量（归档不保存个人路径，也不提供指向作者机器的默认值）：
#   CLANG    pinned clang++  (sha256 35898d127974570667e06f828dbd3fbffcef6b42bc5acd82d7c0b1686222a2e4)
#   LD       pinned ld.lld   (sha256 d57959a1fe1941fa0bf20060398bac74012cf14a5a42bcc262b987f44f8ae0bb)
#   SYSROOT  linx64v5-musl sysroot (headers/libs only)
# 可选环境变量（默认按 REPO/归档相对路径解析）：
#   REPO     SuperNPUBench 根目录（默认：本归档的上两级）
#   API      PTO API 快照（tileop-api 目录）
#   REF      eval/perf_data/hashtable_lookup/hash_6144_PE4.elf
#   DATA     keys32.bin/values.bin/batch_map.bin 目录
#   REPRO    outputs/hash64_batchmap_20261105/slots_sweep/repro_matrix
set -e
A=$(cd "$(dirname "$0")/.." && pwd)
R=${REPO:-$(cd "$A/../.." && pwd)}
: "${CLANG:?set CLANG to the pinned clang++ (see dependencies/compiler_llvm_pr146)}"
: "${LD:?set LD to the pinned ld.lld (see dependencies/compiler_llvm_pr146)}"
: "${SYSROOT:?set SYSROOT to the linx64v5 sysroot used for the build}"
API=${API:-$R/outputs/hash256_soa_20261105/optimization/dependencies/tileop-api}
REF=${REF:-$R/eval/perf_data/hashtable_lookup/hash_6144_PE4.elf}
DATA=${DATA:-$R/outputs/hash256_batchmap_20261105/data}
REPRO=${REPRO:-$R/outputs/hash64_batchmap_20261105/slots_sweep/repro_matrix}
for p in "$CLANG" "$LD" "$SYSROOT" "$API" "$REF" "$DATA" "$REPRO"; do
  [ -e "$p" ] || { echo "missing required path: $p" >&2; exit 2; }
done
COMMON=(--target=linx64v5-unknown-linux-musl -mlxbc -fenable-matrix -O3 -std=c++20
  -D__linx -DENABLE_TENSOR_INSTR -ffreestanding -fno-exceptions -fno-rtti
  -fno-vectorize -fno-slp-vectorize
  -mllvm -linxv5-enable-generic-element=true
  -mllvm -enable-all-vector-as-tilereg=true
  --sysroot=$SYSROOT -I$API -I$R -I$R/benchmark/one-level-arch/kernels
  -DHASH_SKIP_NUM_CHECK=1 -mllvm -linxv5-force-element-tile-mask=true)
mode=$1; shift
case "$mode" in
  matrix)
    lanes=$1; variant=$2; out=$3
    mkdir -p "$out"
    cp $REPRO/fixtures/fixture${lanes}.cpp "$out/fixture.cpp"
    cp $REPRO/kernels/kernel_${variant}.hpp "$out/kernel.hpp"
    cp $REPRO/link.ld "$out/link.ld"
    cd "$out"
    $CLANG "${COMMON[@]}" \
      -DHASH_DATA_ELF_PATH="\"$REF\"" -DHASH_KEY_DATA="\"$DATA/keys32.bin\"" \
      -DHASH_VALUE_DATA="\"$DATA/values.bin\"" \
      -DHASH_BATCH_MAP_DATA="\"$REPRO/maps/map${lanes}.bin\"" \
      -DHASH_QUERY_REPEATS=1 -DHASH_WIDE_LANES=$lanes \
      -c fixture.cpp -o perf.o
    $LD -T link.ld perf.o -o perf.elf
    rm -f perf.o
    echo "built $out/perf.elf"
    ;;
  soa_operator)
    out=$1; mkdir -p "$out"
    cp $R/outputs/hash256_batchmap_20261105/src/fixture.cpp "$out/fixture.cpp"
    cp $R/outputs/hash256_batchmap_20261105/src/link.ld "$out/link.ld"
    cd "$out"
    sed -i '' 's|#include "kernel.hpp"|#include "basic_op/hashtable_lookup/hashtable_lookup_soa_wide.hpp"|' fixture.cpp
    sed -i '' 's|using supernpu::hash_lookup::wide_mask::TableEntry;|using supernpu::hash_lookup::soa_wide::TableEntry;|' fixture.cpp
    sed -i '' 's|supernpu::hash_lookup::wide_mask::run<|supernpu::hash_lookup::soa_wide::run<|' fixture.cpp
    $CLANG "${COMMON[@]}" \
      -DHASH_DATA_ELF_PATH="\"$REF\"" -DHASH_KEY_DATA="\"$DATA/keys32.bin\"" \
      -DHASH_VALUE_DATA="\"$DATA/values.bin\"" \
      -DHASH_BATCH_MAP_DATA="\"$DATA/batch_map.bin\"" \
      -DHASH_QUERY_REPEATS=1 -DHASH_WIDE_LANES=256 \
      -c fixture.cpp -o perf.o
    $LD -T link.ld perf.o -o perf.elf
    rm -f perf.o
    echo "built $out/perf.elf"
    ;;
  *) echo "usage: $0 matrix <lanes> <variant> <outdir> | soa_operator <outdir>" >&2; exit 2;;
esac
