#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

: "${COMPILER_DIR:?set COMPILER_DIR to the Linx LLVM bin directory}"
: "${MUSL_SYSROOT:?set MUSL_SYSROOT to the installed Linx musl sysroot}"
: "${GFRUN:?set GFRUN to the gfrun executable}"
: "${TILEOP_API_ROOT:?set TILEOP_API_ROOT to the Linx-TileOP-API checkout}"

CLANG=${CLANG:-$COMPILER_DIR/clang}
if [ ! -x "$CLANG" ] && [ -x "$COMPILER_DIR/clang-23" ]; then
    CLANG="$COMPILER_DIR/clang-23"
fi
NM=${NM:-$COMPILER_DIR/llvm-nm}
TARGET=${LINX_TARGET:-linx64-unknown-linux-musl}
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/output/hosted-group-runtime-smoke}
RUNTIME_LIB=${LINX_RUNTIME_LIB:-$MUSL_SYSROOT/lib/liblinx_builtin_rt.a}

if [ ! -f "$RUNTIME_LIB" ]; then
    echo "missing Linx runtime archive: $RUNTIME_LIB" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"

"$CLANG" -target "$TARGET" --sysroot "$MUSL_SYSROOT" -D__linx \
    -fno-pie -O2 -Wall -Wextra -Werror \
    -I "$TILEOP_API_ROOT/include" \
    -c "$SCRIPT_DIR/hosted_group_runtime_smoke.cpp" \
    -o "$OUT_DIR/hosted_group_runtime_smoke.o"

"$CLANG" -target "$TARGET" \
    -c "$SCRIPT_DIR/hosted_group_runtime_start.s" \
    -o "$OUT_DIR/hosted_group_runtime_start.o"

"$CLANG" -target "$TARGET" -static -fuse-ld=lld -nostdlib \
    "$OUT_DIR/hosted_group_runtime_start.o" \
    "$OUT_DIR/hosted_group_runtime_smoke.o" \
    "$RUNTIME_LIB" \
    -Wl,--gc-sections -Wl,--image-base=0x40000000 \
    -o "$OUT_DIR/hosted_group_runtime_smoke"

SYMBOLS=$("$NM" -g "$OUT_DIR/hosted_group_runtime_smoke")
for symbol in linx_group_run __linx_group_worker_start \
    __linx_group_worker_main; do
    if ! printf '%s\n' "$SYMBOLS" | grep -Eq "[[:space:]]${symbol}$"; then
        echo "missing hosted group runtime symbol: $symbol" >&2
        exit 2
    fi
done

"$GFRUN" -f "$OUT_DIR/hosted_group_runtime_smoke" \
    -s softcore.multiThreadNum=4
