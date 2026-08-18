#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)

M=${M:-16}
N=${N:-256}
tM=${tM:-8}
tN=${tN:-128}
GFRUN=${GFRUN:-/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun}
TIMEOUT=${TIMEOUT:-120}

ELF_NAME="kernel_norm_rms_norm_M${M}_N${N}_tM${tM}_tN${tN}"
ELF="$REPO_ROOT/benchmark/one-level-arch/output/kernel/norm/elf/${ELF_NAME}.elf"
DATA_DIR="$REPO_ROOT/benchmark/one-level-arch/compare/${ELF_NAME}"
GOLDEN_SCRIPT="$SCRIPT_DIR/src/rms_norm_golden.py"

python3 "$GOLDEN_SCRIPT" check \
    --elf "$ELF" \
    --gfrun "$GFRUN" \
    --data-dir "$DATA_DIR" \
    --rows "$M" \
    --cols "$N" \
    --timeout "$TIMEOUT"
