#!/bin/bash
# Embed src/data/*.bin as .o with _binary_<name>_data_{start,end} symbols.
set -euo pipefail

: "${COMPILER_DIR:?Set COMPILER_DIR}"

DATA_DIR="${1:?data dir}"
OUTPUT_DIR="${2:?output dir}"
NAME="${3:?blob name without extension}"

mkdir -p "$OUTPUT_DIR"
DATA_FILE="${DATA_DIR}/${NAME}.bin"
ASM_FILE="${OUTPUT_DIR}/${NAME}.s"
OBJ_FILE="${OUTPUT_DIR}/${NAME}.o"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "missing $DATA_FILE" >&2
  exit 1
fi

# Absolute path for .incbin (assembler cwd-independent).
DATA_ABS="$(cd "$(dirname "$DATA_FILE")" && pwd)/$(basename "$DATA_FILE")"

cat > "$ASM_FILE" << EOF
.section .data
.global _binary_${NAME}_data_start
_binary_${NAME}_data_start:
.incbin "${DATA_ABS}"
.global _binary_${NAME}_data_end
_binary_${NAME}_data_end:
.global _binary_${NAME}_data_size
.equ _binary_${NAME}_data_size, .-_binary_${NAME}_data_start
EOF

"$COMPILER_DIR/clang++" -target linx64v5 -c "$ASM_FILE" -o "$OBJ_FILE"
echo "built $OBJ_FILE from $DATA_FILE"
