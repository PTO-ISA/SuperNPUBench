# Source before running the guard: `source env.sh`
#
# Point at your linx toolchain (linx_blockisa_llvm_musl) + SuperScalarModel build.
# Two ways:
#   A) export COMPILER_DIR / GFRUN (and optionally GFSIM) yourself; or
#   B) export LINX_ROOT = the directory that contains linx-toolchain-build/ and
#      SuperScalarModel/, then source this file to derive the paths below.
if [ -n "${LINX_ROOT:-}" ]; then
  export COMPILER_DIR="${COMPILER_DIR:-$LINX_ROOT/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin}"
  export GFRUN="${GFRUN:-$LINX_ROOT/SuperScalarModel/bin/gfrun}"
  export GFSIM="${GFSIM:-$LINX_ROOT/SuperScalarModel/bin/gfsim}"
fi
# run_guard.sh errors out if COMPILER_DIR / GFRUN are still unset.
