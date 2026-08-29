# Agent Instructions

## Compiler worktree

All compilations in this repo MUST use the main `linx-toolchain-build` checkout as the compiler. Do not use the `linx-toolchain-build-latest` worktree or the sibling prebuilt `DV4/linx_blockisa_llvm_musl/` copy unless the user explicitly requests one of them.

- Compiler checkout root: `/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build`
- Toolchain bin dir (set this as `COMPILER_DIR` before running any `make`/`compile.all`/`compile_all.sh`):
  ```
  export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
  ```
- Provides `clang`/`clang++` (AS/CC/CXX/LINK), `llvm-objdump` (DUMP), `llvm-objcopy` (COPY), `ld.lld`. Target: `linx64v5-unknown-linux-musl`.

When building a single operator, output can be redirected to `/tmp` by overriding `OBJ_ROOT` on the make command line (Makefile.common defines `OBJ_ROOT` with `:=` and no `override`, so a command-line value wins; `ELF_DIR`/`OBJ_DIR`/`LINK_SCRIPT` all derive from it), e.g.:

```
make -C <testcase-dir> TESTCASE=<name> COMPILER_DIR="$COMPILER_DIR" <params> OBJ_ROOT=/tmp/<build>
```
