---
name: rms-norm-binary
description: >-
  Build, run, and debug the one-level rms_norm_binary kernel (R-split RMSNorm)
  with SuperNPUBench + run_op.py + gfrun/gfsim precision checks. Use when
  editing rms_norm_binary_pto.hpp, rms_norm_binary tests, workspace/GetCacheId
  reduce, TADD cross-tile sum, or verifying [1,8192] fp16 binary RMSNorm.
  Shape dims are A (outer) and R (reduce): g_a/g_r, tile_a/tile_r, tA/tR.
---

# rms_norm_binary — kernel & verification

Default root: `/home/wangyu/Code/SuperScalar`（下文 `$ROOT`）。

## Shape naming

| Old | New | Meaning |
|-----|-----|---------|
| M / `g_m` / `tM` / `tile_m` | **A** / `g_a` / `tA` / `tile_a` | outer / row |
| N / `g_n` / `tN` / `tile_n` | **R** / `g_r` / `tR` / `tile_r` | reduce / col |
| `Nb` | `Rb` | `# R-tiles = ceil(g_r / tile_r)` |

`tiling[4] = {g_a, g_r, tile_a, tile_r}`.

## What it is

One-level PTO RMSNorm when `g_r > tile_r`:

```text
out[a] = x[a] * rsqrt(mean(x[a]^2) + eps)
```

R is split into `Rb = ceil(g_r / tile_r)` tiles. Each tile does local
`TROWSUM(x^2)`, then tiles are reduced to a full-row sum.

Current default test shape: **`[1, 8192]`**, `tile_r=1024` → **`Rb=8`**, fp16.

## Key paths

| Role | Path |
|------|------|
| Kernel | `$ROOT/SuperNPUBench/benchmark/one-level-arch/kernels/normalization/rms_norm_binary/rms_norm_binary_pto.hpp` |
| Reference (single-tile) | `.../kernels/normalization/rms_norm/rms_norm_pto.hpp` |
| Testcase | `.../test/kernel/normalization/rms_norm_binary/` |
| Host entry | `.../rms_norm_binary/src/rms_norm_binary.cpp` |
| Gen golden | `.../rms_norm_binary/src/gen_rms_norm_binary_data.py` |
| Compare | `.../rms_norm_binary/src/rms_norm_binary_data_compare.py` |
| Runner | `$ROOT/run_op.py` preset `rms_norm_binary` |
| Toolchain | `$ROOT/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin` → `COMPILER_DIR` |
| Sims | `$ROOT/SuperScalarModel/bin/gfrun`, `gfsim` |
| Related skill | `.../kernels/reduction/binary-accumulation-cache-id/SKILL.md` |

ELF after build:

```text
.../output/kernel/normalization/rms_norm_binary/elf/
  kernel_normalization_rms_norm_binary_rms_norm_binary_DType__half.elf
```

Compare dir (precision):

```text
.../compare/kernel_normalization_rms_norm_binary_rms_norm_binary_DType__half/
  input.bin  golden.bin  output.bin  tiling_info.bin
```

## Kernel pipeline (current)

File: `rms_norm_binary_pto.hpp`. **No `rms_norm_dyn_ops.hpp`.** TEPL style like
`rms_norm_pto.hpp`.

```text
Pass1:
  TLOAD(sum ← zeros)
  per R-tile: TLOAD → TCVT → TMUL(x,x) → TROWSUM → TADD(sum, sum, cur)

Pass1.5:
  TMULS(mean, sum, 1/g_r) → TADDS(eps) → Newton rsqrt → rms

Pass2  (per R-tile):
  TLOAD → TCVT → TROWEXPANDMUL(x, rms) → TCVT → TSTORE
```

Important implementation notes:

1. **Cross-tile sum is streaming** (`sum += cur`), not GetCacheId carry-merge.
2. Zero-init `sum` outside the R loop; uniform `TADD` inside (no first-tile branch).
3. `tile_v`: `Cols=32`, **static `Valid=1,1`** (`tile_a==1`) so TEPL `B.DIM`
   immediates are legal.
4. `workspace` argument is kept in the API but **currently unused**.
5. Do **not** put early-return parameter checks in the kernel (caller owns tiling).

## How to verify

```bash
export COMPILER_DIR=$ROOT/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
cd $ROOT
python3 run_op.py rms_norm_binary
```

What `run_op.py` does:

1. `gen_rms_norm_binary_data.py` → write `input.bin` / `golden.bin` / tiling
2. `make TESTCASE=rms_norm_binary DType=__half res_check=on` → ELF with I/O
3. `gfrun` functional sim → writes `output.bin`
4. `rms_norm_binary_data_compare.py` → atol/rtol vs golden
5. Rebuild **without** `res_check` (res_check ELF often crashes `gfsim`)
6. `gfsim` performance / cycle sim
7. After compile: write `<elf>.diss` via `llvm-objdump -dl` (disable: `--no-diss`)

Useful flags:

```bash
python3 run_op.py rms_norm_binary --func-only          # gfrun + precision only
python3 run_op.py rms_norm_binary --perf-only          # gfsim only
python3 run_op.py rms_norm_binary --compile-only
python3 run_op.py rms_norm_binary --no-check-precision
python3 run_op.py rms_norm_binary --skip-compile
python3 run_op.py rms_norm_binary --no-diss
```

Manual make (same case):

```bash
cd $ROOT/SuperNPUBench/benchmark/one-level-arch/test/kernel/normalization/rms_norm_binary
make TESTCASE=rms_norm_binary DType=__half COMPILER_DIR=$COMPILER_DIR
# or: bash compile.all
```

## Expected results (as of current kernel)

| Step | Typical result |
|------|----------------|
| Compile | OK |
| gfrun | **PASS** (`Success to Reach the End`) |
| Precision | **PASS** (`max_abs` often `0.0` on `[1,8192]`) |
| gfsim | **FAIL** |

### gfsim failure (known)

Symptom:

```text
TMOV ... INVALID
FATAL: gfsim received signal 11
# or: Bank store offset out of range!
```

Cause: compiler-inserted `TMOV` (tile rename) for small `tile_v` kept live across
R-split loops; timing sim corrupts tile metadata. **Not a golden mismatch** —
gfrun + compare already pass.

Baseline `rms_norm` (no cross-tile accumulate / second R loop) usually **PASS**es gfsim.

## Testcase layout

`rms_norm_binary.cpp` defaults:

```cpp
G_A=1, G_R=8192, TILE_A=1, TILE_R=1024
workspace_buf[K_MAX_LEVELS * G_A * K_WS_COLS]  // kept for ABI; unused by kernel
```

Precision scripts default shape `--g-r 8192`, `--tile-r 1024`.

`run_op.py` preset name is exactly **`rms_norm_binary`** (no size suffix).

## Agent checklist when changing the kernel

1. Keep compute TEPL-only; do not reintroduce `rms_norm_dyn_ops.hpp` unless asked.
2. Prefer `Valid=1,1` on `tile_v` when using TEPL ops with NTTP `B.DIM`.
3. Avoid taking addresses of `tile_v` / large pointer arrays of tiles (Liveouts /
   illegal spill).
4. Do not mix `TROWSUM` u-reg lineage with `TLOAD` of small reduce tiles in the
   same hot path without verifying Match Instruction / gfsim.
5. After edits: `python3 run_op.py rms_norm_binary` (or `--func-only` if only
   checking correctness).
6. If implementing true GetCacheId carry + workspace reload, also read
   `binary-accumulation-cache-id/SKILL.md` and expect toolchain/sim constraints
   above.

## Anti-patterns

- Naming the run_op preset `rms_norm_binary_1x8192` / `..._1x32768` — canonical
  name is `rms_norm_binary`.
- Treating gfsim FAIL as a precision bug when gfrun+compare already PASS.
- Putting workspace spill between `rsqrt` and `TROWEXPANDMUL` (clobbers `rms`).
- Using TEPL `TADD` with `Valid=-1` (`Match Instruction Error`).
- Reverting shape names to M/N — use **A/R** consistently.
