# Four-PE Kernels

This directory contains kernels whose implementation explicitly depends on
four PEs. They use three execution models:

- Cooperative execution: shared matrix operands and four-PE `TMATMUL` for
  Matmul and FlashAttention.
- Contiguous SPMD partitioning: `get_thread_idx()` assigns equal, disjoint
  element or row ranges. The single-PE Tile kernel is reused on each range.
- Tile-block SPMD partitioning: Conv2D output-spatial blocks and Transpose
  input-row blocks are assigned round-robin while preserving the full global
  tensor layout.

The common `ContiguousPartition` helper requires its work dimension to be
divisible by the PE count. This is checked at compile time.

## Current implementations

| Operator | Header |
|---|---|
| Broadcast | `broadcast/broadcast.hpp` |
| Concat gather/scatter | `concat/concat_gather.hpp`, `concat/concat_scatter.hpp` |
| 1x1 Conv2D | `conv2d/v300_conv2d.hpp` |
| TADD | `element_wise/tadd_multithread.hpp` |
| GELU | `element_wise/gelu.hpp` |
| FlashAttention | `fa/fa_2d_unroll_gmma.hpp` |
| Gather | `gather/gather.hpp` |
| Shared Matmul | `matmul/matmul_shared.hpp` |
| Shared-B-reuse Matmul | `matmul/matmul_shared_reuseB.hpp` |
| Low-precision Matmul | `matmul/matmul_shared_lowp.hpp` |
| RMSNorm | `normalization/rms_norm/rms_norm.hpp` |
| Binary-accumulation RMSNorm | `normalization/rms_norm_binary/rms_norm_binary.hpp` |
| Row Cumsum | `reduction/cumsum_rowvec.hpp` |
| Row Max/Prod/Sum | `reduction/reducemax_rowvec.hpp`, `reduction/reduceprod_rowvec.hpp`, `reduction/reducesum_rowvec.hpp` |
| 2D Transpose | `transpose/transpose.hpp` |

## Mirrored directory layout

Kernel and test paths mirror the single-PE tree. For example:

| Single PE | Four PE |
|---|---|
| `kernels/single_thread/gather/gather.hpp` | `kernels/multi_thread/gather/gather.hpp` |
| `test/kernel/gather/` | `test/kernel/multi_thread/gather/` |
| `test/kernel/element_wise/gelu/` | `test/kernel/multi_thread/element_wise/gelu/` |
| `test/kernel/normalization/rms_norm_binary/` | `test/kernel/multi_thread/normalization/rms_norm_binary/` |

Each operator directory has its own `Makefile`, `compile.all`, and `src/`
instead of sharing a mixed test source. One model failure therefore does not
hide the status of other operator classes.

## Four-PE operator execution regression

Build a single operator through its mirrored test directory:

```bash
make -C benchmark/one-level-arch/test/kernel/multi_thread/element_wise/gelu \
  TESTCASE=gelu COMPILER_DIR="$COMPILER_DIR" diss
```

The 2026-08-29 full regression used main `linx-toolchain-build` commit
`e6a31ef` and SuperScalarModel commit `d8903938`. It compiled every configuration
listed by the `multi_thread` `compile.all` files and ran each ELF with four
simulated PEs.

| Operator family | ELF | PASS | FAIL |
|---|---:|---:|---:|
| Broadcast | 1 | 1 | 0 |
| Concat | 2 | 2 | 0 |
| Conv2D | 1 | 1 | 0 |
| Element-wise | 1 | 1 | 0 |
| FlashAttention | 16 | 10 | 6 |
| Gather | 1 | 1 | 0 |
| Matmul | 9 | 9 | 0 |
| Normalization | 2 | 2 | 0 |
| Row reduction | 4 | 4 | 0 |
| Transpose | 1 | 1 | 0 |
| Vector | 1 | 1 | 0 |
| **Total** | **39** | **33** | **6** |

There were no timeouts. The remaining six failures are FlashAttention
conversion-surface limitations rather than PE partition failures:

- Four HIF4/MXFP4 configurations require BF16-to-packed-FP4 conversion, while
  PTO 0.58 TCVT requires matching source and destination logical shapes.
- Two HIF8 configurations require FP32-to-HIF8 conversion, which the current
  model reports as an unregistered `.fs -> .hifb` conversion.

The tested Broadcast shape expands only leading unit dimensions, so each PE
uses the direct full-input-copy fast path. Other Broadcast shapes still use
the generic MGATHER offset path, whose compiler-generated raw-tile spill is a
known compiler/model carrier limitation.

The table above records compilation and gfrun completion. It does not by
itself prove that the generated tensor is numerically correct. The separate
`RES_CHECK` regression below supplies inputs and compares the complete output
against a host reference.

## Four-PE RES_CHECK numerical validation

### Validation mechanism

The numerical check is a host-driven flow. The ELF reads prepared binary
inputs and writes its result, while Python computes the reference and performs
the comparison; the ELF does not calculate the Golden result itself.

```text
build a RES_CHECK ELF
        -> generate binary inputs and a NumPy Golden result
        -> execute the ELF with four gfrun PEs
        -> let PE0 export the complete output tensor
        -> compare every element with NumPy allclose
```

Passing `res_check=on` to make causes `test/common/Makefile.common` to:

- define `RES_CHECK` and `ENABLE_BINARY_OUTPUT`;
- define `CHK_DIR` as `benchmark/one-level-arch/compare/<elf-stem>`;
- link the hosted file-I/O and group-worker support needed by the testcase.

Only an ELF built with `res_check=on` can be used for this flow. A later normal
build may overwrite the same output ELF with file I/O disabled, in which case
running the Python checker against that ELF does not constitute a numerical
check.

Hosted four-PE execution runs the testcase entry on PE0 through PE3. Each
testcase therefore follows these ownership and synchronization rules:

1. Input, output, scale, and workspace arrays used by multiple PEs are placed
   in shared static storage rather than on a PE-private stack.
2. PE0 is the only thread that calls `readBinaryFile` and
   `writeBinaryFile`. For cooperative FA and Matmul, this is also the PE used
   by the single-PE shared-tile load path.
3. After PE0 loads all inputs, it publishes `input_ready`; PE1 through PE3
   wait before entering the kernel.
4. Every PE writes a separate completion slot after its final Tile operation.
   PE0 waits for all four slots before exporting the complete output tensor.
5. The Python runner pre-creates an output file of the expected type and size,
   runs gfrun with `softcore.multiThreadNum=4`, then reads that file back.
6. The runner checks the gfrun return code, output element count, and every
   output value using the testcase's `atol` and `rtol`.

The shared synchronization implementation is
`test/common/multi_thread_res_check.h`. The input generation, gfrun launch,
and Golden comparison are implemented by
`test/kernel/multi_thread/res_check_all.py`.

For SPMD testcases, `res_check_publish_inputs()` prevents PE1 through PE3 from
reading uninitialized inputs and `res_check_wait_for_all()` prevents PE0 from
exporting a partial result. Cooperative testcases launched through
`linx_group_run()` use the group runtime to start and join their worker PEs;
file ownership remains on PE0.

### Build procedure

All builds must use the main `linx-toolchain-build` checkout:

```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export res_check=on

for script in $(find benchmark/one-level-arch/test/kernel/multi_thread \
    -name compile.all | sort); do
  dir=${script%/compile.all}
  (cd "$dir" && bash compile.all) || exit $?
done
```

This compiles the complete precision/configuration matrix. The 2026-08-29
run produced 39/39 ELFs successfully with compiler commit `e6a31ef`.

To build only one configuration, pass `res_check=on` directly to make. For
example:

```bash
make -C benchmark/one-level-arch/test/kernel/multi_thread/matmul \
  TESTCASE=matmul_shared \
  COMPILER_DIR="$COMPILER_DIR" \
  res_check=on \
  DTYPE=float B=1 M=256 N=256 K=256 \
  tM=128 tN=256 tK=128
```

### Run and compare

Run one or more named cases:

```bash
python3 benchmark/one-level-arch/test/kernel/multi_thread/res_check_all.py \
  broadcast fa matmul_shared rms_norm
```

Run the complete representative numerical portfolio:

```bash
python3 benchmark/one-level-arch/test/kernel/multi_thread/res_check_all.py \
  --timeout 120
```

The runner uses
`/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun`
by default. Use `--gfrun /other/path/to/gfrun` to select another model binary.
For each case the runner creates deterministic input files, keeps the NumPy
Golden result in host memory, preallocates the output file, and invokes:

```bash
gfrun -s softcore.multiThreadNum=4 -f /absolute/path/to/operator.elf
```

It saves `gfrun.log` beside the binary inputs and outputs in the ELF's
`compare/<elf-stem>/` directory. The final status is classified as:

- `PASS`: gfrun exits normally, output length matches, and `np.allclose`
  passes;
- `FAIL`: gfrun returns nonzero, the output length differs, or the numerical
  comparison fails;
- `TIMEOUT`: gfrun exceeds `--timeout`;
- `SKIP`: the requested ELF does not exist.

For a numerical mismatch, the runner reports the maximum absolute error.
Exact data-movement tests can use zero tolerance, while FP16/BF16, Matmul, FA,
and nonlinear operators use case-specific tolerances declared in the `CASES`
table in `res_check_all.py`.

### Numerical result: 2026-08-29

The representative portfolio covers all 18 multi-thread testcase sources.
FA uses the FP32/VecFP32 configuration. Low-precision Matmul uses exact FP8
zero inputs to verify four-PE ownership and writeback without relying on a
host FP8 package. The full 39-ELF compile matrix remains the compilation
coverage for the other precision variants.

| Testcase | Result | Maximum absolute error or failure |
|---|---|---|
| `broadcast` | PASS | 0 |
| `concat_gather` | PASS | 0 |
| `concat_scatter` | PASS | 0 |
| `v300_conv2d` | FAIL | 0.282559; only part of the output is written correctly |
| `gelu` | PASS | 7.62939e-06 |
| `fa_2d_unroll_gmma` FP32/VecFP32 | PASS | 7.12688e-05 |
| `gather` | FAIL | 1.03847e+34; MGATHER index/offset contract mismatch |
| `matmul_shared` | PASS | 0 |
| `matmul_reuseB` | PASS | 0 |
| `matmul_lowp` FP8 | PASS | 0 |
| `rms_norm` | PASS | 0.000976562 |
| `rms_norm_binary` | PASS | 0.000976562 |
| `cumsum_row` | PASS | 0 |
| `reducemax_row` | FAIL | 0.999606; row-result physical stride mismatch |
| `reduceprod_row` | FAIL | 1.06037; row-result physical stride mismatch |
| `reducesum_row` | PASS | 5.72205e-06 |
| `transpose` | PASS | 0 |
| `tadd` | PASS | 0 |
| **Total** | **14 PASS / 4 FAIL / 0 TIMEOUT** | **18 representative cases** |

Two RMSNorm cases initially failed because the old Newton iteration used
`TRECIP(x)` as the inverse-square-root seed. With the current compiler's
`TRSQRT` support, replacing that sequence with `TRSQRT` reduced the maximum
absolute error to 0.000976562 and made both cases pass.

The remaining failures are kernel/API-model issues exposed by numerical
checking, rather than binary I/O or four-PE synchronization failures:

- Conv2D assigns different output-spatial blocks to PEs around a cooperative
  TMATMUL path; its cube result ownership/writeback needs to be redesigned.
- Gather passes row indexes to the current MGATHER path, while the generated
  instruction/model behavior is offset-oriented and returns corrupted data.
- ReduceMax and ReduceProd expose one valid value per row through a padded
  reduction carrier, but their current TSTORE path writes with the physical
  carrier stride instead of a compact `[Rows]` layout.

## Scope limits

This directory only contains implementations with race-free output ownership.
GroupNorm backward, column-wise/global reductions, TopK, and fused kernels
with cross-PE dependencies still require a kernel-level barrier and/or an
inter-PE reduction primitive. They must not be implemented by having four PEs
write the same destination.

## Run with gfrun

Run an ELF built from a multi-thread operator test with four simulated PEs:

```bash
gfrun -t 1 -s softcore.multiThreadNum=4 -f /path/to/operator.elf
```

Do not omit `softcore.multiThreadNum=4`: cooperative kernels require all four
PEs, while SPMD kernels use thread IDs 0 through 3 to partition the work.

The run passes when `gfrun` exits with status 0 and its output contains both
`Reach the End of Benchmark` and `R2 = 0`.
