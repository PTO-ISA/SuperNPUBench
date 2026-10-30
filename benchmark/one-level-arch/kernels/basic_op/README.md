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
| ArgMaxV2 | `argmaxv2/argmaxv2.hpp` |
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
| Row Cumsum | `reduction/cumsum_rowvec.hpp` |
| Row Max/Prod/Sum | `reduction/reducemax_rowvec.hpp`, `reduction/reduceprod_rowvec.hpp`, `reduction/reducesum_rowvec.hpp` |
| 2D Transpose | `transpose/transpose.hpp` |

## Directory layout

Each operator directory holds both the four-PE wrapper and the single-PE base
implementation it partitions. The base header lives under `<op>/detail/` so it
does not collide with the wrapper that shares its name:

| Base implementation | Four-PE wrapper |
|---|---|
| `basic_op/argmaxv2/detail/argmaxv2.hpp` | `basic_op/argmaxv2/argmaxv2.hpp` |
| `basic_op/argmaxv2/detail/argmaxv2_dynamic.hpp` | `basic_op/argmaxv2/argmaxv2_dynamic.hpp` |
| `basic_op/gather/detail/gather.hpp` | `basic_op/gather/gather.hpp` |
| `basic_op/element_wise/detail/gelu.hpp` | `basic_op/element_wise/gelu.hpp` |

Shared helpers live in `basic_op/utils/` (`spmd_partition.hpp`,
`layout_transform.hpp`); the MX-quantized matmul base used by the
`quant_batch_matmul` solution is `basic_op/matmul/matmul_mx.hpp`. Each
operator has its own test directory under `test/kernel/<op>/`
with a `Makefile`, `compile.all`, and `src/`, so one model failure does not
hide the status of other operator classes.

## Four-PE operator execution regression

Build a single operator through its mirrored test directory:

```bash
make -C benchmark/one-level-arch/test/kernel/element_wise/gelu \
  TESTCASE=gelu COMPILER_DIR="$COMPILER_DIR" diss
```

The 2026-08-29 full regression used main `linx-toolchain-build` commit
`e6a31ef` and SuperScalarModel commit `d8903938`. It compiled every configuration
listed by the `basic_op` `compile.all` files and ran each ELF with four
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
`test/kernel/res_check_all.py`.

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

for script in $(find benchmark/one-level-arch/test/kernel \
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
make -C benchmark/one-level-arch/test/kernel/matmul \
  TESTCASE=matmul_shared \
  COMPILER_DIR="$COMPILER_DIR" \
  res_check=on \
  DTYPE=float B=1 M=256 N=256 K=256 \
  tM=128 tN=256 tK=128
```

### Run and compare

Run one or more named cases:

```bash
python3 benchmark/one-level-arch/test/kernel/res_check_all.py \
```

Run the complete representative numerical portfolio:

```bash
python3 benchmark/one-level-arch/test/kernel/res_check_all.py \
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
| `cumsum_row` | PASS | 0 |
| `reducemax_row` | FAIL | 0.999606; row-result physical stride mismatch |
| `reduceprod_row` | FAIL | 1.06037; row-result physical stride mismatch |
| `reducesum_row` | PASS | 5.72205e-06 |
| `transpose` | PASS | 0 |
| `tadd` | PASS | 0 |
| **Total** | **12 PASS / 4 FAIL / 0 TIMEOUT** | **16 representative cases** |

The remaining failures are kernel/API-model issues exposed by numerical
checking, rather than binary I/O or four-PE synchronization failures:

- Conv2D assigns different output-spatial blocks to PEs around a cooperative
  TMATMUL path; its cube result ownership/writeback needs to be redesigned.
- Gather passes row indexes to the current MGATHER path, while the generated
  instruction/model behavior is offset-oriented and returns corrupted data.
- ReduceMax and ReduceProd expose one valid value per row through a padded
  reduction carrier, but their current TSTORE path writes with the physical
  carrier stride instead of a compact `[Rows]` layout.

### Numerical result: 2026-10 — ArgMaxV2

ArgMaxV2 (CANN `ArgMaxV2` / `aclnnArgMax` semantics) is validated against
`np.argmax` goldens (identical semantics: first max index on ties, NaN
treated as maximal with the first NaN index, ±0.0 equal). Comparison is
exact (`atol=0, rtol=0`). Inputs embed tie / ±inf / positive and negative
NaN / ±0.0 / all-lower-bound rows.

`res_check_all.py` registers one representative golden case per code
path (repo norm — see the fa / matmul sections above):

| Testcase | Path | Result | Max abs error |
|---|---|---|---|
| `argmaxv2` fp32 dim=1 [7,3333] | R-split (tail segment tailB=5, sentinels, barrier merge) | PASS | 0 |
| `argmaxv2_dynamic` fp32 dim=1 [133,1000] | dynamic (runtime tiling, ceil partition + tails) | PASS | 0 |

Broader coverage lives in the self-contained suites under
`test/kernel/argmaxv2/` (each runner generates its own inputs and
compares against `np.argmax`): `run_fuzz.py` — 200 randomized shapes
across dtypes, axes, and both static/dynamic/R-split routings;
`run_net_sc.py` — 8 single-core scaled network shapes with per-case
accuracy check and cycle tracking; `run_gfsim_batch.py` — gfsim timing
batch. Historical full variant-matrix validation (19 dtype × axis ×
rank combinations, all PASS, 2026-10-10) preceded the suite
consolidation.

Toolchain: `linx_blockisa_llvm_musl` (clang 15.0.4, linx64v5), gfrun
four-PE `softcore.multiThreadNum=4`. The kernel exposes a single variadic
entry `argmaxv2<DType, IdType, Axis, Dims...>` (any rank, negative axis;
compile-time folding reuses the 2D row/column detail kernels). ArgMaxV2
returns the index of the
first maximum along the reduction axis: ties resolve to the smallest
index, fp16/fp32/bf16 NaN is treated as maximal (first NaN index), and
±0.0 compares equal. Input dtypes: fp16/fp32/bf16/int32 (sortable-key or
direct reduction); int64 input is NOT supported — the tile-op layer has
no 64-bit reduce/compare instructions (the TROWARGMAX/TCOLARGMAX
contracts and the model reduce/compare dtype whitelist cap at 32-bit),
and the earlier u64-split workaround was removed by decision. Output
index dtype is decoupled: int32 direct store or int64 via a TCVT exit.
Last-axis reduction accepts arbitrary row/col remainders.

Routing compiles to one of three last-axis paths:

1. **Static SPMD** — compile-time-specialized partition where exactly
   expressible: outer rows divisible by `PeCount`, or column windows
   satisfying `Cols % (PeCount * TileN) == 0`.
2. **Dynamic** (`argmaxv2_dynamic`) — every other shape (including
   1-D inputs and misaligned column windows) is auto-delegated at
   compile time with a tiling folded from the template shape: runtime
   tiling, dynamic Valid strips, ceil-partition four-PE execution with
   identical results.
3. **R-split** (`argmaxv2_rsplit`) — large-R small-A shapes
   (`ceil(A/PeCount)*4 < tM`, `R >= PeCount*tM`, and `A < PeCount` or
   `R >= PeCount*tM*tN/2`), where row partitioning cannot fill a row
   tile: each PE reduces its segment as `[S/16,16]` reshaped views
   (~full-load blocks instead of 1/tM row utilization), stage winners
   land in per-PE exclusive rows, an eviction barrier
   (`utils/mt_barrier.hpp`) publishes them, and all PEs run a
   redundant strict-GT merge (watchdog-friendly). Two inner forms are
   selected at compile time by block-count bounds: reshape+merge vs.
   row-parallel direct reduce; 4-byte keys with segments <= 1024 use
   a [64,8] tile (single-row-block main view). bf16 is supported by the
   kernel (same 16-bit sortable-key path as fp16) and compile-verified,
   but is not yet in the res_check matrix.

> Note: `test/kernel/res_check_all.py` computed `ROOT` one directory too
> high for the current tree, which made every case SKIP; fixed to
> `parents[2]` in this change.

### Numerical result: 2026-10-14 — ArgMaxV2 dynamic (runtime tiling)

The runtime-tiling counterpart `argmaxv2_dynamic` was added following the
`solution/normalization` static/dynamic split (rms_norm / group_norm_grad):
shape folding moves to the host, which passes
`argmaxv2_dyn::ArgMaxV2TilingData{g_outer, g_axis, g_inner, tile_m, tile_n,
axis_last}`; the kernel uses dynamic Valid Tile carriers (`-1` templates +
runtime constructor masks), fail-closed runtime validation, and ceil PE
partitioning, which relaxes the static entry's divisibility constraints
(last-axis outer rows and column windows no longer need to divide evenly;
the PE0-solo fallback and the `kInner % (PeCount*TileN) == 0` requirement
are gone — the static entry now auto-delegates exactly these shapes to
`argmaxv2_dynamic`, so both constraints disappeared from the static API
too). Neither entry takes tile parameters — physical tile shapes are
fixed from the dtype traits (normalization convention); the tiling's
`tile_m`/`tile_n` only shrink the runtime valid strips. The reduction
instruction chain is identical to the static
version (sortable key → TROWMAX/TROWARGMAX or TCOLMAX/TCOLARGMAX → strict-GT
TSEL merge), so results are bit-exact. At introduction, nine dynamic cases mirroring the static matrix
(deliberately non-divisible shapes: ceil partitions, partial tail
blocks, runtime strips smaller than the physical tile) all passed with
zero error, and the static matrix re-run confirmed no regression
(9 PASS / 0 FAIL, bit-identical). Those per-variant cases were later
consolidated (2026-10-27) into the representative-case registration
plus the self-contained suites in `test/kernel/argmaxv2/` — see the
2026-10 section above; the dynamic path keeps continuous coverage
there through `run_fuzz.py` and the network suites.

## Scope limits

This directory only contains implementations with race-free output ownership.
A kernel-level multi-PE barrier now exists (`utils/mt_barrier.hpp`: directed
set-eviction publish + phase counting), and the R-split ArgMaxV2 path is the
in-tree example of a cross-PE reduction built on it — intermediate results
are staged in per-PE exclusive cacheline-aligned rows, and only PE0 owns the
final output store, so no two PEs ever write the same destination. New
operators with cross-PE dependencies must follow the same pattern
(exclusive staging + single-owner output); group-wise/global reductions,
TopK, and fused cross-PE kernels that cannot express this ownership still
do not belong here.

## Run with gfrun

Run an ELF built from a multi-thread operator test with four simulated PEs:

```bash
gfrun -t 1 -s softcore.multiThreadNum=4 -f /path/to/operator.elf
```

Do not omit `softcore.multiThreadNum=4`: cooperative kernels require all four
PEs, while SPMD kernels use thread IDs 0 through 3 to partition the work.

The run passes when `gfrun` exits with status 0 and its output contains both
`Reach the End of Benchmark` and `R2 = 0`.
