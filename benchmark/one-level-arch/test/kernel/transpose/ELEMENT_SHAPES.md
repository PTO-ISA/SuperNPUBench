# U16 transpose shape builds

All current compile entry points use U16 (uint16_t, 2 bytes/element), four active
PEs and out-of-place tensors. Each PE owns every fourth output block. All data
and index Tiles are Local; SharedTile footprint, including scales, is 0 bytes.

| Case | Input | Output | Read block | Input row stride (elements) |
|---|---|---|---|---:|
| 1 | 129×22×120 | 120×22×129 | 32×1, last block 1×1 | 2640 |
| 2 | 8×4096×3 | 8×3×4096 | 32×1 | 3 |
| 3 | 1476×32 | 32×1476 | 32×1, last block 4×1 | 32 |
| 4 | 8×64×4×16×7 | 8×16×4×64×7 | 8×(4+2+1) | 448 |
| 5 | 8×8×64×64 | 8×64×8×64 | 1×64 | 64 |

Current CellReg layout (2026-10-29): explicit MGATHER uses CubeM32 for
both U16 data and U32 offsets. Specialized TLOAD uses CubeM16 for <=16 rows,
otherwise CubeM32. Seven-column blocks split into 4+2+1 transfers while
preserving the seven-element output row stride.
GlobalTensor remains ND external memory. New fixed8/automatic generic TLOAD
also use M16/M32; see TLOAD_GENERIC.md for their different partition policy.

Case 1 was reduced to 129×22×120 by user request. It has 340,560 U16 elements;
input plus output need 1,362,240 bytes, excluding the small output guard.

Both implementations use the same scalar source/destination address formulas,
block shapes, output ownership and TSTORE. The MGATHER version loads a static
byte-offset Tile once per PE (and one for the tail shape if needed), then reuses
it. The TLOAD version expresses the affine input row stride directly. Padding
keeps physical Local Tiles at legal sizes while valid rows/columns restrict
memory accesses. The Element-if implementation and its build entries were removed on 2026-11-11.

Custom row strides use explicit GlobalTensor/Stride. The current convenience
global_tensor wrapper derives packed strides from Cols, even if its layout
argument carries another RowStride; do not use that wrapper for affine sources
or padded index arrays.

## Build

```sh
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
# All five shapes, both U16 implementations:
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/transpose/compile.all
# Only MGATHER, cases 2 and 5:
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/transpose/compile.all.u16 mgather 2 5
# One shape, both implementations (optional argument: tload or mgather):
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/transpose/compile.all.8x4096x3
```

OBJ_ROOT may override the build directory. Case and implementation receive
separate object directories and ELF names containing U16_G0/G1_PE4.
G0 means TLOAD/TSTORE, G1 means MGATHER/TSTORE.

The legacy Tile ISA/TTRANS headers remain for source compatibility, but are not
used by the new U16 build entries. Their TTRANS instantiation does not compile
with the current compiler. The explicit 64×64 testcase now uses the U16 strided
TLOAD/TSTORE primitive; the default testcase is U16 shape case 2.

## Validation and timing

TRANSPOSE_CHECK=1 selects a full numerical check using deterministic nonzero
U16 data, an independent input-coordinate reference, and output guards. It uses
one gfrun context to execute all four partitions sequentially. Timing ELFs use
four simultaneous gfsim contexts, zero-initialized input and no checking inside
the timed region. These are distinct validations, not a claim of concurrent
four-PE numerical checking.

Timing uses the previously selected HIF4 configuration and 1.8 GHz. Reported
cycles are full-program totals including startup and exit. Old F32 timings
are historical and are not U16 baselines.

Current all-method reproduction and results (128 B/cycle, mean330,
uniform half-width180, four PE, 1.8GHz):
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/transpose_all_cube_20261029/run.py
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/transpose_all_cube_20261029/REPORT.txt

Earlier RowMajor and 156 B/cycle HIF4 results remain historical evidence only;
they must not be combined with this M16/M32 comparison as identical conditions.

## Main integration validation (2026-11-25)

Main compiler e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156 compiled and linked
all five fixed-shape TLOAD cases, six gather-plane cases and six generic TLOAD
cases. Fixed-shape case 4 also compiled and linked with MGATHER and RES_CHECK.
These are build checks, not new numerical or timing results. The mandated
functional-model checkout was absent, so gfrun was not rerun. The previously
published r5 results apply to the frozen archive, not automatically to this
working-tree integration. The old headers are retained; the active benchmark
uses the new U16 implementation.
