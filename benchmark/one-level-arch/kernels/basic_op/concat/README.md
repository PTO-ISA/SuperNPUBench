# Two-input concat

`concat_two.hpp` implements concatenation of two independently allocated,
contiguous row-major tensors with the same element type and matching dimensions
outside the concat axis. Shapes are compile-time parameters. It copies bits using
PTO `TLOAD` / `TSTORE`; no arithmetic, numerical conversion or temporary global
buffer is required.

```cpp
#include "basic_op/concat/concat_two.hpp"

// A: [3, 5], B: [3, 7], concat axis=1 -> C: [3, 12].
// Each of the four PEs calls this with the same pointers.
supernpu::multi_thread::concat_two<float, 3, 5, 7>(a, b, c);

// A: [2, 3, 5, 7], B: [2, 4, 5, 7], axis=1 -> C: [2, 7, 5, 7].
supernpu::multi_thread::concat_two<float, 2, 3, 4, 35>(a, b, c);
```

The template parameters are `T, Outer, LeftAxis, RightAxis, Inner=1,
TileElements=128, PeCount=4`:

- `Outer = product(shape[:axis])` and `Inner = product(shape[axis+1:])`.
  Normalize a negative axis before calculating these products.
- Each output row contains `LeftAxis * Inner` elements from A followed by
  `RightAxis * Inner` elements from B. Input axis lengths may differ or be zero.
  `Outer` and `Inner` must be positive; empty outer/inner dimensions are outside
  this initial interface.
- The output allocation contains `Outer * (LeftAxis + RightAxis) * Inner`
  elements and must not overlap either input. There is no runtime shape/stride
  validation; the caller supplies consistent shapes and sufficient storage.
- Full `(row, tile)` jobs and tail rows are distributed cyclically among PEs.
  `Outer` need not be divisible by four. Tail tiles specify their exact valid
  columns; input and output global views likewise expose only valid elements.
  Different PEs write disjoint output elements.
- Callers must publish inputs before launch and join participating PEs before
  reading the output. The test harness uses `multi_thread_res_check.h` for this.
  `PeCount=1` is also available for callers that run only PE0.
- Tiles are powers of two, limited to 2048 bytes. The test portfolio uses FP32,
  FP16, int32 and uint8. The API is a basic copy implementation, with no
  strided-input, arbitrary-input-list, in-place, or runtime-shape support.

## Relationship to the Ascend A5 reference

The reference inspected was `ops-math/conversion/concat_d` with the Ascend 950 /
arch35 configuration. `concat_d` delegates its host tiling to
`conversion/concat/op_host/arch35/concat_tiling_arch35.cpp::TilingCommon` and
dispatches to kernels under `conversion/concat/op_kernel/arch35/`.

`GetTensorListDim` merges dimensions before the axis into dim0 and dimensions
from the axis onward into each input's dim1. `GenerateOutputShape` sums dim1.
`OneAxisConcatPureCopy::ProcessNoSplitDim1` copies each input with an accumulated
output column offset. This implementation adopts that dimension folding and
copy-by-input mapping. For output row r, with L=LeftAxis*Inner and
R=RightAxis*Inner:

```text
C[r*(L+R)       : r*(L+R)+L] = A[r*L : (r+1)*L]
C[r*(L+R)+L     : (r+1)*(L+R)] = B[r*R : (r+1)*R]
```

A5 uses UB queues, DataCopy and shape/alignment-dependent tiling dispatch.
SuperScalar uses Vec tiles and `TLOAD` / `TSTORE`, with a fixed-size tile and
SPMD job assignment. A5 gather/SIMT specializations and buffering optimizations
are not part of this initial implementation. Existing `concat_gather` and
`concat_scatter` entry points remain available; `concat_two` accepts separate
input pointers rather than requiring prepacked inputs.

## Build and exact validation

Set `COMPILER_DIR` to the Linx toolchain's bin directory and `GFRUN` to the model
executable. From `benchmark/one-level-arch/test/kernel/concat`:

```bash
make TESTCASE=concat_two OUTER=3 LEFT=5 RIGHT=7 INNER=1 TILE=128 \
     DTYPE=float res_check=on
make check-concat-two
# Or run selected cases:
python3 src/check_concat_two.py --case negative_axis --case multi_tile_tail
```

The Python runner compiles each configuration with `make -B -j1`, prepares two
independent inputs, and runs the hosted ELF with four PEs. It checks exit status,
the benchmark-end and `R2=0` markers, worker entry startup, exact output length,
and byte-for-byte equality against `numpy.concatenate`. FP inputs include NaN
payloads, infinities and signed zero. Prefix/suffix canaries detect out-of-bounds
output writes. No floating-point tolerance is applied to a copy operation.

The 14-case portfolio covers aligned copies, first/last/middle axes, unequal
lengths, full tiles plus tails, rows not divisible by four, byte offsets larger
than 65535, small tiles, tiny tensors and zero concat-axis lengths. Logs and
`summary.json` are saved in a unique subdirectory of `--artifacts` (defaults to
`ARTIFACTS`, or ignored build output). Binary inputs, golden and guarded output
are under the ignored `compare/<elf-stem>/` directory. The runner is sequential;
do not run it concurrently against the same build/compare directories.

This validates functional correctness with gfrun. No hardware or gfsim timing
performance claim is made.
