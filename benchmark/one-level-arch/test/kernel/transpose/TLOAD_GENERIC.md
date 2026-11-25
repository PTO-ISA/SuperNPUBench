# Generic U16 TLOAD/TSTORE transpose

Kernel: `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/kernels/basic_op/transpose/transpose_tload.hpp`

```cpp
constexpr auto shape = std::array{8u, 8u, 64u, 64u};
constexpr auto perm = std::array{0u, 2u, 1u, 3u};
supernpu::multi_thread::transpose_tload<shape, perm>(input, output);
```

`Shape` and `Perm` are compile-time unsigned arrays of rank 1..5. Output axis i
corresponds to input axis Perm[i]. Input and output must not overlap.
Optional template parameters are TileRows (default 0 = automatic, explicit
values 1..32), TileCols (default 64, maximum 256), and PeCount (default 4).
Each Local CellReg Tile uses CubeM16 for at most 16 valid rows or CubeM32
for 17..32 valid rows. U16 physical CELL shapes are 16x4 and 32x2;
physical columns round up to the corresponding CELL width. Tail extents
remain compile-time constants. Non-power-of-two column widths split into
power-of-two segments (seven becomes 4+2+1), working around the current
gfsim CUBE descriptor/capacity check. GlobalTensor keeps ND memory layout;
TLOAD/TSTORE perform ND2M16/M162ND or ND2M32/M322ND conversion.
Automatic rows: 32 when the last input axis moves, otherwise 32 for up to
16 valid columns or 8 for wider rows, clamped to the output row-axis extent.
The former RowMajor 64/256-row Tile profiles are historical baselines and
are not part of this CellReg implementation. Explicit TileRows >32 is rejected.
The caller must launch the
matching number of PEs. Each PE owns consecutive output Tile blocks; empty
partitions are allowed. Tensor size is limited to signed 32-bit byte offsets.

If the last input axis stays last, a Tile combines rows and contiguous columns.
Otherwise, a single-column strided Tile handles the last output axis. Row and
column tails have compile-time valid extents. There is no SharedTile allocation.
These mappings are functional generalizations, not a guarantee that every
permutation outperforms MGATHER.

Build entry:
`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/transpose/compile.all.tload [both|perf|check] [case:0..5] [rows:0=auto] [cols:64]`

Cases: 0 = original 8x8x64x64; 1 = 3x11x5x70 (both tails and uneven PE split);
2 = 3x5x7 reversed; 3 = 2x11x3x5x7 perm 0,3,2,1,4; 4 = 17 identity;
5 = 19x13 transposed. Custom shapes can be supplied by the TRANSPOSE_SHAPE and
TRANSPOSE_PERM preprocessor definitions in the benchmark source.

Checks run one gfrun context, explicitly invoking all four PE partitions in
sequence, comparing against an independent coordinate mapping and checking
both output guards. They do not constitute simultaneous four-PE timing tests.

The compile script places different row/column profiles in separate object
roots to prevent stale-object reuse. The benchmark source also accepts
TRANSPOSE_TILE_ROWS and TRANSPOSE_TILE_COLS preprocessor definitions.
Reproduction and all measured profiles:
`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/transpose_tload_tune_20261029/run.py`
