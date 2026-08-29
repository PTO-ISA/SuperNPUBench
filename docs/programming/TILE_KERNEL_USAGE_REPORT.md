# PTO Tile Instruction Usage Report — SuperNPUBench one-level-arch kernels

> This report cross-references the 109 PTO ISA 0.58.0 tile operations (see [`TILE_INSTRUCTIONS.md`](TILE_INSTRUCTIONS.md)) against every header in `SuperNPUBench/benchmark/one-level-arch/kernels/`. It records which tile instruction each operator uses, how many call sites it has, and whether the operator stays within the accepted PTO tile catalog.

## Executive summary

- **Tile operations in PTO 0.58.0 catalog**: 109
- **Tile operations actually used by kernels**: 49 (44%)
- **Tile operations not used by any kernel**: 60 (55%)
- **Kernel header files scanned**: 60
- **Operator categories**: 15
- **Total tile-instruction call sites across all kernels**: 2257

### Most-used tile instructions (by kernel-file count)

| Rank | Mnemonic | Engine | Class | # files | # call sites |
| ---: | --- | :---: | --- | ---: | ---: |
| 1 | `TLOAD` | TLSU | memory-and-data-movement | 58 | 886 |
| 2 | `TSTORE` | TLSU | memory-and-data-movement | 58 | 112 |
| 3 | `TEXPANDS` | VEC | tile-scalar-and-immediate | 33 | 79 |
| 4 | `TADD` | VEC | elementwise-tile-tile | 25 | 113 |
| 5 | `TCVT` | VEC | elementwise-tile-tile | 21 | 58 |
| 6 | `TMULS` | VEC | tile-scalar-and-immediate | 18 | 49 |
| 7 | `TMUL` | VEC | elementwise-tile-tile | 12 | 31 |
| 8 | `TROWSUM` | SFU | reduce-and-expand | 12 | 20 |
| 9 | `TMAX` | VEC | elementwise-tile-tile | 11 | 38 |
| 10 | `TRECIP` | SFU | elementwise-tile-tile | 11 | 17 |
| 11 | `TSUB` | VEC | elementwise-tile-tile | 10 | 18 |
| 12 | `TROWMAX` | SFU | reduce-and-expand | 10 | 18 |
| 13 | `TCOLSUM` | SFU | reduce-and-expand | 9 | 19 |
| 14 | `TROWEXPANDMUL` | SFU | reduce-and-expand | 9 | 14 |
| 15 | `TEXP` | SFU | elementwise-tile-tile | 8 | 18 |

### Coverage by execution engine

| Engine | In catalog | Used by kernels | Usage rate |
| :---: | ---: | ---: | ---: |
| VEC | 35 | 22 | 62% |
| SFU | 52 | 19 | 36% |
| CUBE | 12 | 3 | 25% |
| TLSU | 10 | 5 | 50% |

### Coverage by semantic class

| Class | In catalog | Used by kernels | Usage rate |
| --- | ---: | ---: | ---: |
| `elementwise-tile-tile` | 25 | 13 | 52% |
| `tile-scalar-and-immediate` | 15 | 11 | 73% |
| `reduce-and-expand` | 28 | 12 | 42% |
| `memory-and-data-movement` | 9 | 4 | 44% |
| `matrix-and-matrix-vector` | 12 | 3 | 25% |
| `layout-and-rearrangement` | 7 | 4 | 57% |
| `irregular-and-complex` | 13 | 2 | 15% |

## Operator × tile-instruction cross-table

`✓` = operator uses the instruction; a number = call-site count. Operators are listed in the order of the kernels README.

| Operator | `TLOAD` | `TSTORE` | `TEXPANDS` | `TADD` | `TCVT` | `TMULS` | `TMUL` | `TROWSUM` | `TMAX` | `TRECIP` | `TSUB` | `TROWMAX` | `TCOLSUM` | `TROWEXPANDMUL` | `TEXP` | `TMATMUL` | `TINSERT` | `MGATHER` | `TCOLEXPANDMUL` | `TADDS` | `TCMP` | `TSEL` | `TMATMUL_ACC` | `TCOLMAX` |
| --- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `matmul` | 769 | 28 | 6 | 50 | · | · | 1 | · | · | · | · | · | · | · | · | 234 | · | 32 | 1 | · | · | · | 236 | · |
| `fa` | 23 | 5 | 12 | 31 | 17 | 6 | 7 | 3 | 25 | 5 | 7 | 3 | 4 | 6 | 15 | 5 | · | · | 3 | · | · | · | 1 | 4 |
| `broadcast` | 6 | 8 | 1 | 1 | · | 1 | · | · | · | · | · | · | · | · | · | · | 4 | 2 | · | · | · | · | · | · |
| `reduction` | 37 | 23 | 18 | 14 | 4 | · | 8 | 9 | 10 | · | · | 8 | 12 | · | · | · | 4 | · | · | · | · | · | · | 6 |
| `element_wise` | 4 | 3 | · | 1 | 2 | 1 | 8 | · | · | 1 | · | · | · | · | 1 | · | · | · | · | 7 | · | · | · | · |
| `gather` | 4 | 4 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | 4 | · | · | · | · | · | · |
| `concat` | 2 | 2 | 4 | 8 | 4 | 14 | · | · | · | · | 6 | · | · | · | · | · | · | 2 | · | · | · | · | · | · |
| `transpose` | 6 | 8 | 2 | 2 | 2 | 4 | · | · | · | · | 2 | · | · | · | · | · | · | 2 | · | · | · | · | · | · |
| `control` | 1 | 1 | 6 | 1 | 11 | 7 | · | · | · | · | · | · | · | · | · | · | · | 2 | · | 2 | 2 | 1 | · | · |
| `sort` | 2 | · | 3 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |
| `deepseek/engram` | 3 | 2 | 1 | · | 2 | 1 | 1 | · | · | · | · | · | · | · | · | · | · | · | · | 1 | · | · | · | · |
| `deepseek/mhc` | 10 | 6 | · | 2 | 2 | 2 | 4 | 3 | · | 5 | · | 1 | 3 | 3 | 1 | 1 | · | · | 3 | 7 | · | · | 1 | · |
| `deepseek/moe` | 10 | 13 | 26 | 3 | 5 | 3 | · | 5 | · | 2 | 3 | 2 | · | 2 | · | · | 3 | · | · | 2 | 8 | 7 | · | · |
| `deepseek/quant` | 8 | 8 | · | · | 9 | 10 | 2 | · | 3 | 4 | · | 4 | · | 3 | 1 | · | · | · | 2 | 1 | · | · | · | 2 |
| `deepseek/transpose` | 1 | 1 | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |

---

## Per-operator detail

### <a id="op-matmul"></a>`matmul`

General and quantized matrix multiply (FP32/FP16/FP8, MX microscaling, mask/dynamic/vec variants).

- **Files scanned**: 4
- **Distinct tile instructions used**: 11
- **Total tile-instruction call sites**: 1416
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `matmul/matmul.hpp`
- `matmul/matmul_mx.hpp`
- `matmul/matmul_mx_pto.hpp`
- `matmul/matmul_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 763 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TMATMUL_ACC`](TILE_INSTRUCTIONS.md#tmatmul_acc) | CUBE | matrix-and-matrix-vector | 235 | Multiply matrices and accumulate into the supplied accumulator Tile. |
| [`TMATMUL`](TILE_INSTRUCTIONS.md#tmatmul) | CUBE | matrix-and-matrix-vector | 232 | Multiply the left and right matrices into the destination. |
| [`TMATMUL_MX`](TILE_INSTRUCTIONS.md#tmatmul_mx) | CUBE | matrix-and-matrix-vector | 68 | Multiply matrices using row and column scale Tiles. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 50 | Apply elementwise addition to the two source Tiles. |
| [`MGATHER`](TILE_INSTRUCTIONS.md#mgather) | TLSU | memory-and-data-movement | 32 | Gather GM elements at Tile-provided indices into the destination. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 27 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 6 | Fill the destination Tile by expanding the bound scalar value. |
| [`TCOLEXPANDMUL`](TILE_INSTRUCTIONS.md#tcolexpandmul) | SFU | reduce-and-expand | 1 | Apply multiplication while expanding the bound col vector across the source Tile. |
| [`TMUL`](TILE_INSTRUCTIONS.md#tmul) | VEC | elementwise-tile-tile | 1 | Apply elementwise multiplication to the two source Tiles. |
| [`TCOLEXPAND`](TILE_INSTRUCTIONS.md#tcolexpand) | SFU | reduce-and-expand | 1 | Apply broadcast while expanding the bound col vector across the source Tile. |

### <a id="op-fa"></a>`fa`

Flash Attention family — 2D unroll, unaligned boundary, HIF4 quantized, DCore-optimized, sparse (SFA).

- **Files scanned**: 5
- **Distinct tile instructions used**: 24
- **Total tile-instruction call sites**: 199
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `fa/fa_2d_unroll_gmma.hpp`
- `fa/fa_2d_unroll_pto.hpp`
- `fa/fa_hif4.hpp`
- `fa/fa_hif4_pto.hpp`
- `fa/sfa_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 31 | Apply elementwise addition to the two source Tiles. |
| [`TMAX`](TILE_INSTRUCTIONS.md#tmax) | VEC | elementwise-tile-tile | 25 | Apply elementwise maximum selection to the two source Tiles. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 23 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 17 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TEXP`](TILE_INSTRUCTIONS.md#texp) | SFU | elementwise-tile-tile | 15 | Apply elementwise exponential to the source Tile. |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 12 | Fill the destination Tile by expanding the bound scalar value. |
| [`TMUL`](TILE_INSTRUCTIONS.md#tmul) | VEC | elementwise-tile-tile | 7 | Apply elementwise multiplication to the two source Tiles. |
| [`TSUB`](TILE_INSTRUCTIONS.md#tsub) | VEC | elementwise-tile-tile | 7 | Apply elementwise subtraction to the two source Tiles. |
| [`TROWEXPANDMUL`](TILE_INSTRUCTIONS.md#trowexpandmul) | SFU | reduce-and-expand | 6 | Apply multiplication while expanding the bound row vector across the source Tile. |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 6 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TMATMUL_MX`](TILE_INSTRUCTIONS.md#tmatmul_mx) | CUBE | matrix-and-matrix-vector | 6 | Multiply matrices using row and column scale Tiles. |
| [`TRECIP`](TILE_INSTRUCTIONS.md#trecip) | SFU | elementwise-tile-tile | 5 | Apply elementwise reciprocal to the source Tile. |
| [`TMATMUL`](TILE_INSTRUCTIONS.md#tmatmul) | CUBE | matrix-and-matrix-vector | 5 | Multiply the left and right matrices into the destination. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 5 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TCOLEXPANDSUB`](TILE_INSTRUCTIONS.md#tcolexpandsub) | SFU | reduce-and-expand | 5 | Apply subtraction while expanding the bound col vector across the source Tile. |
| [`TCOLMAX`](TILE_INSTRUCTIONS.md#tcolmax) | SFU | reduce-and-expand | 4 | Reduce each source col to its maximum. |
| [`TCOLSUM`](TILE_INSTRUCTIONS.md#tcolsum) | SFU | reduce-and-expand | 4 | Reduce each source col to its sum. |
| [`TROWEXPANDSUB`](TILE_INSTRUCTIONS.md#trowexpandsub) | SFU | reduce-and-expand | 3 | Apply subtraction while expanding the bound row vector across the source Tile. |
| [`TROWMAX`](TILE_INSTRUCTIONS.md#trowmax) | SFU | reduce-and-expand | 3 | Reduce each source row to its maximum. |
| [`TROWSUM`](TILE_INSTRUCTIONS.md#trowsum) | SFU | reduce-and-expand | 3 | Reduce each source row to its sum. |
| [`TCOLEXPANDMUL`](TILE_INSTRUCTIONS.md#tcolexpandmul) | SFU | reduce-and-expand | 3 | Apply multiplication while expanding the bound col vector across the source Tile. |
| [`TQUANT`](TILE_INSTRUCTIONS.md#tquant) | SFU | irregular-and-complex | 2 | Quantize source elements using scale, zero point, rounding, and saturation controls. |
| [`TMATMUL_ACC`](TILE_INSTRUCTIONS.md#tmatmul_acc) | CUBE | matrix-and-matrix-vector | 1 | Multiply matrices and accumulate into the supplied accumulator Tile. |
| [`TMOV`](TILE_INSTRUCTIONS.md#tmov) | TLSU | layout-and-rearrangement | 1 | Copy the source Tile payload and definedness into the destination. |

### <a id="op-broadcast"></a>`broadcast`

Multi-dimensional broadcast (2D–5D, vectorized, mscatter / nocopyout / simple variants).

- **Files scanned**: 4
- **Distinct tile instructions used**: 11
- **Total tile-instruction call sites**: 28
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `broadcast/broadcast_pto.hpp`
- `broadcast/broadcast_vec_019_pto.hpp`
- `broadcast/broadcast_vec_039_pto.hpp`
- `broadcast/broadcast_vec_07_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 8 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 6 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TINSERT`](TILE_INSTRUCTIONS.md#tinsert) | SFU | layout-and-rearrangement | 4 | Insert the source Tile into the destination region at the encoded row and column offsets. |
| [`MGATHER`](TILE_INSTRUCTIONS.md#mgather) | TLSU | memory-and-data-movement | 2 | Gather GM elements at Tile-provided indices into the destination. |
| [`TROWEXPAND`](TILE_INSTRUCTIONS.md#trowexpand) | SFU | reduce-and-expand | 2 | Apply broadcast while expanding the bound row vector across the source Tile. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 1 | Apply elementwise addition to the two source Tiles. |
| [`TCI`](TILE_INSTRUCTIONS.md#tci) | SFU | irregular-and-complex | 1 | Initialize destination elements as an ascending or descending counter sequence. |
| [`TDIVS`](TILE_INSTRUCTIONS.md#tdivs) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise division between the source Tile and bound scalar. |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TREMS`](TILE_INSTRUCTIONS.md#trems) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise remainder between the source Tile and bound scalar. |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 1 | Fill the destination Tile by expanding the bound scalar value. |

### <a id="op-reduction"></a>`reduction`

Row/column reductions: sum / max / prod / cumsum; single-tree and unaligned variants.

- **Files scanned**: 16
- **Distinct tile instructions used**: 14
- **Total tile-instruction call sites**: 158
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `reduction/cumsum_colvec_pto.hpp`
- `reduction/cumsum_rowvec_pto.hpp`
- `reduction/reducemax_colvec_pto.hpp`
- `reduction/reducemax_colvec_unalign_120_8_pto.hpp`
- `reduction/reducemax_rowvec_pto.hpp`
- `reduction/reducemax_rowvec_single_tree.hpp`
- `reduction/reducemax_rowvec_single_tree_pto.hpp`
- `reduction/reduceprod_colvec_pto.hpp`
- `reduction/reduceprod_rowvec_pto.hpp`
- `reduction/reducesum_colvec_pto.hpp`
- `reduction/reducesum_colvec_single_tree.hpp`
- `reduction/reducesum_colvec_single_tree_pto.hpp`
- `reduction/reducesum_colvec_unalign_120_8_pto.hpp`
- `reduction/reducesum_rowvec_pto.hpp`
- `reduction/reducesum_rowvec_single_tree.hpp`
- `reduction/reducesum_rowvec_single_tree_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 36 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 22 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 18 | Fill the destination Tile by expanding the bound scalar value. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 14 | Apply elementwise addition to the two source Tiles. |
| [`TCOLSUM`](TILE_INSTRUCTIONS.md#tcolsum) | SFU | reduce-and-expand | 12 | Reduce each source col to its sum. |
| [`TMAX`](TILE_INSTRUCTIONS.md#tmax) | VEC | elementwise-tile-tile | 10 | Apply elementwise maximum selection to the two source Tiles. |
| [`TROWSUM`](TILE_INSTRUCTIONS.md#trowsum) | SFU | reduce-and-expand | 8 | Reduce each source row to its sum. |
| [`TMUL`](TILE_INSTRUCTIONS.md#tmul) | VEC | elementwise-tile-tile | 8 | Apply elementwise multiplication to the two source Tiles. |
| [`TROWMAX`](TILE_INSTRUCTIONS.md#trowmax) | SFU | reduce-and-expand | 8 | Reduce each source row to its maximum. |
| [`TCOLMAX`](TILE_INSTRUCTIONS.md#tcolmax) | SFU | reduce-and-expand | 6 | Reduce each source col to its maximum. |
| [`TINSERT`](TILE_INSTRUCTIONS.md#tinsert) | SFU | layout-and-rearrangement | 4 | Insert the source Tile into the destination region at the encoded row and column offsets. |
| [`TCOLPROD`](TILE_INSTRUCTIONS.md#tcolprod) | SFU | reduce-and-expand | 4 | Reduce each source col to its product. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 4 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TROWPROD`](TILE_INSTRUCTIONS.md#trowprod) | SFU | reduce-and-expand | 4 | Reduce each source row to its product. |

### <a id="op-element_wise"></a>`element_wise`

Elementwise activations (GELU polynomial-fit, exact erf and tanh approximations).

- **Files scanned**: 2
- **Distinct tile instructions used**: 11
- **Total tile-instruction call sites**: 30
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `element_wise/gelu_pto.hpp`
- `element_wise/tadd_multithread.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TMUL`](TILE_INSTRUCTIONS.md#tmul) | VEC | elementwise-tile-tile | 8 | Apply elementwise multiplication to the two source Tiles. |
| [`TADDS`](TILE_INSTRUCTIONS.md#tadds) | VEC | tile-scalar-and-immediate | 7 | Apply elementwise addition between the source Tile and bound scalar. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 4 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 3 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 2 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 1 | Apply elementwise addition to the two source Tiles. |
| [`TEXP`](TILE_INSTRUCTIONS.md#texp) | SFU | elementwise-tile-tile | 1 | Apply elementwise exponential to the source Tile. |
| [`TRECIP`](TILE_INSTRUCTIONS.md#trecip) | SFU | elementwise-tile-tile | 1 | Apply elementwise reciprocal to the source Tile. |
| [`TMAXS`](TILE_INSTRUCTIONS.md#tmaxs) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise maximum selection between the source Tile and bound scalar. |
| [`TMINS`](TILE_INSTRUCTIONS.md#tmins) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise minimum selection between the source Tile and bound scalar. |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise multiplication between the source Tile and bound scalar. |

### <a id="op-gather"></a>`gather`

Large-scale gather with multiple indexing modes.

- **Files scanned**: 1
- **Distinct tile instructions used**: 3
- **Total tile-instruction call sites**: 12
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `gather/gather_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`MGATHER`](TILE_INSTRUCTIONS.md#mgather) | TLSU | memory-and-data-movement | 4 | Gather GM elements at Tile-provided indices into the destination. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 4 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 4 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |

### <a id="op-concat"></a>`concat`

Concat via gather-based or scatter-based composition.

- **Files scanned**: 2
- **Distinct tile instructions used**: 11
- **Total tile-instruction call sites**: 58
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `concat/concat_gather_pto.hpp`
- `concat/concat_scatter_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 14 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TDIVS`](TILE_INSTRUCTIONS.md#tdivs) | VEC | tile-scalar-and-immediate | 10 | Apply elementwise division between the source Tile and bound scalar. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 8 | Apply elementwise addition to the two source Tiles. |
| [`TSUB`](TILE_INSTRUCTIONS.md#tsub) | VEC | elementwise-tile-tile | 6 | Apply elementwise subtraction to the two source Tiles. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 4 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TCI`](TILE_INSTRUCTIONS.md#tci) | SFU | irregular-and-complex | 4 | Initialize destination elements as an ascending or descending counter sequence. |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 4 | Fill the destination Tile by expanding the bound scalar value. |
| [`MSCATTER`](TILE_INSTRUCTIONS.md#mscatter) | TLSU | memory-and-data-movement | 2 | Scatter source Tile elements to GM addresses selected by Tile indices. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 2 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`MGATHER`](TILE_INSTRUCTIONS.md#mgather) | TLSU | memory-and-data-movement | 2 | Gather GM elements at Tile-provided indices into the destination. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 2 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |

### <a id="op-transpose"></a>`transpose`

3D–6D transpose and vectorized transpose (007/050).

- **Files scanned**: 3
- **Distinct tile instructions used**: 11
- **Total tile-instruction call sites**: 43
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `transpose/transpose_pto.hpp`
- `transpose/transpose_vector_007_pto.hpp`
- `transpose/transpose_vector_050_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TTRANS`](TILE_INSTRUCTIONS.md#ttrans) | SFU | layout-and-rearrangement | 9 | Transpose the source Tile into the destination. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 8 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 6 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TDIVS`](TILE_INSTRUCTIONS.md#tdivs) | VEC | tile-scalar-and-immediate | 4 | Apply elementwise division between the source Tile and bound scalar. |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 4 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 2 | Apply elementwise addition to the two source Tiles. |
| [`TSUB`](TILE_INSTRUCTIONS.md#tsub) | VEC | elementwise-tile-tile | 2 | Apply elementwise subtraction to the two source Tiles. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 2 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TCI`](TILE_INSTRUCTIONS.md#tci) | SFU | irregular-and-complex | 2 | Initialize destination elements as an ascending or descending counter sequence. |
| [`MGATHER`](TILE_INSTRUCTIONS.md#mgather) | TLSU | memory-and-data-movement | 2 | Gather GM elements at Tile-provided indices into the destination. |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 2 | Fill the destination Tile by expanding the bound scalar value. |

### <a id="op-control"></a>`control`

Pure tile-op kernels (no SIMT) such as hash-table lookup.

- **Files scanned**: 1
- **Distinct tile instructions used**: 16
- **Total tile-instruction call sites**: 54
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `control/hashtable_lookup_simd.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 11 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TSHRS`](TILE_INSTRUCTIONS.md#tshrs) | VEC | tile-scalar-and-immediate | 8 | Apply elementwise right shift between the source Tile and bound scalar. |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 7 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 6 | Fill the destination Tile by expanding the bound scalar value. |
| [`TXOR`](TILE_INSTRUCTIONS.md#txor) | VEC | elementwise-tile-tile | 5 | Apply elementwise bitwise XOR to the two source Tiles. |
| [`TREM`](TILE_INSTRUCTIONS.md#trem) | VEC | elementwise-tile-tile | 3 | Apply elementwise remainder to the two source Tiles. |
| [`TAND`](TILE_INSTRUCTIONS.md#tand) | VEC | elementwise-tile-tile | 2 | Apply elementwise bitwise AND to the two source Tiles. |
| [`TCMP`](TILE_INSTRUCTIONS.md#tcmp) | VEC | elementwise-tile-tile | 2 | Apply elementwise comparison to the two source Tiles. |
| [`MGATHER`](TILE_INSTRUCTIONS.md#mgather) | TLSU | memory-and-data-movement | 2 | Gather GM elements at Tile-provided indices into the destination. |
| [`TADDS`](TILE_INSTRUCTIONS.md#tadds) | VEC | tile-scalar-and-immediate | 2 | Apply elementwise addition between the source Tile and bound scalar. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 1 | Apply elementwise addition to the two source Tiles. |
| [`TOR`](TILE_INSTRUCTIONS.md#tor) | VEC | elementwise-tile-tile | 1 | Apply elementwise bitwise OR to the two source Tiles. |
| [`TSEL`](TILE_INSTRUCTIONS.md#tsel) | VEC | elementwise-tile-tile | 1 | Select each destination element from the true or false source under the mask Tile. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 1 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 1 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TSHLS`](TILE_INSTRUCTIONS.md#tshls) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise left shift between the source Tile and bound scalar. |

### <a id="op-sort"></a>`sort`

Top-K via radix-bucket histogram.

- **Files scanned**: 1
- **Distinct tile instructions used**: 4
- **Total tile-instruction call sites**: 8
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `sort/topk_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 3 | Fill the destination Tile by expanding the bound scalar value. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 2 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TSHRS`](TILE_INSTRUCTIONS.md#tshrs) | VEC | tile-scalar-and-immediate | 2 | Apply elementwise right shift between the source Tile and bound scalar. |
| [`TANDS`](TILE_INSTRUCTIONS.md#tands) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise bitwise AND between the source Tile and bound scalar. |

### <a id="op-deepseek-engram"></a>`deepseek/engram`

DeepSeek engram hash + fused weight (TileKernels migration).

- **Files scanned**: 2
- **Distinct tile instructions used**: 9
- **Total tile-instruction call sites**: 13
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `deepseek/engram/engram_hash_pto.hpp`
- `deepseek/engram/fused_weight_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 3 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 2 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 2 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TXOR`](TILE_INSTRUCTIONS.md#txor) | VEC | elementwise-tile-tile | 1 | Apply elementwise bitwise XOR to the two source Tiles. |
| [`TADDS`](TILE_INSTRUCTIONS.md#tadds) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise addition between the source Tile and bound scalar. |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TREMS`](TILE_INSTRUCTIONS.md#trems) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise remainder between the source Tile and bound scalar. |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 1 | Fill the destination Tile by expanding the bound scalar value. |
| [`TMUL`](TILE_INSTRUCTIONS.md#tmul) | VEC | elementwise-tile-tile | 1 | Apply elementwise multiplication to the two source Tiles. |

### <a id="op-deepseek-mhc"></a>`deepseek/mhc`

DeepSeek MHC expand / Sinkhorn / backward / multilayer recompute / norm (5+1 kernels).

- **Files scanned**: 5
- **Distinct tile instructions used**: 17
- **Total tile-instruction call sites**: 55
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `deepseek/mhc/expand_to_mhc_bwd_pto.hpp`
- `deepseek/mhc/expand_to_mhc_pto.hpp`
- `deepseek/mhc/multilayer_recompute_pto.hpp`
- `deepseek/mhc/norm_fn_pto.hpp`
- `deepseek/mhc/sinkhorn_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 10 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TADDS`](TILE_INSTRUCTIONS.md#tadds) | VEC | tile-scalar-and-immediate | 7 | Apply elementwise addition between the source Tile and bound scalar. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 6 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TRECIP`](TILE_INSTRUCTIONS.md#trecip) | SFU | elementwise-tile-tile | 5 | Apply elementwise reciprocal to the source Tile. |
| [`TMUL`](TILE_INSTRUCTIONS.md#tmul) | VEC | elementwise-tile-tile | 4 | Apply elementwise multiplication to the two source Tiles. |
| [`TCOLEXPANDMUL`](TILE_INSTRUCTIONS.md#tcolexpandmul) | SFU | reduce-and-expand | 3 | Apply multiplication while expanding the bound col vector across the source Tile. |
| [`TCOLSUM`](TILE_INSTRUCTIONS.md#tcolsum) | SFU | reduce-and-expand | 3 | Reduce each source col to its sum. |
| [`TROWEXPANDMUL`](TILE_INSTRUCTIONS.md#trowexpandmul) | SFU | reduce-and-expand | 3 | Apply multiplication while expanding the bound row vector across the source Tile. |
| [`TROWSUM`](TILE_INSTRUCTIONS.md#trowsum) | SFU | reduce-and-expand | 3 | Reduce each source row to its sum. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 2 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 2 | Apply elementwise addition to the two source Tiles. |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 2 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TEXP`](TILE_INSTRUCTIONS.md#texp) | SFU | elementwise-tile-tile | 1 | Apply elementwise exponential to the source Tile. |
| [`TROWEXPANDSUB`](TILE_INSTRUCTIONS.md#trowexpandsub) | SFU | reduce-and-expand | 1 | Apply subtraction while expanding the bound row vector across the source Tile. |
| [`TROWMAX`](TILE_INSTRUCTIONS.md#trowmax) | SFU | reduce-and-expand | 1 | Reduce each source row to its maximum. |
| [`TMATMUL`](TILE_INSTRUCTIONS.md#tmatmul) | CUBE | matrix-and-matrix-vector | 1 | Multiply the left and right matrices into the destination. |
| [`TMATMUL_ACC`](TILE_INSTRUCTIONS.md#tmatmul_acc) | CUBE | matrix-and-matrix-vector | 1 | Multiply matrices and accumulate into the supplied accumulator Tile. |

### <a id="op-deepseek-moe"></a>`deepseek/moe`

DeepSeek MoE expand-to-fused, unique group indices, top-k gate, fused mapping, reduce, normalize, mask, group count (8+1 kernels).

- **Files scanned**: 8
- **Distinct tile instructions used**: 23
- **Total tile-instruction call sites**: 106
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `deepseek/moe/expand_to_fused_pto.hpp`
- `deepseek/moe/get_fused_mapping_pto.hpp`
- `deepseek/moe/group_count_aux_fi_pto.hpp`
- `deepseek/moe/inplace_unique_group_indices_pto.hpp`
- `deepseek/moe/mask_indices_by_tp_pto.hpp`
- `deepseek/moe/normalize_weight_pto.hpp`
- `deepseek/moe/reduce_fused_pto.hpp`
- `deepseek/moe/topk_gate_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | VEC | tile-scalar-and-immediate | 26 | Fill the destination Tile by expanding the bound scalar value. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 13 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 10 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TCMP`](TILE_INSTRUCTIONS.md#tcmp) | VEC | elementwise-tile-tile | 8 | Apply elementwise comparison to the two source Tiles. |
| [`TSEL`](TILE_INSTRUCTIONS.md#tsel) | VEC | elementwise-tile-tile | 7 | Select each destination element from the true or false source under the mask Tile. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 5 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TROWSUM`](TILE_INSTRUCTIONS.md#trowsum) | SFU | reduce-and-expand | 5 | Reduce each source row to its sum. |
| [`TINSERT`](TILE_INSTRUCTIONS.md#tinsert) | SFU | layout-and-rearrangement | 3 | Insert the source Tile into the destination region at the encoded row and column offsets. |
| [`TSUB`](TILE_INSTRUCTIONS.md#tsub) | VEC | elementwise-tile-tile | 3 | Apply elementwise subtraction to the two source Tiles. |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | VEC | elementwise-tile-tile | 3 | Apply elementwise addition to the two source Tiles. |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 3 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TEXTRACT`](TILE_INSTRUCTIONS.md#textract) | SFU | layout-and-rearrangement | 2 | Extract a rectangular source region at the encoded row and column offsets. |
| [`TROWMAX`](TILE_INSTRUCTIONS.md#trowmax) | SFU | reduce-and-expand | 2 | Reduce each source row to its maximum. |
| [`TREMS`](TILE_INSTRUCTIONS.md#trems) | VEC | tile-scalar-and-immediate | 2 | Apply elementwise remainder between the source Tile and bound scalar. |
| [`TRECIP`](TILE_INSTRUCTIONS.md#trecip) | SFU | elementwise-tile-tile | 2 | Apply elementwise reciprocal to the source Tile. |
| [`TROWEXPANDMUL`](TILE_INSTRUCTIONS.md#trowexpandmul) | SFU | reduce-and-expand | 2 | Apply multiplication while expanding the bound row vector across the source Tile. |
| [`TADDS`](TILE_INSTRUCTIONS.md#tadds) | VEC | tile-scalar-and-immediate | 2 | Apply elementwise addition between the source Tile and bound scalar. |
| [`TAND`](TILE_INSTRUCTIONS.md#tand) | VEC | elementwise-tile-tile | 2 | Apply elementwise bitwise AND to the two source Tiles. |
| [`TDIVS`](TILE_INSTRUCTIONS.md#tdivs) | VEC | tile-scalar-and-immediate | 2 | Apply elementwise division between the source Tile and bound scalar. |
| [`TCI`](TILE_INSTRUCTIONS.md#tci) | SFU | irregular-and-complex | 1 | Initialize destination elements as an ascending or descending counter sequence. |
| [`TROWEXPAND`](TILE_INSTRUCTIONS.md#trowexpand) | SFU | reduce-and-expand | 1 | Apply broadcast while expanding the bound row vector across the source Tile. |
| [`TROWEXPANDSUB`](TILE_INSTRUCTIONS.md#trowexpandsub) | SFU | reduce-and-expand | 1 | Apply subtraction while expanding the bound row vector across the source Tile. |
| [`TSUBS`](TILE_INSTRUCTIONS.md#tsubs) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise subtraction between the source Tile and bound scalar. |

### <a id="op-deepseek-quant"></a>`deepseek/quant`

DeepSeek per-token cast / cast-back / swiglu-fused cast (3+2 kernels).

- **Files scanned**: 3
- **Distinct tile instructions used**: 14
- **Total tile-instruction call sites**: 60
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `deepseek/quant/cast_back_pto.hpp`
- `deepseek/quant/per_token_cast_pto.hpp`
- `deepseek/quant/swiglu_fused_cast_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | VEC | tile-scalar-and-immediate | 10 | Apply elementwise multiplication between the source Tile and bound scalar. |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | VEC | elementwise-tile-tile | 9 | Convert source elements to the destination data type under rounding and saturation controls. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 8 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 8 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |
| [`TRECIP`](TILE_INSTRUCTIONS.md#trecip) | SFU | elementwise-tile-tile | 4 | Apply elementwise reciprocal to the source Tile. |
| [`TROWMAX`](TILE_INSTRUCTIONS.md#trowmax) | SFU | reduce-and-expand | 4 | Reduce each source row to its maximum. |
| [`TMAX`](TILE_INSTRUCTIONS.md#tmax) | VEC | elementwise-tile-tile | 3 | Apply elementwise maximum selection to the two source Tiles. |
| [`TROWEXPANDMUL`](TILE_INSTRUCTIONS.md#trowexpandmul) | SFU | reduce-and-expand | 3 | Apply multiplication while expanding the bound row vector across the source Tile. |
| [`TMAXS`](TILE_INSTRUCTIONS.md#tmaxs) | VEC | tile-scalar-and-immediate | 3 | Apply elementwise maximum selection between the source Tile and bound scalar. |
| [`TCOLEXPANDMUL`](TILE_INSTRUCTIONS.md#tcolexpandmul) | SFU | reduce-and-expand | 2 | Apply multiplication while expanding the bound col vector across the source Tile. |
| [`TCOLMAX`](TILE_INSTRUCTIONS.md#tcolmax) | SFU | reduce-and-expand | 2 | Reduce each source col to its maximum. |
| [`TMUL`](TILE_INSTRUCTIONS.md#tmul) | VEC | elementwise-tile-tile | 2 | Apply elementwise multiplication to the two source Tiles. |
| [`TEXP`](TILE_INSTRUCTIONS.md#texp) | SFU | elementwise-tile-tile | 1 | Apply elementwise exponential to the source Tile. |
| [`TADDS`](TILE_INSTRUCTIONS.md#tadds) | VEC | tile-scalar-and-immediate | 1 | Apply elementwise addition between the source Tile and bound scalar. |

### <a id="op-deepseek-transpose"></a>`deepseek/transpose`

DeepSeek batched transpose.

- **Files scanned**: 1
- **Distinct tile instructions used**: 3
- **Total tile-instruction call sites**: 4
- **Catalog conformance**: `CONFORMS`

**Kernel files**:

- `deepseek/transpose/batched_transpose_pto.hpp`

**Tile instructions used** (sorted by call count):

| Mnemonic | Engine | Class | Call sites | Summary |
| --- | :---: | --- | ---: | --- |
| [`TTRANS`](TILE_INSTRUCTIONS.md#ttrans) | SFU | layout-and-rearrangement | 2 | Transpose the source Tile into the destination. |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | TLSU | memory-and-data-movement | 1 | Load the valid GM rectangle into a Tile using the encoded base and logical row stride. |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | TLSU | memory-and-data-movement | 1 | Store the valid Tile rectangle to GM using the encoded base and logical row stride. |

---

## Tile instructions not used by any one-level-arch kernel

The following 60 PTO 0.58.0 tile operations have no call site in any scanned kernel header. They are still part of the accepted ISA catalog but are not exercised by the current SuperNPUBench operator set.

| Mnemonic | Engine | Class | Summary |
| --- | :---: | --- | --- |
| [`GMOV`](TILE_INSTRUCTIONS.md#gmov) | TLSU | memory-and-data-movement | Copy the resolved peer-PE Tile fragment selected by the bound peer TID. |
| [`MGATHER_CAS`](TILE_INSTRUCTIONS.md#mgather_cas) | TLSU | memory-and-data-movement | Atomically compare and conditionally replace GM elements at Tile-provided indices. |
| [`MGATHER_MASK`](TILE_INSTRUCTIONS.md#mgather_mask) | TLSU | memory-and-data-movement | Gather masked GM elements at Tile-provided indices into the destination. |
| [`MSCATTER_MASK`](TILE_INSTRUCTIONS.md#mscatter_mask) | TLSU | memory-and-data-movement | Scatter masked source elements to GM addresses selected by Tile indices. |
| [`TABS`](TILE_INSTRUCTIONS.md#tabs) | VEC | elementwise-tile-tile | Apply elementwise absolute value to the source Tile. |
| [`TCMPS`](TILE_INSTRUCTIONS.md#tcmps) | VEC | tile-scalar-and-immediate | Apply elementwise comparison between the source Tile and bound scalar. |
| [`TCOLARGMAX`](TILE_INSTRUCTIONS.md#tcolargmax) | SFU | reduce-and-expand | Reduce each source col to its maximum index. |
| [`TCOLARGMIN`](TILE_INSTRUCTIONS.md#tcolargmin) | SFU | reduce-and-expand | Reduce each source col to its minimum index. |
| [`TCOLEXPANDADD`](TILE_INSTRUCTIONS.md#tcolexpandadd) | SFU | reduce-and-expand | Apply addition while expanding the bound col vector across the source Tile. |
| [`TCOLEXPANDDIV`](TILE_INSTRUCTIONS.md#tcolexpanddiv) | SFU | reduce-and-expand | Apply division while expanding the bound col vector across the source Tile. |
| [`TCOLEXPANDEXPDIF`](TILE_INSTRUCTIONS.md#tcolexpandexpdif) | SFU | reduce-and-expand | Apply exponential difference while expanding the bound col vector across the source Tile. |
| [`TCOLEXPANDMAX`](TILE_INSTRUCTIONS.md#tcolexpandmax) | SFU | reduce-and-expand | Apply maximum selection while expanding the bound col vector across the source Tile. |
| [`TCOLEXPANDMIN`](TILE_INSTRUCTIONS.md#tcolexpandmin) | SFU | reduce-and-expand | Apply minimum selection while expanding the bound col vector across the source Tile. |
| [`TCOLMIN`](TILE_INSTRUCTIONS.md#tcolmin) | SFU | reduce-and-expand | Reduce each source col to its minimum. |
| [`TCONCAT`](TILE_INSTRUCTIONS.md#tconcat) | SFU | layout-and-rearrangement | Concatenate two source Tiles along the selected axis. |
| [`TDEQUANT`](TILE_INSTRUCTIONS.md#tdequant) | SFU | irregular-and-complex | Dequantize source elements using scale, zero point, rounding, and saturation controls. |
| [`TDIV`](TILE_INSTRUCTIONS.md#tdiv) | VEC | elementwise-tile-tile | Apply elementwise division to the two source Tiles. |
| [`TFILLPAD`](TILE_INSTRUCTIONS.md#tfillpad) | SFU | layout-and-rearrangement | Copy the source and fill destination padding elements with the bound scalar. |
| [`TFMA`](TILE_INSTRUCTIONS.md#tfma) | VEC | elementwise-tile-tile | Compute a fused elementwise left-times-right plus addend result. |
| [`TGATHER`](TILE_INSTRUCTIONS.md#tgather) | SFU | irregular-and-complex | Gather source elements by Tile indices into the destination. |
| [`TGEMV`](TILE_INSTRUCTIONS.md#tgemv) | CUBE | matrix-and-matrix-vector | Multiply the matrix by the vector into the destination. |
| [`TGEMV_ACC`](TILE_INSTRUCTIONS.md#tgemv_acc) | CUBE | matrix-and-matrix-vector | Multiply the matrix by the vector and accumulate into the supplied Tile. |
| [`TGEMV_BIAS`](TILE_INSTRUCTIONS.md#tgemv_bias) | CUBE | matrix-and-matrix-vector | Multiply the matrix by the vector and add the bias Tile. |
| [`TGEMV_MX`](TILE_INSTRUCTIONS.md#tgemv_mx) | CUBE | matrix-and-matrix-vector | Multiply the matrix by the vector using row and column scale Tiles. |
| [`TGEMV_MX_ACC`](TILE_INSTRUCTIONS.md#tgemv_mx_acc) | CUBE | matrix-and-matrix-vector | Multiply the scaled matrix and vector and accumulate into the supplied Tile. |
| [`TGEMV_MX_BIAS`](TILE_INSTRUCTIONS.md#tgemv_mx_bias) | CUBE | matrix-and-matrix-vector | Multiply the scaled matrix and vector and add the bias Tile. |
| [`THISTOGRAM`](TILE_INSTRUCTIONS.md#thistogram) | SFU | irregular-and-complex | Accumulate a histogram from source values and selected-byte indices. |
| [`TIMG2COL`](TILE_INSTRUCTIONS.md#timg2col) | SFU | layout-and-rearrangement | Transform an image Tile into kernel-column layout using kernel, stride, padding, and fill operands. |
| [`TLOG`](TILE_INSTRUCTIONS.md#tlog) | SFU | elementwise-tile-tile | Apply elementwise logarithm to the source Tile. |
| [`TMATMUL_BIAS`](TILE_INSTRUCTIONS.md#tmatmul_bias) | CUBE | matrix-and-matrix-vector | Multiply matrices and add the bias Tile into the destination. |
| [`TMATMUL_MX_ACC`](TILE_INSTRUCTIONS.md#tmatmul_mx_acc) | CUBE | matrix-and-matrix-vector | Multiply scaled matrices and accumulate into the supplied accumulator Tile. |
| [`TMATMUL_MX_BIAS`](TILE_INSTRUCTIONS.md#tmatmul_mx_bias) | CUBE | matrix-and-matrix-vector | Multiply scaled matrices and add the bias Tile. |
| [`TMIN`](TILE_INSTRUCTIONS.md#tmin) | VEC | elementwise-tile-tile | Apply elementwise minimum selection to the two source Tiles. |
| [`TMRGSORT`](TILE_INSTRUCTIONS.md#tmrgsort) | SFU | irregular-and-complex | Merge two sorted source Tiles in the selected ascending or descending order. |
| [`TNEG`](TILE_INSTRUCTIONS.md#tneg) | VEC | elementwise-tile-tile | Apply elementwise arithmetic negation to the source Tile. |
| [`TNOT`](TILE_INSTRUCTIONS.md#tnot) | VEC | elementwise-tile-tile | Apply elementwise bitwise complement to the source Tile. |
| [`TORS`](TILE_INSTRUCTIONS.md#tors) | VEC | tile-scalar-and-immediate | Apply elementwise bitwise OR between the source Tile and bound scalar. |
| [`TPARTADD`](TILE_INSTRUCTIONS.md#tpartadd) | SFU | irregular-and-complex | Combine corresponding source partitions by addition. |
| [`TPARTMAX`](TILE_INSTRUCTIONS.md#tpartmax) | SFU | irregular-and-complex | Combine corresponding source partitions by maximum selection. |
| [`TPARTMIN`](TILE_INSTRUCTIONS.md#tpartmin) | SFU | irregular-and-complex | Combine corresponding source partitions by minimum selection. |
| [`TPARTMUL`](TILE_INSTRUCTIONS.md#tpartmul) | SFU | irregular-and-complex | Combine corresponding source partitions by multiplication. |
| [`TPREFETCH`](TILE_INSTRUCTIONS.md#tprefetch) | TLSU | memory-and-data-movement | Prefetch the requested GM byte range without producing a Tile destination. |
| [`TRELU`](TILE_INSTRUCTIONS.md#trelu) | VEC | elementwise-tile-tile | Apply elementwise rectified-linear activation to the source Tile. |
| [`TROWARGMAX`](TILE_INSTRUCTIONS.md#trowargmax) | SFU | reduce-and-expand | Reduce each source row to its maximum index. |
| [`TROWARGMIN`](TILE_INSTRUCTIONS.md#trowargmin) | SFU | reduce-and-expand | Reduce each source row to its minimum index. |
| [`TROWEXPANDADD`](TILE_INSTRUCTIONS.md#trowexpandadd) | SFU | reduce-and-expand | Apply addition while expanding the bound row vector across the source Tile. |
| [`TROWEXPANDDIV`](TILE_INSTRUCTIONS.md#trowexpanddiv) | SFU | reduce-and-expand | Apply division while expanding the bound row vector across the source Tile. |
| [`TROWEXPANDEXPDIF`](TILE_INSTRUCTIONS.md#trowexpandexpdif) | SFU | reduce-and-expand | Apply exponential difference while expanding the bound row vector across the source Tile. |
| [`TROWEXPANDMAX`](TILE_INSTRUCTIONS.md#trowexpandmax) | SFU | reduce-and-expand | Apply maximum selection while expanding the bound row vector across the source Tile. |
| [`TROWEXPANDMIN`](TILE_INSTRUCTIONS.md#trowexpandmin) | SFU | reduce-and-expand | Apply minimum selection while expanding the bound row vector across the source Tile. |
| [`TROWMIN`](TILE_INSTRUCTIONS.md#trowmin) | SFU | reduce-and-expand | Reduce each source row to its minimum. |
| [`TRSQRT`](TILE_INSTRUCTIONS.md#trsqrt) | SFU | elementwise-tile-tile | Apply elementwise reciprocal square root to the source Tile. |
| [`TSCATTER`](TILE_INSTRUCTIONS.md#tscatter) | SFU | irregular-and-complex | Scatter source elements by Tile indices into the destination. |
| [`TSELS`](TILE_INSTRUCTIONS.md#tsels) | VEC | tile-scalar-and-immediate | Select each destination element from the Tile source or scalar alternative under the mask Tile. |
| [`TSHL`](TILE_INSTRUCTIONS.md#tshl) | VEC | elementwise-tile-tile | Apply elementwise left shift to the two source Tiles. |
| [`TSHR`](TILE_INSTRUCTIONS.md#tshr) | VEC | elementwise-tile-tile | Apply elementwise right shift to the two source Tiles. |
| [`TSORT`](TILE_INSTRUCTIONS.md#tsort) | SFU | irregular-and-complex | Sort source groups, returning ordered values and original U32 indices. |
| [`TSQRT`](TILE_INSTRUCTIONS.md#tsqrt) | SFU | elementwise-tile-tile | Apply elementwise square root to the source Tile. |
| [`TTRI`](TILE_INSTRUCTIONS.md#ttri) | SFU | irregular-and-complex | Initialize the selected upper or lower triangular region relative to the diagonal. |
| [`TXORS`](TILE_INSTRUCTIONS.md#txors) | VEC | tile-scalar-and-immediate | Apply elementwise bitwise XOR between the source Tile and bound scalar. |

---

## Appendix: reverse index — files per tile instruction

Only instructions with at least one call site are listed. Counts are call sites per file (after stripping comments and string literals).

| Mnemonic | # files | # call sites | Files (file:count) |
| --- | ---: | ---: | --- |
| [`MGATHER`](TILE_INSTRUCTIONS.md#mgather) | 7 | 44 | `broadcast/broadcast_pto.hpp`:2, `concat/concat_gather_pto.hpp`:2, `control/hashtable_lookup_simd.hpp`:2, `gather/gather_pto.hpp`:4, `matmul/matmul_mx.hpp`:16, `matmul/matmul_mx_pto.hpp`:16, `transpose/transpose_pto.hpp`:2 |
| [`MSCATTER`](TILE_INSTRUCTIONS.md#mscatter) | 1 | 2 | `concat/concat_scatter_pto.hpp`:2 |
| [`TADD`](TILE_INSTRUCTIONS.md#tadd) | 25 | 113 | `broadcast/broadcast_pto.hpp`:1, `concat/concat_gather_pto.hpp`:6, `concat/concat_scatter_pto.hpp`:2, `control/hashtable_lookup_simd.hpp`:1, `deepseek/mhc/norm_fn_pto.hpp`:2, `deepseek/moe/group_count_aux_fi_pto.hpp`:2, `deepseek/moe/reduce_fused_pto.hpp`:1, `element_wise/tadd_multithread.hpp`:1, `fa/fa_2d_unroll_gmma.hpp`:2, `fa/fa_2d_unroll_pto.hpp`:8, `fa/fa_hif4.hpp`:10, `fa/fa_hif4_pto.hpp`:9, `fa/sfa_pto.hpp`:2, `matmul/matmul.hpp`:24, `matmul/matmul_mx.hpp`:1, `matmul/matmul_mx_pto.hpp`:1, `matmul/matmul_pto.hpp`:24, `reduction/cumsum_colvec_pto.hpp`:1, `reduction/cumsum_rowvec_pto.hpp`:1, `reduction/reducesum_colvec_pto.hpp`:4, `reduction/reducesum_colvec_single_tree_pto.hpp`:2, `reduction/reducesum_colvec_unalign_120_8_pto.hpp`:1, `reduction/reducesum_rowvec_pto.hpp`:4, `reduction/reducesum_rowvec_single_tree_pto.hpp`:1, `transpose/transpose_pto.hpp`:2 |
| [`TADDS`](TILE_INSTRUCTIONS.md#tadds) | 7 | 20 | `control/hashtable_lookup_simd.hpp`:2, `deepseek/engram/engram_hash_pto.hpp`:1, `deepseek/mhc/norm_fn_pto.hpp`:2, `deepseek/mhc/sinkhorn_pto.hpp`:5, `deepseek/moe/normalize_weight_pto.hpp`:2, `deepseek/quant/swiglu_fused_cast_pto.hpp`:1, `element_wise/gelu_pto.hpp`:7 |
| [`TAND`](TILE_INSTRUCTIONS.md#tand) | 2 | 4 | `control/hashtable_lookup_simd.hpp`:2, `deepseek/moe/mask_indices_by_tp_pto.hpp`:2 |
| [`TANDS`](TILE_INSTRUCTIONS.md#tands) | 1 | 1 | `sort/topk_pto.hpp`:1 |
| [`TCI`](TILE_INSTRUCTIONS.md#tci) | 5 | 8 | `broadcast/broadcast_pto.hpp`:1, `concat/concat_gather_pto.hpp`:2, `concat/concat_scatter_pto.hpp`:2, `deepseek/moe/topk_gate_pto.hpp`:1, `transpose/transpose_pto.hpp`:2 |
| [`TCMP`](TILE_INSTRUCTIONS.md#tcmp) | 6 | 10 | `control/hashtable_lookup_simd.hpp`:2, `deepseek/moe/get_fused_mapping_pto.hpp`:1, `deepseek/moe/group_count_aux_fi_pto.hpp`:2, `deepseek/moe/inplace_unique_group_indices_pto.hpp`:1, `deepseek/moe/mask_indices_by_tp_pto.hpp`:2, `deepseek/moe/topk_gate_pto.hpp`:2 |
| [`TCOLEXPAND`](TILE_INSTRUCTIONS.md#tcolexpand) | 1 | 1 | `matmul/matmul_mx.hpp`:1 |
| [`TCOLEXPANDMUL`](TILE_INSTRUCTIONS.md#tcolexpandmul) | 7 | 9 | `deepseek/mhc/norm_fn_pto.hpp`:1, `deepseek/mhc/sinkhorn_pto.hpp`:2, `deepseek/quant/cast_back_pto.hpp`:1, `deepseek/quant/per_token_cast_pto.hpp`:1, `fa/fa_hif4_pto.hpp`:2, `fa/sfa_pto.hpp`:1, `matmul/matmul_mx_pto.hpp`:1 |
| [`TCOLEXPANDSUB`](TILE_INSTRUCTIONS.md#tcolexpandsub) | 3 | 5 | `fa/fa_hif4.hpp`:1, `fa/fa_hif4_pto.hpp`:2, `fa/sfa_pto.hpp`:2 |
| [`TCOLMAX`](TILE_INSTRUCTIONS.md#tcolmax) | 6 | 12 | `deepseek/quant/per_token_cast_pto.hpp`:2, `fa/fa_hif4.hpp`:1, `fa/fa_hif4_pto.hpp`:2, `fa/sfa_pto.hpp`:1, `reduction/reducemax_colvec_pto.hpp`:4, `reduction/reducemax_colvec_unalign_120_8_pto.hpp`:2 |
| [`TCOLPROD`](TILE_INSTRUCTIONS.md#tcolprod) | 1 | 4 | `reduction/reduceprod_colvec_pto.hpp`:4 |
| [`TCOLSUM`](TILE_INSTRUCTIONS.md#tcolsum) | 9 | 19 | `deepseek/mhc/expand_to_mhc_bwd_pto.hpp`:1, `deepseek/mhc/sinkhorn_pto.hpp`:2, `fa/fa_hif4.hpp`:1, `fa/fa_hif4_pto.hpp`:2, `fa/sfa_pto.hpp`:1, `reduction/reducesum_colvec_pto.hpp`:4, `reduction/reducesum_colvec_single_tree.hpp`:3, `reduction/reducesum_colvec_single_tree_pto.hpp`:3, `reduction/reducesum_colvec_unalign_120_8_pto.hpp`:2 |
| [`TCVT`](TILE_INSTRUCTIONS.md#tcvt) | 21 | 58 | `concat/concat_gather_pto.hpp`:2, `concat/concat_scatter_pto.hpp`:2, `control/hashtable_lookup_simd.hpp`:11, `deepseek/engram/fused_weight_pto.hpp`:2, `deepseek/mhc/expand_to_mhc_bwd_pto.hpp`:2, `deepseek/moe/group_count_aux_fi_pto.hpp`:1, `deepseek/moe/reduce_fused_pto.hpp`:2, `deepseek/moe/topk_gate_pto.hpp`:2, `deepseek/quant/cast_back_pto.hpp`:2, `deepseek/quant/per_token_cast_pto.hpp`:4, `deepseek/quant/swiglu_fused_cast_pto.hpp`:3, `element_wise/gelu_pto.hpp`:2, `fa/fa_2d_unroll_gmma.hpp`:3, `fa/fa_2d_unroll_pto.hpp`:3, `fa/fa_hif4.hpp`:5, `fa/fa_hif4_pto.hpp`:3, `fa/sfa_pto.hpp`:3, `reduction/reducemax_rowvec_single_tree_pto.hpp`:1, `reduction/reducesum_colvec_single_tree_pto.hpp`:2, `reduction/reducesum_rowvec_single_tree_pto.hpp`:1, `transpose/transpose_pto.hpp`:2 |
| [`TDIVS`](TILE_INSTRUCTIONS.md#tdivs) | 5 | 17 | `broadcast/broadcast_pto.hpp`:1, `concat/concat_gather_pto.hpp`:6, `concat/concat_scatter_pto.hpp`:4, `deepseek/moe/mask_indices_by_tp_pto.hpp`:2, `transpose/transpose_pto.hpp`:4 |
| [`TEXP`](TILE_INSTRUCTIONS.md#texp) | 8 | 18 | `deepseek/mhc/sinkhorn_pto.hpp`:1, `deepseek/quant/swiglu_fused_cast_pto.hpp`:1, `element_wise/gelu_pto.hpp`:1, `fa/fa_2d_unroll_gmma.hpp`:2, `fa/fa_2d_unroll_pto.hpp`:2, `fa/fa_hif4.hpp`:4, `fa/fa_hif4_pto.hpp`:4, `fa/sfa_pto.hpp`:3 |
| [`TEXPANDS`](TILE_INSTRUCTIONS.md#texpands) | 33 | 79 | `broadcast/broadcast_pto.hpp`:1, `concat/concat_gather_pto.hpp`:2, `concat/concat_scatter_pto.hpp`:2, `control/hashtable_lookup_simd.hpp`:6, `deepseek/engram/engram_hash_pto.hpp`:1, `deepseek/moe/expand_to_fused_pto.hpp`:1, `deepseek/moe/get_fused_mapping_pto.hpp`:4, `deepseek/moe/group_count_aux_fi_pto.hpp`:10, `deepseek/moe/inplace_unique_group_indices_pto.hpp`:1, `deepseek/moe/mask_indices_by_tp_pto.hpp`:4, `deepseek/moe/reduce_fused_pto.hpp`:1, `deepseek/moe/topk_gate_pto.hpp`:5, `fa/fa_2d_unroll_gmma.hpp`:2, `fa/fa_2d_unroll_pto.hpp`:2, `fa/fa_hif4.hpp`:3, `fa/fa_hif4_pto.hpp`:2, `fa/sfa_pto.hpp`:3, `matmul/matmul_mx.hpp`:4, `matmul/matmul_mx_pto.hpp`:2, `reduction/cumsum_colvec_pto.hpp`:1, `reduction/cumsum_rowvec_pto.hpp`:1, `reduction/reducemax_colvec_pto.hpp`:2, `reduction/reducemax_colvec_unalign_120_8_pto.hpp`:1, `reduction/reducemax_rowvec_pto.hpp`:2, `reduction/reducemax_rowvec_single_tree_pto.hpp`:1, `reduction/reduceprod_colvec_pto.hpp`:2, `reduction/reduceprod_rowvec_pto.hpp`:2, `reduction/reducesum_colvec_pto.hpp`:2, `reduction/reducesum_colvec_unalign_120_8_pto.hpp`:1, `reduction/reducesum_rowvec_pto.hpp`:2, `reduction/reducesum_rowvec_single_tree_pto.hpp`:1, `sort/topk_pto.hpp`:3, `transpose/transpose_pto.hpp`:2 |
| [`TEXTRACT`](TILE_INSTRUCTIONS.md#textract) | 1 | 2 | `deepseek/moe/inplace_unique_group_indices_pto.hpp`:2 |
| [`TINSERT`](TILE_INSTRUCTIONS.md#tinsert) | 7 | 11 | `broadcast/broadcast_vec_019_pto.hpp`:2, `broadcast/broadcast_vec_039_pto.hpp`:2, `deepseek/moe/group_count_aux_fi_pto.hpp`:2, `deepseek/moe/inplace_unique_group_indices_pto.hpp`:1, `reduction/reducemax_rowvec_single_tree.hpp`:1, `reduction/reducesum_colvec_single_tree.hpp`:2, `reduction/reducesum_rowvec_single_tree.hpp`:1 |
| [`TLOAD`](TILE_INSTRUCTIONS.md#tload) | 56 | 879 | `broadcast/broadcast_vec_019_pto.hpp`:2, `broadcast/broadcast_vec_039_pto.hpp`:2, `broadcast/broadcast_vec_07_pto.hpp`:2, `concat/concat_scatter_pto.hpp`:2, `control/hashtable_lookup_simd.hpp`:1, `deepseek/engram/engram_hash_pto.hpp`:1, `deepseek/engram/fused_weight_pto.hpp`:2, `deepseek/mhc/expand_to_mhc_bwd_pto.hpp`:1, `deepseek/mhc/expand_to_mhc_pto.hpp`:1, `deepseek/mhc/multilayer_recompute_pto.hpp`:4, `deepseek/mhc/norm_fn_pto.hpp`:3, `deepseek/mhc/sinkhorn_pto.hpp`:1, `deepseek/moe/expand_to_fused_pto.hpp`:1, `deepseek/moe/get_fused_mapping_pto.hpp`:1, `deepseek/moe/group_count_aux_fi_pto.hpp`:2, `deepseek/moe/inplace_unique_group_indices_pto.hpp`:1, `deepseek/moe/mask_indices_by_tp_pto.hpp`:1, `deepseek/moe/normalize_weight_pto.hpp`:2, `deepseek/moe/reduce_fused_pto.hpp`:1, `deepseek/moe/topk_gate_pto.hpp`:1, `deepseek/quant/cast_back_pto.hpp`:4, `deepseek/quant/per_token_cast_pto.hpp`:2, `deepseek/quant/swiglu_fused_cast_pto.hpp`:2, `deepseek/transpose/batched_transpose_pto.hpp`:1, `element_wise/gelu_pto.hpp`:2, `element_wise/tadd_multithread.hpp`:2, `fa/fa_2d_unroll_gmma.hpp`:3, `fa/fa_2d_unroll_pto.hpp`:3, `fa/fa_hif4.hpp`:6, `fa/fa_hif4_pto.hpp`:6, `fa/sfa_pto.hpp`:5, `gather/gather_pto.hpp`:4, `matmul/matmul.hpp`:238, `matmul/matmul_mx.hpp`:148, `matmul/matmul_mx_pto.hpp`:136, `matmul/matmul_pto.hpp`:241, `reduction/cumsum_colvec_pto.hpp`:1, `reduction/cumsum_rowvec_pto.hpp`:1, `reduction/reducemax_colvec_pto.hpp`:4, `reduction/reducemax_colvec_unalign_120_8_pto.hpp`:1, `reduction/reducemax_rowvec_pto.hpp`:4, `reduction/reducemax_rowvec_single_tree.hpp`:1, `reduction/reducemax_rowvec_single_tree_pto.hpp`:1, `reduction/reduceprod_colvec_pto.hpp`:4, `reduction/reduceprod_rowvec_pto.hpp`:4, `reduction/reducesum_colvec_pto.hpp`:4, `reduction/reducesum_colvec_single_tree.hpp`:2, `reduction/reducesum_colvec_single_tree_pto.hpp`:2, `reduction/reducesum_colvec_unalign_120_8_pto.hpp`:1, `reduction/reducesum_rowvec_pto.hpp`:4, `reduction/reducesum_rowvec_single_tree.hpp`:1, `reduction/reducesum_rowvec_single_tree_pto.hpp`:1, `sort/topk_pto.hpp`:2, `transpose/transpose_pto.hpp`:4, `transpose/transpose_vector_007_pto.hpp`:1, `transpose/transpose_vector_050_pto.hpp`:1 |
| [`TMATMUL`](TILE_INSTRUCTIONS.md#tmatmul) | 7 | 238 | `deepseek/mhc/multilayer_recompute_pto.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:2, `fa/sfa_pto.hpp`:3, `matmul/matmul.hpp`:112, `matmul/matmul_mx.hpp`:4, `matmul/matmul_mx_pto.hpp`:5, `matmul/matmul_pto.hpp`:111 |
| [`TMATMUL_ACC`](TILE_INSTRUCTIONS.md#tmatmul_acc) | 5 | 237 | `deepseek/mhc/multilayer_recompute_pto.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `matmul/matmul.hpp`:117, `matmul/matmul_mx_pto.hpp`:1, `matmul/matmul_pto.hpp`:117 |
| [`TMATMUL_MX`](TILE_INSTRUCTIONS.md#tmatmul_mx) | 5 | 74 | `fa/fa_hif4.hpp`:3, `fa/fa_hif4_pto.hpp`:3, `matmul/matmul.hpp`:2, `matmul/matmul_mx_pto.hpp`:64, `matmul/matmul_pto.hpp`:2 |
| [`TMAX`](TILE_INSTRUCTIONS.md#tmax) | 11 | 38 | `deepseek/quant/per_token_cast_pto.hpp`:2, `deepseek/quant/swiglu_fused_cast_pto.hpp`:1, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:7, `fa/fa_hif4.hpp`:8, `fa/fa_hif4_pto.hpp`:8, `fa/sfa_pto.hpp`:1, `reduction/reducemax_colvec_pto.hpp`:4, `reduction/reducemax_colvec_unalign_120_8_pto.hpp`:1, `reduction/reducemax_rowvec_pto.hpp`:4, `reduction/reducemax_rowvec_single_tree_pto.hpp`:1 |
| [`TMAXS`](TILE_INSTRUCTIONS.md#tmaxs) | 3 | 4 | `deepseek/quant/per_token_cast_pto.hpp`:2, `deepseek/quant/swiglu_fused_cast_pto.hpp`:1, `element_wise/gelu_pto.hpp`:1 |
| [`TMINS`](TILE_INSTRUCTIONS.md#tmins) | 1 | 1 | `element_wise/gelu_pto.hpp`:1 |
| [`TMOV`](TILE_INSTRUCTIONS.md#tmov) | 1 | 1 | `fa/fa_hif4_pto.hpp`:1 |
| [`TMUL`](TILE_INSTRUCTIONS.md#tmul) | 12 | 31 | `deepseek/engram/fused_weight_pto.hpp`:1, `deepseek/mhc/norm_fn_pto.hpp`:4, `deepseek/quant/swiglu_fused_cast_pto.hpp`:2, `element_wise/gelu_pto.hpp`:8, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `fa/fa_hif4.hpp`:2, `fa/fa_hif4_pto.hpp`:2, `fa/sfa_pto.hpp`:1, `matmul/matmul_mx.hpp`:1, `reduction/reduceprod_colvec_pto.hpp`:4, `reduction/reduceprod_rowvec_pto.hpp`:4 |
| [`TMULS`](TILE_INSTRUCTIONS.md#tmuls) | 18 | 49 | `broadcast/broadcast_pto.hpp`:1, `concat/concat_gather_pto.hpp`:10, `concat/concat_scatter_pto.hpp`:4, `control/hashtable_lookup_simd.hpp`:7, `deepseek/engram/engram_hash_pto.hpp`:1, `deepseek/mhc/norm_fn_pto.hpp`:2, `deepseek/moe/group_count_aux_fi_pto.hpp`:1, `deepseek/moe/mask_indices_by_tp_pto.hpp`:1, `deepseek/moe/reduce_fused_pto.hpp`:1, `deepseek/quant/per_token_cast_pto.hpp`:6, `deepseek/quant/swiglu_fused_cast_pto.hpp`:4, `element_wise/gelu_pto.hpp`:1, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `fa/fa_hif4.hpp`:1, `fa/fa_hif4_pto.hpp`:1, `fa/sfa_pto.hpp`:2, `transpose/transpose_pto.hpp`:4 |
| [`TOR`](TILE_INSTRUCTIONS.md#tor) | 1 | 1 | `control/hashtable_lookup_simd.hpp`:1 |
| [`TQUANT`](TILE_INSTRUCTIONS.md#tquant) | 2 | 2 | `fa/fa_hif4.hpp`:1, `fa/fa_hif4_pto.hpp`:1 |
| [`TRECIP`](TILE_INSTRUCTIONS.md#trecip) | 11 | 17 | `deepseek/mhc/norm_fn_pto.hpp`:1, `deepseek/mhc/sinkhorn_pto.hpp`:4, `deepseek/moe/normalize_weight_pto.hpp`:2, `deepseek/quant/per_token_cast_pto.hpp`:2, `deepseek/quant/swiglu_fused_cast_pto.hpp`:2, `element_wise/gelu_pto.hpp`:1, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `fa/fa_hif4.hpp`:1, `fa/fa_hif4_pto.hpp`:1, `fa/sfa_pto.hpp`:1 |
| [`TREM`](TILE_INSTRUCTIONS.md#trem) | 1 | 3 | `control/hashtable_lookup_simd.hpp`:3 |
| [`TREMS`](TILE_INSTRUCTIONS.md#trems) | 4 | 4 | `broadcast/broadcast_pto.hpp`:1, `deepseek/engram/engram_hash_pto.hpp`:1, `deepseek/moe/mask_indices_by_tp_pto.hpp`:1, `deepseek/moe/topk_gate_pto.hpp`:1 |
| [`TROWEXPAND`](TILE_INSTRUCTIONS.md#trowexpand) | 2 | 3 | `broadcast/broadcast_vec_07_pto.hpp`:2, `deepseek/moe/topk_gate_pto.hpp`:1 |
| [`TROWEXPANDMUL`](TILE_INSTRUCTIONS.md#trowexpandmul) | 9 | 14 | `deepseek/mhc/norm_fn_pto.hpp`:1, `deepseek/mhc/sinkhorn_pto.hpp`:2, `deepseek/moe/normalize_weight_pto.hpp`:2, `deepseek/quant/cast_back_pto.hpp`:1, `deepseek/quant/per_token_cast_pto.hpp`:1, `deepseek/quant/swiglu_fused_cast_pto.hpp`:1, `fa/fa_2d_unroll_gmma.hpp`:2, `fa/fa_2d_unroll_pto.hpp`:2, `fa/fa_hif4.hpp`:2 |
| [`TROWEXPANDSUB`](TILE_INSTRUCTIONS.md#trowexpandsub) | 5 | 5 | `deepseek/mhc/sinkhorn_pto.hpp`:1, `deepseek/moe/topk_gate_pto.hpp`:1, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `fa/fa_hif4.hpp`:1 |
| [`TROWMAX`](TILE_INSTRUCTIONS.md#trowmax) | 10 | 18 | `deepseek/mhc/sinkhorn_pto.hpp`:1, `deepseek/moe/topk_gate_pto.hpp`:2, `deepseek/quant/per_token_cast_pto.hpp`:2, `deepseek/quant/swiglu_fused_cast_pto.hpp`:2, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `fa/fa_hif4.hpp`:1, `reduction/reducemax_rowvec_pto.hpp`:4, `reduction/reducemax_rowvec_single_tree.hpp`:2, `reduction/reducemax_rowvec_single_tree_pto.hpp`:2 |
| [`TROWPROD`](TILE_INSTRUCTIONS.md#trowprod) | 1 | 4 | `reduction/reduceprod_rowvec_pto.hpp`:4 |
| [`TROWSUM`](TILE_INSTRUCTIONS.md#trowsum) | 11 | 19 | `deepseek/mhc/norm_fn_pto.hpp`:1, `deepseek/mhc/sinkhorn_pto.hpp`:2, `deepseek/moe/get_fused_mapping_pto.hpp`:1, `deepseek/moe/group_count_aux_fi_pto.hpp`:2, `deepseek/moe/normalize_weight_pto.hpp`:2, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `fa/fa_hif4.hpp`:1, `reduction/reducesum_rowvec_pto.hpp`:4, `reduction/reducesum_rowvec_single_tree.hpp`:2, `reduction/reducesum_rowvec_single_tree_pto.hpp`:2 |
| [`TSEL`](TILE_INSTRUCTIONS.md#tsel) | 6 | 8 | `control/hashtable_lookup_simd.hpp`:1, `deepseek/moe/get_fused_mapping_pto.hpp`:1, `deepseek/moe/group_count_aux_fi_pto.hpp`:2, `deepseek/moe/inplace_unique_group_indices_pto.hpp`:1, `deepseek/moe/mask_indices_by_tp_pto.hpp`:1, `deepseek/moe/topk_gate_pto.hpp`:2 |
| [`TSHLS`](TILE_INSTRUCTIONS.md#tshls) | 1 | 1 | `control/hashtable_lookup_simd.hpp`:1 |
| [`TSHRS`](TILE_INSTRUCTIONS.md#tshrs) | 2 | 10 | `control/hashtable_lookup_simd.hpp`:8, `sort/topk_pto.hpp`:2 |
| [`TSTORE`](TILE_INSTRUCTIONS.md#tstore) | 56 | 110 | `broadcast/broadcast_pto.hpp`:2, `broadcast/broadcast_vec_019_pto.hpp`:2, `broadcast/broadcast_vec_039_pto.hpp`:2, `broadcast/broadcast_vec_07_pto.hpp`:2, `concat/concat_gather_pto.hpp`:2, `control/hashtable_lookup_simd.hpp`:1, `deepseek/engram/engram_hash_pto.hpp`:1, `deepseek/engram/fused_weight_pto.hpp`:1, `deepseek/mhc/expand_to_mhc_bwd_pto.hpp`:1, `deepseek/mhc/expand_to_mhc_pto.hpp`:1, `deepseek/mhc/multilayer_recompute_pto.hpp`:1, `deepseek/mhc/norm_fn_pto.hpp`:2, `deepseek/mhc/sinkhorn_pto.hpp`:1, `deepseek/moe/expand_to_fused_pto.hpp`:2, `deepseek/moe/get_fused_mapping_pto.hpp`:1, `deepseek/moe/group_count_aux_fi_pto.hpp`:2, `deepseek/moe/inplace_unique_group_indices_pto.hpp`:1, `deepseek/moe/mask_indices_by_tp_pto.hpp`:1, `deepseek/moe/normalize_weight_pto.hpp`:4, `deepseek/moe/reduce_fused_pto.hpp`:1, `deepseek/moe/topk_gate_pto.hpp`:1, `deepseek/quant/cast_back_pto.hpp`:2, `deepseek/quant/per_token_cast_pto.hpp`:4, `deepseek/quant/swiglu_fused_cast_pto.hpp`:2, `deepseek/transpose/batched_transpose_pto.hpp`:1, `element_wise/gelu_pto.hpp`:2, `element_wise/tadd_multithread.hpp`:1, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `fa/fa_hif4.hpp`:1, `fa/fa_hif4_pto.hpp`:1, `fa/sfa_pto.hpp`:1, `gather/gather_pto.hpp`:4, `matmul/matmul.hpp`:10, `matmul/matmul_mx.hpp`:2, `matmul/matmul_mx_pto.hpp`:3, `matmul/matmul_pto.hpp`:12, `reduction/cumsum_colvec_pto.hpp`:1, `reduction/cumsum_rowvec_pto.hpp`:1, `reduction/reducemax_colvec_pto.hpp`:2, `reduction/reducemax_colvec_unalign_120_8_pto.hpp`:1, `reduction/reducemax_rowvec_pto.hpp`:2, `reduction/reducemax_rowvec_single_tree.hpp`:1, `reduction/reducemax_rowvec_single_tree_pto.hpp`:1, `reduction/reduceprod_colvec_pto.hpp`:2, `reduction/reduceprod_rowvec_pto.hpp`:2, `reduction/reducesum_colvec_pto.hpp`:2, `reduction/reducesum_colvec_single_tree.hpp`:1, `reduction/reducesum_colvec_single_tree_pto.hpp`:1, `reduction/reducesum_colvec_unalign_120_8_pto.hpp`:1, `reduction/reducesum_rowvec_pto.hpp`:2, `reduction/reducesum_rowvec_single_tree.hpp`:1, `reduction/reducesum_rowvec_single_tree_pto.hpp`:1, `transpose/transpose_pto.hpp`:6, `transpose/transpose_vector_007_pto.hpp`:1, `transpose/transpose_vector_050_pto.hpp`:1 |
| [`TSUB`](TILE_INSTRUCTIONS.md#tsub) | 10 | 18 | `concat/concat_gather_pto.hpp`:4, `concat/concat_scatter_pto.hpp`:2, `deepseek/moe/mask_indices_by_tp_pto.hpp`:1, `deepseek/moe/topk_gate_pto.hpp`:2, `fa/fa_2d_unroll_gmma.hpp`:1, `fa/fa_2d_unroll_pto.hpp`:1, `fa/fa_hif4.hpp`:2, `fa/fa_hif4_pto.hpp`:2, `fa/sfa_pto.hpp`:1, `transpose/transpose_pto.hpp`:2 |
| [`TSUBS`](TILE_INSTRUCTIONS.md#tsubs) | 1 | 1 | `deepseek/moe/mask_indices_by_tp_pto.hpp`:1 |
| [`TTRANS`](TILE_INSTRUCTIONS.md#ttrans) | 3 | 11 | `deepseek/transpose/batched_transpose_pto.hpp`:2, `transpose/transpose_pto.hpp`:8, `transpose/transpose_vector_007_pto.hpp`:1 |
| [`TXOR`](TILE_INSTRUCTIONS.md#txor) | 2 | 6 | `control/hashtable_lookup_simd.hpp`:5, `deepseek/engram/engram_hash_pto.hpp`:1 |

## Methodology

- **Source of truth for tile instructions**: `pto-spec` repo, `asl/tile/**/*.asl` files, parsed from each file's `// PTO-INSTRUCTION: {JSON}` header (109 entries).
- **Kernel scan root**: `SuperNPUBench/benchmark/one-level-arch/kernels/` (68 files: `.hpp`/`.h`/`.cpp`).
- **Pre-processing**: each kernel file is stripped of C++ block comments (`/* */`), line comments (`//`), and string/char literals before matching, to avoid false positives from prose or doc-strings.
- **Match rule**: a tile mnemonic `XYZ` is counted when it appears as a standalone identifier (bounded by non-identifier characters), e.g. `TADD(...)` counts; `TADDS` does not count as `TADD`, and `TADDSC` does not count as `TADD` or `TADDS`.
- **Conformance verdict**: an operator `CONFORMS` when every distinct tile mnemonic it invokes belongs to the accepted 109-entry PTO 0.58.0 catalog.
- **Known limitation**: the scan covers textual call sites. Macro expansion, templated helpers in `common/pto_tileop.hpp` wrappers, and inline helpers may wrap a single mnemonic many times; counts reflect textual references, not dynamic execution frequency.
