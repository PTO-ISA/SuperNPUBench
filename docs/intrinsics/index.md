# C++ tile intrinsic reference

This reference contains the 111 operations in the authoritative API
workbook and the two execution-identity functions used by kernels. The
workbook defines the programming interface; compiler implementation status
does not determine whether an operation belongs in this reference.

Read [operand and tile conventions](conventions.md) before relying on an
individual operation's constraints.

## Execution identity

| Function | Purpose |
| --- | --- |
| [`get_thread_idx()`](get_thread_idx.md) | Select the current thread's fragment or control path. |
| [`get_block_idx()`](get_block_idx.md) | Select the block's global-tensor partition. |

## Element-wise binary operations

### Arithmetic

| Intrinsic | Operation |
| --- | --- |
| [`TADD`](tadd.md) | Elementwise add of two tiles. |
| [`TSUB`](tsub.md) | Elementwise subtract of two tiles. |
| [`TMUL`](tmul.md) | Elementwise multiply of two tiles. |
| [`TMAX`](tmax.md) | Elementwise maximum of two tiles. |
| [`TMIN`](tmin.md) | Elementwise minimum of two tiles. |

### Bitwise and comparison

| Intrinsic | Operation |
| --- | --- |
| [`TAND`](tand.md) | Elementwise bitwise AND of two tiles. |
| [`TOR`](tor.md) | Elementwise bitwise OR of two tiles. |
| [`TXOR`](txor.md) | Elementwise bitwise XOR of two tiles. |
| [`TSHL`](tshl.md) | Elementwise shift-left of two tiles. |
| [`TSHR`](tshr.md) | Elementwise shift-right of two tiles. |
| [`TCMP`](tcmp.md) | Compare two tiles and write a packed predicate mask. |
| [`TSEL`](tsel.md) | Select between two tiles using a mask tile (per-element selection). |

## Element-wise unary operations

### Logical and activation operations

| Intrinsic | Operation |
| --- | --- |
| [`TABS`](tabs.md) | Elementwise absolute value of a tile. |
| [`TNOT`](tnot.md) | Elementwise bitwise NOT of a tile. |
| [`TNEG`](tneg.md) | Elementwise negation of a tile. |
| [`TRELU`](trelu.md) | Elementwise ReLU of a tile. |

## Mathematical functions

### Mathematical functions

| Intrinsic | Operation |
| --- | --- |
| [`TDIV`](tdiv.md) | Elementwise division of two tiles. |
| [`TREM`](trem.md) | Elementwise remainder of two tiles. |
| [`TSQRT`](tsqrt.md) | Elementwise square root. |
| [`TLOG`](tlog.md) | Elementwise natural logarithm of a tile. |
| [`TRECIP`](trecip.md) | Elementwise reciprocal of a tile. |
| [`TEXP`](texp.md) | Elementwise exponential. |
| [`TRSQRT`](trsqrt.md) | Elementwise reciprocal square root. |

## Tile-scalar operations

### Arithmetic

| Intrinsic | Operation |
| --- | --- |
| [`TADDS`](tadds.md) | Elementwise add a scalar to a tile. |
| [`TAXPY`](taxpy.md) | AXPY-style fused update: multiply a tile by a scalar and accumulate into the destination tile. |
| [`TSUBS`](tsubs.md) | Elementwise subtract a scalar from a tile. |
| [`TMULS`](tmuls.md) | Elementwise multiply a tile by a scalar. |
| [`TDIVS`](tdivs.md) | Elementwise division with a scalar (tile/scalar or scalar/tile). |
| [`TMINS`](tmins.md) | Elementwise minimum of a tile and a scalar. |
| [`TMAXS`](tmaxs.md) | Elementwise max of a tile and a scalar: `max(src, scalar)`. |
| [`TREMS`](trems.md) | Elementwise remainder with a scalar: `remainder(src, scalar)`. |

### Bitwise, comparison, and shifts

| Intrinsic | Operation |
| --- | --- |
| [`TANDS`](tands.md) | Elementwise bitwise AND of a tile and a scalar. |
| [`TORS`](tors.md) | Elementwise bitwise OR of a tile and a scalar. |
| [`TXORS`](txors.md) | Elementwise bitwise XOR of a tile and a scalar. |
| [`TCMPS`](tcmps.md) | Compare a tile against a scalar and write per-element comparison results. |
| [`TSELS`](tsels.md) | Select between source tile and scalar using a mask tile (per-element selection for source tile). |
| [`TSHLS`](tshls.md) | Elementwise shift-left a tile by a scalar. |
| [`TSHRS`](tshrs.md) | Elementwise shift-right a tile by a scalar. |

## Reductions

### Reduce rows to a column

| Intrinsic | Operation |
| --- | --- |
| [`TROWSUM`](trowsum.md) | Reduce each row by summing across columns. |
| [`TROWPROD`](trowprod.md) | Reduce each row by multiplying across columns. |
| [`TROWMAX`](trowmax.md) | Reduce each row by taking the maximum across columns. |
| [`TROWMIN`](trowmin.md) | Reduce each row by taking the minimum across columns. |
| [`TROWARGMAX`](trowargmax.md) | Get the column index of the maximum element for each row. |
| [`TROWARGMIN`](trowargmin.md) | Get the column index of the minimum element for each row. |

### Reduce columns to a row

| Intrinsic | Operation |
| --- | --- |
| [`TCOLSUM`](tcolsum.md) | Reduce each column by summing across rows. |
| [`TCOLPROD`](tcolprod.md) | Reduce each column by multiplying across rows. |
| [`TCOLMAX`](tcolmax.md) | Reduce each column by taking the maximum across rows. |
| [`TCOLMIN`](tcolmin.md) | Reduce each column by taking the minimum across rows. |
| [`TCOLARGMAX`](tcolargmax.md) | Get the row index of the maximum element for each column. |
| [`TCOLARGMIN`](tcolargmin.md) | Get the row index of the minimum element for each column. |

## Broadcast operations

### Row-wise broadcast

| Intrinsic | Operation |
| --- | --- |
| [`TROWEXPAND`](trowexpand.md) | Broadcast the first element of each source row across the destination row. |
| [`TROWEXPANDADD`](trowexpandadd.md) | Row-wise broadcast add: add a per-row scalar vector. |
| [`TROWEXPANDSUB`](trowexpandsub.md) | Row-wise broadcast subtract: subtract a per-row scalar vector `src1` from each row of `src0`. |
| [`TROWEXPANDMUL`](trowexpandmul.md) | Row-wise broadcast multiply: multiply each row of `src0` by a per-row scalar vector `src1`. |
| [`TROWEXPANDDIV`](trowexpanddiv.md) | Row-wise broadcast divide: divide each row of `src0` by a per-row scalar vector `src1`. |
| [`TROWEXPANDMAX`](trowexpandmax.md) | Row-wise broadcast max with a per-row scalar vector. |
| [`TROWEXPANDMIN`](trowexpandmin.md) | Row-wise broadcast min with a per-row scalar vector. |
| [`TROWEXPANDEXPDIF`](trowexpandexpdif.md) | Row-wise exp-diff: compute exp(src0 - src1) with per-row scalars. |

### Column-wise broadcast

| Intrinsic | Operation |
| --- | --- |
| [`TCOLEXPAND`](tcolexpand.md) | Broadcast the first element of each source column across the destination column. |
| [`TCOLEXPANDADD`](tcolexpandadd.md) | Column-wise broadcast add with per-column scalar vector. |
| [`TCOLEXPANDSUB`](tcolexpandsub.md) | Column-wise broadcast subtract: subtract a per-column scalar vector from each column. |
| [`TCOLEXPANDMUL`](tcolexpandmul.md) | Column-wise broadcast multiply: multiply each column by a per-column scalar vector. |
| [`TCOLEXPANDDIV`](tcolexpanddiv.md) | Column-wise broadcast divide: divide each column by a per-column scalar vector. |
| [`TCOLEXPANDMAX`](tcolexpandmax.md) | Column-wise broadcast max with per-column scalar vector. |
| [`TCOLEXPANDMIN`](tcolexpandmin.md) | Column-wise broadcast min with per-column scalar vector. |
| [`TCOLEXPANDEXPDIF`](tcolexpandexpdif.md) | Column-wise exp-diff: compute exp(src0 - src1) with per-column scalars. |

## Matrix operations

### Matrix-matrix operations

| Intrinsic | Operation |
| --- | --- |
| [`TMATMUL`](tmatmul.md) | Matrix multiply (GEMM) producing an accumulator/output tile. |
| [`TMATMUL_BIAS`](tmatmul_bias.md) | Matrix multiply with bias add. |
| [`TMATMUL_ACC`](tmatmul_acc.md) | Matrix multiply with accumulator input (fused accumulate). |
| [`TMATMUL_MX`](tmatmul_mx.md) | Matrix multiply (GEMM) with additional scaling tiles for mixed-precision / quantized matmul on supported targets. |

### Matrix-vector operations

| Intrinsic | Operation |
| --- | --- |
| [`TGEMV`](tgemv.md) | General Matrix-Vector multiplication producing an accumulator/output tile. |
| [`TGEMV_BIAS`](tgemv_bias.md) | GEMV with bias add. |
| [`TGEMV_ACC`](tgemv_acc.md) | GEMV with explicit accumulator input/output tiles. |
| [`TGEMV_MX`](tgemv_mx.md) | GEMV with additional scaling tiles for mixed-precision / quantized matrix-vector compute. |

## Memory access

### Regular memory access

| Intrinsic | Operation |
| --- | --- |
| [`TLOAD`](tload.md) | Load data from a GlobalTensor (GM) into a Tile. |
| [`TSTORE`](tstore.md) | Store data from a Tile into a GlobalTensor (GM), optionally using atomic write or quantization parameters. |
| [`TPREFETCH`](tprefetch.md) | Prefetch data from global memory into a tile-local cache/buffer (hint). |

### Indexed memory access

| Intrinsic | Operation |
| --- | --- |
| [`MGATHER`](mgather.md) | Gather-load elements from global memory into a tile using per-element indices. |
| [`MSCATTER`](mscatter.md) | Scatter-store elements from a tile into global memory using per-element indices. |

## Transform and utility operations

### Initialization

| Intrinsic | Operation |
| --- | --- |
| [`TEXPANDS`](texpands.md) | Broadcast a scalar into a destination tile. |
| [`TCI`](tci.md) | Generate a contiguous integer sequence into a destination tile. |
| [`TTRI`](ttri.md) | Generate a triangular (lower/upper) mask tile. |
| [`TFILLPAD`](tfillpad.md) | Copy+pad a tile outside the valid region with a compile-time pad value. |

### Data conversion

| Intrinsic | Operation |
| --- | --- |
| [`TCVT`](tcvt.md) | Elementwise type conversion with a specified rounding mode. |
| [`TQUANT`](tquant.md) | Quantize a tile (e.g. FP32 to FP8) producing exponent/scaling/max outputs. |
| [`TDEQUANT`](tdequant.md) | Dequantize an integer tile into a floating-point tile using scale and offset tiles. |

### Layout transformation

| Intrinsic | Operation |
| --- | --- |
| [`TEXTRACT`](textract.md) | Extract a sub-tile from a source tile. |
| [`TINSERT`](tinsert.md) | Insert a sub-tile into a destination tile at an (indexRow, indexCol) offset. |
| [`TGATHER`](tgather.md) | Gather/select elements using either an index tile or a compile-time mask pattern. |
| [`TSCATTER`](tscatter.md) | Scatter rows of a source tile into a destination tile using per-element row indices. |
| [`TCONCAT`](tconcat.md) | Concatenate two source tiles along the column dimension into a destination tile. |
| [`TTRANS`](ttrans.md) | Transpose with an implementation-defined temporary tile. |
| [`TIMG2COL`](timg2col.md) | Image-to-column transform for convolution-like workloads. |
| [`TMOV`](tmov.md) | Move/copy between tiles, optionally applying implementation-defined conversion modes. |
| [`TGATHERB`](tgatherb.md) | Gather elements using byte offsets. |
| [`TDEINTERLEAVE`](tdeinterleave.md) | De-interleave one or two source tiles into an even-position destination and an odd-position destination. This is the inverse of TINTERLEAVE. |
| [`TINTERLEAVE`](tinterleave.md) | Interleave even and odd source sequences into two destination tiles. This is the inverse of TDEINTERLEAVE. |
| [`TRESHAPE`](treshape.md) | Reinterpret a tile as another tile type/shape while preserving the underlying bytes. |

### Sorting and histogram

| Intrinsic | Operation |
| --- | --- |
| [`TSORT`](tsort.md) | Sort each 32-element block of src together with the corresponding indices from idx, and write the sorted value-index pairs into dst. |
| [`TMRGSORT`](tmrgsort.md) | Merge sort for multiple sorted lists (implementation-defined element format and layout). |
| [`THISTOGRAM`](thistogram.md) | Accumulate histogram bin counts from source values using an index tile. |

### Partial-tile operations

| Intrinsic | Operation |
| --- | --- |
| [`TPARTADD`](tpartadd.md) | Partial elementwise add with implementation-defined handling of mismatched valid regions. |
| [`TPARTMUL`](tpartmul.md) | Partial elementwise multiply with implementation-defined handling of mismatched valid regions. |
| [`TPARTMAX`](tpartmax.md) | Partial elementwise max with implementation-defined handling of mismatched valid regions. |
| [`TPARTMIN`](tpartmin.md) | Partial elementwise min with implementation-defined handling of mismatched valid regions. |
| [`TPARTARGMAX`](tpartargmax.md) | Select the maximum values and their indices across two partial tile inputs. |
| [`TPARTARGMIN`](tpartargmin.md) | Select the minimum values and their indices across two partial tile inputs. |

## Coordination and resources

### Coordination and resources

| Intrinsic | Operation |
| --- | --- |
| [`TPUSH`](tpush.md) | Push a tile into a pipe or FIFO producer endpoint. |
| [`TPOP`](tpop.md) | Pop a tile from a pipe or FIFO consumer endpoint. |
| [`TALLOC`](talloc.md) | Associate a pipeline object with a global-tensor resource for the lifetime managed by the corresponding TFREE operation. |
| [`TFREE`](tfree.md) | Release the currently held pipe or FIFO slot back to the producer. |
