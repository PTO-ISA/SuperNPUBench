# Issue: gfrun cooperative TMATMUL MX ScaleB runtime check mismatches ISA spec for non-transposed Shared B

## Summary

When running a 4-PE cooperative `TMATMUL_MX` with `__fp4_e2m1x2` inputs and
Shared `ScaleB` tiles (`TransB=0`), gfrun's `ExecuteOrSuspendCollective`
runtime assertion fails with:

```
ASSERTION FAILED: (!rightNeedsScale || scaleInfoLegal(
    rightScaleShared->tileInfo, rightScaleBlocks, n,
    MatrixScaleCarrierType(rightType))) &&
    "cooperative TMATMUL MX scale descriptor is illegal"
```

The emulator's `scaleInfoLegal` call passes `(rightScaleBlocks, n)` as
`(rows, cols)`, checking `validRow == KBlocksB && validCol == N`.

But both the ISA header (`template_asm.hpp:4297`) and the emulator's own
decoder (`Block.cpp:524-525`) expect the opposite for `TransB=0`:
`validRow == N && validCol == KBlocksB`.

The result is that any correctly-declared Shared non-transposed B scale tile
is rejected at runtime, making cooperative MX matmul with Shared B+ScaleB
unusable on gfrun.

## Environment

- **gfrun build**: `python3 build.py all -j8` from `SuperScalarModel` main
- **Toolchain**: `linx-toolchain-build` output `linx_blockisa_llvm_musl`
  (clang 15.0.4, TileOP-API headers v0.58.3 reinstalled from source)
- **Kernel**: `quant_batch_matmul_mxfp4_mt.hpp` (4-PE cooperative MX fp4 matmul)
- **Test shape**: M=64, N=32, K=64, tM=64, tN=32, tK=64, B=1

## Reproduction

```bash
# Compile the res_check ELF
cd SuperNPUBench/benchmark/one-level-arch/test/solution/quant_batch_matmul_test
make TESTCASE=quant_batch_matmul_test_mxfp4_mt \
    M=64 N=32 K=64 tM=64 tN=32 tK=64 B=1 \
    res_check=on PLAT=linx \
    COMPILER_DIR=<toolchain>/bin

# Run on gfrun
bin/gfrun -s softcore.multiThreadNum=4 -f <elf>
```

### Expected behavior

gfrun completes execution and writes `res.bin` for precision comparison.

### Actual behavior

gfrun aborts with an illegal-instruction assertion at
`SoftCore.cpp:2402-2405`:

```
gfrun: illegal instruction: ASSERTION FAILED:
  (!leftNeedsScale || ...) &&
  (!rightNeedsScale || scaleInfoLegal(
      rightScaleShared->tileInfo, rightScaleBlocks, n,
      MatrixScaleCarrierType(rightType))) &&
  "cooperative TMATMUL MX scale descriptor is illegal"
, func ExecuteOrSuspendCollective, file SoftCore.cpp:2397
```

## Root cause analysis

Three locations validate the MX ScaleB shape; two agree, one does not:

### 1. ISA header (compile-time, `template_asm.hpp:4294-4300`) — CORRECT

```cpp
// Shared non-transposed: stored [N, KBlocks]
if constexpr (is_shared_tile_v<ScaleB> && !Attr.TransB) {
    static_assert(ScaleB::ValidRow == N && ScaleB::ValidCol == KBlocksB,
                  "Shared non-transposed MX ScaleB is declared as its "
                  "physical [N, ceil(K/groupB)] shape");
}
```

Expects: `ValidRow == N, ValidCol == KBlocksB` → physical `[N, K]`.

### 2. Block.cpp decoder (`Block.cpp:518-526`) — CORRECT

```cpp
if (rightNeedsScale) {
    const auto& scale = block.srcTile[rightScaleIndex];
    ASSERT(scale && scale->tileInfo &&
           ...
           scale->tileInfo->validRow == n &&
           scale->tileInfo->validCol == rightScaleBlocks &&
           "TMATMUL_MX right scale has an illegal carrier or shape");
}
```

Expects: `validRow == n (N), validCol == rightScaleBlocks (KBlocksB)`.
Matches the ISA header for `TransB=0`.

### 3. SoftCore.cpp `ExecuteOrSuspendCollective` (`SoftCore.cpp:2402-2404`) — BUG

```cpp
// SoftCore.cpp:2391-2395
const auto scaleInfoLegal = [](const TileInfoPtr& info, size_t rows,
                               size_t cols, DataType carrier) {
    return info && info->dataType == carrier &&
           info->layout == TileLayout::ROW_MAJOR &&
           info->validRow == rows &&     // <-- checks validRow
           info->validCol == cols && ...  // <-- checks validCol
};

// SoftCore.cpp:2402-2404  (the buggy call)
(!rightNeedsScale || scaleInfoLegal(
    rightScaleShared->tileInfo, rightScaleBlocks, n,   // <-- swapped!
    MatrixScaleCarrierType(rightType)))
```

Passes `(rows=rightScaleBlocks, cols=n)`, checking:
`validRow == KBlocksB && validCol == N`.

This is the **transpose** of what the ISA header and Block.cpp expect for
`TransB=0`. The emulator does not branch on `TransB` here — it always uses
the `(KBlocksB, N)` ordering, which is only correct for `TransB=1`.

### Contrast: left scale (correct in all three locations)

```cpp
// SoftCore.cpp:2398-2400
scaleInfoLegal(leftScaleShared->tileInfo,
               expectedLeftRows,       // rows = M
               leftScaleBlocks,        // cols = KBlocksA
               MatrixScaleCarrierType(leftType))
```

Left scale is checked as `(M, KBlocksA)` → `validRow == M, validCol == KBlocksA`.
This matches ISA header (`template_asm.hpp:4281`) and Block.cpp:514-515.
The left-scale call is correct; only the right-scale call is wrong.

## Proposed fix

In `SoftCore.cpp` around line 2402, branch on `block->transB` (or
`rightPhysicalRows/rightPhysicalCols` already computed at line 2332-2333)
to pass the correct `(rows, cols)` order:

```cpp
// Option A: use the already-computed rightPhysicalRows/Cols
if (!rightNeedsScale ||
    (block->transB
        ? scaleInfoLegal(rightScaleShared->tileInfo,
                         rightScaleBlocks, n,
                         MatrixScaleCarrierType(rightType))
        : scaleInfoLegal(rightScaleShared->tileInfo,
                         n, rightScaleBlocks,
                         MatrixScaleCarrierType(rightType))))
```

Or equivalently, derive from `rightPhysicalRows` / `rightPhysicalCols`
which already account for `transB`:

```cpp
// rightPhysicalRows = transB ? n : k    (line 2332)
// rightPhysicalCols = transB ? k : n    (line 2333)
// For ScaleB: rows/cols follow B's primary, with K replaced by KBlocksB
const size_t scaleBRows = block->transB ? rightScaleBlocks : n;
const size_t scaleBCols = block->transB ? n : rightScaleBlocks;
(!rightNeedsScale || scaleInfoLegal(
    rightScaleShared->tileInfo, scaleBRows, scaleBCols,
    MatrixScaleCarrierType(rightType)))
```

## Impact

- All 4-PE cooperative `TMATMUL_MX` / `TMATMUL_MX_ACC` kernels using Shared B
  tiles with MX scales (`__fp4_e2m1x2`, `__fp8_e4m3`, etc.) fail at runtime
  on gfrun.
- Kernels using `TransB=1` (B stored as `[K, N]`) are unaffected because the
  emulator's hardcoded order happens to match for that case.
- The ones-input precision test passes only because all-zero scale blocks
  produce a degenerate tile info that coincidentally satisfies the wrong
  check; random inputs trigger the failure.

## Files involved

| File | Role | Status |
|------|------|--------|
| `tileop-api/jcore/template_asm.hpp:4294-4300` | ISA compile-time spec | Correct |
| `SuperScalarModel/isa/Block.cpp:518-526` | Decode-time validation | Correct |
| `SuperScalarModel/emulator/SoftCore.cpp:2402-2404` | Runtime validation | **BUG** |

## Additional context

During this investigation the following kernel-level fixes were also made
to `quant_batch_matmul_mxfp4_mt.hpp` (separate from this emulator bug):

1. `gM`/`gN`/`gKv` changed from runtime function parameters to template
   parameters (required by `RowMajor<gM, gN>` static layout).
2. B tile declared as `SharedMatrixRight<dtypeB, tN, tKv>` (physical `[N, K]`
   per PTO spec #257, `TransB=0`).
3. Scale offsets computed manually instead of via `global_iterator` (the
   latter mis-computes offsets for padded scale tiles).

These kernel fixes are correct per the ISA spec; the remaining failure is
solely the emulator bug described above.
