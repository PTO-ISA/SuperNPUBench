# FP32 Tile and element `if` kernel

`src/elementwise_if_for.cpp` implements one complete kernel using the official
`ElementTile<float, 128>` API. It loads both input Tiles, visits four logical
32-element parts, copies each part with raw `TMOV`, executes an ordinary
`#pragma pto element for` loop with `a > 0 ? a + b : a - b`, then stores that
part. The source exposes no physical layout or lane and uses no numeric
conversion to transport a value.

The original case retains three `alignas(32)` 512-byte buffers, zero
initialization, 128 FP32 elements, and natural `main` completion. The separate
`elementwise_if_for_strong` case uses finite exact dyadic values, both branches,
cancellation, all four signed-zero input pairs, and 64-byte output guards.
Independent raw-bit goldens check both inputs and the complete output allocation;
`-0 - +0` must remain `-0`. This profile makes no source-level floating exception
ordering or `fenv` claim.

Both cases delegate to the existing Tile/element suite build. Supply its fresh
compiler, musl sysroot, resource directory, and official API include directory,
then build from this directory with:

```sh
make TESTCASE=elementwise_if_for diss
make TESTCASE=elementwise_if_for_strong diss
```

The existing suite runner selects either case using `--case`. Its IR/object
canaries require masked native FP32 comparison and the actual arithmetic graph:
`TNEG` → `TSEL` → `TADD` after standard LLVM folds the diamond, or masked
`TADD`/`TSUB` joined by `TSEL`. Both require raw part Tile transport and value merge, with no residual element marker, scalar extraction
loop, or extra floating conversion. FP32 functional acceptance requires fresh compiler/API builds and exact gfrun
memory comparison, selected with `MODELS=gfrun`. FP32 gfsim execution, physical
definedness, and raw-part transport are a separate timing PR; a functional PASS
does not establish timing parity.
