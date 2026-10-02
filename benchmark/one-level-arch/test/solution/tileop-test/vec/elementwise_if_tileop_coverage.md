# PTO 0.59 element-wise-if coverage

This matrix records the boundary exercised by the element-wise-if examples.
The positive path is the compiler-generated `TEXPANDS` → `TCMP` → masked
operation(s) → `TSEL` sequence.  A negative result means the compiler rejects
the shape explicitly; it must not silently emit an unmasked tile operation.

| TileOP in an if arm | LLVM element-wise source | Mask-tile/predicate lowering | gfrun/gfsim |
| --- | --- | --- | --- |
| TADD | supported | `ew.tadd.masked`, predicate + `PredInv` | pass (predicate matrix) |
| TSUB | supported | `ew.tsub.masked`, predicate + `PredInv` | pass (predicate matrix) |
| TMUL | rejected by frontend | no masked intrinsic | not runnable |
| TDIV | rejected by frontend | no masked intrinsic | not runnable |
| TREM | rejected by frontend | no masked intrinsic | not runnable |
| TMAX/TMIN | rejected by frontend | no masked intrinsic | not runnable |
| TAND/TOR/TXOR | rejected by frontend | no masked intrinsic | not runnable |
| TSHL/TSHR | rejected by frontend | no masked intrinsic | not runnable |

Predicate coverage for the supported TADD/TSUB pair is now `GT`, `LT`, `GE`,
`LE`, `EQ`, `NE`, and `!(GT)`.  The LLVM test checks the TCMP mode values
`3,2,5,4,0,1,3` and verifies that `!(GT)` swaps the add/sub `PredInv` values.
The runnable benchmark seeds positive, negative, and zero lanes before calling
all seven functions.

The negative tile-op cases live in
`llvm-project/clang/test/LinxV5/elementwise-if-unsupported-ops.cpp`.  They are
intentional coverage probes for the remaining compiler work; the model-side
PTO 0.59 staged validator already recognizes these operation families, but
that does not imply that LLVM can currently produce their masked bundles.
