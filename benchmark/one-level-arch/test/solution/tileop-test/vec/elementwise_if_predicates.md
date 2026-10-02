# Element-wise `if` predicate matrix

`src/elementwise_if_predicates.cpp` runs the same masked `TADD`/`TSUB` arms
under `GT`, `LT`, `GE`, `LE`, `EQ`, `NE`, and `!(GT)` conditions.  Its input
contains positive, negative, and zero lanes, so both predicate arms are live
in the functional model.

Build and disassemble it with:

```sh
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
make TESTCASE=elementwise_if_predicates diss
```

The disassembly should contain seven `TCMP` operations with compare modes
`GT`, `LT`, `GE`, `LE`, `EQ`, `NE`, and `GT` respectively.  The final `GT`
case must swap the `PredInv` values of the masked add/sub arms.  This test
does not claim that other tile operations are compiler-supported in an
element-wise branch; the LLVM coverage test records those current gaps.
