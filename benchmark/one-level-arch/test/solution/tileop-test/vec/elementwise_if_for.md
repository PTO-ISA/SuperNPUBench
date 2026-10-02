# Element-wise `if` smoke test

`src/elementwise_if_for.cpp` is the for-loop-with-`if` example from
LinxISA/llvm-project issues #110/#114.  Build it from this directory with:

```sh
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
make TESTCASE=elementwise_if_for diss
```

The expected lowering is `TLOAD` → `TEXPANDS` → `TCMP` → masked `TADD` and
`TSUB` → `TSEL` → `TSTORE`.  For a 16×8 FP32 CUBE shape the disassembly must
show an explicit 32-byte row stride, `TCMP` `LB0=8/LB1=16/LB2=8`, and
`DTYPE_NONE` in the secondary DATR field of `TCMP`, `TADD`, and `TSUB`.

The resulting ELF is placed under the normal `output/.../elf` tree and is the
input for the matching `gfrun` and `gfsim` commands described in the repository
README.  This test intentionally stays separate from the existing 0.58 vector
microbenchmarks: it exercises the 0.59 ExecutionMask development contract.
