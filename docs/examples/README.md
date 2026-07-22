# Compile-checked documentation examples

These files are included verbatim in the MkDocs tutorials. Run them from the
repository root with the active Linx LLVM toolchain:

```bash
export COMPILER_DIR=/path/to/linx-isa/compiler/llvm/build-linxisa-clang/bin
export LINX_SYSROOT=/path/to/linx-isa/out/libc/musl/install/phase-b
bash docs/examples/check.sh
```

`cpp_language.cpp` covers the general C++20 features described in the guide.
`tile_add.cpp` and `gemm_tile.cpp` use the canonical builtin-backed one-level
PTO surface from the sibling `pto_kernels` workload, which is also used by Linx
QEMU tile validation.
`flash_attention.cpp` compiles a larger QK/PV dataflow probe from that same
canonical surface. It intentionally omits online softmax; the full benchmark
sources provide the numerically complete algorithm.
They are compile-only teaching kernels; linked, runnable programs live in the
benchmark test harness.
