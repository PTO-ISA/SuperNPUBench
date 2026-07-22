# Compile-checked documentation examples

These files are included verbatim in the MkDocs tutorials. Run them from the
repository root with the active target compiler:

```bash
export COMPILER_DIR=/path/to/compiler/bin
export LINX_SYSROOT=/path/to/target/sysroot
bash docs/examples/check.sh
```

`cpp_language.cpp` covers the general C++20 features described in the guide.
`tile_add.cpp`, `shared_tile_types.cpp`, and `gemm_tile.cpp` use the public
`pto_kernel/tile.hpp` header,
which keeps target-specific implementation namespaces behind one include.
`flash_attention.cpp` compiles a larger QK/PV dataflow probe from that same
surface. It intentionally omits online softmax; the full benchmark sources
provide the numerically complete algorithm.
They are compile-only teaching kernels; linked, runnable programs live in the
benchmark test harness.
