# Benchmarks

This is the primary navigation surface for active SuperNPUBench benchmark sources. These suites are intended to build through the shared make harness with `PLAT=linx COMPILER_DIR=<linx-isa-llvm-bin>` unless a local `compile*.all` file explicitly selects another platform for comparison.

## Layout

| Path | Purpose |
| --- | --- |
| [`common`](common) | Shared make harness and benchmark-local utility headers. |
| [`api/tileop`](api/tileop) | Focused TileOP API operation benchmarks. |
| [`npu/cube`](npu/cube) | Cube/matmul NPU benchmark cases. |
| [`npu/fusion`](npu/fusion) | Flash-attention and fusion NPU cases. |
| [`npu/nddma`](npu/nddma) | NDDMA transpose cases. |
| [`npu/vec_simd`](npu/vec_simd) | Vector SIMD NPU cases. |
| [`npu/vec_simt`](npu/vec_simt) | Vector SIMT NPU cases, including embedded data-object cases. |
| [`kernels/control`](kernels/control) | Control-flow/hash-table kernels. |
| [`kernels/element_wise`](kernels/element_wise) | Element-wise kernels. |
| [`kernels/gemm`](kernels/gemm) | GEMM/matmul kernels. |
| [`kernels/fusion`](kernels/fusion) | Kernel-level fusion cases. |
| [`kernels/memory`](kernels/memory) | Broadcast, gather, scatter, concat, and transpose memory kernels. |
| [`kernels/reduction`](kernels/reduction) | Row/column reduction kernels. |
| [`kernels/sort`](kernels/sort) | Sort/top-k kernels with embedded data-object support. |
| [`kernels/composite`](kernels/composite) | Composite kernels formerly grouped under `orther`. |
| [`models/deepseekv3`](models/deepseekv3) | DeepSeekV3 model-level benchmark kernels. |
| [`microbench`](microbench) | Cube, memory, and vector microbenchmark suites. |
| [`scripts`](scripts) | Batch and recursive helper scripts. |

## Build Pattern

Run local make commands from the suite directory:

```sh
cd benchmarks/api/tileop
make TESTCASE=TAdd PLAT=linx COMPILER_DIR=/path/to/linx/compiler/bin
```

Run suite batches from the same directory as the script:

```sh
cd benchmarks/kernels/gemm/matmul
bash compile.all
```

For the complete source and build catalog, use [`INDEX.md`](INDEX.md).
