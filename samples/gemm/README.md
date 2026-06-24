# GEMM Sample

`gemm_avs_tile_smoke.diss` is a checked-in `llvm-objdump -dl` disassembly of a
compiler-produced Linx object for an AVS tile smoke GEMM case.

Related SuperNPUBench sources:

| Path | Role |
| --- | --- |
| [`../../benchmarks/npu/vec_simd/gemm_18x128x256`](../../benchmarks/npu/vec_simd/gemm_18x128x256) | Active NPU GEMM benchmark case. |
| [`../../benchmarks/kernels/composite/src/gemm.cpp`](../../benchmarks/kernels/composite/src/gemm.cpp) | Composite GEMM benchmark entrypoint. |
| [`../../benchmarks/kernels/gemm/matmul`](../../benchmarks/kernels/gemm/matmul) | Matmul/GEMM kernel benchmark suite. |

Regenerate from a compatible Linx compiler object with:

```sh
llvm-objdump -dl gemm.o > gemm_avs_tile_smoke.diss
```
