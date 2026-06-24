# Samples

This tree keeps small checked-in compiler-output examples for quick inspection.
Samples are reference artifacts, not build gates, and are intentionally separate
from generated local output under `output/`.

| Workload | Sample | Related SuperNPUBench source | Provenance |
| --- | --- | --- | --- |
| Flash attention | [`flash_attention/flash_attention_block_template.diss`](flash_attention/flash_attention_block_template.diss) | [`../benchmarks/npu/fusion`](../benchmarks/npu/fusion), [`../benchmarks/kernels/composite/src/flash_attention.cpp`](../benchmarks/kernels/composite/src/flash_attention.cpp) | `pto_objdump`/`llvm-objdump` disassembly of a larger compiler-produced flash attention object, including block-template TileOP sequences such as `BSTART.TLOAD`, `BSTART.TMATMUL`, `BSTART.TCVT`, and `BSTART.TSTORE`. |
| GEMM | [`gemm/gemm_avs_tile_smoke.diss`](gemm/gemm_avs_tile_smoke.diss) | [`../benchmarks/npu/vec_simd/gemm_18x128x256`](../benchmarks/npu/vec_simd/gemm_18x128x256), [`../benchmarks/kernels/composite/src/gemm.cpp`](../benchmarks/kernels/composite/src/gemm.cpp) | `llvm-objdump -dl` of a compiler-produced `gemm.o` from the Linx superproject AVS tile smoke outputs. |

A compatible Linx compiler can disassemble these objects, but direct
SuperNPUBench NPU source compilation may still require frontend support for the
block-vector builtins and tile-register inline-assembly constraints used by the
benchmark headers.
