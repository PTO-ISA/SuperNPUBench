# SuperNPUBench

SuperNPUBench is a Linx/PTO TileOP benchmark workspace. Active benchmark
entrypoints live under `benchmarks/`, reusable library code stays under
`include/`, `kernels/`, and `models/`, non-benchmark correctness checks live
under `tests/`, and superseded material is preserved under `archive/outdated/`.

## Repository Map

| Path | Purpose |
| --- | --- |
| [`benchmarks`](benchmarks) | Primary active Linx-buildable benchmark tree. Start here for benchmark source and build commands. |
| [`benchmarks/INDEX.md`](benchmarks/INDEX.md) | Source catalog with benchmark paths, build commands, category, status, and required data objects. |
| [`benchmarks/common`](benchmarks/common) | Shared make harness, platform selection, compiler flags, simulator targets, and benchmark-local helper headers. |
| [`include/common`](include/common) | Shared TileOP API surface, data types, tensor helpers, layouts, and compile-time utilities. |
| [`include/benchmark_support`](include/benchmark_support) | Benchmark-only support headers, including NPU helper APIs used by active suites. |
| [`include/cpu_sim`](include/cpu_sim) | CPU simulation backend used by checks built with `PLAT=cpu`. |
| [`include/aarch64`](include/aarch64) | Host/Arm-oriented backend headers and TileOP API variants. |
| [`include/accelerator`](include/accelerator) | Versioned device API headers consumed by Linx/NPU code. |
| [`include/jcore`](include/jcore) | Linx/JCore backend headers used by `PLAT=linx`. |
| [`kernels`](kernels) | Reusable kernel implementations shared by benchmark entrypoints. |
| [`models`](models) | Reusable model-level implementation code shared by model benchmarks. |
| [`tests`](tests) | Non-benchmark correctness material that is not the primary Linx benchmark navigation surface. |
| [`archive/outdated`](archive/outdated) | Preserved duplicate, superseded, generated, or unusable historical material with replacement notes. |
| `output/` | Generated build products. Treat this as local output, not source. |

## Getting Started: Linx Compiler And QEMU

These commands assume this workload lives at `$LINXISA_ROOT/workloads/SuperNPUBench` and the Linx superproject is checked out at `/Users/zhoubot/linx-isa`. Adjust `LINXISA_ROOT` if your checkout is elsewhere.

Build the Linx LLVM compiler from `compiler/llvm`:

```bash
export LINXISA_ROOT=/Users/zhoubot/linx-isa
cd "$LINXISA_ROOT"

cmake -S compiler/llvm/llvm -B compiler/llvm/build-linxisa-clang -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD=Linx
cmake --build compiler/llvm/build-linxisa-clang --target clang lld llvm-mc llvm-objdump llvm-objcopy

export COMPILER_DIR="$LINXISA_ROOT/compiler/llvm/build-linxisa-clang/bin"
```

For incremental compiler rebuilds after the CMake tree exists:

```bash
cd "$LINXISA_ROOT"
bash tools/bringup/run_llvm_incremental_build.sh clang lld llvm-mc llvm-objdump llvm-objcopy
```

Build the Linx QEMU target and run a QEMU smoke suite:

```bash
cd "$LINXISA_ROOT"
QEMU="$(bash tools/bringup/run_qemu_build_local.sh)"

cd "$LINXISA_ROOT/avs/qemu"
CLANG="$COMPILER_DIR/clang" LLD="$COMPILER_DIR/ld.lld" QEMU="$QEMU" ./run_tests.sh --suite arithmetic --timeout 10
```

Compile one SuperNPUBench case with the rebuilt compiler:

```bash
cd "$LINXISA_ROOT/workloads/SuperNPUBench"
make -C benchmarks/api/tileop TESTCASE=TAdd PLAT=linx COMPILER_DIR="$COMPILER_DIR"
```

The benchmark `sim` target invokes `$(QEMU) -run-supertest -blk_optimize force_tb_chained <elf>`. Use it with a SuperTest-compatible QEMU binary:

```bash
make -C benchmarks/api/tileop TESTCASE=TAdd PLAT=linx COMPILER_DIR="$COMPILER_DIR" QEMU=/path/to/supertest-compatible-qemu sim
```

For standard `qemu-system-linx64 -machine virt` validation, use the `avs/qemu` runner shown above.

## Benchmark Navigation

Active benchmark source is grouped by workload intent:

| Path | Category |
| --- | --- |
| [`benchmarks/api/tileop`](benchmarks/api/tileop) | TileOP API operation benchmarks. |
| [`benchmarks/npu`](benchmarks/npu) | NPU cube, fusion, NDDMA, vec SIMD, and vec SIMT suites. |
| [`benchmarks/kernels`](benchmarks/kernels) | Control, element-wise, GEMM, fusion, memory, reduction, sort, and composite kernel suites. |
| [`benchmarks/models/deepseekv3`](benchmarks/models/deepseekv3) | DeepSeekV3 model-level benchmark cases. |
| [`benchmarks/microbench`](benchmarks/microbench) | Cube, memory, and vector microbenchmarks. |
| [`benchmarks/scripts`](benchmarks/scripts) | Batch and recursive helper scripts for benchmark workflows. |

Use [`benchmarks/INDEX.md`](benchmarks/INDEX.md) when you need the exact source path, suite-level build command, or data-object requirement for a case.

## Benchmark Names

The table below is generated from active benchmark source files. It currently lists 166 C++ benchmark entrypoints across 26 suites and 44 batch scripts.

| Suite | Benchmark names |
| --- | --- |
| [`benchmarks/api/tileop`](benchmarks/api/tileop) | `Cus_Template_ASM`, `MatMacc`, `MatMul`, `MatMul_e4m3`, `Print`, `TAbs`, `TAdd`<br>`TAdd_mask`, `TAdds`, `TAnd`, `TAssemble`, `TCI`, `TCast`, `TCmp`<br>`TCopy`, `TLoad`, `TStore`, `TCvt`, `TDiv`, `TDivs`, `TExp`<br>`TExpandCol`, `TExpandRow`, `TExpandScalar`, `TExtract`, `TFillPad`, `TGather`, `TMax`<br>`TMaxs`, `TMin`, `TMins`, `TMul`, `TMuls`, `TOr`, `TPad`<br>`TRSqrt`, `TRecip`, `TRem`, `TReshape`, `TRowMax`, `TRowMaxExpand`, `TRowSum`<br>`TRowSumExpand`, `TScatter`, `TSelect`, `TSqrt`, `TSub`, `TSubs`, `TTrans`<br>`test_MatMacc`, `test_MatMmxac`, `test_MatMul`, `test_MatMulmx` |
| [`benchmarks/npu/cube`](benchmarks/npu/cube) | `LLAMA3_70B_attn_matmul_decode_bs_192`, `LLAMA3_70B_ffn_matmul_3_decode_bs_192`, `QuantBatchMatmulV3_292_hif4`, `QuantBatchMatmulV3_293_hif4`, `QuantBatchMatmulV3_294_hif4`, `QuantBatchMatmulV3_295_hif4`, `QuantBatchMatmulV3_296_hif4`<br>`QuantBatchMatmulV3_297_hif4`, `dsv3_q_up_proj_mxfp8`, `llama3_70b_w8_bs_1_case_4`, `llama_train_mm_2_A16W4`, `llama_train_mm_2_A16W8`, `llama_train_mm_2_mxfp8_mxfp4`, `llava1_6_6`<br>`mat_mul_o1_align_0001`, `matmul_1_bs16_fp8_GB_test`, `model_graph_graph7_mat_mul_0279_fp8_GB_DN_nbuf`, `moe_w1w3_bs16_fp8_GB_DN_nbuf`, `mx_a8w4_float8_e4m3fn_float4_e2m1_bfloat16_0022`, `mx_a8w4_nz_0001_float8_e4m3fn_float4_e2m1_bfloat16`, `xinghuo_13b_tp8_matmul_01_A16W8`<br>`xinghuo_13b_tp8_matmul_01_mxfp8_modified`, `xinghuo_13b_tp8_matmul_01_mxfp8_mxfp4` |
| [`benchmarks/npu/fusion`](benchmarks/npu/fusion) | `fa1`, `fa10`, `fa11`, `fa2`, `fa3`, `fa4`, `fa5`<br>`fa6`, `fa7`, `fa8`, `fa9`, `fa_fp4`, `flashmla13` |
| [`benchmarks/npu/nddma`](benchmarks/npu/nddma) | `transpose_053_mgather`, `transpose_053_tload` |
| [`benchmarks/npu/vec_simd`](benchmarks/npu/vec_simd) | `Add_ND_bfloat16_float32_DeepSeek_V3_000028`, `LayerNormV4_ND_bfloat16_IDZJ06_25B_8K_LORA_R6144_000001_grad_chip_generic`, `LayerNormV4_ND_bfloat16_float32_X1_ViT175B_R12288_000020_grad_chip_generic`, `LayerNormV4_ND_bfloat16_float32_X1_ViT175B_R24576_000020_grad_GENERIC_AIV`, `gemm_18x128x256`, `layernorm_vcadd_vaddx3_12288_fp16`, `moe_gating_top_k_deepseekv3_16_fp32_GENERIC_AIV`<br>`rmsnorm_reduce_1_16384_fp16`, `rmsnorm_reduce_2_8192_fp16`, `rmsnorm_reduce_4_4096_fp16`, `rmsnorm_reduce_4_5120_fp16`, `rope_32_40_1_64_bf16`, `softmax_8_34_fp16`, `softmax_LLM_2`<br>`softmax_vaddx3_vcadd_1_4096_bf16`, `softmax_vaddx3_vcadd_1_4096_fp16`, `swiglu_64_1024_fp16` |
| [`benchmarks/npu/vec_simt`](benchmarks/npu/vec_simt) | `npu_hashtable_insert_cmp_host`, `npu_hashtable_lookup_cmp_host`, `hashfind` |
| [`benchmarks/kernels/composite`](benchmarks/kernels/composite) | `flash_attention`, `flash_attention_mask`, `gemm`, `linear`, `matmul`, `normalization`, `onlinesoftmax`<br>`softmax` |
| [`benchmarks/kernels/control`](benchmarks/kernels/control) | `hashfind`, `hashtable_lookup_simd`, `hashtable_lookup_simt`, `hashtable_lookup_simt_v2`, `hkv` |
| [`benchmarks/kernels/element_wise/gelu`](benchmarks/kernels/element_wise/gelu) | `gelu` |
| [`benchmarks/kernels/fusion`](benchmarks/kernels/fusion) | `fa_hif4` |
| [`benchmarks/kernels/gemm/matmul`](benchmarks/kernels/gemm/matmul) | `A16W4`, `HiF4_HiF4` |
| [`benchmarks/kernels/memory/broadcast`](benchmarks/kernels/memory/broadcast) | `broadcast`, `broadcast_019`, `broadcast_039`, `broadcast_07`, `broadcast_Hunyuan`, `broadcast_mscatter`, `broadcast_nostore`<br>`broadcast_nomg`, `broadcast_tst` |
| [`benchmarks/kernels/memory/broadcast_vec`](benchmarks/kernels/memory/broadcast_vec) | `broadcast_vec_019`, `broadcast_vec_039`, `broadcast_vec_07` |
| [`benchmarks/kernels/memory/concat_gather`](benchmarks/kernels/memory/concat_gather) | `concat_gather` |
| [`benchmarks/kernels/memory/concat_scatter`](benchmarks/kernels/memory/concat_scatter) | `concat_scatter` |
| [`benchmarks/kernels/memory/gather`](benchmarks/kernels/memory/gather) | `gather` |
| [`benchmarks/kernels/memory/transpose`](benchmarks/kernels/memory/transpose) | `transpose` |
| [`benchmarks/kernels/reduction/reducemax_col`](benchmarks/kernels/reduction/reducemax_col) | `reducemax_col` |
| [`benchmarks/kernels/reduction/reducemax_row`](benchmarks/kernels/reduction/reducemax_row) | `reducemax_row` |
| [`benchmarks/kernels/reduction/reducesum_col`](benchmarks/kernels/reduction/reducesum_col) | `reducesum_col` |
| [`benchmarks/kernels/reduction/reducesum_row`](benchmarks/kernels/reduction/reducesum_row) | `reducesum_row` |
| [`benchmarks/kernels/sort/topk`](benchmarks/kernels/sort/topk) | `topk` |
| [`benchmarks/models/deepseekv3`](benchmarks/models/deepseekv3) | `concat`, `expand`, `gate`, `mask`, `mla`, `mlp`, `moe`<br>`permute`, `projection`, `rmsnorm`, `rope`, `split`, `topk`, `transformer` |
| [`benchmarks/microbench/cube`](benchmarks/microbench/cube) | `matop` |
| [`benchmarks/microbench/lmbench`](benchmarks/microbench/lmbench) | `mem` |
| [`benchmarks/microbench/vec`](benchmarks/microbench/vec) | `lat_bw` |

## Building Benchmarks

Most benchmark directories include [`benchmarks/common/Makefile.common`](benchmarks/common/Makefile.common). The common harness is controlled mainly by `TESTCASE`, `PLAT`, `COMPILER_DIR`, and `QEMU`.

```sh
cd benchmarks/api/tileop
make clean
make TESTCASE=TAdd PLAT=cpu COMPILER_DIR=/usr/bin
make TESTCASE=TAdd PLAT=linx COMPILER_DIR=/path/to/linx/compiler/bin
make TESTCASE=TAdd PLAT=linx QEMU=/path/to/supertest-compatible-qemu sim
```

| Variable | Meaning |
| --- | --- |
| `TESTCASE` | Source basename or suite-local case name, for example `TAdd` in `benchmarks/api/tileop/src/TAdd.cpp`. |
| `PLAT=cpu` | Builds against the CPU simulation backend and defines `__cpu_sim__`. |
| `PLAT=linx` | Builds for the Linx target and defines `__linx`. |
| `PLAT=arm_sme` | Builds Arm SME-oriented cases and defines `__ARM_FEATURE_SME`. |
| `COMPILER_DIR` | Directory containing `clang`, `clang++`, `llvm-objdump`, `llvm-objcopy`, and related tools. |
| `QEMU` | Simulator binary used by `make sim` for Linx-targeted execution. |

Common targets:

```sh
make TESTCASE=TAdd all
make TESTCASE=TAdd diss
make TESTCASE=TAdd sim
make TESTCASE=TAdd debug
make clean
make clean_all
```

Generated objects, executables, disassembly, and logs are written below `output/`.

## Batch Suites

Many suites provide a local `compile*.all` file. Run these from the suite directory so relative paths and local make variables resolve as intended.

Examples:

```sh
cd benchmarks/api/tileop && bash compile.all
cd benchmarks/npu/vec_simt && bash compile.all
cd benchmarks/kernels/gemm/matmul && bash compile.all
cd benchmarks/models/deepseekv3 && bash compile.all
```

For recursive compile/run automation, see [`benchmarks/scripts/recursive/README.md`](benchmarks/scripts/recursive/README.md).

## Header Installation

The top-level `Makefile` installs the TileOP header tree into the active Clang resource directory under `include/tileop-api`.

```sh
make -n CLANG_PREFIX=/usr
make install CLANG_PREFIX=/path/to/clang/prefix
make uninstall CLANG_PREFIX=/path/to/clang/prefix
```

`CLANG_PREFIX` should point to a prefix containing `bin/clang`. The dry run is useful because the install location is derived from `clang -print-resource-dir`.

## Adding Work

Use the current directory ownership when adding code:

1. Add API-facing definitions or declarations in [`include/common`](include/common).
2. Add backend behavior in [`include/cpu_sim`](include/cpu_sim), [`include/jcore`](include/jcore), [`include/aarch64`](include/aarch64), or the relevant support include directory.
3. Add reusable compute kernels under the matching [`kernels`](kernels) domain.
4. Add Linx-buildable benchmark entrypoints under the matching [`benchmarks`](benchmarks) category.
5. Add non-benchmark correctness material under [`tests`](tests).
6. Move superseded or duplicate material to [`archive/outdated`](archive/outdated) with a replacement note instead of deleting it.

New make-driven benchmark directories should keep the local `Makefile` small and include [`benchmarks/common/Makefile.common`](benchmarks/common/Makefile.common) for shared platform flags, output paths, simulator targets, and cleanup behavior.

## Generated Files

Do not commit generated files from `output/`, object files, executable artifacts, local logs, or disassembly files. Keep source changes in `include/`, `kernels/`, `models/`, `benchmarks/`, `tests/`, and `archive/outdated/`.
