# SuperNPUBench 算子编译列表

工具链：`llvm-project temp/shared-tload-integration-20260811 @ eb64de8af` + `Linx-TileOP-API temp/shared-tload-integration-20260811 @ 72f8255`

全部 28 个配置编译通过（0 error），每个 `.elf` 同目录有 `.elf.diss`。

| # | 类别 | 配置 | dtype | ELF 路径 |
|---|---|---|---|---|
| 1 | broadcast | vec_07 | half | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/broadcast/elf/kernel_broadcast_broadcast_vec_07__DType__half_tM16_IN_SHAPE1443_1_OUT_SHAPE1443_129.elf` |
| 2 | broadcast | vec_019 | half | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/broadcast/elf/kernel_broadcast_broadcast_vec_019__DType__half_tM8_kInner49_IN_SHAPE1280_1_49_OUT_SHAPE1280_8_49.elf` |
| 3 | element_wise | gelu | bf16 | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/element_wise/gelu/elf/kernel_element_wise_gelu_gelu_Approximate_false_DType__bf16_tM2048_SHAPE24_8_1024.elf` |
| 4 | gather | gather | fp32 | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/gather/elf/kernel_gather_gather_DType__fp32_OTypeuint32_t_gKs131072_gMs32_gNs256_tMs32_tNs64.elf` |
| 5 | fa | 2d_unroll Tm16 Tk16 | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/fa/elf/kernel_fa/fa_2d_unroll_Sq256_Skv512_Tm16_Tk16_X1_Y2.elf` |
| 6 | concat | concat_gather | int32 | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/concat/elf/kernel_concat_concat_gather_DTypeint32_t_tM512_IN_SHAPE64_2_OUT_SHAPE64_2000.elf` |
| 7 | concat | concat_scatter | int32 | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/concat/elf/kernel_concat_concat_scatter_DTypeint32_t_tM512_IN_SHAPE64_2_OUT_SHAPE64_2000.elf` |
| 8 | control | hashtable_lookup | — | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/control/elf/hashtable_lookup_simd_kNum6144_kMaxProbe512_knum_col256_debug_off.elf` |
| 9 | norm | rms_norm | — | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/norm/elf/kernel_norm_rms_norm_M16_N256_tM8_tN128.elf` |
| 10 | reduction | reducesum_row | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/reduction/reducesum_row/elf/kernel_reduction_reducesum_row_reducesum_row_DTypefloat_tM16_tN128_GM16_GN8192.elf` |
| 11 | reduction | reducesum_col | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/reduction/reducesum_col/elf/kernel_reduction_reducesum_col_reducesum_col_DTypefloat_tM32_tN64_GM2048_GN64.elf` |
| 12 | reduction | reducesum_col | int32 | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/reduction/reducesum_col/elf/kernel_reduction_reducesum_col_reducesum_col_DTypeint32_t_tM32_tN64_GM2048_GN64.elf` |
| 13 | reduction | reducesum_col | half | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/reduction/reducesum_col/elf/kernel_reduction_reducesum_col_reducesum_col_DType__half_tM32_tN64_GM2048_GN64.elf` |
| 14 | reduction | reducemax_row | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/reduction/reducemax_row/elf/kernel_reduction_reducemax_row_reducemax_row_DTypefloat_tM16_tN128_GM16_GN8192.elf` |
| 15 | reduction | reducemax_row | int32 | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/reduction/reducemax_row/elf/kernel_reduction_reducemax_row_reducemax_row_DTypeint32_t_tM16_tN128_GM16_GN8192.elf` |
| 16 | reduction | reducemax_col | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/reduction/reducemax_col/elf/kernel_reduction_reducemax_col_reducemax_col_DTypefloat_tM32_tN64_GM2048_GN64.elf` |
| 17 | reduction | reducemax_col | int32 | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/reduction/reducemax_col/elf/kernel_reduction_reducemax_col_reducemax_col_DTypeint32_t_tM32_tN64_GM2048_GN64.elf` |
| 18 | sort | topk | — | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/sort/elf/topk.elf` |
| 19 | transpose | unroll | half | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/transpose/elf/kernel_transpose/transpose_unroll_DType__half_tM512_IN1476_32_OUT32_1476.elf` |
| 20 | deepseek | fused_weight | — | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/deepseek/elf/kernel_deepseek/fused_weight.elf` |
| 21 | deepseek | rms_norm | — | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/deepseek/elf/kernel_deepseek/rms_norm.elf` |
| 22 | matmul | MASK_FP32 256x256x256 tM32tN32tK32 | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK32.elf` |
| 23 | matmul | MASK_FP32 256x256x256 tM32tN32tK64 | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N256_K256_tM32_tN32_tK64.elf` |
| 24 | matmul | MASK_FP32 512x512x512 tM32tN32tK64 | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M512_N512_K512_tM32_tN32_tK64.elf` |
| 25 | matmul | MASK_FP32 256x2048x2048 tM32tN32tK64 | float | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/matmul/elf/kernel_matmul/matmul_MASK_MASK_FP32_M256_N2048_K2048_tM32_tN32_tK64.elf` |
| 26 | multi_thread | vec/tadd | — | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/multi_thread/vec/elf/kernel_multi_thread_vec_Rows16_Cols16.elf` |
| 27 | multi_thread | vec/trowsum | — | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/multi_thread/vec/elf/kernel_multi_thread_vec_trowsum_Rows16_Cols16.elf` |
| 28 | flashMLA | Sq64 Dk512 | — | `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/output/kernel/flashMLA/elf/kernel_flashMLA/flashMLA_Sq64_QHeadPerHK1_NumBlocks2_Dk512_Dv512_DChunk128_VChunk128_Tm16_Tk16.elf` |
