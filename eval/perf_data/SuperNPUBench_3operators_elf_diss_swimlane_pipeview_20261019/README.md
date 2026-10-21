# Operator trace files

The operator directories are flat under this folder. Directory and file-name prefixes follow the configuration fields in `operator_performance_full_list_20261019_matmul_fullKcache_C23_tK_sweep.xlsx`.

Naming fields:

- FA: category, Cube quant, vector quant, Sq, Skv, HeadDim, implementation, Tm, and Tk.
- Matmul: category, dtype, M/K/N shape, implementation, cache count, tM, tN, and tK.

## Excel mapping

| Source package case | Operator directory | Excel location |
| --- | --- | --- |
| AlgD D256 | `FA_MXFP4_BF16_Sq256_Skv65536_D256_AlgoD_doubleQ_Tm128_Tk256` | `FA原始数据!A12:I12` |
| AlgD D192 | `FA_MXFP4_BF16_Sq256_Skv65536_D192_AlgoD_doubleQ_Tm128_Tk256` | `FA原始数据!A13:I13` |
| AlgD D128 | `FA_MXFP4_BF16_Sq256_Skv65536_D128_AlgoD_doubleQ_Tm128_Tk256` | `FA原始数据!A14:I14` |
| AlgD D64 | `FA_MXFP4_BF16_Sq256_Skv65536_D64_AlgoD_doubleQ_Tm128_Tk256` | `FA原始数据!A15:I15` |
| FA row14 | `FA_FP8_FP32_Sq128_Skv8192_D256_fa_gmma_opt_Tm128_Tk128` | `FA原始数据!A39:I39` |
| FA row16 | `FA_FP8_FP32_Sq128_Skv8192_D128_fa_gmma_opt_Tm128_Tk128` | `FA原始数据!A41:I41` |
| FA row17 | `FA_FP8_FP32_Sq128_Skv8192_D64_fa_gmma_opt_Tm128_Tk128` | `FA原始数据!A42:I42` |
| FA row18 | `FA_FP8_BF16_Sq128_Skv8192_D256_fa_gmma_opt_Tm128_Tk128` | `FA原始数据!A43:I43` |
| FA row20 | `FA_FP8_BF16_Sq128_Skv8192_D128_fa_gmma_opt_Tm128_Tk256` | `FA原始数据!A45:I45` |
| FA row21 | `FA_FP8_BF16_Sq128_Skv8192_D64_fa_gmma_opt_Tm128_Tk256` | `FA原始数据!A46:I46` |
| FA row22 | `FA_BF16_FP32_Sq128_Skv8192_D256_fa_gmma_opt_Tm128_Tk64` | `FA原始数据!A47:I47` |
| FA row24 | `FA_BF16_FP32_Sq128_Skv8192_D128_fa_gmma_opt_Tm128_Tk128` | `FA原始数据!A49:I49` |
| FA row25 | `FA_BF16_FP32_Sq128_Skv8192_D64_fa_gmma_opt_Tm128_Tk128` | `FA原始数据!A50:I50` |
| HIF4 M48/K7168/N96 | `matmul_HIF4_M48_K7168_N96_reuseA_fullK_C2_tM128_tN64_tK1024` | `Matmul原始数据!A127:Q127` |
| HIF4 M256/K4096/N1280 | `matmul_HIF4_M256_K4096_N1280_reuseA_fullK_C2_tM128_tN128_tK512` | `Matmul原始数据!A128:Q128` |

Each operator directory uses the same prefix for its ELF, disassembly, SwimLane, PipeView, and related logs. The AlgD D192 source package does not contain a SwimLane file.

The root-level `*_source_*` files preserve the source package metadata and snapshots. Paths embedded in those provenance files refer to their original generation locations.
