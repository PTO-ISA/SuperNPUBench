# transpose solution（最终版）

只保留性能最优的实现与依赖闭包。

## 文件

| 文件 | 说明 |
|---|---|
| `transpose_nd_v3_1.hpp` | 场景分派（TENSOR_MOVE / N_LAST / GATHER），全编译期；**v3.2.1 的依赖** |
| `transpose_nd_v3_2.hpp` | v3.1 的动态 shape 版；提供 `v32_detail` 规范化/路由；**v3.2.1 的依赖** |
| **`transpose_nd_v3_2_1.hpp`** | **最终版**：`MakeV32Tiling`（Host-tiling）+ `transpose_nd_v3_2_1`（HT kernel）+ `transpose_nd_v3_2_1_ct`（编译期特化）+ `dispatch` 路由 |
| `transpose_nd_simt_elementif.hpp` | SIMT（element-if，对标 ops-math SMALL_SHAPE 10001） |

## 最终性能（gfsim 4 PE，i32，`-DPERF_ONLY`）

| case | shape / perm | 机制 | cycles |
|---|---|---|---|
| 0 | `[4096]` (0) | CT / TENSOR_MOVE | **599** |
| 1 | `[64,64]` (1,0) | CT / MGATHER | **1232** |
| 5 | `[8,8,8,8]` (0,2,1,3) | CT / MGATHER | **1367** |
| 4 | `[16,32,64]` (2,0,1) | CT / MGATHER | **6780** |
| 7 | `[8,8,8,16,32]` (0,1,4,3,2) | HT / MGATHER | **57867** |
| 3 | `[32,64,512]` (0,2,1) | HT / MGATHER | **202238** |
| 2 | `[64,64,256]` (1,0,2) | HT / N_LAST | **202268** |
| 6 | `[1,16,936,128]` (0,2,1,3) | HT / N_LAST | **368100** |

小 case 用 **CT**（编译期特化，循环常量展开）比 v3.2 快 **-28%~-70%**；5D 用 **HT** 快 **-32%**；大 case 持平。

## 构建与运行

```bash
export COMPILER_DIR=<linx_blockisa_llvm_musl>/bin

# 小 shape（CT 路径由 dispatch 选择；Element<=64K）
make -C benchmark/one-level-arch/test/solution/transpose TESTCASE=transpose_v32 V32_CASE=5 ccopts="-DPERF_ONLY"
<gfsim> -s softcore.multiThreadNum=4 -f output/solution/transpose/elf/*_transpose_v32_PE4.elf

# SIMT（需冻结工具链 + --pto059-execution-mask-dev）
make -C benchmark/one-level-arch/test/solution/transpose TESTCASE=transpose_simt SIMT_CASE=0 SIMT_PE=1
<gfsim> --pto059-execution-mask-dev -s softcore.multiThreadNum=1 -f ..._transpose_simt_PE4.elf
```

- **性能测量必须加 `-DPERF_ONLY`**：否则测试自身的初始化循环会被算进 gfsim 的 `Total Cycles`（小 case 会虚高数十倍）。
- SIMT 需要冻结组合：llvm `482b4207` / TileOP `b223de6` / bench `9076bd0` / model `8e8d8ac7`，
  且必须加 `--pto059-execution-mask-dev`（否则报 `illegal TCI`）。

## 已知遗留
- `transpose_nd_v3_2_1_dispatch` 目前未路由到 CT（1592 vs 直接调 CT 的 599），
  建议调用方用 `if constexpr` 直接选入口；详见 issue LinxISA/SuperScalarModel#1133。
- 4 线程/4 PE 的 SIMT 受模型 `iex_pe_rslv_array` 尺寸与 store-buffer 可见性限制（#1133）。
