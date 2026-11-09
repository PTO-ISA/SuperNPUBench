# transpose solution（最终版）

与 ops-math 原算子对齐：**Host 侧算 tiling（运行时 shape）+ kernel 按参数搬运**，不使用编译期 shape 特化。

## 文件

| 文件 | 说明 |
|---|---|
| `transpose_nd_v3_1.hpp` | 场景分派（TENSOR_MOVE / N_LAST / GATHER），全编译期；**v3.2.1 的依赖** |
| `transpose_nd_v3_2.hpp` | v3.1 的动态 shape 版；提供 `v32_detail` 规范化/路由；**v3.2.1 的依赖** |
| **`transpose_nd_v3_2_1.hpp`** | **最终版**：`MakeV32Tiling`（Host-tiling）+ `transpose_nd_v3_2_1`（HT kernel）+ `dispatch` 入口（全部走 HT） |
| `transpose_nd_simt_elementif.hpp` | SIMT（element-if，对标 ops-math SMALL_SHAPE 10001） |

## 最终性能（gfsim 4 PE，i32，`-DPERF_ONLY`）

| case | shape / perm | 场景 | cycles |
|---|---|---|---|
| 0 | `[4096]` (0) | TENSOR_MOVE | **1754** |
| 1 | `[64,64]` (1,0) | GATHER | **3700** |
| 5 | `[8,8,8,8]` (0,2,1,3) | GATHER | **4650** |
| 4 | `[16,32,64]` (2,0,1) | GATHER | **9587** |
| 7 | `[8,8,8,16,32]` (0,1,4,3,2) | GATHER | **57909** |
| 3 | `[32,64,512]` (0,2,1) | GATHER | **202487** |
| 2 | `[64,64,256]` (1,0,2) | N_LAST | **202501** |
| 6 | `[1,16,936,128]` (0,2,1,3) | N_LAST | **368765** |
| S1 | `[8,8,8,8]` (0,2,1,3) | **SIMT（element-if, 10001）**，**1 PE / uint32** | **59043** |

> **S1（SIMT）与上表其余行口径不同**：`uint32` 数据、**仅 1 PE**（4 PE 受模型 `iex_pe_rslv_array`/store-buffer 限制，
> 见 issue #1133）、需冻结工具链 + `--pto059-execution-mask-dev`，因此绝对值不能与 4 PE 行直接比较。

## 正确性验证

8 个 case 均已与 numpy 参考实现逐元素比对通过：

```bash
export COMPILER_DIR=<linx_blockisa_llvm_musl>/bin
make -C benchmark/one-level-arch/test/solution/transpose TESTCASE=transpose_v32 V32_CASE=<0..7>

# 用 gfrun 导出 output 缓冲区（gfsim 不支持 guest 的 write() syscall）
<gfrun> -f <elf> -s softcore.multiThreadNum=4 \
        --dump-memory 0x<output 符号地址>:<字节数>:/tmp/out.bin
```

- **验证不能用 `-DPERF_ONLY`**（该开关会跳过 input 初始化，结果无意义）。
- 功能验证请用 `gfrun`；`gfsim` 里的 `write()` 走 `acrc 1`，会直接 `signal 6`。

## 构建与运行

```bash
export COMPILER_DIR=<linx_blockisa_llvm_musl>/bin

make -C benchmark/one-level-arch/test/solution/transpose TESTCASE=transpose_v32 V32_CASE=5 \
     CC_OPTS="-DPERF_ONLY"
<gfsim> -s softcore.multiThreadNum=4 -f output/solution/transpose/elf/*_transpose_v32_PE4.elf

# SIMT（需冻结工具链 + --pto059-execution-mask-dev）
make -C benchmark/one-level-arch/test/solution/transpose TESTCASE=transpose_simt SIMT_CASE=0 SIMT_PE=1
<gfsim> --pto059-execution-mask-dev -s softcore.multiThreadNum=1 -f ..._transpose_simt_PE4.elf
```

- **性能测量必须加 `-DPERF_ONLY`**：否则测试自身的初始化循环会被算进 gfsim 的 `Total Cycles`（小 case 会虚高数十倍）。
- SIMT 需要冻结组合：llvm `482b4207` / TileOP `b223de6` / bench `9076bd0` / model `8e8d8ac7`，
  且必须加 `--pto059-execution-mask-dev`（否则报 `illegal TCI`）。

## 已知遗留

- 4 线程/4 PE 的 SIMT 受模型 `iex_pe_rslv_array` 尺寸与 store-buffer 可见性限制（#1133）。
- 校验过程中发现两个模型缺陷（与 kernel 无关）：
  1. `constexpr` 常量数组的部分元素被错误物化（`{64,64}` 读成 `{64,0}`，与 #1038 同类）；
  2. 相邻的 32 位标量写会被合并成一次 64 位栈写并丢掉高半 → 测试里用 `volatile` 逐元素写绕过。
