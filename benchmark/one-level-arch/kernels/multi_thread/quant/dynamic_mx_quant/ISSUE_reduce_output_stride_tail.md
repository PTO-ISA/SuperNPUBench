# boxed 尾块（validRow < 物理行高）的 TROWMAX 输出物理 stride 被模型反推塌成 1 → 下游 TCVT 形状契约断言崩溃

## 摘要

当一个 reduce（`TROWMAX`）的**物理 tile 行高 `TileM` 大于有效行 `validRow`**（即 boxed 尾块，
`validRow < TileM` 且 `validRow` 不整除物理分配字节数）时，emulator 侧对 reduce 输出 tile 的
**物理列 stride** 用启发式 `dst->size / (validRow * elemBytes)` 反推。该式在 `validRow` 不整除
物理分配时**塌成 1**，使 `TROWMAX` 输出被记为 `col = 1`；而紧随其后的 `TCVT` 目标物理列由 B.DIM
的 `lb2 = BlockSize`（=32）给出，`col = 32`。二者不等 → `Block.cpp` `ValidateOperandContract`
的 “PTO 0.58 TCVT requires matching source/destination logical shapes” 断言直接崩溃。

**关键点：这不是工具链 codegen 缺陷。** full-tile 与 tail 的 `TROWMAX`/`TCVT` 指令**编码逐位相同**，
唯一差别是运行期寄存器里的 `validRow`（64 vs 58）。崩溃纯粹发生在 emulator 构造 reduce 输出
tile 描述符时的 stride 反推逻辑上。

## 环境

| 项 | 值 |
|---|---|
| 工具链 | `clang version 15.0.4 (linx64v5-musl-local 611105f2be11fab9a8ef20bd02b740f2c5d786b3)`（工作目录）；`adcb8794`（env_test ops-20260828）——两者发的指令编码一致 |
| 模型（gfrun） | **本 issue 精确锁定工作目录 SuperScalarModel `f0c488c8`（`Block.cpp:1155` `ValidateOperandContract`）**：full-tile 通过、仅 boxed 尾块崩。env_test `d8903938`（`Block.cpp:1650`）是**另一机制**（rowReduce stride 硬编码 =1，连 full-tile 都崩），见下方「影响」与「附：与 env_test 回归的关系」 |
| kernel | `dynamic_mx_quant_tail_ocp_fp8.hpp`（正式 kernel，固定 SPMD 4-PE；去 probe 前为 `probe_dynamic_mx_quant_tail_ocp_fp8_newcalc_mt.hpp`），BlockSize=32，half in / e4m3 out |
| 触发配置 | `M % (kPeNum*TileM) != 0`，即某 PE 的 `SubM % TileM != 0`（seg_tail>0）。例：M=1000/4PE → SubM=250 → seg_tail=58 |

## 根因

### reduce 输出 stride 反推（`SuperScalarModel/isa/Block.cpp`，rowReduce 分支）

工作目录版：

```cpp
if (rowReduce) {
    const uint64_t stride = dataType == DataType::UINT16
        ? 1
        : (elemBytes == 0 || validRow == 0 ||
           dst->size % (validRow * elemBytes) != 0     // ← 不整除 → 塌成 1
               ? 1
               : dst->size / (validRow * elemBytes));
    dst->tileInfo->validCol = 1;
    dst->tileInfo->col = stride;                        // 物理列 = stride
    ...
}
```

物理 tile 恒 `TileM × BlockSize` half = `64 × 32 × 2 = 4096` 字节（`dst->size`）。

| validRow | `4096 % (validRow*2)` | 反推 stride = `col` | TCVT dst `col`（lb2=BlockSize） | 匹配 |
|---|---|---|---|---|
| 64（full-tile） | 0 | `4096/(64*2)` = **32** | 32 | ✅ 通过 |
| 58（boxed 尾块） | 36 ≠ 0 | **1** | 32 | ❌ 崩溃 |

full-tile 之所以一直没暴露，是因为 `validRow == 物理 TileM`，`dst->size` 恰好整除，反推 stride
= 真实物理列 BlockSize。boxed 尾块 `validRow < TileM`（且不整除）时反推失败。

### 为什么反推会失败：物理 stride 在 bundle 里没有独立编码

reduce 输出逻辑上是「每行一个值」的列向量（`validCol=1`），但物理上嵌在 `TileM × BlockSize` 的
分配里，行间物理步长 = BlockSize。这个**物理步长没有单独的 B.DIM 字段承载**——模型只能从
`dst->size`（物理分配总字节）反推。反推式隐含假设 `dst->size == validRow × 物理列 × elemBytes`，
即 `validRow == 物理行高`；一旦 boxed（`validRow < 物理行高`），`dst->size` 含了 padding 行的
字节，`dst->size / (validRow*elemBytes)` 不再等于物理列，非整除时更直接塌成 1。

## 最小复现

```bash
export COMPILER_DIR=<linx toolchain>/bin
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant

# 崩溃：M=1000（4PE → SubM=250 → seg_full=3, seg_tail=58 boxed）
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP8 CC_OPTS="-DPM=1000 -DPN=32" diss
<gfrun> -f <...>_probe_ocp_fp8_newcalc.elf          # -> ValidateOperandContract 崩

# 对照通过：M=1024（4PE → SubM=256 → seg_full=4, seg_tail=0 无尾块）
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP8 CC_OPTS="-DPM=1024 -DPN=32" diss
<gfrun> -f <...>_probe_ocp_fp8_newcalc.elf          # -> Suaccelss ... R2 = 0
```

唯一变量是 M（1024→1000），使某 PE 的 `SubM % TileM` 从 0 变 58。

## 指令编码对照（证明非 codegen 分叉）

```
M=1024 full-tile:   11610: 24119181  BSTART.TEPL TROWMAX, FP16
                    11626: 21b19181  BSTART.TEPL TCVT,    FP16
M=1000 tail:        11612: 24119181  BSTART.TEPL TROWMAX, FP16   ← 逐位相同
                    11628: 21b19181  BSTART.TEPL TCVT,    FP16   ← 逐位相同
```

崩溃前 trace（M=1000，WD gfrun `-t 2`）：TROWMAX 的 `B.DIM` 发 `lb0=0x20(32)`、`lb1=0x3a(58)`、
`lb2=0x20(32)`；随后 TCVT 目标同样 `lb1=58`、`lb2=32`——**工具链发的维度是自洽的**，是模型对
reduce 输出的物理列反推塌成 1 才导致 src(1) ≠ dst(32)。

## 影响

- **全 mx_quant 家族共有**：凡「reduce 输出直接喂 elementwise（TCVT 等物理列由 lb2 声明的 op）」
  且发生 boxed 尾块的 kernel 都会中招。此前所有注册配置的 M 都能被 `kPeNum×TileM` 整除
  （无 boxed 尾块），故一直未触发。
- 与 **env_test ops-20260828 的 full-tile 回归**是**两个独立问题**，见下节。

## 附：与 env_test（`d8903938`）回归的关系

env_test 最新版**不含**本 issue 的反推式。`grep` 全文件确认其 rowReduce 分支**只有硬编码**：

```cpp
// env_test d8903938  Block.cpp:2059  rowReduce 分支
/* The destination capacity may contain additional physical rows,
 * but its physical column count is always one. */
const uint64_t stride = 1;          // ← 写死 1，不存在 dst->size/(validRow*elemBytes) 反推
dst->tileInfo->col = stride;
dst->tileInfo->validCol = 1;
dst->tileInfo->validRow = validRow;
dst->tileInfo->row = stride == 0 || elemBytes == 0 ? 0 : dst->size / (stride * elemBytes);
```

`grep 'dst->size % (validRow\|dst->size / (validRow'` 在 env_test 版**无任何命中**——反推式只在工作目录 `f0c488c8` 存在。

| 模型 | reduce 输出物理列 `col` 来源 | full-tile（validRow=64） | boxed 尾块（validRow=58） |
|---|---|---|---|
| **WD `f0c488c8`**（本 issue） | `dst->size/(validRow*elemBytes)` 反推 | =32 ✅ | 非整除塌成 1 ❌ |
| **ET `d8903938`**（另一 issue） | **硬编码 =1** | =1 ❌（≠ TCVT dst 32） | =1 ❌ |

**含义**：ET 版直接假设 reduce 输出物理列恒 1（密集列向量），期望 kernel 把 reduce 输出声明成
`physical Cols=1`。而 kernel 现用的 `physical=BlockSize / ValidCol=1`（行跨步，历史上被
`ISSUE_32B_align.md` 的 32B 列对齐 static_assert 逼出）在 ET 版彻底不兼容——这是「physical=1 迁移」
的动因，是独立于本尾块 issue 的前瞻项，不在本 issue 修复范围内。

## 修复建议

### 根本解（模型/spec 侧）

reduce 输出的**物理列 stride 应有权威来源，而非从 `dst->size` 反推**。两条路：

1. **用物理行高而非 validRow 反推**：若 bundle 能提供物理 tile 行高（`TileM`），则
   `stride = dst->size / (TileM * elemBytes)`——full-tile/boxed 都得物理列 BlockSize。
2. **spec 补字段**：让 reduce 输出显式携带物理列步长（类似 TCVT 的 lb2），emulator 不再猜。

### kernel 侧规避（本仓库当前可选，二选一）

- **方案 C（约束 + static_assert）**：要求 `SubM % TileM == 0`（等价 `M % (kPeNum*TileM) == 0`），
  编译期 `static_assert` 拦住 boxed 尾块，并在此 issue 挂钩说明；牺牲任意 M 的通用性。
- **方案 D（幂次分解尾块为 full-tile）**：把 `seg_tail`（如 58）按 2 的幂分解成若干合法物理行高的
  full-tile（58 = 32+16+8+2，各自 `validRow == 物理行高`，绕开反推失败），代价是多几个小 tile 迭代。

## 附：本仓库相关文件

- kernel：`kernels/multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8.hpp`（去 probe 转正后的 TAIL_OCP_FP8 正式 kernel）
- 相关历史：本目录 `RECORD.md` 问题13（TCVT 不发 lb2）、问题16（TCVT 形状契约）、`ISSUE_tcvt_no_lb2.md`
- 相关约束：`ISSUE_32B_align.md`（旧的 physical=BlockSize/valid=1 规避即源于 32B 对齐 static_assert）
