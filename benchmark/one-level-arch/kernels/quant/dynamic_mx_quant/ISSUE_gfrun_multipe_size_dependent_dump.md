# gfrun 多-PE（multiThreadNum=4）res_check 落盘随问题规模出现「仅 PE0 段有数据、其余块行为零」——小尺寸复现、大尺寸不复现（根因未定位）

> **本 issue 记录一个已稳定复现但根因尚未定位的现象。** 请勿把下文「可能方向」当结论——它们都是**未经证实的猜测**。
> 现象**规模相关**、且在两个**独立 kernel**（`nontail_cublas_fp8`、`nontail_ocp_fp4`）上签名完全一致，
> 故大概率在 gfrun 多-PE 执行 / res_check harness 侧，而非某一个 kernel。**问题隐蔽**（同一份编译产物、
> 仅数据规模不同即一过一挂），请务必按下方**完整版本清单 + 复现步骤**对齐环境，否则无法定位。

---

## 1. 摘要

非尾轴 mx_quant kernel 做 **4-PE SPMD**（按归约块行 `kb` 连续切 4 段，每 PE 写不重叠的 `y` 行段 + `scale` 行段）后，
用 `gfrun -s softcore.multiThreadNum=4` 跑 res_check ELF，落盘的 `output.bin` / `scale_output.bin`：

- **仅 PE0 段（`kb ∈ [0, numKb/4)`）有正确数据，其余块行为零**，另有末块（`kb = numKb−1`）出现少量非零「碎片」。
- 输出文件为**满尺寸**（非截断），`writeBinaryFile` 无报错、gfrun 正常 `R2 = 0` 到底。

同一 kernel、同一 4-PE 结构，**仅改自由轴长度 `Post`**（`Axis=512, BlockSize=32` 固定）即出现确定性的过/挂翻转：

| `Post` | `numN`(=Post/TileN) | 落盘覆盖（非零 `kb`） | `output` 官方比对 |
|---|---|---|---|
| 64 / 96 / 128 / 192 | 2 / 3 / 4 / 6 | 仅 `kb 0–3`（PE0 段）+ `kb15` 碎片 | **fail**（MSE 极大） |
| 256 | 8 | **全部 16 块行完整** | **pass**（MSE=0, MaxAE=0.011719） |

临界点落在 `Post=192` 与 `256` 之间。**`multiThreadNum=1` 单 PE 跑任意尺寸均写满全部块行、逐字节正确** → kernel 计算逻辑无误。

---

## 2. 环境 / 组件版本清单（务必逐一对齐）

工作区：三仓库均在同一分支 `dmxq-ops-20260828`（linx-toolchain-build 在 `main`）。

| 组件 | 基线 commit | 说明 |
|---|---|---|
| **SuperScalarModel**（gfrun） | **`762a72c3`**（BotGen2Dev，官方可见最新） | **+ 附录 A 复现前置补丁**（在此官方 commit 上打，见下 §2.1 / 附录 A）。**不要用 `762a72c3` 裸版**——裸版跑不通暴露 kernel。 |
| **SuperNPUBench**（kernel + driver + gen/compare 脚本） | **Bench PR [#102](https://github.com/PTO-ISA/SuperNPUBench/pull/102)** | 复现所需的 kernel（`nontail_cublas_fp8`）/ driver / `gen_*` / `*_compare.py` / `_start.s` / `fileop.h` 均由该 PR 提供 |
| **linx-toolchain-build**（构建编排） | `main` `e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156` | 工具链组件见下表；工作树干净 |

工具链组件（`linx-toolchain-build/src/*`，由 `make init-src` 拉取；本次实测 commit）：

| 组件 | commit |
|---|---|
| llvm-project（LLVM 15 + LinxV5 后端，clang 15.0.4） | `0f878a871d3241289c7cb96a63b005a8253d2eaf` |
| Linx-TileOP-API（TileOP 头，含 `pto_tileop.hpp` / `template_asm.hpp`） | `54b405be8cfa59b9249d2f3782ac5400e87c87db` |
| musl | `af0dfc2066272563fa5607cb6ae8cf974baaa415` |
| linux-linxisa | `1055a743f16eaebfc371e0aabec8c861ab44858f` |
| jemalloc | `4495309cd11cae1a0a2d008c65acd9c410076f06` |

### 2.1 复现前置补丁（重要——隐蔽问题必读）

- gfrun 的 SPMD 执行与终止逻辑（`emulator/SoftCore.cpp`）——**即本 bug 本体所在**——在官方 `762a72c3` 上**已存在、
  未被下述前置补丁改动**。换言之 bug 是官方现有的。
- 但**触发它需要一个足够复杂的 kernel**（本 issue 用 `nontail_cublas_fp8`）：仅用基础算子（TLOAD/TCVT/TSTORE、
  甚至 TABS+TCOLMAX+TCOLEXPANDMUL）的最小程序**无法触发**（实测均全覆盖、不复现）；必须是含**归约 + scale→E8M0
  转换 + 比较/选择守卫**的完整量化 kernel。而该 kernel 在官方 `762a72c3` 上跑**需要两处 v0.58.4 适配**，否则运行前置就崩、
  到不了落盘那步：
  - **E8M0（SF8）TCVT 转换**：`ConvertFloatToE8M0` + `DataFormatCvt` 的 `dstType==SF8` 分支。官方
    `762a72c3` **无**此执行路径（历史 `#253 / 52f56d5` 曾加、后被 `930d9981` 连坐删）→ scale→E8M0 崩。
  - **compare/select 源的 carrier-width 兼容判定**：`ValidateCompareSelectTepl` 用
    `IsCompatibleOperationDataTile` 替 `IsCompatibleDataTile`，接纳 `reinterpret_tile` 过的守卫源。
- **这两处是复现前置、不是本 bug 的修复**。完整 diff 见**附录 A**（基于官方 `762a72c3`）。另含一处**可选**的
  `GFRUN_FORCE_DIRECTBOOT_ABI` env 开关（`main.cpp`）：仅当你的 gfrun 对 res_check 直接引导 ELF 误判 syscall-ABI
  时需要；**本 issue 的裸文件 I/O 路径实测不需要它**（`writeBinaryFile` 用裸 `open/write`，RES_CHECK 下静音 stdio）。
- **复现步骤**（§3）：checkout 官方 `762a72c3` → 打附录 A 补丁 → `python3 build.py build --target gfrun -j8`
  重建 gfrun（**勿复用旧二进制**）→ 用 Bench PR #102 的 kernel/driver 编 res_check ELF → 跑。
- **验证锚点**：附录 A = `git diff 762a72c3..<本地 HEAD>`；本地 HEAD 即「`762a72c3` + 附录 A」，本 issue 所有实测
  数据均在该状态上取得，故「`762a72c3` + 附录 A」**必然复现**（Post≤192 挂 / 256 过）。

> 完整版本对照（本地实测所用，供交叉核对；官方复现以「762a72c3 + 附录 A」为准）：SuperScalarModel 本地 HEAD
> `19e2e97a`（=762a72c3 + 附录 A 的三个 commit `60ce26fd`/`3ca5744c`/`19e2e97a`）；SuperNPUBench 本地 HEAD
> `006c5d10`（内容以 Bench PR #102 为准）。**三仓 + 工具链 5 个 src 组件工作树均干净**，无游离未提交补丁。

运行时/harness 相关（均在上述 SuperNPUBench commit 内，非本次改动）：

- SPMD worker 入口：`benchmark/one-level-arch/test/common/src/group_worker_runtime.c`
  （gfrun 把 PE0 置 musl `_start`、PE1..3 置 `__linx_group_worker_start`；worker 跑完 `main()` 后 park，
  PE0 的 `SYS_exit_group` 释放全组）。
- 落盘：`benchmark/one-level-arch/test/common/writeBinary.h` / `readBinary.h`（`write(fd, buf, size)` 原子写；
  `RES_CHECK` 下静音 stdout printf，见 `ISSUE_gfrun_res_check_writev_hang.md`，与本问题无关）。
- `GFRUN_FORCE_DIRECTBOOT_ABI`：res_check 直接引导 ELF 的 syscall-ABI 规避（RECORD 问题23），**已含在附录 A**
  （`main.cpp`）。§3 命令带上它以求稳妥；但因落盘走裸 `open/write`（无 stdio），最小 raw-I/O 路径实测可不依赖它，
  故该开关在附录 A 中标为**可选**（见 §2.1）。

---

## 3. 复现步骤（最小、确定性）

用 `nontail_cublas_fp8`（4-PE，`TYPE=NONTAIL_CUBLAS_FP8_4PE`，fp16 in / e4m3 out）在**同一份代码**上仅变 `Post` 即可复现过/挂翻转。

### 3.0 工具链环境

```bash
R=<workspace>                    # SuperNPU 根
# --- gfrun：官方 762a72c3 + 附录 A 前置补丁，重建 ---
cd $R/SuperScalarModel
git checkout 762a72c3
git apply <附录A.patch>          # 附录 A 全文（基于 762a72c3）
python3 build.py build --target gfrun -j8      # 勿复用旧二进制
GFRUN=$R/SuperScalarModel/bin/gfrun
# --- 工具链 + kernel/driver（来自 Bench PR #102）---
export COMPILER_DIR=$R/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=$R/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot/usr
BASE=$R/SuperNPUBench/benchmark/one-level-arch
SRC=$BASE/test/kernel/quant/dynamic_mx_quant/src
```

driver：`$SRC/nontail_cublas_fp8_4pe.cpp`（默认 `PAXIS=512`，`PPOST=256`；下方用 `-DPPOST=` 覆盖）。
编译 flag（与 Makefile 一致）：

```bash
CF="-c -mlxbc -fenable-matrix -O2 -mllvm -enable-all-vector-as-tilereg=true \
 -mllvm -linxv5-enable-HL-Inst-Opt=true -mllvm -linxv5-enable-dim-opt=true \
 -mllvm -linxv5-enable-ldst-bridge=false -mllvm -linxv5-enable-continuous-mem-opt=true \
 -mllvm -linxv5-enable-tile-clock-hand=false -mllvm -linxv5-enable-simt-clock-hand=true \
 -mllvm -enable-misched=false -std=c++20"
INC="-I$BASE/include -I$BASE/test/common -I$BASE/test/common/src -I$BASE/kernels -I$BASE/models -D__linx -DENABLE_TENSOR_INSTR"
```

### 3.1 挂：Post=128

```bash
CHK=$BASE/compare/repro_128; mkdir -p $CHK
$COMPILER_DIR/clang++ $CF -DPPOST=128 -DRES_CHECK -DENABLE_BINARY_OUTPUT -DCHK_DIR=\"$CHK\" $INC \
  $SRC/nontail_cublas_fp8_4pe.cpp -o /tmp/r128.o
$COMPILER_DIR/clang++ -nostartfiles $BASE/test/common/_start.s /tmp/r128.o -o /tmp/r128.elf
python3 $SRC/gen_dynamic_mx_quant_data.py --algo CUBLAS --kernel nontail --dtype FP8 \
  --in-dtype fp16 --scale-layout compact --M 512 --K 128 --block-size 32 --seed 42 -o $CHK
GFRUN_FORCE_DIRECTBOOT_ABI=1 $GFRUN -f /tmp/r128.elf -s softcore.multiThreadNum=4
python3 $SRC/dynamic_mx_quant_data_compare.py --dtype FP8 --scale-layout compact \
  --cmp-root $BASE/compare -d repro_128
# 期望（BUG）：output=fail（MSE 极大）；output.bin 仅 rows 0–127(kb0–3) 非零 + rows 480–511(kb15) 碎片
```

### 3.2 过：Post=256（唯一改动 = `-DPPOST=256` + `--K 256`）

```bash
CHK=$BASE/compare/repro_256; mkdir -p $CHK
$COMPILER_DIR/clang++ $CF -DPPOST=256 -DRES_CHECK -DENABLE_BINARY_OUTPUT -DCHK_DIR=\"$CHK\" $INC \
  $SRC/nontail_cublas_fp8_4pe.cpp -o /tmp/r256.o
$COMPILER_DIR/clang++ -nostartfiles $BASE/test/common/_start.s /tmp/r256.o -o /tmp/r256.elf
python3 $SRC/gen_dynamic_mx_quant_data.py --algo CUBLAS --kernel nontail --dtype FP8 \
  --in-dtype fp16 --scale-layout compact --M 512 --K 256 --block-size 32 --seed 42 -o $CHK
GFRUN_FORCE_DIRECTBOOT_ABI=1 $GFRUN -f /tmp/r256.elf -s softcore.multiThreadNum=4
python3 $SRC/dynamic_mx_quant_data_compare.py --dtype FP8 --scale-layout compact \
  --cmp-root $BASE/compare -d repro_256
# 期望（正常）：output=pass (MSE=0.000000, MaxAE=0.011719)；16 块行全覆盖
# （scale=fail 是无关的 problem 5 parity-interleave 布局差，非本 issue）
```

### 3.3 逐块行覆盖检查（判定用）

```python
import numpy as np
POST = 128   # 或 256
Y = np.fromfile(f"<CHK>/output.bin", dtype=np.uint8).reshape(512, POST)
print([kb for kb in range(16) if (Y[kb*32:(kb+1)*32] != 0).any()])
# Post=128 → [0,1,2,3,15]（BUG）；Post=256 → range(16)（正常）
```

### 3.4 第二个独立 kernel 佐证（非 kernel 特有）

`nontail_ocp_fp4`（`TYPE=NONTAIL_OCP_FP4_4PE`，Post=64）出现**同一签名** `kb=[0,1,2,3]+kb15 碎片`；
`--algo OCP --kernel nontail --dtype FP4 --M 512 --K 64`。其 `output` 另受 fp4 写侧缺失影响
（`ISSUE_fp4_pack_tcvt_regression.md`），但**块行覆盖**的过/挂现象与 cublas 一致。

---

## 4. 已排除的方向（每条带证据；对定位隐蔽问题至关重要）

1. **非 kernel 代码差异。** `nontail_cublas_fp8` 的 `Post=128` 与 `256` 反汇编逐条对比（`llvm-objdump -d`）：
   tile 指令流同序、同 op，tile 维度 `lb0/lb1/lb2 = 64/32/64` 相同；差异仅
   ①内层 `numN` 循环次数（4 vs 8）、②访存 stride 立即数（输入行 stride 256B vs 512B，正比于 Post，各自正确）、
   ③寄存器分配、④256 侧多一条空 `C.BSTART.STD`。**无任何会改变计算的代码区别** → 同一份逻辑仅因运行期规模不同。
2. **非 write 截断/超时。** 输出文件**满尺寸**（Post=64: 32768B / 1024B），`writeBinaryFile` 无 "faild"；
   落盘内容本身即「半成品」内存快照。
3. **非通用内存/dump 竞争。** 最小**标量**探针（每 PE 用标量 store 把自己 `kb` 段全写成 `tid+1`，无 tile 指令）
   在任意尺寸、甚至每块插入大自旋延迟后，落盘**均完整**（4 段齐全、值正确）。
4. **非单纯 tile 存储路径。** `TLOAD + TCVT + TSTORE` 的 identity-copy **tile** 探针在 Post=64 落盘**也完整**。
5. **非每 PE 工作量不均。** 过/挂两种情况 4 个 PE 的 `Total Block number` 相同，`Total Inst number` 仅 PE0 多约 3 条
   （其 main I/O 收尾），均衡。
6. **非 reduce 输出物理形状。** 把非尾轴 reduce 输出/下游行向量 tile 从 `physical row=BlockSize, valid row=1`
   改为 `physical row=1`（对齐模型 `Block.cpp:2349` colReduce 语义）后，Post=256 仍 `output=pass`、Post=128 仍挂，
   **签名不变**。

---

## 5. 确定性事实（供维护者定位）

- **执行模型（gfrun functional）**：`SuperScalarModel/emulator/SoftCore.cpp` `SoftCore::Step()` 为
  **round-robin**（`for thread in 0..multiThreadNum`，thread 0 先跑，每 thread 每步跑 `execWidth` 个 block）。
- **组终止**：`SoftCore.cpp:379-388` —— *"The test finisher is a core-level device. A passing write from one PE
  terminates the complete SPMD program instance."* 任一 PE 到达终点即 `for participant: threadStatus[p].simEnd=true`
  并 `return`，**立即终止全组**（当步其余 thread 不再执行）。`MarkSyscallExit(exitGroup=true)`（`SoftCore.cpp:349`）
  对 `SYS_exit_group` / PE0 的 `SYS_exit` 同样置全组 `simEnd`。
- **所有 4 个 PE 都执行完整 `main()`**：`readBinaryFile` → kernel → 两次 `writeBinaryFile`；worker 跑完 park
  （`group_worker_runtime.c`）。
- 复现规模的每 PE 计数（Post=64 cublas）：4 PE 均 `Block=494`，`Inst≈2355`（PE0=2358）。

---

## 6. 可能方向（**均为未证实猜测，非结论**）

- 多-PE 下 **tile 存储对 host 侧 `writeBinaryFile` 读取的可见时序**，与「首个 PE 到达终点即终止全组」之间的交互：
  worker 的 tile-store 是否在 PE0 触发终止前已提交到 host 可见内存，未验证。（注：曾在 PE0 落盘前插入大自旋延迟，
  Post=64 仍只落 PE0 段——若为「worker 慢、可见延迟」本应被延迟掩盖，故此方向存疑。）
- **访存 stride / 对齐的规模相关性**：Post=256 时输入行 stride=512B（恰为常见 bank/对齐宽度 512B），
  ≤192 时为 256B/384B 等；是否影响多-PE 写回落 bank / 可见性，未验证。
- **round-robin 步进与「首个 PE 终止」的边界**：为何大尺寸下 worker 的写在 PE0 终止前完成、小尺寸不完成，
  临界点（Post 192→256）的具体成因未在 block 级锁死。
- 曾试「仅 PE0 落盘 + 共享 `.bss` 标志 barrier」：PE0 自旋等标志时**死锁**（标量 `.bss` 跨 PE 读似不即时可见），
  说明**标量与 tile 存储的跨-PE 可见性可能不同**——仅为线索，非定论。

---

## 7. 影响与现状

- **kernel 逻辑无误**（单 PE 逐字节正确佐证）；受影响的是**无 QEMU 环境下用 gfrun 4 线程跑 res_check 精度验证**这一路径。
- 现象**规模相关**：Post≥256 量级可得完整落盘并 `output=pass`；≤192 落盘不完整。既有 `nontail_cublas_fp8`
  Post=256「精度通过」结论正建立在阈值之上——**并非规模无关的可靠保证**。
- **未定位、未修**。SuperNPUBench 侧记于 `RECORD.md` 问题24。请维护者从「§5 确定性事实 + §6 可能方向」入手，
  优先厘清 gfrun 多-PE 下 tile-store 提交/可见时序与 TestFinisher 组终止的先后关系。

---

## 8. 附：诊断口径（可直接复用）

- **判「是否只落 PE0 段」**：见 §3.3 逐块行覆盖脚本。
- **隔离 tile vs 标量**：标量探针（§4-③）完整 → 排除通用内存竞争；tile identity-copy（§4-④）完整 → 排除单纯 tile store；
  → 复现失败需**完整复杂 kernel（含归约 + scale 存储 + 守卫）+ 多-PE + 小尺寸**三者叠加。
- **规模扫描**：`Post ∈ {64,96,128,192,256}` 固定其余参数，观察覆盖从 `[0,1,2,3,15]` 到 `range(16)` 的翻转与临界点。

---

## 附录 A：复现前置补丁（apply to official `762a72c3`；**非本 bug 的修复**）

让暴露 bug 的 `nontail_cublas_fp8` 量化 kernel 能在官方 `762a72c3` 上跑通所需的两处 v0.58.4 适配
（E8M0 TCVT + compare/select carrier-width），外加一处**可选**的 res_check ABI env 开关。
`git diff 762a72c3..<本地HEAD>`，可直接 `git apply`：

```diff
diff --git a/emulator/engine/AccumulateBlockInfo.cpp b/emulator/engine/AccumulateBlockInfo.cpp
index f77d94f6..5286d590 100644
--- a/emulator/engine/AccumulateBlockInfo.cpp
+++ b/emulator/engine/AccumulateBlockInfo.cpp
@@ -550,13 +550,16 @@ void ValidateCompareSelectTepl(const BlockFuncPtr& block,
            "compare/select dimensions do not fit the destination allocation");
 
     const size_t priorSources = block->srcTile.size();
+    // 问题14(reinterpret_tile / issue254):比较源可能是 reinterpret 过的 tile
+    // (carrier dtype=BF16、op 域=UINT16),严格 dataType 相等会误拒。改用与本文件
+    // elementwise 路径(686/710/743)一致的 carrier-width 兼容判定 IsCompatibleOperationDataTile。
     if (block->tileOp == TileOp::TCMP) {
         ASSERT(priorSources == 0 && inst->srcs.size() == 3 &&
                inst->dsts.size() == 1 &&
                inst->dsts[0]->size >= predicateBytes &&
-               IsCompatibleDataTile(inst->srcs[1], block->dataType, validRow,
+               IsCompatibleOperationDataTile(inst->srcs[1], block->dataType, validRow,
                                     validCol, physicalCol, dataBytes) &&
-               IsCompatibleDataTile(inst->srcs[2], block->dataType, validRow,
+               IsCompatibleOperationDataTile(inst->srcs[2], block->dataType, validRow,
                                     validCol, physicalCol, dataBytes) &&
                "TCMP requires two compatible Tile sources");
         return;
@@ -565,7 +568,7 @@ void ValidateCompareSelectTepl(const BlockFuncPtr& block,
         ASSERT(priorSources == 0 && inst->srcs.size() == 2 &&
                inst->dsts.size() == 1 &&
                inst->dsts[0]->size >= predicateBytes &&
-               IsCompatibleDataTile(inst->srcs[1], block->dataType, validRow,
+               IsCompatibleOperationDataTile(inst->srcs[1], block->dataType, validRow,
                                     validCol, physicalCol, dataBytes) &&
                "TCMPS requires one compatible Tile source");
         return;
@@ -582,18 +585,20 @@ void ValidateCompareSelectTepl(const BlockFuncPtr& block,
         return;
     }
     if (priorSources == 0) {
+        // 问题14:TSEL 就地首拍 true 源可能是 reinterpret 过的 tile,同上用 carrier-width 判定。
         ASSERT(inst->srcs.size() == 3 && inst->dsts.empty() &&
                IsCompatibleLogicalTile(inst->srcs[1], validRow, validCol,
                                        physicalCol) &&
-               IsCompatibleDataTile(inst->srcs[2], block->dataType, validRow,
+               IsCompatibleOperationDataTile(inst->srcs[2], block->dataType, validRow,
                                     validCol, physicalCol, dataBytes) &&
                "select first B.IOT requires mask then true/source Tile");
         return;
     }
+    // 问题14:TSEL 就地次拍 false 源(prior-dst,recip_u16)同为 reinterpret tile。
     ASSERT(block->tileOp == TileOp::TSEL && priorSources == 2 &&
            inst->srcs.size() == 2 && inst->dsts.size() == 1 &&
            inst->dsts[0]->size >= dataBytes &&
-           IsCompatibleDataTile(inst->srcs[1], block->dataType, validRow,
+           IsCompatibleOperationDataTile(inst->srcs[1], block->dataType, validRow,
                                 validCol, physicalCol, dataBytes) &&
            "TSEL second B.IOT requires one compatible false-source Tile");
 }
diff --git a/emulator/engine/CubeEngine.cpp b/emulator/engine/CubeEngine.cpp
index fbb202d4..34171b6c 100644
--- a/emulator/engine/CubeEngine.cpp
+++ b/emulator/engine/CubeEngine.cpp
@@ -945,6 +945,20 @@ void SoftCore::DataFormatCvt(std::vector<uint64_t>& dst, superScalar::DataType s
         return;
     }
 
+    if (dstType == DataType::SF8) {
+        // [reactive port of ad288c24 / 52f56d5 (#253) 被 930d9981 连坐删的 e8m0 转换;
+        //  codex/pr-0.58.4 未含。见 ISSUE_fp4_pack_tcvt_regression / 问题15]
+        // PTO-TCVT-E8M0-PROFILE-001: FP16/BF16/FP32 -> E8M0 code(MX 每行共享 scale
+        // 落盘的必经手法)。shared 已是 2^k,RNE 下精确。
+        for (uint64_t &raw : dst) {
+            raw = superScalar::Calculate::ConvertFloatToE8M0(
+                raw, OpCvtType(srcType),
+                superScalar::Calculate::PtoRoundingMode::RNE,
+                /*saturating=*/false, nullptr);
+        }
+        return;
+    }
+
     for (uint64_t i = 0; i < dst.size(); i++) {
         dst[i] = superScalar::Calculate::ConvertAggre(dst[i], OpCvtType(srcType), OpCvtType(dstType), getFRM());
     }
diff --git a/emulator/main.cpp b/emulator/main.cpp
index f927defb..139448e9 100644
--- a/emulator/main.cpp
+++ b/emulator/main.cpp
@@ -114,6 +114,15 @@ int main(int argc, char *argv[])
     core->hostedGroupRuntime =
         hostedSupport == HostedMultiPeSupport::SUPPORTED;
     core->directBootSyscallAbi = !addrRet.hosted_runtime;
+    // [TEMP / 本地临时·问题23] res_check ELF 被 HasHostedRuntime 误判为 hosted,读 A7
+    // (垃圾)而非 direct-boot 的 X1(真号)→ Bad Syscall。env=1 强制 direct-boot ABI。
+    // 纯本地(QEMU 不可用时跑 res_check 精度流程),默认不改行为。
+    if (const char *e = getenv("GFRUN_FORCE_DIRECTBOOT_ABI")) {
+        // =1 强制 direct-boot(读 X1);=0 强制 hosted(读 A7)。实测(WV-DIAG):新 clang
+        // (0f878a8)把 syscall 号放 X1,res_check ELF 又被 HasHostedRuntime 误判为 hosted
+        // (默认读 A7=垃圾→Bad Syscall),故新 combo 跑 res_check 必须 =1 强制读 X1。
+        core->directBootSyscallAbi = (e[0] == '1');
+    }
 
     const uint64_t stackTop = addrRet.sp_addr + sizeof(uint64_t);
     if (addrRet.sp_addr != static_cast<uint64_t>(-1) &&
diff --git a/isa/calculate/FloatPointUtils.cpp b/isa/calculate/FloatPointUtils.cpp
index 6bada98d..f42e6638 100644
--- a/isa/calculate/FloatPointUtils.cpp
+++ b/isa/calculate/FloatPointUtils.cpp
@@ -1697,6 +1697,193 @@ static void InitConvertMapInt(std::map<std::pair<OPConvertType, OPConvertType>,
                                 };
 }
 
+// ReferenceE8M0HighestSetBit (pto-spec asl/arch/profile/e8m0-conversion.asl):
+// the highest '1' bit position of an integer significand, 0-based.
+static int E8M0HighestSetBit(uint64_t significand)
+{
+    assert(significand != 0);
+    int highest = 0;
+    for (int position = 0; position < 64; ++position) {
+        if (((significand >> position) & UINT64_C(1)) != 0) {
+            highest = position;
+        }
+    }
+    return highest;
+}
+
+// ReferenceE8M0RoundExponent: round the base-two exponent of the exact value
+// UInt(significand) * 2^exponent to an integer under the selected PTO mode.
+// Exact powers of two are always exact; the midpoint is detected by comparing
+// significand^2 against 2^(2*highest+1) as in the ASL.
+static std::pair<int, bool> E8M0RoundExponent(uint64_t significand, int exponent,
+                                              PtoRoundingMode mode)
+{
+    const int highest = E8M0HighestSetBit(significand);
+    const int floorCandidate = exponent + highest;
+    assert(floorCandidate >= -149 && floorCandidate <= 127);
+    const int floorExponent = floorCandidate;
+    const bool exactPower = significand == (UINT64_C(1) << highest);
+    if (exactPower) {
+        return {floorExponent, true};
+    }
+
+    const int ceilingExponent = floorExponent + 1;
+    switch (mode) {
+        case PtoRoundingMode::RTM:
+            return {floorExponent, false};
+        case PtoRoundingMode::RTP:
+            return {ceilingExponent, false};
+        case PtoRoundingMode::RTZ:
+            if (floorExponent < 0) {
+                return {ceilingExponent, false};
+            }
+            return {floorExponent, false};
+        case PtoRoundingMode::RTO:
+            if (floorExponent % 2 != 0) {
+                return {floorExponent, false};
+            }
+            return {ceilingExponent, false};
+        default:
+            break;
+    }
+
+    // RNE, RNA, and RHB compare against the exact midpoint boundary.
+    const unsigned __int128 square =
+        static_cast<unsigned __int128>(significand) * significand;
+    const unsigned __int128 boundary =
+        static_cast<unsigned __int128>(UINT64_C(1)) << (2 * highest + 1);
+    if (square < boundary) {
+        return {floorExponent, false};
+    }
+    if (square > boundary) {
+        return {ceilingExponent, false};
+    }
+    if (mode == PtoRoundingMode::RNE) {
+        if (floorExponent % 2 == 0) {
+            return {floorExponent, false};
+        }
+        return {ceilingExponent, false};
+    }
+    if (mode == PtoRoundingMode::RNA) {
+        if (floorExponent < 0) {
+            return {floorExponent, false};
+        }
+        return {ceilingExponent, false};
+    }
+    assert(mode == PtoRoundingMode::RHB);
+    return {ceilingExponent, false};
+}
+
+// ReferenceFloatToE8M0: the authoritative PTO profile conversion.  Source
+// decomposition follows TileNumericFiniteDecomposition for FP16/BF16/FP32, so
+// no host floating-point arithmetic participates in the result.
+uint64_t ConvertFloatToE8M0(uint64_t data, OPConvertType srcType,
+                            PtoRoundingMode mode, bool saturating,
+                            float_status* status)
+{
+    auto setFlags = [status](uint32_t flags) {
+        if (status != nullptr) {
+            status->float_exception_flags |= static_cast<uint8_t>(flags);
+        }
+    };
+
+    int sign = 0;
+    int exponentField = 0;
+    uint64_t fraction = 0;
+    int allOnesExponent = 0;
+    int subnormalExponent = 0;
+    int normalExponentBias = 0;
+    int implicitBit = 0;
+    switch (srcType) {
+        case OPConvertType::OPCVT_FP32:
+            sign = static_cast<int>((data >> 31) & UINT64_C(1));
+            exponentField = static_cast<int>((data >> 23) & 0xffu);
+            fraction = data & 0x7fffffu;
+            allOnesExponent = 0xff;
+            subnormalExponent = -149;
+            normalExponentBias = 150;
+            implicitBit = 23;
+            break;
+        case OPConvertType::OPCVT_FP16:
+            sign = static_cast<int>((data >> 15) & UINT64_C(1));
+            exponentField = static_cast<int>((data >> 10) & 0x1fu);
+            fraction = data & 0x3ffu;
+            allOnesExponent = 0x1f;
+            subnormalExponent = -24;
+            normalExponentBias = 25;
+            implicitBit = 10;
+            break;
+        case OPConvertType::OPCVT_BF16:
+            sign = static_cast<int>((data >> 15) & UINT64_C(1));
+            exponentField = static_cast<int>((data >> 7) & 0xffu);
+            fraction = data & 0x7fu;
+            allOnesExponent = 0xff;
+            subnormalExponent = -133;
+            normalExponentBias = 134;
+            implicitBit = 7;
+            break;
+        default:
+            // HardwareTCVTE8M0SourceTypeSupported accepts only these sources.
+            assert(0 && "TCVT to E8M0 accepts only FP16, BF16, and FP32 sources");
+            return 0xff;
+    }
+
+    if (exponentField == allOnesExponent) {
+        if (fraction != 0 || sign != 0) {
+            // NaN or negative infinity: 0xFF with NV.
+            setFlags(float_flag_invalid);
+            return 0xff;
+        }
+        // Positive infinity follows the overflow rule: NX|OF, finite endpoint
+        // 0xFE when Sat is one.
+        setFlags(float_flag_inexact | float_flag_overflow);
+        return saturating ? 0xfe : 0xff;
+    }
+    if (exponentField == 0 && fraction == 0) {
+        // Positive or negative zero: 0xFF with NV.
+        setFlags(float_flag_invalid);
+        return 0xff;
+    }
+    if (sign != 0) {
+        // Negative finite values: 0xFF with NV.
+        setFlags(float_flag_invalid);
+        return 0xff;
+    }
+
+    const uint64_t significand =
+        exponentField == 0 ? fraction
+                           : (UINT64_C(1) << implicitBit) | fraction;
+    const int exponent = exponentField == 0
+        ? subnormalExponent
+        : exponentField - normalExponentBias;
+    assert(significand != 0);
+
+    const int highest = E8M0HighestSetBit(significand);
+    const int floorCandidate = exponent + highest;
+    assert(floorCandidate >= -149 && floorCandidate <= 127);
+    const int floorExponent = floorCandidate;
+    const bool exactPower = significand == (UINT64_C(1) << highest);
+    if (floorExponent < -127) {
+        // Finite values below 2^-127: NX|UF, finite endpoint 0x00 when Sat
+        // is one.
+        setFlags(float_flag_inexact | float_flag_underflow);
+        return saturating ? 0x00 : 0xff;
+    }
+    if (floorExponent == 127 && !exactPower) {
+        // Finite values above 2^127: NX|OF, finite endpoint 0xFE when Sat is
+        // one.
+        setFlags(float_flag_inexact | float_flag_overflow);
+        return saturating ? 0xfe : 0xff;
+    }
+
+    const auto rounded = E8M0RoundExponent(significand, exponent, mode);
+    assert(rounded.first >= -127 && rounded.first <= 127);
+    if (!rounded.second) {
+        setFlags(float_flag_inexact);
+    }
+    return static_cast<uint64_t>(rounded.first + 127);
+}
+
 uint64_t ConvertAggre(uint64_t data, OPConvertType from, OPConvertType to, uint32_t FRM, bool needCheck)
 {
     static std::map<std::pair<OPConvertType, OPConvertType>, std::function<uint64_t(uint64_t, FloatRoundMode)>> funcMap;
diff --git a/isa/calculate/FloatPointUtils.h b/isa/calculate/FloatPointUtils.h
index 32b88645..4df16f11 100644
--- a/isa/calculate/FloatPointUtils.h
+++ b/isa/calculate/FloatPointUtils.h
@@ -18,6 +18,28 @@ bool IsSNaN(OPConvertType cvt,uint64_t data);
 bool IsNaNOrSNan(OPConvertType cvt,uint64_t data);
 
 FloatRoundMode FRM2FloatRoundMode(uint32_t FRM);
+// PTO NumericRoundingMode (pto-spec asl/arch/data-types/rounding.asl).  Kept
+// separate from softfloat's FloatRoundMode so that E8M0 exponent rounding can
+// honor RHB exactly (round-half-bias rounds the midpoint toward the ceiling).
+enum class PtoRoundingMode {
+    RNE = 0,
+    RTM = 1,
+    RTP = 2,
+    RTZ = 3,
+    RNA = 4,
+    RTO = 5,
+    RHB = 6,
+};
+
+// PTO-TCVT-E8M0-PROFILE-001 (pto-spec asl/arch/profile/e8m0-conversion.asl):
+// convert an FP16/BF16/FP32 carrier to an E8M0 code.  Zero, negative values,
+// and NaNs produce 0xFF with NV; positive infinity and finite range
+// overflow/underflow produce 0xFF (or the finite endpoint when saturating)
+// with NX|OF / NX|UF.  Optional softfloat flags are accumulated on `status`.
+uint64_t ConvertFloatToE8M0(uint64_t data, OPConvertType srcType,
+                            PtoRoundingMode mode, bool saturating,
+                            struct float_status* status = nullptr);
+
 uint64_t ConvertAggre(uint64_t data, OPConvertType from, OPConvertType to, uint32_t FRM, bool needCheck = true);
 uint64_t Fadd(OPConvertType cvt, uint64_t data1, uint64_t data2, FloatRoundMode frm);
 uint64_t Fsub(OPConvertType cvt, uint64_t data1, uint64_t data2, FloatRoundMode frm);
```
