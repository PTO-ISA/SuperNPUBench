# 窄化 TCVT（bf16→e8m0）目的物理 row 被 128B 最小 tile 档撑翻倍 → TCVT 形状契约断言崩溃

## 摘要

一个 **1 字节 e8m0**（或任何 1 字节窄类型）的**列向量** tile `[TileM, 1]`（每行 1 个 scale 值）逻辑上只有
`TileM` 字节，但 PTO TSize 分档**最小就是 128B**——工具链 `round_capacity` 把它向上取到 128B 档并编进
B.IOT 的 `sizeClass`。emulator 解码出 `dst->size = 128`，随后用 `row = dst->size / (physicalCol × elemBytes)`
反推物理 row，对 1 字节类型得 `128 / (1×1) = 128`（而同形状 bf16 源得 `128 / (1×2) = 64`）。
`TCVT` 的 `ValidateOperandContract` 要求 src/dst **物理 row 逐位相等**，`64 ≠ 128` → 断言崩溃。

**关键点：这不是某一侧单独的 bug，而是编译器与 model 对「sub-128B 窄 tile」这一场景的处理方案未对齐。**
编译器把不足 128B 的窄 tile 向上取到 128B 最小档、并把该档编进 `sizeClass`；model 收到后用「档位字节数 ÷
字节宽」反推物理 row。两侧各自都自洽——编译器发的指令合法、`validRow` 两侧都是 64——但**对「被最小档撑大的
字节数该如何还原回物理 row」缺一条共同约定**：编译器语义里 128B 只是分配下限、逻辑 row 仍是 64，model 却把
128B 全部当有效数据反推成 row=128。跨 dtype 收窄的 TCVT 里（elemBytes 改变、同一逻辑形状的物理 row 随字节宽
变化），这个未对齐就暴露成 `row 64≠128` 的契约断言。

## 环境

| 项 | 值 |
|---|---|
| 工具链 | `clang 15.0.4 (linx64v5-musl-local adcb879481d8)`；TileOP-API `f94bc12`、llvm-project `adcb879` |
| 模型（gfrun） | `SuperScalarModel d8903938`（`Tag_0817-289-gd8903938`，"fix(gfrun): align reduce expand dtype gates"） |
| Bench 分支 | `dmxq-ops-20260828`（tag `ops-20260828` + 38 commit） |
| kernel | `dynamic_mx_quant_tail_ocp_fp8.hpp`（固定 SPMD 4-PE，half in / e4m3 out / e8m0 scale，BlockSize=32） |
| 触发前提 | reduce 输出列向量链已迁移 physical `Cols=1`（见「背景」），scale pass 末端 `TCVT(bf16→e8m0)` |
| 复现无需 | 4-PE / boxed 尾块 / golden —— full-tile（`M % (kPeNum×TileM) == 0`）单块即崩 |

## 背景（本 issue 如何暴露）

`d8903938` 的 rowReduce 分支把 reduce 输出物理 col **无条件强制为 1**（`Block.cpp:2069`
`const uint64_t stride = 1;`）。为对齐该语义，`tail_ocp_fp8` 的 reduce 输出及其下游列向量 tile
（`t_hb/t_bfb/t_e8b/t_fb`）已声明成 physical `Cols=1`（「physical=1 迁移」，见
`ISSUE_reduce_output_stride_tail.md` 附录）。迁移后 `reduce→TCVT`（`half→fp32`）已通过；
`ValidateScalarLogicalTepl` 的严格 dtype 检查也已反应式移植放宽（`3739068c`：dtype→位宽，接受
`reinterpret_tile` 视图）。**本 issue 是这两步之后前移暴露的第三个、独立的崩点**，且**仅 kernel 侧无解**。

## 根因（完整链条，两侧指实）

scale pass 末端把每行 shared-scale 从 bf16 转成 e8m0 落盘：
`… → TMULS(shared_bf) → TCVT(scale_e8m0, shared_bf) → TSTORE`。崩在 `TCVT(bf16 → e8m0)`。

### 实测 src/dst 描述符（gfrun 断言前打印）

```
TCVT#1 half→fp32 :  src[row=64 col=1 vr=64 vc=1 sz=128]   dst[row=64  col=1 vr=64 vc=1 sz=256]   ✓
TCVT#2 fp32→bf16 :  src[row=64 col=1 vr=64 vc=1 sz=256]   dst[row=64  col=1 vr=64 vc=1 sz=128]   ✓
TCVT#3 bf16→e8m0 :  src[row=64 col=1 vr=64 vc=1 sz=128]   dst[row=128 col=1 vr=64 vc=1 sz=128]   ✗
```

唯一不等项 = **物理 `row`：源 64、目的 128**。`validRow` 两侧都对（64），`col/validCol` 也对。

### 从填充到崩溃的逐行链条

| 步 | 侧 | 位置 | e8m0 `[64,1]` 取值 |
|---|---|---|---|
| ① 原始逻辑字节 | 工具链头 | `pto_tile.hpp:894` `kBytes = (Rows*Cols*bits+7)/8` | `64*1*8/8 = 64` |
| ② **向上取到 128B 最小档** | 工具链头 | `pto_tile.hpp:861-869` `round_capacity`（`int capacity = 128; while (capacity < bytes) capacity *= 2;`） | `round_capacity(64) = 128` |
| ③ TSize 码 | 工具链头 | `pto_tile.hpp:929` `TilesizeCode`（`LogicalTileBytes==128 → __tilesize_128B`） | `__tilesize_128B = 1` |
| ④ 编进 B.IOT 立即数 | 工具链 codegen | B.IOT `sizeClass`（SRC0 UIMM） | `1` |
| ⑤ 解码 `dst->size` | 模型 | `isa/calculate/blockArgs/BlockArgs.cpp:24` `128U * (1U << (sizeClass-1))` | `128*2^0 = 128` |
| ⑥ 反推物理 row | 模型 | `isa/Block.cpp:2106` `dst->tileInfo->row = dst->size / (dstPhysicalCol * elemBytes)` | `128/(1×1) = 128` |
| ⑦ 契约断言 | 模型 | `isa/Block.cpp:1653` `srcTile[0]->tileInfo->row == dstTile[0]->tileInfo->row` | `64 ≠ 128` → 崩 |

- 步②的 128B 是**硬件 tile 最小档**（`capacity` 从 128 起步）。half/bf16 `[64,1]`=128B、fp32=256B 都正好
  落档、不被抬；**唯独 1 字节 e8m0 的 64B 被抬到 128B**。
- 步⑥的 `physicalCol` = 1 = e8m0 目的 tile 的 `dst::Cols`。TCVT 发 `lb2 = dst::Cols`（`template_asm.hpp:136
  TCVT_T` → `159: "B.DIM zero, %c7, ->lb2"`，`%7 = tile_shape_out::Cols`），模型取 `physicalCol = lb2`
  （`Block.cpp:1897-1902`：TCVT 且 `lb2 != 0` 时 `physicalCol = lb2`）。e8m0 目的 tile 的 `Cols` 被锁死为 1
  （见「为什么 kernel 侧无解」），故 `lb2 = 1`、`physicalCol = 1`，row 由 `size/(1×elemBytes)` 决定，1 字节
  类型相对 2 字节源翻倍。
- **填充在编译器侧、还原在 model 侧**：`dst->size=128` 是 `BlockArgs.cpp:24` 忠实解码编译器编进的
  `sizeClass=1`，真正把 64B 抬到 128B 的是编译器 `pto_tile.hpp:861 round_capacity`；model 只是拿到 128B 后
  按 `size÷(col×bytes)` 还原。两步各自都对，缺的是「128B 里哪部分是最小档填充、哪部分是真实 row」这条跨侧约定。

## 最小复现

探针 `test/kernel/multi_thread/quant/dynamic_mx_quant/src/tcvt_reduce_shape_probe.cpp`（`TYPE=TCVT_REDUCE_SHAPE_PROBE`）：
`TLOAD half → TABS → TROWMAX(→[R,1]) → TCVT(→fp32) → TANDS → TCVT(→bf16) → TCVT(→e8m0) → TSTORE`，
全部列向量 tile physical `Cols=1`。

```bash
export COMPILER_DIR=<linx toolchain>/bin
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant
make TESTCASE=dynamic_mx_quant TYPE=TCVT_REDUCE_SHAPE_PROBE diss     # EXIT=0（编译期无阻）
<gfrun> -f <...>_tcvt_reduce_shape_probe.elf                          # -> Block.cpp:1653 崩

# 分步验证（逐步加 TCVT 定位）：
#   仅 reduce→TCVT(fp32)                        -> R2=0（过）
#   + TCVT(fp32→bf16) + 夹层 TANDS              -> R2=0（过）
#   + TCVT(bf16→e8m0)                           -> Block.cpp:1653 崩（本 issue）
```

崩溢出仅由链末的 `bf16→e8m0` 引入，前两个 TCVT（16b/32b，`size/elemBytes` 反推 row 恰=64）均通过。

## 影响

- **全 mx_quant 家族共有**：凡「每行 1 值的 scale 以 1 字节窄类型（e8m0）经 `TCVT` 从 bf16/fp32 收窄」的
  kernel 都会中招——只要该 e8m0 列向量物理字节 < 128B（即 `TileM < 128`），就被 `round_capacity` 抬到
  128B、row 反推翻倍。本 kernel `TileM = 64`（BS=32，受 fp32 中间量 ≤8KB 约束）恒触发。
- 与 `ISSUE_reduce_output_stride_tail.md` 是**同一类未对齐的另一面**：ISA 编码只携带档位字节数、不含独立的
  物理 row/col 字段，model 一律 `size ÷ (col×bytes)` 反推——前者因 reduce 强制 col=1 而与编译器声明失配，
  本 issue 因 1 字节窄 tile 被最小档填充而失配。二者根都在「物理几何需由字节数反推、而填充与 col 约定两侧
  未统一」。

## 为什么 kernel 侧无解

1. reduce 在 `d8903938` 被无条件强制 **col=1**（`Block.cpp:2069` `const uint64_t stride = 1;`），故紧邻的
   reduce→TCVT 目的 tile（max_f）**必须声明 `Cols=1`** 才能与运行期 src col=1 匹配（否则 TCVT 发 `lb2=Cols`、
   `physicalCol=Cols`，与 src col=1 失配，直接崩 reduce→TCVT）。
2. 编译期 `TileLogicalShapeMatch`（`template_asm.hpp:143` `tile_shape_out::Cols == tile_shape_in::Cols`）要求
   **每个 TCVT 的 src/dst 物理 `Cols` 逐位相等**，于是 `Cols=1` 从 max_f 沿链 `max_bf → shared_bf → scale_e8m0`
   **强制传播**——e8m0 目的 tile 的 `Cols` 无从取到 >1。
3. `Cols=1` ⇒ TCVT 发 `lb2=1` ⇒ 模型 `physicalCol=1`；e8m0 1 字节、`[TileM,1]`（`TileM<128`）物理必 <128B、被
   `round_capacity` 抬到 128B ⇒ `row = 128/(1×1) = 128` 恒翻倍。
4. 想把 e8m0 dst 声明成 `Cols=2`（则 `lb2=2`、`physicalCol=2`、`row=64` 本可匹配）——**但第 2 条编译期传播要求
   其 src `shared_bf` 也 `Cols=2`，逆推到 max_f 也 `Cols=2`，而第 1 条又要求 max_f `Cols=1`，冲突**；`Cols=2` 的
   max_f 在 reduce→TCVT 运行期即崩。故无法在保持 reduce→TCVT 通过的前提下把链上任何一环放宽到 `Cols>1`。

四条锁死：**只要最终要落每行 1 值的 e8m0 scale，1 字节列向量的 `Cols` 就被 reduce-col=1 与编译期形状传播共同
钉死为 1，物理 row 被 128B 最小档反推翻倍，撞上严格 TCVT 契约，kernel 无论怎么声明都过不了**。

## 对齐方向（两侧建立共同约定即可，任择其一）

问题本质是「128B 最小档填充后、物理 row 如何还原」缺一条编译器与 model 共享的约定。以下任一处补齐该约定都能
消除断言，不预设哪一侧「错」：

### 方向一：model 按逻辑不变量校验（改动最小）

`TCVT` 的 `ValidateOperandContract`（`Block.cpp:1650-1660`）改为**只比较 `validRow / validCol / layout`**，
去掉物理 `row == row` / `col == col` 两条 conjunct——物理 row/col 是随 dtype 字节宽变化的 `size÷(col×bytes)`
派生量，跨收窄/展宽 TCVT 天然不等，本就不适合作为这条契约的判据。与
`ISSUE_tcvt_fp4_shape_contract.md` / `ISSUE_reduce_output_stride_tail.md` 是同一处，一处放宽同时覆盖
reduce→TCVT 与本 e8m0 收窄。

### 方向二：model 还原 row 时扣除最小档填充

保留物理校验，改 `Block.cpp:2106` 的 row 反推：当 `dst->size` 因最小档填充而 `> validRow×physicalCol×
elemBytes` 时，物理 row 取 `validRow`（或从 emit 的 lb1 取真实物理行）而非 `size/(col×bytes)`。更外科，
但只覆盖本 issue。

### 方向三：编译器额外传递真实逻辑 row

若希望 model 侧不改校验，可由编译器在 B.IOT（或伴随字段）额外编码真实逻辑 row / 有效字节，使 model 能区分
「最小档填充」与「真实数据」。代价是 ISA 编码扩展。注意 `round_capacity` 的 128B 下限本身是硬件 TSize 最小档
（`__tilesize_128B`），属分配语义、不宜去除——要补的是「填充量」的跨侧传递，而非取消填充。

## 附：本仓库内相关文件

- 复现探针：`test/kernel/multi_thread/quant/dynamic_mx_quant/src/tcvt_reduce_shape_probe.cpp`（`TYPE=TCVT_REDUCE_SHAPE_PROBE`）
- 受影响 kernel：`kernels/multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8.hpp`
- 同类/相关：`ISSUE_reduce_output_stride_tail.md`（reduce 强制 col=1 的另一面）、
  `ISSUE_tcvt_fp4_shape_contract.md`（同一 TCVT 物理形状契约在 fp4 打包上的表现）、
  `RECORD.md` 问题16/22
