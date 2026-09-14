# [Issue] emulator `ValidateScalarLogicalTepl`：零指令 `reinterpret_tile` 的位重解释在运行期不可见，dtype 相等断言误杀

## 摘要

`dynamic_mx_quant` OCP 探针 kernel 用 v0.58 的 `reinterpret_tile<uint16_t>(max_bf)` 把一个
bf16 tile **零指令**重解释成 u16，再对其做 `TANDS`（按位清尾数，只留指数位），随后仍按 bf16
读回做 `TMULS`。编译与反汇编均正确，但 `gfrun` 执行时命中 emulator 运行期断言：

```
scalar logical TEPL source dtype/shape/stride is incompatible
AccumulateBlockInfo.cpp:440  ValidateScalarLogicalTepl
```

根因是**编译期 / 运行期不一致**：`reinterpret_tile` 只在编译期改操作数的静态 DType（于是 TANDS
指令编码 datatype=U16），**不发射任何指令**；而 emulator 给每个物理 tile 挂了一个运行期
`tileInfo->dataType` 标签，只由**产出它的指令**设置（TCVT → BF16）。断言 441 要求
`source->tileInfo->dataType == block->dataType`，即 `BF16 == U16` → 失败。缺陷所在仓
**`SuperScalarModel`（emulator 建模层）**，非 ISA、非工具链、非 kernel 逻辑。

## 环境

| 项 | 值 |
|---|---|
| 编译仓 | SuperNPUBench `benchmark/one-level-arch` |
| 执行仓 | SuperScalarModel（`bin/gfrun` 执行 elf） |
| 工具链 | `linx_blockisa_llvm_musl`，clang-15，target `linx64v5-linux-musl` |
| 复现根目录 | `benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant` |
| 触发 kernel | `kernels/multi_thread/quant/dynamic_mx_quant/probe_dynamic_mx_quant_tail_ocp_fp8.hpp`（`TYPE=PROBE_OCP_FP8`） |

### 各仓代码分支（复现基线）

| 仓库 | 分支 / HEAD | commit |
|---|---|---|
| SuperNPUBench（我的 fork，含探针） | `feat/dmxq-rel0812` (`github.com:ziyang-cheng/SuperNPUBench.git`)| `79d9140` |
| SuperScalarModel | `feat/pto-v058-adaptation` | `c3051e3a` |
| linx-toolchain-build | `main` | `e6a31ef` |
| └ Linx-TileOP-API | `feat/v058-reinterpret-cmpmode-backfill`（`github.com/ziyang-cheng/Linx-TileOP-API`） | `cb47f6d` |

> **基线说明**：
> - **SuperScalarModel**：`feat/pto-v058-adaptation @ c3051e3a`（tip，含 ADDTPC page-offset 系列
>   `430e7bbb`/`e2c8ad23`/`5adc5d8a`/`8baf731e`，musl crt startup 正常通过，执行流进到 kernel 首条
>   `TANDS` 触发本断言）。**缺陷在此基线原样存在**：`ValidateScalarLogicalTepl`
>   （`AccumulateBlockInfo.cpp`）的 `source->tileInfo->dataType == block->dataType` 严格相等校验未放松
>   （周围维度检查已用 `BytesOf`）。
> - **Linx-TileOP-API**：`https://github.com/ziyang-cheng/Linx-TileOP-API.git` 的
>   `feat/v058-reinterpret-cmpmode-backfill @ cb47f6d`，单个 checkout 即得完整复现基线（含本 issue
>   所用的 `reinterpret_tile`）。

## 复现命令

```bash
export COMPILER_DIR=.../linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=.../linx_blockisa_llvm_musl/sysroot/usr
cd benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant
make TESTCASE=dynamic_mx_quant TYPE=PROBE_OCP_FP8 diss   # 编译+反汇编，clean

SuperScalarModel/bin/gfrun -t 1 -f "$PWD/output/.../dynamic_mx_quant_probe_ocp_fp8.elf"
```

## 报错

```
B8  T0 BPC 0x1134e [TEPL] [FALL] [TANDS]
M38  TPC:0x1134e  BSTART.TEPL TANDS UINT16
...
gfrun: illegal instruction: ASSERTION FAILED:
  source && OperandTypeIsTile(source->type) && source->tileInfo &&
  source->tileInfo->dataType == block->dataType &&        // <-- BF16 != UINT16
  source->tileInfo->validRow == validRow && ...
  "scalar logical TEPL source dtype/shape/stride is incompatible"
  func ValidateScalarLogicalTepl, file .../AccumulateBlockInfo.cpp:440   (EXIT=1)
```

## 根因分析

### 反汇编：数据流本身正确（掩码没丢）

追踪三条 TEPL 块的 tilereg（同一个 512B 的 `t`，就地读写，中间无 copy）：

```
11328: TCVT, BF16   -> 11330: B.IOT t#1 ... ->t<512B>   max_bf = TCVT(max_h)
1134e: TANDS, U16    -> 1135c: B.IOT t#1 ... ->t<512B>   就地 AND 掩码, 写回同一 t
1136e: TMULS, BF16   -> 1137c: B.IOT t#1 ... ->t<512B>   读同一 t, 乘 RECIP_EMAX
```

`-D__linx` intrinsic 通过 `"=Tr"(dst.data())` / `"Tr"(src.data())` 绑定操作数，而
`reinterpret_tile` 返回的 view `.data()` 转发到源 tile 的**同一个 `data_` 寄存器**
（`common/pto_tile.hpp:1352`）。所以 TANDS 的 U16 掩码就地写进 `max_bf` 的寄存器，TMULS 读的
正是被掩码后的值 —— **抽象"对象身份 SSA"模型在 linx 后端不适用，共享 carrier 把值串起来了**。
结论：kernel 数据流无 bug。

### 缺陷在 emulator 的 per-tile 运行期 dtype 标签

- **编译期**：`reinterpret_tile<uint16_t>(max_bf)` 改操作数静态 DType → TANDS 指令
  `block->dataType = UINT16`。零指令，不产出任何 tile。
- **运行期**：emulator 每个物理 tile 有 `tileInfo->dataType`，由**最后一条产出它的指令**
  `block->dataType` 设置（`SoftCore.cpp:685` 一类赋值）。产出该 tile 的是 TCVT → 标签 BF16。
  reinterpret 没发指令 → **运行期标签永远停在 BF16**。
- **断言 440-445** 要求源 tile 运行期标签 `== block->dataType`：`BF16 == UINT16` → 失败。

即：位重解释的信息只活在**编译期的指令 datatype 字段**里，emulator 的**物理 tile 运行期标签**
接收不到它。这是编译期 bitcast 与运行期 tile 元数据的接口不一致。

### 关键澄清 1：424 的"逻辑op必须整数"是对的，必须保留

同函数更早的断言（`AccumulateBlockInfo.cpp:424`）：

```cpp
ASSERT(IsLogicalIntegerTeplDataType(block->dataType) &&
       "scalar logical TEPL tuple is not defined by PTO ISA v0.2");
```

`TANDS`/`TORS`/`TXORS` 这类逻辑op的 datatype 字段本就**只允许整数类型**。这正是为什么必须
先 `reinterpret_tile` 到 u16：把 datatype 合法化成整数、同时保 bit 不变。**因此不能直接
`TANDS(max_bf, max_bf, mask)` 跳过 reinterpret**——那会让 `block->dataType=BF16`，先挂在
424；且 bf16 tile-scalar 立即数会数值转换掉掩码 bit 并崩 LinxV5 后端（见
`ISSUE_rel0812_toolchain_defects` / 记忆 `reference_linx_bf16_scalar_backend_defects`）。

### 关键澄清 2：HBM 往返为何"能过"

旧规避（scratch-HBM 位重解释）里 `TSTORE(bf16) → TLOAD(uint16)` 的 `TLOAD` 是**一条真指令**，
它把物理 tile 的运行期标签重新盖成 U16。所以 HBM 往返无意中同时干了两件事：位重解释 + 重打标签。
它能过断言纯属副作用，代价是一次 HBM round-trip。零指令 reinterpret 只干位重解释，才暴露出
emulator 这条断言没建模 bitcast。

## 定性

- **属 emulator 建模缺陷（过严校验）**，非 ISA/工具链/kernel 缺陷。
- reinterpret_tile 是 v0.58 合法特性、工具链发射的 bit 正确、真实硬件按 datatype 字段当场解释
  bit 即可运行——硬件不存在"源 tile 必须与本op datatype 同历史标签"的持久约束。
- 断言把"本op如何解释 bit"（`block->dataType`）与"上一条产出此 tile 的 dtype"
  （`tileInfo->dataType`）强行划等号，从而禁掉一切"零指令 bitcast 后被异类型op消费"的合法用法。

## 修复建议（emulator 侧）

把 dtype **相等**约束放松为 **bit 宽相等**，保留 shape/stride/valid 校验不动。

`AccumulateBlockInfo.cpp:441`（`ValidateScalarLogicalTepl`）：

```cpp
// 原：
source->tileInfo->dataType == block->dataType &&
// 改：允许零指令 bitcast —— 源 tile 运行期 dtype 与本op datatype 只需位宽一致
BytesOf(source->tileInfo->dataType) == BytesOf(block->dataType) &&
```

同族相等断言（为一致性应同步评估，按需同改）：
- `:270` `IsCompatibleDataTile`（TLOAD/TSTORE/TMOV 兼容性）
- `:410` scalar-arith TEPL 源 dtype
- `:475` basic-binary TEPL 源 dtype

**保留不动**：424（逻辑op要整数）、435-439（维度/分配）、442-444（validRow/validCol/col/size）。
放松后 boxed valid-col=1、physical col、byteSize 等仍全量校验，仅解除 dtype **名义**相等。

### 备选（不改 emulator）

保留 HBM 往返规避（kernel 里已有注释块）：`TSTORE(bf16)→TLOAD(u16)→TANDS→TSTORE(u16)→TLOAD(bf16)`。
`TLOAD` 重打标签使断言通过，但每 block 引入 2 次 HBM round-trip，抵消 reinterpret 零指令的收益。

## 影响面

- 阻断 `probe_dynamic_mx_quant_tail_ocp_fp8` 的 gfrun 执行（OCP 指数位提取链）。
- 泛化影响：任何"`reinterpret_tile` 到整数域 → 逻辑op（`TANDS`/`TORS`/`TXORS`）→ 读回原域"
  的位操作 idiom，在 emulator 上都会被 440 误杀。这是 MX-quant OCP 公式（清尾数取 2^E_max）的
  核心手法。
