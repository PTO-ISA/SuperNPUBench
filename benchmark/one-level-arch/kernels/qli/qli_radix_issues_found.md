# QLI radix-select 过程中发现的可提 ISSUE 汇总（2026-08-24）

> 状态：待用户确认后提交。均为 SuperScalarModel（仿真器/ISA 层）或
> Linx-TileOP-API 侧问题，非 QLI 算子自身问题。

---

## Issue R1: THISTOGRAM ByteId 位域解码错误（Byte0/1/2 恒解码为 Byte3）

**组件**: SuperScalarModel — `isa/Block.cpp:716`（HandleBDATR）
**版本**: a68dba29 / v058 60fea39f

### 现象
工具链生成 THISTOGRAM B.DATR：
- Byte0 → `0x19901023`
- Byte1 → `0x19941023`
- Byte2 → `0x19981023`
- Byte3 → `0x199c1023`

ByteId 编码在 **bits[19:18]**（LLVM LinxV5InstrInfo.td:1581 `let Inst{19-18}=ByteId`）。
但解码器 `HandleBDATR` 从 `inst.srcs[SRC5_IDX]` 读 selectedByte，而 decode 表
`%padValue_27_28` 绑定 bits[28:27]（该位恒为 3）。**结果所有 ByteId 都解码成 Byte3**。

### 影响
THISTOGRAM 只能统计最高字节；Byte2/Byte1/Byte0 统计与 Idx 前缀收窄全部失效。
P3 单轮 Byte3 用法未暴露；多轮 radix-select（radix 直方图）被阻塞。

### 根因
decode 表 `B_DATR %padValue_27_28` 与工具链 `ByteId` 编码位不一致。
ASL 规范（cmd_pipe_as.md:248）称 ByteId 复用 CMode 字段，也与工具链不一致——
存在"规范/解码/编码"三方位的歧义。

### 修复（本地）
```cpp
// isa/Block.cpp HandleBDATR, THISTOGRAM 分支:
blockAttr->selectedByte = static_cast<uint8_t>((inst.binary >> 18u) & 0x3u);
```

### 建议
上游确认 ByteId 的官方位域（bits[19:18] 或 padValue/CMode 复用），
统一 decode 表与 ASL 文档。

### 附带影响（需同步更新测试）
`tests/isa_test/main.cpp:285` `DecodeHistogramByte3UsesPadUnionField`
使用 `0x19901023`（Bits28:27=3 = padValue=Byte3 的"旧 canonical"编码）
并 `EXPECT_EQ(selectedByte, 3u)`。按工具链真实编码（ByteId 在 bits[19:18]），
`0x19901023` 的 bits[19:18]=0，应解码为 **Byte0**（selectedByte=0）。
R1 修复后该测试断言会失败——测试假设的"ByteId 在 padValue 字段"与工具链
实际编码（bits[19:18]）不一致。上游正式修复时需**同步更新测试**为一个
真实工具链 Byte3 编码（如 `0x199c1023`），或采用两字段兼容逻辑。

---

## Issue R2: TROWEXPAND 广播输出的 tileInfo 与 TCMP 校验不兼容

**组件**: SuperScalarModel — `isa/Block.cpp`（TROWEXPAND dst tileInfo）
**版本**: a68dba29 / v058 60fea39f

### 现象
`TROWEXPAND(mxbc, mx)` 中 mx 为 `[1,32] valid[1,1]`，mxbc 声明为 `[1,2048]`。
工具链 asm 的 B.DIM lb0/lb1/lb2 取自 **src(mx)**（1/1/32），
`UpdateDstTileInfo`/执行器据此发布 dst tileInfo，导致后续
`TCMP(ismax, mv, mxbc)` 校验期望 `[1,2048] validRow=1 validCol=2048 col=2048`，
实际 mxbc tileInfo 为 `validCol=1, col=32` → `IsCompatibleDataTile` 失败。

### 影响
任何"reduce 结果 → TROWEXPAND 广播 → 再消费"的链式用法崩在 TCMP/TCMP 系校验。
P2/P3 未用 TROWEXPAND 广播（直接用 TROWEXPANDMUL 等二元 op），故未暴露；
radix pop（tile argmax 消零）触发。

### 规避（本地 QLI 侧）
改用 `TSTORE(mx→标量 GM) → TEXPANDS(标量→[1,2048])` 广播。

### 建议
TROWEXPAND dst 的 tileInfo 应按 **dst 声明形状** 发布（或至少
lb0/lb1 用 dst valid），而不是继承 src。

---

## Issue R3: Tile 作函数参数时 LinxV5 后端生成错误编码

**组件**: Linx-TileOP-API / LLVM LinxV5 后端
**版本**: TileOP-API 72f8255 / LLVM eb64de8afcbd

### 现象
把 tile（如 `TKey<2048>& mv`）作为普通 C++ 函数参数传入并在函数内使用，
后端为其生成 `TLOAD S64 [1,1024]`（8KB）甚至改 src dtype 为 INT64/S64，
导致 TCMP 源类型不匹配（UINT32 vs INT64）。

复现：`RadixPopN(TKey<2048>& mv, ...)` 函数版（已改为宏内联规避）；
Spike 用同函数内局部 tile 则正常。

### 影响
"tile 作为形参"的模板函数不可靠，编译静默生成错误编码，gfrun 校验崩溃。

### 规避（本地 QLI 侧）
pop/eq 逻辑全部内联为宏（`QLI_RADIX_POP_N` 等），tile 均为局部变量。

### 建议
上游检查 LinxV5 对 tile 类型函数形参的 lowering（疑似把 tile 引用当
内存对象处理，生成 S64 8KB TLOAD）。

---

## Issue R4（备选）: TCMPS 校验不允许 UINT32（B2 类）

**组件**: SuperScalarModel — `AccumulateBlockInfo.cpp:277` IsCompareSelectTeplDataType
**现象**: `TCMPS` dtype 允许列表为 INT32/FP32/FP16/UINT16/INT16，无 UINT32。
**影响**: 无法用 `TCMPS<GT>(key, (uint32)kth)` 单源标量比较（radix 需要）。
**本地**: 本轮未改（改用 TCMP + TEXPANDS kth 广播 tile），但 bucket 版曾需。
**建议**: 与 B2 一致放宽 UINT32（若上游决定 TCMPS 支持 UINT32 域）。

---

## 汇总建议优先级

| # | 组件 | 类别 | 严重度 | 建议动作 |
|---|---|---|---|---|
| R1 | SuperScalarModel ISA decode | Bug | 高（阻塞多轮 radix/THISTOGRAM 全功能） | 上游确认 ByteId 位域并修 |
| R2 | SuperScalarModel TROWEXPAND | Bug | 中（阻塞 TROWEXPAND 广播链式用法） | 修 dst tileInfo 发布 |
| R3 | LLVM LinxV5 后端 | Bug（编译器） | 高（tile 作形参普遍不可靠） | 上游修 lowering |
| R4 | SuperScalarModel TCMPS | 门禁放宽 | 低 | 视 TCMPS UINT32 支持而定 |

*Generated 2026-08-24 · 来源：QLI qli_topk_radix 实现过程中实测定位*