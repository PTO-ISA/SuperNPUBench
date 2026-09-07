# [gfrun][NA] SuperScalarModel 模型缺口（tileop-guard 第二轮 · 2026-09-03 09:29）

本 issue 汇总 TileOP-API v0.58 文档看护**第二轮**（放宽纯文档驱动、读工具链头订正签名后）
新发现的、**报错发生在仿真器 gfrun 且无法靠补文档解决**的模型缺口。接口按头文件正确写、编译通过，
但 gfrun 侧运行期拒绝。与首轮已提 issue 不重复；以问题为单位。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | PTO-ISA/SuperNPUBench | **PR #96** | https://github.com/PTO-ISA/SuperNPUBench/pull/96 |
| **SuperScalarModel(gfrun,本 issue 目标)** | LinxISA/SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` (762a72c34305f7f1df6964e7dfe202bd3e63a951) |
| Linx-TileOP-API(头) | LinxISA/Linx-TileOP-API | `linx` | `6f230c5` (6f230c598dd674d0007f6a0b6634ab4183c0ce48) |
| llvm-project(clang/lld) | LinxISA/llvm-project | `dev-llvm15_56` | `25677bb` (25677bb1a6c758d8867cc4e1b42acaf6626f9316) |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

> 工具链指纹（对应上表 llvm `25677bb`）：clang++ md5 `b6201631d2fdb77c6ad541c2c769460e`、
> gfrun md5 `04ca39ece7533eb35c805a4741996ebb`。

## 通用复现步骤

```bash
source microbenchmark/tileop-guard/env.sh   # COMPILER_DIR=linx clang / GFRUN=SuperScalarModel/bin/gfrun
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## gfrun-N1 · TCMP 处理器拒绝 reinterpret_tile 视图源（官方 reinterpret 修复在模型层遗漏 TCMP）

**涉及接口**：TCMP + reinterpret_tile。
**组件**：SuperScalarModel（TCMP 运行期校验）。

**问题**：`reinterpret_tile<int32_t>(fp32Tile)` 返回的视图（`ReinterpretedTileView`）作为 **TCMP 的源操作数**时，
**编译通过**（`TCMP<tile_out, tile_in>` 分离模板参，允许普通 predicate 输出 + 视图源），但 gfrun 在 TCMP
处理器崩溃。对照实验证明**差异就是视图源**：
- 同样两个 int32 视图喂 **TANDS** → 跑通 + 精度PASS（`misc/reinterpret_tile`，golden=abs）；
- 普通 int32 tile 喂 **TCMP** → 跑通 + 精度PASS（`vec/tcmp`）；
- int32 视图喂 **TCMP** → gfrun 崩。

即 TANDS 等 elementwise 处理器已接受 reinterpret 视图源，**唯独 TCMP 处理器的 `IsCompatibleDataTile`
不认视图源描述符**——官方 reinterpret 修复遗漏了 TCMP 这一路。

**复现**：`bash run_guard.sh misc reinterpret_tcmp`。

**错误信息**：
```
gfrun: illegal instruction: ASSERTION FAILED: priorSources == 0 && inst->srcs.size() == 3 &&
  inst->dsts.size() == 1 && inst->dsts[0]->size >= predicateBytes &&
  IsCompatibleDataTile(inst->srcs[1], block->dataType, validRow, validCol, physicalCol, dataBytes) &&
  IsCompatibleDataTile(inst->srcs[2], block->dataType, validRow, validCol, physicalCol, dataBytes) &&
  "TCMP requires two compatible Tile sources"
```

**附加说明**：崩在 TCMP 本身（早于任何 TSTORE）。最小复现即 `reinterpret_tcmp.cpp`：两 fp32 tile →
`reinterpret_tile<int32_t>` → `TCMP<GT>(mask, vA, vB)`。修好后该 case 转精度PASS（golden 已就绪：
`where(int_bits(a)>int_bits(b), tru, prior)`）。建议让 TCMP 处理器像 TANDS 一样接受 reinterpret 视图源描述符。

---

## gfrun-N2 · TEXTRACT 任何真实子抽取都被判非法（block 维取自源 valid 维）

**涉及接口**：TEXTRACT。
**组件**：SuperScalarModel（`ValidateV058SpecialTepl` 的 TEXTRACT 分支）＋工具链头 emit（见附加说明）。

**问题**：`TEXTRACT(dst, src, indexRow, indexCol)` 照头签名写、编译通过，但只要是**真实的子抽取**
（indexRow/indexCol > 0，或 dst 小于 src）就被 gfrun 判非法；仅 `offset=0 且 dst 与 src 同为全尺寸`的
**退化恒等拷贝**能过。

根因（模型 `emulator/engine/AccumulateBlockInfo.cpp` TEXTRACT 校验）：
```cpp
rowOffset + validRow > source->tileInfo->validRow ||   // -> bIsIllegal
colOffset + validCol > source->tileInfo->validCol ||
inst->dsts[0]->size < validRow * block->lb2 * BytesOf(block->dataType)
```
其中 `validRow=lb1`、`validCol=lb0` 是 block 维，而工具链头 `TEXTRACT` emit 把 **源 tile 的 ValidRow/ValidCol**
发成 lb1/lb0。于是 `validRow == source->tileInfo->validRow`，`rowOffset + validRow > validRow` ⟺ `rowOffset > 0`
恒成立 → 任何非零 offset 必非法；且 dst 尺寸校验也按源全尺寸算，dst（抽取区域）小于源即失败。

**复现**：`bash run_guard.sh sfu textract`（demo 用 src 16×32、dst 8×16、offset(4,8)）。

**错误信息**：
```
gfrun: illegal instruction at 0x0: illegal TEXTRACT operand or descriptor contract
```

**附加说明**：本质是 **block 维应表示「抽取区域(dst)维」而非「源维」**的契约不一致。修复方向二选一：
①工具链头 TEXTRACT emit 改发抽取区域(dst)的 valid 维作 lb0/lb1；②模型校验改用区域维（并允许区域 < 源）。
同族的 TINSERT 用相同 emit 却能运行（其模型校验无此 `offset+validRow>source` 分支），可作对照。

---

## gfrun-N3 · region TileArray + TASSEMBLY 只落 slot0，其余 slot 归零（拼接 producer 未补齐）

**涉及接口**：region::TileArray / TASSEMBLY（＋ region TCVT slot producer）。
**组件**：SuperScalarModel（region 拼接 producer 执行路径）。

**问题**：按 `docs/tileop-usage/range-modifiers.md` 的 "TileArray region API" 写法，把 4 个 `PM×FN=32×16`
的行主序 Fragment 分别经 `TCVT(dst[0][k], srck)` 写入 4 个 slot，再 `TASSEMBLY<Parent>` 组装成
`PM×PN=32×64` 的 parent。**编译通过、gfrun 跑到 `Reach the End of Benchmark` 不再崩**（早期
`raw tile spill source does not fit the carrier shape` 断言已消失），但 **parent 只有第 0 个 slot（列 0..15）
写对，slot1..3（列 16..63）整块为 0**。

**独立 golden 判据**：parent 按列拼接 4 片 → `ref[:, k*16:(k+1)*16] = fragment_k`。逐元素比对：
- **前 512 元素（fragment0 满 32×16）逐字节正确**（证明 golden 的列拼接布局无误）；
- **后 1536 元素（fragment1..3）全为 0.0**。

即 region 拼接 producer 只发布了首个 slot 的 payload，其余 slot 未落地。

**复现**：`bash run_guard.sh tlsu region_tilearray`。

**看护输出（golden.py 独立比对）**：
```
[region_tilearray] REGIONASM MISMATCH 1536/2048 first@16 got=0.0 ref=23.893508911132812
```
（`first@16` = parent[0,16] = fragment1[0,0]；期望 23.8935，实测 0。）

**附加说明**：gfrun 无 assert（不是非法指令，是**数据缺口**）。golden 已就绪并自洽（列拼接语义与
`test/tileop_api/src/TileArrayRegionAsm.cpp` 的 4×32×16→32×64 组装一致）；模型补齐 slot1..N 的
拼接 producer 后本 case 自动转精度正确。demo 已改 res_check（读 `in_a.bin`=4 片 block-major，dump
`out.bin`=32×64 parent），无需再改。

---

## gfrun-N4 · B.FPATR PreQuant 消费 FP19 scale 参数时丢弃缩放后 payload（量化输出恒=offset/0）

**涉及接口**：TMATMUL + `fixp::s8` / `fixp::scalar<QF322S8Pre>` / `fixp::s8(tile)`(VQF322S8Pre) /
`fixp::vector<VQF322F16Pre>` / `fixp::s8(desc).lrelu(fp19)`（即所有消费 **FP19 scale 参数**的 B.FPATR
PreQuant 模式）。
**组件**：SuperScalarModel（CUBE B.FPATR 矩阵后处理 PreQuant 缩放路径）。

**问题**：按 `docs/tileop-usage/.../matrix-postprocess.md` 写 `TMATMUL(out,a,b,fixp::s8(desc))`，
descriptor 含合法 FP19 scale（bits[31:13]）+ S9 offset（bits[45:37]）。**编译通过、gfrun 跑到
`Reach the End of Benchmark` 不崩**，但量化输出**丢弃了 `value·scale` 缩放后的 payload**——S8 模式
输出**恒等于 offset**，F16 模式输出**恒等于 0**，与累加器数值无关。

**根因定位（pto-spec `arch/profile/matrix-postprocess.asl` 规范契约）**：`MatrixPostQuantBaseWithFlags`
把 scale 与激活融合成单一 multiplier（`MatrixSelectedMultiplier`：正值→`FP19FiniteValue(quant_param[31:13])`
即 scale，负值在 LReLU/PReLU 下→slope），`activated = source_value · multiplier`，再
`MatrixQuantizedAffine(activated, 1.0, offset, S9, ...)`。实测模型把**该 multiplier 的 scale 分支当作 0**：
`activated = value·0 = 0`，故 S8 输出 = `round_S9(0)+offset = offset`，F16 输出 = `fp16(0) = 0`。

**关键隔离证据（同批 demo 精确对照，golden 均钉 pto-spec 语义）**：
- **s8_scalar / scalar_generic / s8_vector**（QF322S8Pre / VQF322S8Pre，scale=16.0 offset=5）：
  输出**逐元素恒=5**（993/1024 mismatch，唯 D≈0 的少数恰好对上）。
- **vquant_f16**（VQF322F16Pre，scale=16.0）：输出**逐元素恒=0**（1023/1024）。
- **lrelu**（QF322S8Pre + LReLU slope=8.0）：**负值分支（multiplier=slope）100% 正确**、
  **正值分支（multiplier=scale）恒=offset**（491/1024，正好约半——正负各半）。
- **prelu**（`fixp::f16()` convert，scale=1.0，**不消费 FP19 quant 参数** + PReLU slope=0.5）：
  **全 PASS**——正值·1.0、负值·slope 均正确。
- **rowmax_acc / group_max / chain / cscale**（keep_acc，**无 PreQuant scale**）：**全 PASS**。

即：**activation slope 乘子路径正确、offset 路径正确、无 quant 参数的 convert 正确，唯独 FP19 quant
scale 乘子被当作 0**。这与 tile 侧 **gfrun-5（TQUANT 忽略 mult/zp）**同属"量化 scale 参数未生效"缺陷族，
但发生在 CUBE B.FPATR matrix-postprocess 路径。

**复现**：
```bash
bash run_guard.sh fixp s8_scalar     # got≡5(offset)
bash run_guard.sh fixp vquant_f16    # got≡0
bash run_guard.sh fixp lrelu         # 负值对、正值≡offset
bash run_guard.sh fixp prelu         # PASS(无 quant 参数对照)
```

**看护输出（golden.py 独立比对）**：
```
[s8_scalar]  MQUANT-S8  MISMATCH 993/1024  first@0 got=5     ref=-24     (scale=16.0 off=5 tol=1)
[vquant_f16] MQUANT-F16 MISMATCH 1023/1024 first@0 got=-0.0  ref=-29.03  (scale=16.0)
[lrelu]      MQUANT-S8  MISMATCH 491/1024  first@9 got=5      ref=11      (scale=16.0 off=5 tol=1)
```

**附加说明**：gfrun 无 assert（数据缺口）。golden 依据 pto-spec `arch/data-types/fp19.asl`（FP19 解码）
+ `arch/profile/matrix-postprocess.asl`（量化/激活融合乘子 + S9 中间 + offset）+
`arch/profile/matrix-quantization.asl`（`MatrixQuantizedAffine`），已离线 round-trip 自测并通过
"scale 被忽略则 witness 触发"判别。模型让 FP19 quant scale 乘子生效后，5 个 case 自动转精度正确，
无需改 demo/golden。建议与 gfrun-5 一并排查量化 scale 参数的解码/应用路径。
