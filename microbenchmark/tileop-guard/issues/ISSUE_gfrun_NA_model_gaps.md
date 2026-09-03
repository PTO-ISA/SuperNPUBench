# [gfrun][NA] SuperScalarModel 模型未实现 / 模型侧契约拒绝

本 issue 汇总 TileOP-API v0.58 文档看护中,**报错发生在仿真器(gfrun)** 且**无法靠补文档解决**的问题:
接口按文档正确写、编译通过,但 gfrun 侧要么是**未实现桩**,要么在运行期拒绝按文档示例生成的
descriptor。文档类缺陷、编译器/后端类缺陷另行分别提 issue。

以问题为单位,每条含复现与实测错误信息。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | PTO-ISA/SuperNPUBench | `ops-20260828` | `d1604cf` |
| **SuperScalarModel(本 issue 目标)** | LinxISA/SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` |
| Linx-TileOP-API | LinxISA/Linx-TileOP-API | `linx` | `6f230c5` |
| llvm-project(linx clang/lld) | LinxISA/llvm-project | `dev-llvm15_56` | `25677bb` |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

> 看护 demo 代码见 **SuperNPUBench PR #96**:https://github.com/PTO-ISA/SuperNPUBench/pull/96
> 工具链指纹(2026-09-03,第三~五轮):clang++ md5 `b6201631d2fdb77c6ad541c2c769460e`、
> gfrun md5 `04ca39ece7533eb35c805a4741996ebb`。gfrun-5/gfrun-6 为本轮(第四轮)新增,在此指纹下复现。

## 通用复现步骤

```bash
# 1. checkout 看护 demo(SuperNPUBench PR #96)
# 2. 指向 env_test 工具链
source microbenchmark/tileop-guard/env.sh   # COMPILER_DIR=linx clang / GFRUN=SuperScalarModel/bin/gfrun
# 3. 单 case 三态验证(编译 → gfrun → host golden);以下 case 均「编译通过、gfrun 崩」
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## gfrun-1 · TIMG2COL 未实现桩(卷积 im2col)

**涉及接口**:TIMG2COL。

**问题**:照 `TIMG2COL.md` 示例(普通 `Tile<Vec,float,8,256>` + TLOAD + `TIMG2COL(dst,src,3,5)`)写,
编译通过,gfrun 直接命中未实现断言——模型侧卷积窗口/repeat/padding 契约未实现。

**复现**:`bash run_guard.sh sfu timg2col`。

**错误信息**:
```
TIMG2COL not yet fully implemented
```

**附加说明**:文档示例本身不用持久 Matrix-location feature-map 描述符源,故 demo 亦忠实按普通 Vec tile 写;
运行受阻纯属模型缺口,非 demo 写法问题。

---

## gfrun-2 · TMRGSORT 未实现桩(归并排序)

**涉及接口**:TMRGSORT。

**问题**:`sort.md` 原示例 shape 不可编译(见 docs issue Doc-8);修正为两源各 1×128、dst 1×256 后编译通过,
gfrun 命中未实现断言。

**复现**:`bash run_guard.sh sfu tmrgsort`(demo 已用修正后的合法 shape)。

**错误信息**:
```
TMRGSORT not yet fully implemented
```

**附加说明**:排序语义 golden 已可实现,待模型补齐后即可转精度校验。

---

## gfrun-3 · TileArray region producer 路径未实现

**涉及接口**:region::TileArray + TASSEMBLY + TPARTVIEW(TCVT slot producer 路径)。

**问题**:绑定问题修好(见 docs issue Doc-15)后,用 TPARTVIEW 父 strided 子视图作源、或独立紧凑 Fragment
作源,gfrun 均命中同一断言。结合 `range-modifiers-developer-guide.md` 文档**自身告警**
「until the … path is implemented and validated」,判定 region producer 路径在当前模型上尚不可运行。

**复现**:`bash run_guard.sh tlsu region_tilearray`。

**错误信息**:
```
raw tile spill source does not fit the carrier shape
```

**附加说明**:文档已自述该路径「尚未实现并验证」,属已知模型缺口,登记待实现。

---

## gfrun-4 · range::Subview 运行期拒绝文档示例 descriptor

**涉及接口**:`range::Subview`。

**问题**:照 `range-modifiers.md` 示例参数编译通过(签名/汇编层无缺口),gfrun 在运行期拒绝按文档示例参数
生成的 descriptor。

**复现**:`bash run_guard.sh tlsu range_subview`。

**错误信息**:
```
illegal TSTORE operand or descriptor contract
```

**附加说明**:签名/汇编层无问题、编译通过,受阻在运行期 descriptor 契约校验。若模型判定为合法用法,应放行
该 subview descriptor;若属文档示例参数问题,请回推正确参数域(此时应转 docs issue)。

---

## gfrun-5 · TQUANT 静默忽略 multiplier / zeroPoint(第四轮新发现)

**涉及接口**:TQUANT。

**问题**:`TQUANT.md` 文档签名为
`TQUANT<RoundMode Mode=RNE, bool Sat>(dst, src, float multiplier=1.0f, int32_t zeroPoint=0)`,
参数说明「multiplier=量化乘数、zeroPoint=量化零点」,bundle 结构也明确编码
`B.IOR MultiplierFP32, ZeroPoint`。语义应为 `q = clamp(round(src*multiplier)+zeroPoint, -128, 127)`。
但**实测 gfrun 忽略这两个参数**:传 `multiplier=0.5f, zeroPoint=1` 时,输出**逐字节等于
`clamp(round_RNE(src), -128, 127)`**(即按 multiplier=1.0 / zeroPoint=0 计算),0/2048 元素全部落在
mult=1/zp=0 的参考上,对 multiplier 做线性拟合斜率≈1.0。

**根因已隔离到模型侧(非工具链)**:对 `TQUANT<RNE,true>(d,s,0.5f,1)` 反汇编,`BSTART.TEPL TQUANT, FP32`
之后确有 `B.IOR` 操作数,且 0.5f 的常量 `0x3f000000` 经 `lui 0x3f000` 被 materialize 后送入 bundle
——即**后端已正确发射 multiplier operand,是 gfrun 执行 TQUANT 时未消费该 operand**。

**复现**:把 `sfu/src/tquant.cpp` 的 mult/zp 改成非 identity(如 `0.5f, 1`),`bash run_guard.sh sfu tquant`
→ golden(按文档语义 `round(src*mult)+zp`)MISMATCH;改回 identity(`1.0f, 0`)即 PASS。

**当前看护处置**:demo 已用 identity 参数(mult=1.0/zp=0)精确看护 TQUANT **确实执行**的核心
——RNE 舍入 + S8 饱和(输入跨 ±256 触发双端 clamp),被忽略的缩放路径不在 golden 里断言。模型补齐
multiplier/zeroPoint 消费后,即可把 demo 换回非 identity 参数、golden 转全语义校验。

---

## gfrun-6 · TLOG 计算 log₂ 而非文档所述自然对数(第四轮新发现)

**涉及接口**:TLOG。

**问题**:`TLOG.md` 开宗明义「computes the same-type **natural logarithm** of every valid Local Tile
element」,即应为 ln(base-e)。但**实测 gfrun 计算的是 log₂(base-2)**:输入 4.25 时输出 2.0875
(=log₂4.25),而非 ln(4.25)=1.4469。配套 TEXP 实测为 base-e(与「natural」一致),故本条是 TLOG 单点
与文档语义相反(base-2 vs 文档声明的 base-e)。

**复现**:`bash run_guard.sh sfu tlog`。golden 若按文档 `np.log`(自然对数)会 MISMATCH;按 `np.log2`
逐元素通过。当前 demo 的 golden 已按**实测的 log₂**钉死(见 golden.py `check_transcend` fn['log']=np.log2),
误差 ~6e-8(近 fp32 精确)。

**需澄清**:属**模型实现与文档不一致**——要么模型应改为自然对数以符合 `TLOG.md`,要么文档应更正为
「以 2 为底的对数」。在澄清前,看护 golden 按实测 log₂ 固定,以免把口径分歧误报为精度失败。
