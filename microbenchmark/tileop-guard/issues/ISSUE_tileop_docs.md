# [docs] Linx-TileOP-API `docs/tileop-usage` 文档缺陷汇总

本 issue 汇总 TileOP-API v0.58 文档看护(逐接口写最小 demo)中发现的**文档类缺陷**:签名缺失、
示例不可编译、dtype/形状契约未写、语义未说明。均为「补文档即可解决」的问题——现象即便表现为
gfrun 断言,也是 gfrun 在正确执行一条文档漏写的契约;真正的模型/后端缺口另行分别提 issue
(`[gfrun][NA]` / `[linx]`)。

以问题为单位,每条含复现与实测错误信息。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | PTO-ISA/SuperNPUBench | `ops-20260828` | `d1604cf` |
| SuperScalarModel(gfrun) | LinxISA/SuperScalarModel | `codex/pr-0.58.4-shared-model` | `762a72c` |
| **Linx-TileOP-API(本 issue 目标)** | LinxISA/Linx-TileOP-API | `linx` | `d6a52b8` |
| llvm-project(linx clang/lld) | LinxISA/llvm-project | `dev-llvm15_56` | `0f878a8` |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

> 看护 demo 代码见 **SuperNPUBench PR #96**:https://github.com/PTO-ISA/SuperNPUBench/pull/96

## 通用复现步骤

```bash
# 1. checkout 看护 demo(SuperNPUBench PR #96)
# 2. 指向 env_test 工具链
source microbenchmark/tileop-guard/env.sh   # COMPILER_DIR=linx clang / GFRUN=SuperScalarModel/bin/gfrun
# 3. 单 case 三态验证(编译 → gfrun → host golden)
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## Doc-1 · VEC 全族无 C++ 调用签名

**涉及接口**:TADD/TSUB/TMUL/TDIV/TMAX/TMIN、TABS/TNEG/TRELU/TNOT、TFMA、`*S` 标量族、TEXPANDS 等
(engines.md 覆盖的全部 VEC 算子)。

**问题**:除 `cmp.md` 外,VEC 算子在文档中**无一给出 C++ 调用签名或示例**。`engines.md` 是「名字 +
汇编投影 + 分类」三列的汇编投影表,并非 C++ API 文档。只能靠 dst-first 惯例 `(dst, src0, src1)` 反猜;
遇到非常规签名(见 Doc-2/Doc-4)必然猜错。

**复现**:`bash run_guard.sh vec tadd`(惯例猜中可编译);查阅 `docs/tileop-usage/.../engines.md` 无签名。

**建议**:为每个 VEC 算子补一行 C++ 签名 + 一个最小示例。

---

## Doc-2 · TSEL / TSELS 三源元组契约未文档化

**涉及接口**:TSEL、TSELS。

**问题**:文档(engines.md)仅列名字与分类,不给签名与 select 语义。实测 `(dst,src0,src1)` 三参可编译,
但 gfrun 运行期要求 **mask + true/false 三源元组**,并未在文档说明「按什么条件选哪个」。

**复现**:`bash run_guard.sh vec tsel` / `bash run_guard.sh vec tsels`(编译过,gfrun 崩)。

**错误信息**:
```
TSEL : select first B.IOT requires mask then true/source Tile   (srcs.size()==3 断言)
TSELS: TSELS requires mask and source Tile operands
```

**建议**:补 TSEL/TSELS 完整签名 + 三源结构(mask/true/false)+ 选择语义;`*S` 变体标量位置反直觉,需示例。

---

## Doc-3 · TCMP / TCMPS 结果为 packed predicate,消费方式未说明

**涉及接口**:TCMP、TCMPS。

**问题**:`cmp.md` 给出完整 `TCMP<Mode>(dst,src0,src1)` / `TCMPS<Mode>(dst,src,scalar)` 签名 + dtype 表 +
示例,编译通过。但比较结果是 **packed predicate 载体**,与常规 tile 的 TSTORE 源契约不符,文档未说明
如何落盘/消费该结果。

**复现**:`bash run_guard.sh vec tcmp` / `bash run_guard.sh vec tcmps`。

**错误信息**:
```
Local TSTORE requires one legal source Tile descriptor   (srcs.size()==2 断言)
```

**建议**:在 cmp.md 说明 predicate 结果的存储/消费路径(如何 TSTORE、如何喂给 TSEL),或给端到端示例。

---

## Doc-4 · TEXPANDS 签名/分类误导

**涉及接口**:TEXPANDS。

**问题**:真实签名 `TEXPANDS(dst, scalar)` 仅 2 参、无 src tile——「标量广播填充整个 tile」。但分类栏标
`tile-scalar-and-immediate`,看似 `tile ⊕ scalar` 的二元标量运算,极易误写成 3 参。

**复现**:`bash run_guard.sh vec texpands`(按 2 参写才可编译 + 精度PASS)。

**建议**:明确 TEXPANDS 为纯标量填充,给 `(dst, scalar)` 签名,修正分类描述。

---

## Doc-5 · TROW*/TCOL* reduce 输出 tile 形状契约缺失

**涉及接口**:TROWSUM/MAX/MIN/PROD、TCOLSUM/MAX/MIN/PROD。

**问题**:一元 `(dst, src)`,src 为 M×N。但 dst 形状规则文档未写:row-reduce 的 dst 须**物理 M×N +
ValidCol=1**;col-reduce 的 dst 须**物理 M×N + ValidRow=1**(物理 M×1 非法:fp32 需 `Cols×bits%256==0`)。
只能靠 gfrun 断言反推此「物理满宽 + valid 轴=1」规则。

**复现**:`bash run_guard.sh sfu trowsum` / `tcolsum`(按正确形状写才通过,已精度PASS)。

**建议**:补 reduce 族输出形状契约(物理形状 + valid 轴)。

---

## Doc-6 · expand 族广播源形状规则缺失且左右不对称

**涉及接口**:TROWEXPAND{ADD,SUB,MUL,DIV,MAX,MIN,EXPDIF}、TCOLEXPAND{...}、TROWEXPAND、TCOLEXPAND。

**问题**:
- **arith 变体**(各 7,gfrun 通过):二元 `(dst M×N, src0 M×N, src1_broadcast)`。col-expand 的 src1 须
  **物理 M×N + ValidRow=1**(1×N 行广播);row-expand 的 src1 须**物理 M×1** 列广播。两侧广播源形状规则
  **不对称**,文档均未给。
- **copy-expand**(TROWEXPAND/TCOLEXPAND):纯 `(dst, src)`,gfrun 断言要求广播源,文档无构造说明 →
  **run-fail**。

**复现**:`bash run_guard.sh sfu tcolexpandadd`(需正确广播源形状);`bash run_guard.sh sfu trowexpand`(崩)。

**错误信息**(copy-expand):
```
COPY expansion requires one broadcast source
```

**建议**:补 expand 族广播源形状规则(明确 row/col 两侧的物理形状与 valid 轴),copy-expand 给广播源构造示例。

---

## Doc-7 · argmax/argmin 输出 dtype 契约缺失

**涉及接口**:TROWARGMAX、TROWARGMIN、TCOLARGMAX、TCOLARGMIN。

**问题**:编译通过,gfrun 断言索引输出的 dtype 与 block dtype 不匹配;文档未给 argmax 输出的 dtype/形状契约。

**复现**:`bash run_guard.sh sfu trowargmax`(编译过,gfrun 崩)。

**错误信息**:
```
typed Local TSTORE source dtype must match the block dtype
```

**建议**:说明 argmax/argmin 索引输出的 dtype(整型索引域)与形状约束。

---

## Doc-8 · TMRGSORT 示例本身不可编译

**涉及接口**:TMRGSORT。

**问题**:`sort.md` 示例三个 tile 全用 1×256:`TMRGSORT(out, a, b)`,但 static_assert 要求
`dst.ValidCol == left.ValidCol + right.ValidCol`,`256 != 256+256` 直接编译失败。修正为两源各 1×128、
dst 1×256 后方可编译(编译后 gfrun 另有模型未实现问题,另见 `[gfrun][NA]` issue)。

**复现**:`bash run_guard.sh sfu tmrgsort`。

**错误信息**(照原示例):
```
static_assert: dst.ValidCol must equal left.ValidCol + right.ValidCol
```

**建议**:改示例为 `left/right = 1×128, dst = 1×256`。

---

## Doc-9 · layout / irregular 大批算子无签名

**涉及接口**:TCONCAT、TEXTRACT、TINSERT、TFILLPAD、TTRANS、TPART{ADD,MUL,MAX,MIN}、TTRI、THISTOGRAM、
TGATHER、TSCATTER。

**问题**:`layout.md` 只有散文,不给任何 C++ 签名;TTRI/THISTOGRAM/TGATHER/TSCATTER/TPART* 连专题文档都无。
靠 dst-first 惯例:
- 惯例命中可编译:TCONCAT/TFILLPAD/TTRANS/TPART{ADD,MUL,MAX,MIN}。
- 猜错编译失败:TEXTRACT/TINSERT/TTRI/THISTOGRAM(`no matching function`)。
- TGATHER/TSCATTER:惯例签名可编,但后端发 `Match Instruction Error`。因文档无签名,**无法确认是猜错
  intrinsic 还是真后端缺口**——根因待补出文档签名后方可判定,故仍归此文档条目。

**复现**:`bash run_guard.sh sfu textract`(`no matching function`)。

**建议**:layout/irregular 每算子补签名 + 示例。

---

## Doc-10 · TPREFETCH src 类型未说明

**涉及接口**:TPREFETCH。

**问题**:文档签名 `TPREFETCH(src, valid_col, valid_row)` 未说明 src 类型。传 iterator 视图编译报
`no member named 'Cols'`;改传带 compile-time 列数的 static RowMajor `global_tensor` 本体后通过。

**复现**:`bash run_guard.sh tlsu tprefetch`。

**建议**:点明 src 须为带 compile-time 列数的 `global_tensor` 本体(非 iterator 视图)。

---

## Doc-11 · Shared TMOV 四变体 + TSTORE_PART 无签名、无 Shared tile 构造示例

**涉及接口**:TMOV_L2S_INSERT、TMOV_L2S_PUBLISH、TMOV_S2L_BROADCAST、TMOV_S2L_EXTRACT、TSTORE_PART。

**问题**:`tlsu.md` 提到这些接口,但既未给签名,也未给 **Shared-location tile 的 C++ 声明/构造方式**。
`docs/tileop-usage` 全目录无一处 Shared tile 构造示例,仅凭文档**无法写出可编译 demo**。这是 TLSU Shared
面的系统性文档缺口。

**复现**:尝试为上述接口写 demo——无 Shared tile 构造示例,卡在类型声明,无法编译。

**建议**:补 Shared-location tile 的构造/生命周期示例 + 上述接口签名。

---

## Doc-12 · CUBE Bias 形状契约 + TLOAD_CUBE/TSTORE_CUBE 无签名

**涉及接口**:TMATMUL_BIAS(Bias 辅助参数)、TLOAD_CUBE、TSTORE_CUBE。

**问题**:
- `matrix-postprocess.md` 只说 Bias「辅助参数仍是普通 Local Tile」,未说 dtype/valid shape。实测 Bias 须
  **普通 RowMajor + 派生 AccType(fp16 输入→FP32)+ valid `1×N`**,且用**普通 `TLOAD`(非 TLOAD_CUBE)**。
- `cube.md` 只说用 TLOAD_CUBE/TSTORE_CUBE「做 GM 转换边界」,不给 C++ 签名(`(cube_tile, global_tensor)`
  二参靠惯例命中)。

**复现**:`bash run_guard.sh cube tmatmul_bias`(需按上述隐含契约写)。

**建议**:补 Bias 的 dtype/valid-shape/加载方式;补 TLOAD_CUBE/TSTORE_CUBE 签名。

---

## Doc-13 · FIXP `QF322S16Pre` 与输入矩阵 dtype 的兼容矩阵缺失

**涉及接口**:`fixp::scalar<QF322S16Pre>()`(matrix-postprocess.md B.FPATR PreQuantMode 表)。

**问题**:B.FPATR 表列 `QF322S16Pre → S16`,但 fp16 矩阵输入 + S16 dst 编译即被 static_assert 拒。表只给
「mode → dst dtype」,未给 **mode ↔ 输入矩阵 dtype 兼容矩阵**(S16 量化疑似要求整数矩阵输入、派生 S32
accumulator)。

**复现**:按 `QF322S16Pre` + fp16 输入写 scalar-quant demo → 编译失败;改 `QF322S8Pre` 才通过。

**错误信息**:
```
static_assert: PreQuantMode incompatible with the derived matrix accumulator type
```

**建议**:补 PreQuantMode ↔ 输入矩阵 dtype 的合法组合表。

---

## Doc-14 · MGATHER / MSCATTER 示例 offset dtype 违反同页 dtype 表

**涉及接口**:MGATHER、MSCATTER。

**问题**:`MGATHER.md` / `MSCATTER.md` 签名 + 示例完整,按签名 `MGATHER(dst, gmSrc, offsetTile)` /
`MSCATTER(base_gm, src, offsetTile)`(offset 存**字节位移**)重写后跑通 + golden 逐元素 PASS。**但**两页
使用示例都把 offset 声明为 `Tile<Vec,uint16_t,...>`,而**同页 dtype 表**规定索引 Tile 必须是
S32/U32/S64/U64。照示例用 uint16 能编过,gfrun 拒 descriptor 契约。

**复现**:`bash run_guard.sh tlsu mgather`(遵从 dtype 表用 U32 → PASS;照示例用 uint16 → 崩)。

**错误信息**(照示例 uint16):
```
illegal MGATHER operand or descriptor contract
```

**建议**:把示例 offset 类型从 `uint16_t` 改为 `uint32_t`,与同页 dtype 表一致。

---

## Doc-15 · range / TileArray region 示例不可直接落地

**涉及接口**:`range::Assemble`、TileArray region API(TPARTVIEW / TileArray / TASSEMBLY 的示例代码)。

**问题**:
- **range::Assemble**:照 `range-modifiers.md` 示例参数(`ParentSizeCode=12`)编译即被 static_assert 拒——
  SizeCode 反推容量 < 声明容量,示例参数自相矛盾。
- **TileArray region 示例**:`range-modifiers-developer-guide.md` 示例 `TCVT(destinations[0][2], source_tile)`
  用临时量,而 region `TCVT` 签名是 `TCVT(TileArrayOutputRef& dst, In& src)`(非 const 左值引用),临时量
  无法绑定;示例源变量名 `source` 与 `source_tile` 不一致——照抄不可编译,须改绑具名左值。
  (绑定修好后另有模型侧 run-fail,见 `[gfrun][NA]` issue。)

**复现**:`bash run_guard.sh tlsu range_assemble`;`bash run_guard.sh tlsu region_tilearray`。

**错误信息**(range::Assemble):
```
static_assert: B.ASSEMBLE length cannot exceed the parent Tile capacity
```

**建议**:修正 range::Assemble 示例的 `ParentSizeCode` 使之与声明容量自洽;修正 TileArray region 示例的
临时量绑定与变量命名,使其可直接编译。
