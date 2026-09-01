# TileOP-API v0.58 文档看护报告

本报告依据 `Linx-TileOP-API/docs/tileop-usage/*.md`,为每个接口写最小看护 demo,记录三态
状态与文档中缺失、不足或与实现不符之处。

- **依据**:demo 仅依据 `docs/tileop-usage/` 写(签名/示例/dtype 全取自文档,不读 intrinsic 源码)。
  独立 golden 允许依据接口语义(SuperNPUBench docs/intrinsics + ISA 规格)实现。
- **三态**:①**编译**成 `.elf`;②**gfrun** 跑到 `Reach the End of Benchmark`(无崩);
  ③**精度**——host 侧独立 golden 逐元素比对通过。
- demo 目录:`microbenchmark/tileop-guard/{vec,sfu,tlsu,cube,fixp,misc}/`;跑批:`source env.sh && bash run_guard.sh <sub>`。

文档问题分类标签:
- **[签名]** 签名/模板参数/参数个数缺失,只能靠编译器报错反推
- **[语义]** 计算语义未说明
- **[dtype]** dtype/shape 约束未说明,靠 gfrun assert 反推
- **[示例]** 文档示例本身不可编译/与自身约束矛盾
- **[后端]** 文档签名完整,工具链后端无法汇编该指令
- **[契约]** 编译通过,gfrun 拒绝按文档参数生成的 descriptor 契约

## 基线指纹

全部状态来自**一次干净重编重跑**:`make clean_all` 后全量 `run_guard.sh`(共享头改动不触发
增量重编,须清缓存才可签字)。

```
run-date : 2026-09-01 18:40
toolchain: env_test/linx-toolchain-build/output/linx_blockisa_llvm_musl
           clang++ md5 c5a5edef0d9ca809dc368de7dbc2ad28
gfrun    : env_test/SuperScalarModel/bin/gfrun
           gfrun   md5 04ca39ece7533eb35c805a4741996ebb
```

## 状态总表

**合计 125 case:52 精度PASS / 47 run-only / 12 编译失败 / 14 run-fail(崩)。**

| 域 | 精度PASS | run-only | 编译失败 | run-fail | 合计 |
|----|---------|----------|----------|----------|------|
| vec  | 30 | 1  | 0 | 4 | 35 |
| sfu  | 13 | 29 | 6 | 8 | 56 |
| tlsu | 2  | 5  | 4 | 2 | 13 |
| cube | 6  | 2  | 1 | 0 | 9  |
| fixp | 1  | 10 | 0 | 0 | 11 |
| misc | 0  | 0  | 1 | 0 | 1  |
| 合计 | **52** | **47** | **12** | **14** | **125** |

- **精度PASS**:编译 + gfrun 跑通 + host 独立 golden 逐元素比对通过(真「算对」)。
- **run-only**:编译 + gfrun 跑通,但未写独立 golden(语义/舍入未由文档钉死),仅验「能跑」。
- **编译失败**:文档无法据以写出可编译调用,或后端拒绝汇编。
- **run-fail**:gfrun 崩(assert/fault),无输出可校。

**各表「精度」列**:✅ = 独立 golden 逐元素通过;❌ = golden 比对失败;— = 未做精度校验
(编译/运行失败无输出,或该接口未写 golden)。当前无 ❌(precision-fail = 0)。

## 精度校验机制(res_check + host 独立 golden)

```
host  golden.py gen  → 按 case 语义 numpy 生成输入 compare/<sub>/<case>/in_*.bin
ELF   guard_read_bin(read() 系统调用整块读入) → TLOAD/op/TSTORE → guard_dump_bin(out.bin)
host  golden.py check → 读 in_*.bin + out.bin,numpy 独立算 ref,逐元素带容差比 → exit 0/1
```

- **独立性**:golden 用 numpy 从接口语义独立实现,不读 emulator 的 tile 实现,是真独立 oracle。
- **输入 host 生成 + `read()` 读入**:设备端填充会被 tile 后端错编导致输入退化(两 fill 调用
  被混叠、一个操作数塌成 0),故由 host 拥有输入、ELF 用 `read()` 整块读入(emulator syscall
  直接写内存,绕过 tile-register 提升)。
- **无 printf dump**:工具链自带 `writeBinaryFile` 尾部 `printf/fflush` 会死循环卡死 gfrun,改用
  自写 `guard_dump_bin`(open/write/close)。
- 校验代码:`golden/golden.py`(注册表 + 语义)、`common/guard_io.{h,c}`(fill/read/dump,以
  `-mlxbc -O2` 无 matrix flags 编译)、`common/guard_case.hpp`(case 宏)。

### 已带独立 golden 的接口(52 精度PASS)

| 域 | 接口 | golden 语义 | 容差 |
|----|------|-------------|------|
| VEC 二元 | tadd/tsub/tmul/tdiv/tmax/tmin(f32)、tand/tor/txor/trem/tshl/tshr(i32) | elementwise;移位/rem 取正保证逻辑=算术、无溢出 | 整数 eps=0 / f32 1e-4 |
| VEC 一元/标量 | tabs/tneg/trelu/tnot、tfma、t{add,sub,mul,div,max,min}s、t{and,or,xor,rem,shl,shr}s、texpands | abs/neg/relu/not、a*b+c、tile⊕标量、标量填充 | 同上 |
| SFU reduce | trow{sum,max,min,prod}、tcol{sum,max,min,prod} | 沿轴 sum/max/min/prod;row 比 out[r*N+0]、col 比 out[c] | 1e-3 |
| SFU create-index | tci、tci_desc、tci_s16、tci_u32、tci_u16 | iota:asc=start+k / desc=start−k(按元素位宽 wrap),ValidRow=1 | 整数 eps=0 |
| CUBE matmul | tmatmul、_acc、_bias、_f16、_relu、_rowmax | A@B(f16→f32 累加);+C / +bias(1×N) / relu / f16 输出 | f16 rel 2e-2~3e-2 |
| TLSU gather/scatter | mgather、mscatter | GM base + U32 字节位移;gather=base[off//4]、scatter=base[off//4]←src(单射无碰撞) | eps=0 |
| FIXP | convert | A@B 后 cast f16 | 3e-2 |

实测 tmatmul `max|out−A@B|=0.0`(f16 matmul 与 numpy 逐字节一致)。

### 未写 golden 的接口(run-only,原因)

- **舍入模式未由文档钉死**:tcvt(f32→i32,trunc vs RNE)。
- **语义需读 ISA 才能独立实现**:expand-arith(t{row,col}expand{add,sub,mul,div,max,min}/expdif)、
  tpart*、tconcat/ttrans/tfillpad、tsort。
- **量化需 RNE+饱和 spec golden**:tquant/tdequant、fixp s8/vector quant/lrelu/prelu/rowmax_acc/
  group_max/cscale/chain。
- **超越函数需容差建模**:texp/tlog/trecip/tsqrt/trsqrt。
- **CUBE 变体形状/scale 语义未钉**:tmatmul_mx、tgemv。
- **通路/搬运类**:tload/tstore/tmov/tprefetch/mgather_cas(无独立数值语义可校)。

---

## VEC 族(elementwise / scalar / compare)

| 接口 | 编译 | gfrun | 精度 | 说明 / 文档问题 |
|------|------|-------|------|-----------------|
| TADD/TSUB/TMUL/TDIV | ✅ | ✅ | ✅ | [签名] 基础算术无签名,`(dst,s0,s1)` 靠 dst-first 惯例推断 |
| TREM/TAND/TOR/TXOR/TSHL/TSHR | ✅ | ✅ | ✅ | [签名] 同上;整数域 |
| TMAX/TMIN | ✅ | ✅ | ✅ | [签名] 同上 |
| TABS/TNEG/TRELU | ✅ | ✅ | ✅ | [签名] 一元 `(dst,src)` 推断 |
| TNOT | ✅ | ✅ | ✅ | [签名] 同上;整数 |
| TFMA | ✅ | ✅ | ✅ | [签名] 三元 `(dst,a,b,c)` 推断 |
| TADDS/TSUBS/TMULS/TDIVS | ✅ | ✅ | ✅ | [签名] `*S` 标量类 `(dst,src,scalar)` 推断 |
| TREMS/TANDS/TORS/TXORS/TSHLS/TSHRS | ✅ | ✅ | ✅ | [签名] 同上;整数 |
| TMAXS/TMINS | ✅ | ✅ | ✅ | [签名] 同上 |
| TEXPANDS | ✅ | ✅ | ✅ | **[签名]** 真实 `(dst,scalar)` 仅 2 参,标量填充整个 tile(详述 2) |
| TCVT | ✅ | ✅ | — | [签名] `(dst,src)` 异 dtype;舍入(trunc/RNE)文档未钉,未写 golden |
| TSEL | ✅ | ❌ | — | **[签名][语义][dtype]** gfrun 要 mask+true+false 三源元组(详述 1) |
| TSELS | ✅ | ❌ | — | **[签名][语义]** gfrun 要 mask+source 三源(`TSELS requires mask and source Tile`)(详述 1) |
| TCMP | ✅ | ❌ | — | **[语义]** cmp.md 签名完整;gfrun 结果为 packed predicate,与 TSTORE 源契约不符(详述 4) |
| TCMPS | ✅ | ❌ | — | 同上 |

VEC 小结:35 case,**30 精度PASS / 1 run-only(tcvt)/ 4 run-fail(tsel/tsels/tcmp/tcmps)**。
除 cmp.md 外,VEC 全族文档不给 C++ 签名(engines.md 只有「名字 + 汇编 + 分类」三列,详述 3);
比较/选择族(TSEL/TSELS/TCMP/TCMPS)编译通过但 gfrun 拒当前源契约。

## SFU 族(reduce / expand / transcendental / layout / irregular)

| 接口 | 编译 | gfrun | 精度 | 说明 / 文档问题 |
|------|------|-------|------|-----------------|
| TROWSUM/TROWMAX/TROWMIN/TROWPROD | ✅ | ✅ | ✅ | **[dtype]** 无签名;dst 须物理 M×N + ValidCol=1(详述 5) |
| TCOLSUM/TCOLMAX/TCOLMIN/TCOLPROD | ✅ | ✅ | ✅ | **[dtype]** 无签名;dst 须物理 M×N + ValidRow=1(详述 5) |
| TCI(vci,5 变体) | ✅ | ✅ | ✅ | TCI.md 签名+示例完整;asc/desc × S32/S16/U32/U16,golden=iota |
| TEXP/TLOG/TRECIP/TSQRT/TRSQRT | ✅ | ✅ | — | [签名] 一元 `(dst,src)`;超越函数需容差建模,未写 golden |
| TCOLEXPANDADD/SUB/MUL/DIV/MAX/MIN/EXPDIF | ✅ | ✅ | — | **[dtype]** 二元;src1 须 1×N 行广播源(详述 6);语义未钉,未写 golden |
| TROWEXPANDADD/SUB/MUL/DIV/MAX/MIN/EXPDIF | ✅ | ✅ | — | **[dtype]** 二元;src1 须物理 M×1 列广播源(详述 6) |
| TCONCAT | ✅ | ✅ | — | **[签名]** layout.md 无签名;`(dst,s0,s1)` 惯例命中 |
| TFILLPAD/TTRANS | ✅ | ✅ | — | **[签名]** `(dst,src)` 惯例命中 |
| TPARTADD/TPARTMUL/TPARTMAX/TPARTMIN | ✅ | ✅ | — | **[签名]** 无文档;`(dst,s0,s1)` 惯例命中 |
| TQUANT | ✅ | ✅ | — | 签名完整(quant-and-im2col.md);FP32→S8,量化 golden 未写(RNE+sat) |
| TDEQUANT | ✅ | ✅ | — | 签名完整;跑通,量化 golden 未写 |
| TSORT | ✅ | ✅ | — | 签名完整(sort.md);跑通,排序 golden 未写 |
| TROWEXPAND | ✅ | ❌ | — | **[语义]** copy-expand;gfrun `COPY expansion requires one broadcast source`(详述 6) |
| TCOLEXPAND | ✅ | ❌ | — | 同上 |
| TROWARGMAX/TROWARGMIN/TCOLARGMAX/TCOLARGMIN | ✅ | ❌ | — | **[语义]** gfrun `TSTORE source dtype must match block dtype`(argmax 输出 dtype,详述 7) |
| TIMG2COL(卷积) | ✅ | ❌ | — | 照 TIMG2COL.md 示例(普通 Vec tile);gfrun `TIMG2COL not yet fully implemented`——模型断言桩(详述 10) |
| TMRGSORT | ✅ | ❌ | — | **[示例]** 示例 shape 不可编译;修正后 gfrun `TMRGSORT not yet fully implemented`(详述 8) |
| TEXTRACT/TINSERT | ❌ | - | — | **[签名]** 无签名;`no matching function`(详述 9) |
| TTRI/THISTOGRAM | ❌ | - | — | **[签名]** 无文档;`no matching function`(详述 9) |
| TGATHER/TSCATTER | ❌ | - | — | **[签名]** 无文档;后端 `Match Instruction Error`(详述 9) |

SFU 小结:56 case,**13 精度PASS / 29 run-only / 6 编译失败 / 8 run-fail**。
- 精度PASS(13):8 reduce + 5 TCI。
- run-only(29):5 超越函数 + 14 expand-arith + tconcat/tfillpad/ttrans + 4 tpart + tquant/tdequant/tsort。
  多因语义/舍入未由文档钉死,未写 golden。
- 编译失败(6):TEXTRACT/TINSERT/TTRI/THISTOGRAM/TGATHER/TSCATTER——文档无签名,无法写出可编译调用。
- run-fail(8):copy-expand(2)、argmax/argmin(4)、TIMG2COL、TMRGSORT——模型未实现或 dtype/契约断言。

## TLSU 族(load / store / move / gather / scatter)

| 接口 | 编译 | gfrun | 精度 | 说明 / 文档问题 |
|------|------|-------|------|-----------------|
| TLOAD | ✅ | ✅ | — | 文档充分;全套 case 依赖,通路正常 |
| TSTORE | ✅ | ✅ | — | 文档充分;通路正常 |
| TPREFETCH | ✅ | ✅ | — | **[签名]** src 须 static RowMajor `global_tensor` 本体,非 iterator 视图(详述 11) |
| TMOV | ✅ | ✅ | — | 基础 Local `(dst,src)`;跑通 |
| MGATHER_CAS | ✅ | ✅ | — | 签名完整;base+offset+expected+replacement |
| MGATHER | ✅ | ✅ | ✅ | MGATHER.md 签名+示例完整;GM base + U32 字节位移,golden=base[off//4]。**[示例]** 示例误用 uint16 offset(违反同页 dtype 表),遵从 dtype 表用 U32(详述 17) |
| MSCATTER | ✅ | ✅ | ✅ | MSCATTER.md 签名+示例完整;`MSCATTER(base_gm,src,off)`,base[off//4]←src(单射),golden 比 scatter 后 base。同 [示例] U32 缺口(详述 17) |
| GMOV | ❌ | - | — | **[后端]** 照示例 `GMOV<15>(dst,peer_tid,src)` 后端 `Match Instruction Error`(详述 11) |
| MGATHER_MASK | ❌ | - | — | **[后端]** 签名+示例完整;`MGATHER.MASK` 双 B.IOT bundle 后端无法汇编(详述 18) |
| MSCATTER_MASK | ❌ | - | — | **[后端]** 同上,`MSCATTER.MASK` bundle 后端未支持(详述 18) |
| range::Assemble | ❌ | - | — | **[示例]** 照示例参数(ParentSizeCode=12)编译即被 static_assert 拒 `B.ASSEMBLE length cannot exceed parent Tile capacity`(详述 15) |
| range::Subview | ✅ | ❌ | — | **[契约]** 照示例编译过,gfrun `illegal TSTORE operand or descriptor contract`(详述 15) |
| TileArray region API(region_tilearray) | ✅ | ❌ | — | **[契约]** TPARTVIEW/TileArray/TASSEMBLY;gfrun `raw tile spill source does not fit the carrier shape`(详述 19) |

TLSU 小结:13 case,**2 精度PASS / 5 run-only / 4 编译失败 / 2 run-fail**。
- 精度PASS(2):MGATHER、MSCATTER。
- run-only(5):TLOAD/TSTORE/TPREFETCH/TMOV/MGATHER_CAS(通路/搬运类,无独立数值语义可校)。
- 编译失败(4):GMOV / MGATHER_MASK / MSCATTER_MASK(后端不支持)、range::Assemble(static_assert)。
- run-fail(2):range::Subview、region_tilearray(gfrun descriptor 契约)。
- 注:Shared TMOV 四变体(TMOV_L2S_INSERT/PUBLISH、TMOV_S2L_BROADCAST/EXTRACT)与 TSTORE_PART
  依赖 opaque Shared handle,docs/tileop-usage 全目录无 Shared tile 构造示例,无法写出可编译 demo(详述 13)。

## CUBE 族(matmul / gemv)

| 接口 | 编译 | gfrun | 精度 | 说明 / 文档问题 |
|------|------|-------|------|-----------------|
| TMATMUL | ✅ | ✅ | ✅ | cube.md 给示例;`(out,a,b)` / `(out,a,b,options)` |
| TMATMUL_ACC | ✅ | ✅ | ✅ | matrix-postprocess.md 签名完整 `(Dst,C,A,B,options)` |
| TMATMUL_BIAS | ✅ | ✅ | ✅ | **[dtype]** Bias 须 1×N + RowMajor + 派生 AccType + 普通 TLOAD(详述 13) |
| TMATMUL(f16) | ✅ | ✅ | ✅ | `fixp::f16()` |
| TMATMUL(relu) | ✅ | ✅ | ✅ | `fixp::f16().relu()` 链式 |
| TMATMUL(row_max) | ✅ | ✅ | ✅ | `fixp::keep_acc().row_max(out)` |
| TMATMUL_MX | ✅ | ✅ | — | 无 scale 便捷式;跑通,MX scale 语义未钉,未写 golden(详述 13) |
| TGEMV | ✅ | ✅ | — | 跑通;matrix-vector 输出形状/语义未钉,未写 golden(详述 13) |
| TMATMUL(bf16) | ❌ | - | — | **[示例]** 照 `fixp::bf16()` 写→clang frontend abort(exit 134)(详述 13) |

CUBE 小结:9 case,**6 精度PASS / 2 run-only(tmatmul_mx/tgemv)/ 1 编译失败(bf16)/ 0 run-fail**。
核心 matmul 通路 + B.FPATR options 链式文档足够;bf16 options 触发编译器崩溃。

## FIXP 族(matrix postprocess options,matrix-postprocess.md)

`fixp::` options 挂在 TMATMUL/TGEMV 上配置 `B.FPATR`,承载 PostProcess(scalar/vector quant、
LReLU/PReLU、RowMax、GroupMax、MaxAbs、CScale)。matrix-postprocess.md 是本目录签名/示例最完整的文档。

| 接口 | 编译 | gfrun | 精度 | 说明 / 文档问题 |
|------|------|-------|------|-----------------|
| fixp::convert\<Mode\>() | ✅ | ✅ | ✅ | `convert<F322F16>()` → dst FP16;golden = A@B cast f16 |
| fixp::s8(desc)(scalar) | ✅ | ✅ | — | `make_s8_quant` descriptor builder;量化 golden 未写 |
| fixp::scalar\<Mode\>(desc) | ✅ | ✅ | — | generic;QF322S8Pre(详述 16 dtype 约束) |
| fixp::s8(tile)(vector) | ✅ | ✅ | — | vector-quant 快捷式;quant Tile 物理 2×32 valid 1×32 |
| fixp::vector\<Mode\>(tile) | ✅ | ✅ | — | generic vector-quant;VQF322F16Pre → dst FP16 |
| fixp::s8(desc).lrelu(fp19) | ✅ | ✅ | — | scalar quant + LReLU 链式 |
| fixp::f16().prelu(tile) | ✅ | ✅ | — | convert + PReLU;FP19 Tile 物理 2×32 valid 1×32 |
| fixp::keep_acc().row_max(in,out) | ✅ | ✅ | — | 累加式 RowMax(RowMaxInit=1);RM tile valid M×1 |
| fixp::keep_acc().group_max\<8\>(out) | ✅ | ✅ | — | GroupMax;out valid M×ceil(N/GroupN)=32×4 |
| fixp::keep_acc().cscale(scale) | ✅ | ✅ | — | TMATMUL_ACC + CScale(U8 CUBE_M32);跑通 |
| .row_max().group_max().max_abs() | ✅ | ✅ | — | 链式 max_abs 组合;跑通 |

FIXP 小结:11 case,**1 精度PASS(convert)/ 10 run-only / 0 编译失败 / 0 run-fail**。全部编译 + gfrun 跑通;
除 convert 外均为量化/postprocess,需 RNE+饱和 spec golden,尚未写独立 golden。
- **编译期约束发现**:B.FPATR 表列 QF322S16Pre→S16,但 fp16 矩阵输入 + S16 dst 被 static_assert 拒;
  表未给 PreQuantMode↔输入矩阵 dtype 兼容矩阵(详述 16)。

## misc

| 接口 | 编译 | gfrun | 精度 | 说明 / 文档问题 |
|------|------|-------|------|-----------------|
| reinterpret_tile | ❌ | - | — | **[实现]** reinterpret-tile.md 签名/示例完整;`reinterpret_tile<int32_t>(src)` 视图被后续 `TABS` 拒(`no matching function`)(详述 14) |

---

## 未覆盖清单

matrix-postprocess.md 列出的部分 CUBE 操作族尚未写 demo,集中登记于此。

| 未覆盖项 | 域 | 文档位置 | 原因 |
|----------|----|----------|------|
| TMATMUL_MX 带 scale 全组(单/双 scale + `_BIAS`/`_ACC`) | cube | matrix-postprocess.md L44-49 | 基础形 `tmatmul_mx`(无 scale)已 run-only;带 scale 变体需先钉 scale 载入/语义 |
| TGEMV 全族(`_BIAS`/`_ACC`/`_MX`/`_MX_BIAS`/`_MX_ACC`) | cube | matrix-postprocess.md L50-58 | 基础形 `tgemv` 已 run-only;变体成批复现同一语义待钉项 |
| Shared Right(`SharedTile<RightTile>` 作 B) | cube | matrix-postprocess.md L377-398 | 依赖 Shared tile 构造,docs/tileop-usage 全目录无 Shared tile 构造示例(详述 12),无法写出可编译 demo |

---

## 重点文档 / 实现问题详述

### 1. [签名][语义][dtype] TSEL / TSELS — 三源元组契约缺失

engines.md 仅列名字与分类。实测:
- **TSEL**:`(dst, src0, src1)` 三参编译通过,gfrun 断言 `srcs.size()==3 && ... "select first
  B.IOT requires mask then true/source Tile"`——运行期要求 mask + true/source 三源元组;文档未写
  「按什么条件选哪个」的 select 语义与三源结构。
- **TSELS**:`(dst, src0, scalar, src1)`,标量夹在两 tile 源中间;gfrun 断言 `"TSELS requires mask
  and source Tile operands"`。签名反直觉,语义/源结构文档均缺。

### 2. [签名] TEXPANDS — 无 src 的纯标量填充

真实签名 `TEXPANDS(dst, scalar)` 仅 2 参,无 src tile——「标量广播填充整个 tile」。分类栏
`tile-scalar-and-immediate` 有误导性(看似 tile⊕scalar)。

### 3. [签名] 系统性缺口:VEC 全族缺 C++ 签名

除 cmp.md 外,VEC 算子在文档里无一给出 C++ 调用签名或示例。engines.md 是「汇编投影表」,非
C++ API 文档;只能靠 dst-first 惯例猜,遇 TSEL/TSELS/TEXPANDS 这类非常规签名必然猜错。
**建议**:为每个算子补一行 C++ 签名。

### 4. [语义] TCMP/TCMPS — packed predicate 结果与 TSTORE 源契约不符

cmp.md 给出完整 `TCMP<Mode>(dst,src0,src1)` / `TCMPS<Mode>(dst,src,scalar)` 签名 + dtype 表 + 示例,
编译通过。gfrun 断言 `srcs.size()==2 && ... "Local TSTORE requires one legal source Tile descriptor"`
——比较结果为 packed predicate 载体,与常规 tile 的 TSTORE 源契约不符;文档未说明如何落盘/消费该结果。

### 5. [dtype] TROW*/TCOL* reduce — 输出 tile 形状规则缺失

一元 `(dst, src)`,src 为 M×N。dst 形状:row-reduce dst 须物理 M×N + ValidCol=1;col-reduce dst 须
物理 M×N + ValidRow=1(物理 M×1 非法:fp32 需 Cols×bits%256==0)。文档未写此「物理满宽 + valid 轴=1」
规则,靠 gfrun 断言反推。**建议**:补 reduce 族输出形状契约。

### 6. [dtype] TROWEXPAND*/TCOLEXPAND* — 广播源形状不对称

- **arith 变体(各 7,gfrun 通过)**:二元 `(dst M×N, src0 M×N, src1_broadcast)`。col-expand 的 src1
  用物理 M×N + ValidRow=1(1×N 行广播);row-expand 的 src1 用物理 M×1 列广播(gfrun 断言要求物理单列)。
  两侧广播源形状规则不对称,文档均未给。
- **copy-expand(TROWEXPAND/TCOLEXPAND)**:纯 `(dst, src)`,gfrun 断言 `"COPY expansion requires
  one broadcast source"`;广播源构造文档无说明。

### 7. [语义] argmax/argmin — 输出 dtype 契约缺失

TROWARGMAX/TROWARGMIN/TCOLARGMAX/TCOLARGMIN 编译通过,gfrun 断言 `"typed Local TSTORE source
dtype must match the block dtype"`——索引输出的 dtype 与 block dtype 不匹配;文档未给 argmax 的
输出 dtype/形状契约。

### 8. [示例] TMRGSORT — sort.md 示例本身不可编译

sort.md 示例三个 tile 全用 1×256:`TMRGSORT(out, a, b)`,但 static_assert 要求
`dst.ValidCol == left.ValidCol + right.ValidCol`(`256 != 256+256`)。修正为两源各 1×128、dst 1×256
后编译通过,gfrun `"TMRGSORT not yet fully implemented"`(模型未实现)。**建议**:改示例为
`left/right=1×128, dst=1×256`。

### 9. [签名] layout / irregular 无签名族

layout.md 只有散文,不给 TCONCAT/TEXTRACT/TINSERT/TFILLPAD/TTRANS 任何 C++ 签名;
TTRI/THISTOGRAM/TGATHER/TSCATTER/TPART* 连专题文档都无。靠 dst-first 惯例:
- 惯例命中(9):TCONCAT/TFILLPAD/TTRANS/TPART{ADD,MUL,MAX,MIN}。
- 猜错编译失败(6):TEXTRACT/TINSERT/TTRI/THISTOGRAM(`no matching function`)、TGATHER/TSCATTER
  (后端 `Match Instruction Error`)。
**建议**:layout/irregular 每算子补签名 + 示例。

### 10. [语义] TIMG2COL(卷积)— 模型断言桩

照 TIMG2COL.md 示例(普通 `Tile<Vec,float,8,256>` + TLOAD + `TIMG2COL(dst,src,3,5)`)写,编译通过,
gfrun `"TIMG2COL not yet fully implemented"`——模型侧卷积窗口/repeat/padding 契约未实现。文档示例本身
不用持久 Matrix-location feature-map 描述符源,故 demo 亦不构造(超出文档驱动范围)。运行受阻属模型缺口。

### 11. [签名] TPREFETCH src 类型 / GMOV 后端拒

- **TPREFETCH**:文档签名 `TPREFETCH(src, valid_col, valid_row)` 未说明 src 类型;iterator 视图编译报
  `no member named 'Cols'`,改传 static RowMajor `global_tensor` 本体后通过。应点明 src 须带 compile-time
  列数的 global_tensor 本体。
- **GMOV**:tlsu.md 给示例 `GMOV<15>(dst, peer_tid, src)`,照抄后端 `Match Instruction Error`;文档示例
  与后端不一致。

### 12. [签名][构造] Shared TMOV 四变体 + TSTORE_PART

tlsu.md 提到 Shared TMOV 四变体(TMOV_L2S_INSERT/PUBLISH、TMOV_S2L_BROADCAST/EXTRACT)与 TSTORE_PART,
但既未给签名,也未给 Shared-location tile 的 C++ 声明/构造方式。docs/tileop-usage 全目录无一处 Shared tile
构造示例,仅凭文档无法写出可编译 demo。这是 TLSU Shared 面的系统性文档缺口。

### 13. [签名][示例][dtype] CUBE:Bias 形状 / bf16 / MX / TGEMV

- **[签名] Bias 形状**:matrix-postprocess.md 只说「辅助参数仍是普通 Local Tile」,未说 dtype/valid shape。
  实测 Bias 须普通 RowMajor + 派生 AccType(fp16 输入→FP32)+ valid `1×N`,且用普通 `TLOAD`(非 TLOAD_CUBE)。
- **[签名] TLOAD_CUBE/TSTORE_CUBE 无 C++ 签名**:cube.md 只说「用它们做 GM 转换边界」;`(cube_tile,
  global_tensor)` 二参惯例命中。
- **[示例] tmatmul_bf16**:照 `TMATMUL(dst_bf16,a,b,fixp::bf16())` 写,`__bf16` 累加器,clang frontend
  abort(exit 134)——编译器崩溃。
- **tmatmul_mx / tgemv**:无 scale 便捷式 / 基础 gemv 均编译 + gfrun 跑通(run-only);MX scale 语义与
  matrix-vector 输出形状文档未钉,未写 golden。

### 14. [实现] reinterpret_tile — 返回视图不被下游 tileop 接受

reinterpret-tile.md 签名 + 示例完整(同位宽、Local only、视图非拥有)。demo 照写 `fp32 tile →
reinterpret_tile<int32_t>` 得视图,随后 `TABS` 消费该视图,编译报 `no matching function for call to 'TABS'`
——视图类型不被后续 tileop 的模板匹配接受。问题在视图类型与 tileop 入参兼容性。

### 15. [契约] range::Subview / range::Assemble

range-modifiers.md 给出完整模板签名 + 示例 + 合法值域 + 发射 asm。
- **range::Assemble**:照示例参数(ParentSizeCode=12)编译即被 static_assert 拒 `B.ASSEMBLE length
  cannot exceed the parent Tile capacity`(SizeCode 反推容量 < 声明容量)。
- **range::Subview**:照示例编译通过,gfrun `illegal TSTORE operand or descriptor contract`——运行期拒
  按文档示例参数生成的 descriptor。签名/汇编层无缺口,受阻在编译期容量断言或 gfrun descriptor 契约。

### 16. [契约][dtype] FIXP postprocess

- **11 个 postprocess demo 全部编译 + gfrun 跑通**(含 cscale、max_abs 链式)。除 convert 外均为量化/后处理,
  需 RNE+饱和 spec golden,尚未写独立 golden。
- **[dtype] QF322S16Pre 输入约束**:B.FPATR 表列 QF322S16Pre→S16,但 fp16 矩阵输入 + S16 dst 编译即被
  static_assert 拒(`PreQuantMode incompatible with the derived matrix accumulator type`)。表只给「mode→dst
  dtype」,未给 mode↔输入矩阵 dtype 兼容矩阵(S16 量化疑似要求整数矩阵输入、派生 S32 accumulator)。generic
  scalar demo 遂用 QF322S8Pre。**建议**:补 PreQuantMode↔输入 dtype 合法组合表。

### 17. [示例] MGATHER / MSCATTER — 示例 offset dtype 违反同页 dtype 表

MGATHER.md / MSCATTER.md 给出完整签名 + 示例:`MGATHER(dst, gmSrc, offsetTile)`(base/stride 由 gm 源
携带,offset 存字节位移)、`MSCATTER(base_gm, src, offsetTile)`。照签名重写后 gfrun 跑通,golden 逐元素 PASS。
- **[示例]** 两页的使用示例都把 offset 声明为 `Tile<Vec,uint16_t,...>`,但同页 dtype 表规定索引 Tile 必须是
  S32/U32/S64/U64。照示例用 uint16 编译能过,gfrun 拒 `"illegal MGATHER operand or descriptor contract"`。
  demo 遵从规范 dtype 表(U32)。**建议**:把示例 offset 类型改为 uint32_t。

### 18. [后端] MGATHER_MASK / MSCATTER_MASK — 文档签名完整,后端无法汇编

MGATHER_MASK.md / MSCATTER_MASK.md 给出完整签名 + dtype 表 + 带 `TmaPadValue::Zero` 的示例
(`MGATHER_MASK<out,off,mask,gm,Pad>(dst, base_gm, offset, mask)`,mask=uint8)。照签名写,clang 报
`Match Instruction Error!`——发射的 `BSTART.TLSU MGATHER.MASK` 双 `B.IOT`(IndexTile + MaskTile)bundle
形式无法被后端匹配/汇编。
- offset 用 U32(dtype 表)或 uint16(示例原样)**都崩在同一行同一错误**(static_assert 已过,崩在指令匹配),
  排除操作数类型因素;同族无 mask 的 MGATHER/MSCATTER 照文档写一次通过 + golden PASS。属后端未实现该指令
  编码,与 GMOV/TMOV 的 `Match Instruction Error` 同类。
- golden 已按语义就绪(`golden.py` fam=`gather_mask`:`where(mask==1, base[off//4], 0)`),待后端补齐即可转精度PASS。
- **建议**:补齐后端 `MGATHER.MASK`/`MSCATTER.MASK` 汇编支持,或在文档标注该变体当前不可用。

### 19. [契约] TileArray region API(TPARTVIEW / TileArray / TASSEMBLY)

range-modifiers.md / range-modifiers-developer-guide.md 新增的 TileArray region API。demo 看护
`region::TileArray` + `TASSEMBLY` + TCVT slot producer。
- **文档示例不可直接落地**:示例 `TCVT(destinations[0][2], source_tile)` 用临时量,而 region `TCVT` 签名是
  `TCVT(TileArrayOutputRef& dst, In& src)`(非 const 左值引用),临时量无法绑定;示例源变量名 `source` 与
  `source_tile` 不一致——照抄不可编译,须绑具名左值。
- 用 TPARTVIEW 父 strided 子视图作源、或独立紧凑 Fragment 作源,gfrun 均断言 `raw tile spill source does
  not fit the carrier shape`。结合文档自身 "until the … path is implemented and validated" 告警,判定
  region producer 路径在当前模型上尚不可运行(run-fail)。
