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
run-date : 2026-09-03 (第四轮:tcvt/tmatmul_mx/tgemv 补独立 golden)
toolchain: env_test/linx-toolchain-build/output/linx_blockisa_llvm_musl
           clang++ md5 b6201631d2fdb77c6ad541c2c769460e  (第二轮为 1ee479a3;工具链已升级)
gfrun    : env_test/SuperScalarModel/bin/gfrun
           gfrun   md5 04ca39ece7533eb35c805a4741996ebb  (不变)
```

> **工具链升级影响（第三轮干净重建发现）**：clang++ 由 `1ee479a3`→`b6201631`。新后端修复了
> 一批 `*.MASK`/GMOV/gather-scatter 的 `Match Instruction Error`：**MGATHER_MASK 现编译+跑通+golden
> 通过（run-fail→精度PASS）**、MSCATTER_MASK 现可跑（补 golden 后精度PASS）；而 TGATHER/TSCATTER/GMOV
> 由「编译失败(后端)」转为「gfrun 运行断言」(编译失败→run-fail，暴露下游模型缺口)。故本轮编译失败
> 7→2、run-fail 5→8 的口径变化中，一部分来自工具链升级而非 demo 改动。

## 状态总表

**合计 126 case:94 精度PASS / 22 run-only / 2 编译失败 / 8 run-fail(崩)。**
（第一轮 125=52/47/12/14；第二轮读头修正签名 61/53/7/5；第三轮为可钉语义的 run-only 补独立 golden
91/25/2/8；第四轮 tcvt/tmatmul_mx/tgemv 补 golden **精度PASS +3**、run-only −3。编译失败/run-fail
口径变化含工具链升级副作用，见上「工具链升级影响」。）

| 域 | 精度PASS | run-only | 编译失败 | run-fail | 合计 |
|----|---------|----------|----------|----------|------|
| vec  | 35 | 0  | 0 | 0 | 35 |
| sfu  | 45 | 5  | 1 | 5 | 56 |
| tlsu | 4  | 7  | 0 | 2 | 13 |
| cube | 8  | 0  | 1 | 0 | 9  |
| fixp | 1  | 10 | 0 | 0 | 11 |
| misc | 1  | 0  | 0 | 1 | 2  |
| 合计 | **94** | **22** | **2** | **8** | **126** |

- **精度PASS**:编译 + gfrun 跑通 + host 独立 golden 逐元素比对通过(真「算对」)。
- **run-only**:编译 + gfrun 跑通,但未写独立 golden(语义/舍入未由文档钉死),仅验「能跑」。
- **编译失败**:文档无法据以写出可编译调用,或后端拒绝汇编。
- **run-fail**:gfrun 崩(assert/fault),无输出可校。

**各表「精度」列**:✅ = 独立 golden 逐元素通过;❌ = golden 比对失败;— = 未做精度校验
(编译/运行失败无输出,或该接口未写 golden)。当前无 ❌(precision-fail = 0)。

## 第二轮修正:读头文件订正签名/契约（2026-09-02）

放宽「纯文档驱动」限制后，允许读工具链头 `jcore/template_asm.hpp`（+ 少量模型断言）订正
**因文档信息不足而写错的 demo**。甄别原则：凡「签名/形状/dtype 猜错」→ 本轮修正；凡后端
`Match Instruction Error`、clang abort、模型未实现桩 → 确保用**正确签名**调用，使失败原因确属
对方缺口（另见三份 issue）。逐条结果：

| demo | 旧态 | 根因（我方 demo 错） | 新态 |
|------|------|----------------------|------|
| TSEL/TSELS/TCMP/TCMPS | run-fail×4 | TCMP 产 packed predicate 不能直接 TSTORE；TSEL/TSELS 的 mask 必须是 predicate。正确用法=`TCMP→predicate→TSEL/TSELS 消费→存 select 结果` | **精度PASS×4** |
| TROW/TCOLARGMAX/ARGMIN | run-fail×4 | 输出 dtype 必须 UINT32（模型 argReduce 强制），旧用 float | **精度PASS×4** |
| TTRI | 编译失败 | 真实签名 1 参 `TTRI(dst)`，旧写 2 参 | run-only |
| TINSERT | 编译失败 | 真实签名 `(dst,src,indexRow,indexCol)`，旧写 3 参；能跑但模型只插 patch 左上部分窗口、且不保留 base，语义未由文档钉死 | run-only |
| TROWEXPAND/TCOLEXPAND(copy) | run-fail×2 | 需广播源(M×1 / 1×N)。用对广播源后**能跑**，但模型填充宽/高被源 ValidCol/Row 钉死为 1（退化 expand，header 与实现矛盾），不设 golden | run-only×2 |
| range::Assemble | 编译失败 | `ParentSizeCode` 须匹配 tile 容量：4×8 float=128B→SizeCode 1，文档示例的 12(256KiB)越界 | run-only |
| MSCATTER_MASK | 编译失败 | 旧写 6 参猜错；真实 4 参 `(base_gm,src,off,mask)`。改对后仍后端 `Match Instruction Error`（与 MGATHER_MASK 同族，真后端缺口） | 编译失败(后端) |
| reinterpret_tile | 编译失败 | 旧 `TABS(普通tile,视图)` 混用类型不合(**非** TABS 拒视图)。订正：TABS/TANDS 已接受视图(双视图即可)；落盘按模型 cross_model 测试 `bf16_backing_tands_u16_then_tmuls_bf16` 的「整数视图 op → 原生类型 op 重置 dtype 标签 → 存」模式 | **精度PASS** |
| reinterpret_tcmp（新增 witness） | — | 同款 bitcast 但消费 op 换 TCMP：`TCMP<out,in>` 分离模板参→普通 predicate 输出 + int32 视图源**编译过**,但仿真器 TCMP 处理器 `IsCompatibleDataTile` **拒视图源**(崩在 TCMP，早于 store);对照 TANDS+同视图 PASS、普通 tile+TCMP PASS→差异是视图。官方 reinterpret 修复在**模型层遗漏 TCMP** | run-fail(模型缺口,witness) |
| TEXTRACT | 编译失败 | 真实签名 `(dst,src,indexRow,indexCol)`。改对后编译过,gfrun 拒 descriptor。**根因(第二轮定)**:header 把**源的 valid 维**发成 block 维,而模型 `ValidateV058SpecialTepl` 校验 `rowOffset+blockValidRow>source.validRow` + dst 尺寸——任何真实抽取(offset>0 或 dst<源)必 illegal,仅 offset=0+dst==源全尺寸的退化恒等才过。header 发的 block 维与「抽取区域维」不一致(头/模型契约不符,非 demo 错) | run-fail(契约缺口) |
| THISTOGRAM | 编译失败 | 真实签名 `(dst,src,Idx,ByteId)`。改对后仍后端 `Match Instruction Error`(TEPL 104) | 编译失败(后端) |
| region_tilearray | run-fail | 干净重编后跑通（此前 run-fail 疑为增量缓存陷阱；gfrun-3 或已随模型版本解） | run-only |

**仍为真缺口（正确签名下失败，属对方）**：编译失败 7=THISTOGRAM/TGATHER/TSCATTER/GMOV/MGATHER_MASK/
MSCATTER_MASK 后端 Match Error + TMATMUL bf16 clang abort；run-fail 5=TEXTRACT/range::Subview 模型
descriptor 契约、TIMG2COL/TMRGSORT 模型未实现桩、**reinterpret_tcmp（TCMP+视图被模型 TCMP 处理器拒，
witness）**。

## 第三轮:为可钉语义的 run-only 补独立 golden（2026-09-03）

对语义能从 ISA 规格/文档独立钉死的 run-only 接口补 host golden，转精度PASS。均走 res_check
（host numpy 生成输入 + ELF 落盘 out.bin + numpy 独立复算）。共 **+29 精度PASS**：

| 族 | 接口（数量） | golden 语义 | 容差 | 备注 |
|----|------|-------------|------|------|
| SFU 超越 | texp/tlog/trecip/tsqrt/trsqrt（5） | exp / **log₂** / 1/x / √x / 1/√x | rel 1e-5 | **TLOG 是 log₂ 不是 ln**（实测证实 TLOG(4.25)=2.0875=log₂）；TEXP 是 base-e。但 TLOG.md 文档写「natural logarithm」→ 模型与文档不一致，已提 gfrun-6。SFU 实测误差 ~6e-8（近 fp32 精确） |
| SFU expand-arith | trow/tcol × add/sub/mul/div/max/min/expdif（14） | dst[i,j]=f(src0[i,j], src1 广播);row 广播 M×1、col 广播 1×N;EXPDIF=exp(src0−src1) | rel 1e-5 | 语义取自 docs/intrinsics/t{row,col}expand*.md |
| SFU layout/sort | ttrans/tconcat/tsort（3） | 转置 src^T / 列拼接 [a\|b] / 逐行升序排序 | 0~1e-6 | tsort 用不重复值避并列（值校验，索引未校） |
| SFU irregular | tpartadd/mul/max/min（4） | elementwise;"PART"=partial-valid-region 非分段（docs 证实），全有效即普通 elementwise | 1e-4 | |
| SFU quant | tquant/tdequant（2） | quant=clamp(round_RNE(src),−128,127);dequant=(src−zp)*mult | 0 | 见下 TQUANT 缺口 |
| TLSU | mscatter_mask（1） | base[off//4]←src where mask==1（单射 offset） | 0 | 工具链升级后可跑;补 golden 即 PASS |

**TQUANT multiplier/zeroPoint 被 emulator 忽略（第三轮发现，已提 issues/ISSUE_gfrun_NA_model_gaps.md gfrun-5，根因反汇编隔离到模型侧）**:TQUANT 签名
`TQUANT<Mode,Sat>(dst,src,float multiplier,int zeroPoint)`。实测传 `multiplier=0.5, zeroPoint=1` 时输出
= `clamp(round_RNE(src), −128,127)`（**逐字节 0/2048 匹配 mult=1/zp=0**），即 **multiplier 与 zeroPoint 被静默
忽略**（斜率拟合 slope≈1.0）。故 demo 改用 identity 参数（mult=1/zp=0）精确看护它**确实执行**的核心
——RNE 舍入 + S8 饱和（输入跨 ±256 触发双端 clamp）;被忽略的缩放路径单独记录，不在此断言。

**仍 run-only（无法钉精度，诚实上限）**:tfillpad/tinsert/ttri/copy-expand（退化/部分语义）、
tload/tstore/tmov/tprefetch/mgather_cas/range_assemble/region_tilearray（纯搬运/视图无数值语义）、
**fixp postprocess 族 10 个**（matmul+FP19 scale 描述符量化，独立 golden 需先解码 FP19 格式 + 匹配硬件
量化路径，是独立深水区，待后续）。

## 第四轮:tcvt / tmatmul_mx / tgemv 补独立 golden（2026-09-03）

第三轮遗留的「好啃」run-only 三个，均转精度PASS（走 res_check，无回归）：

| 接口 | 域 | golden 语义 | 结果 |
|---|---|---|---|
| **tcvt** | vec | fp32→s32 数值转换，舍入=**RNE**（round-half-to-even，实测命中；输入跨双符号+含 .5 中点，`fam='cvt' round='rne'`） | run-only→**精度PASS** |
| **tmatmul_mx** | cube | FP16 pair 无 scale，数学 == 普通 f16 matmul，复用 `fam='matmul'` D=A@B | run-only→**精度PASS** |
| **tgemv** | cube | D(1,N)=Vec(1,K)@Mtx(K,N)，即 M=1 的 matmul（in_a=vec, in_b=mtx），复用 `fam='matmul'` | run-only→**精度PASS** |

- **TCVT 舍入钉死为 RNE**:candidate `np.rint`(round-half-to-even) 首击逐元素通过，无需回退到 RTZ/floor。
- tmatmul_mx/tgemv 无需新 golden 族——GEMV 是 M=1 的 GEMM，MX-f16 是无 scale 的 GEMM，两者数学与 matmul 同，仅复用现成 `check_matmul`。
- MX 的 scale 侧（FP8/FP4 带 scale operand）语义仍未钉，不在此三者内。

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

### 已带独立 golden 的接口(第一~二轮 61 + 第四轮 3 = 64;第三轮 +30 见上「第三轮」节表，合计 94)

| 域 | 接口 | golden 语义 | 容差 |
|----|------|-------------|------|
| VEC 二元 | tadd/tsub/tmul/tdiv/tmax/tmin(f32)、tand/tor/txor/trem/tshl/tshr(i32) | elementwise;移位/rem 取正保证逻辑=算术、无溢出 | 整数 eps=0 / f32 1e-4 |
| VEC 一元/标量 | tabs/tneg/trelu/tnot、tfma、t{add,sub,mul,div,max,min}s、t{and,or,xor,rem,shl,shr}s、texpands | abs/neg/relu/not、a*b+c、tile⊕标量、标量填充 | 同上 |
| **VEC 比较/选择** | tcmp/tcmps/tsel/tsels（**新增**） | 链式 `TCMP→predicate→TSEL/TSELS`;out=where(a⋛b\|s, tru, prior)（tsels: mask?src:标量） | eps=0 |
| SFU reduce | trow{sum,max,min,prod}、tcol{sum,max,min,prod} | 沿轴 sum/max/min/prod;row 比 out[r*N+0]、col 比 out[c] | 1e-3 |
| **SFU argmax** | trow/tcol{argmax,argmin}（**新增**） | 沿轴 arg 索引，UINT32 输出;row 比 out[r*N+0]、col 比 out[0*N+c];源用双重排列避并列 | eps=0 |
| SFU create-index | tci、tci_desc、tci_s16、tci_u32、tci_u16 | iota:asc=start+k / desc=start−k(按元素位宽 wrap),ValidRow=1 | 整数 eps=0 |
| CUBE matmul | tmatmul、_acc、_bias、_f16、_relu、_rowmax、**tmatmul_mx、tgemv（第四轮）** | A@B(f16→f32 累加);+C / +bias(1×N) / relu / f16 输出;MX-f16 无 scale == matmul、tgemv=M=1 GEMM | f16 rel 2e-2~3e-2 |
| **VEC cvt** | tcvt（**第四轮**） | fp32→s32 数值转换，舍入=RNE(round-half-to-even) | 整数 eps=0 |
| TLSU gather/scatter | mgather、mscatter | GM base + U32 字节位移;gather=base[off//4]、scatter=base[off//4]←src(单射无碰撞) | eps=0 |
| FIXP | convert | A@B 后 cast f16 | 3e-2 |
| **misc** | reinterpret_tile（**新增**） | fp32→int32 视图 + TANDS 清符号位 + TMULS 重置标签 → \|x\| | eps=0 |

实测 tmatmul `max|out−A@B|=0.0`(f16 matmul 与 numpy 逐字节一致)。

### 未写 golden 的接口(run-only,原因) — 截至第四轮的真实剩余

（超越/expand-arith/tpart/layout-sort/tquant-dequant/tcvt/tmatmul_mx/tgemv 均已在第三、四轮补齐，
从本表移除。）

- **fixp postprocess 族 10 个**:s8/vector quant、lrelu/prelu、rowmax_acc、group_max、cscale、chain。
  matmul + FP19 scale 描述符量化，独立 golden 需先解码 FP19 格式 + 匹配硬件量化路径，是独立深水区，待后续。
- **退化/部分语义无法设 golden**:tfillpad/tinsert/ttri/copy-expand(t{row,col}expand 退化)。
- **通路/搬运类**:tload/tstore/tmov/tprefetch/mgather_cas/range_assemble/region_tilearray
  (无独立数值语义可校)。

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
| TCVT | ✅ | ✅ | ✅ | 第四轮:fp32→s32 舍入实测=**RNE**(round-half-to-even),补 golden 精度PASS |
| TSEL | ✅ | ❌ | — | **[签名][语义][dtype]** gfrun 要 mask+true+false 三源元组(详述 1) |
| TSELS | ✅ | ❌ | — | **[签名][语义]** gfrun 要 mask+source 三源(`TSELS requires mask and source Tile`)(详述 1) |
| TCMP | ✅ | ❌ | — | **[语义]** cmp.md 签名完整;gfrun 结果为 packed predicate,与 TSTORE 源契约不符(详述 4) |
| TCMPS | ✅ | ❌ | — | 同上 |

VEC 小结:35 case,**35 精度PASS / 0 run-only / 0 run-fail**（第四轮 tcvt 补 golden 转精度PASS，
舍入=RNE；第二轮：比较/选择族 tsel/tsels/tcmp/tcmps 由 run-fail 转精度PASS，见「第二轮修正」表；
上表这四行的旧「❌」状态已过时）。VEC 域已全部精度看护。
除 cmp.md 外,VEC 全族文档不给 C++ 签名(engines.md 只有「名字 + 汇编 + 分类」三列,详述 3)。

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

SFU 小结:56 case,**45 精度PASS / 5 run-only / 1 编译失败 / 5 run-fail**（第三轮更新）。
- 精度PASS(45):8 reduce + 5 TCI + 4 argmax/argmin + **5 超越 + 14 expand-arith + ttrans/tconcat/tsort
  + 4 tpart + tquant/tdequant（第三轮补 golden，见「第三轮」节）**。
- run-only(5):tfillpad/tinsert/ttri/trowexpand/tcolexpand（退化/部分语义，不设 golden）。
- 编译失败(1):THISTOGRAM——后端 `Match Instruction Error`。（TGATHER/TSCATTER 工具链升级后转 run-fail。）
- run-fail(5):TEXTRACT(模型 descriptor 契约)、TIMG2COL/TMRGSORT(模型未实现桩)、**TGATHER/TSCATTER
  (工具链升级后编译过，gfrun `ASSERTION FAILED`——下游模型缺口)**。

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
| range::Subview | ✅ | ❌ | — | **[dtype/契约]** 第二轮根因:parent **须 cube 布局**(模型 Block.cpp:1052 `IsCubeLayout`+`CubeCellDescribeSubview`),Vec tile 被运行期拒;文档示例用 Vec 且未说此要求(详述 15) |
| TileArray region API(region_tilearray) | ✅ | ❌ | — | **[契约]** TPARTVIEW/TileArray/TASSEMBLY;gfrun `raw tile spill source does not fit the carrier shape`(详述 19) |

TLSU 小结:13 case,**4 精度PASS / 7 run-only / 0 编译失败 / 2 run-fail**（第三轮更新）。
- 精度PASS(4):MGATHER、MSCATTER、**MGATHER_MASK（工具链升级后 `*.MASK` bundle 可汇编+跑通+golden 通过）**、
  **MSCATTER_MASK（工具链升级后可跑，第三轮补 golden）**。
- run-only(7):TLOAD/TSTORE/TPREFETCH/TMOV/MGATHER_CAS/range::Assemble/region_tilearray。
- 编译失败(0):——原 GMOV/MGATHER_MASK/MSCATTER_MASK 的后端 `Match Instruction Error` 已随工具链升级消失。
- run-fail(2):range::Subview(模型 descriptor 契约)、**GMOV(工具链升级后编译过，gfrun `GMOV source/dest
  descriptors must match` 断言——下游模型缺口)**。
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
| TMATMUL_MX | ✅ | ✅ | ✅ | 第四轮:FP16 pair 无 scale == 普通 f16 matmul,复用 matmul golden 精度PASS(带 scale 变体未钉) |
| TGEMV | ✅ | ✅ | ✅ | 第四轮:M=1 的 GEMM,复用 matmul golden 精度PASS(_BIAS/_ACC/_MX 变体未钉) |
| TMATMUL(bf16) | ❌ | - | — | **[示例]** 照 `fixp::bf16()` 写→clang frontend abort(exit 134)(详述 13) |

CUBE 小结:9 case,**8 精度PASS / 0 run-only / 1 编译失败(bf16)/ 0 run-fail**（第四轮 tmatmul_mx/tgemv
补 golden 转精度PASS，复用 matmul 族，见「第四轮」节）。核心 matmul 通路 + B.FPATR options 链式文档
足够;唯 bf16 options 触发编译器 clang frontend abort(exit 134)。

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
| reinterpret_tile | ✅ | ✅ | ✅ | **[已解]** 第二轮：旧 `TABS(普通tile,视图)` 混用类型不合(非 TABS 拒视图);TABS/TANDS 已接受视图，落盘按 cube cross_model 的「整数视图 op→原生类型 op 重置标签→存」模式 → **精度PASS(golden=abs)** |
| reinterpret_tcmp | ✅ | ❌ | — | **[模型缺口 witness]** 同款 bitcast 消费 op 换 TCMP：编译过(TCMP out/in 分离模板参),但仿真器 TCMP 处理器 `IsCompatibleDataTile` 拒视图源("TCMP requires two compatible Tile sources",早于 store);官方 reinterpret 修复在模型层遗漏 TCMP(TANDS 接受、TCMP 不接受)。应提 [gfrun][NA] |

---

## 未覆盖清单

matrix-postprocess.md 列出的部分 CUBE 操作族尚未写 demo,集中登记于此。

| 未覆盖项 | 域 | 文档位置 | 原因 |
|----------|----|----------|------|
| TMATMUL_MX 带 scale 全组(单/双 scale + `_BIAS`/`_ACC`) | cube | matrix-postprocess.md L44-49 | 基础形 `tmatmul_mx`(无 scale)第四轮已精度PASS;带 scale 变体需先钉 scale 载入/语义 |
| TGEMV 全族(`_BIAS`/`_ACC`/`_MX`/`_MX_BIAS`/`_MX_ACC`) | cube | matrix-postprocess.md L50-58 | 基础形 `tgemv` 第四轮已精度PASS;变体成批复现同一语义待钉项 |
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
- **tmatmul_mx / tgemv**:无 scale 便捷式 / 基础 gemv 均编译 + gfrun 跑通。第四轮已补 golden 转精度PASS
  (MX-f16 无 scale == 普通 matmul、GEMV == M=1 GEMM，复用 matmul 族)；仅 MX 带 scale 变体的 scale 语义待钉。

### 14. [实现] reinterpret_tile — 返回视图不被下游 tileop 接受

reinterpret-tile.md 签名 + 示例完整(同位宽、Local only、视图非拥有)。demo 照写 `fp32 tile →
reinterpret_tile<int32_t>` 得视图,随后 `TABS` 消费该视图,编译报 `no matching function for call to 'TABS'`
——视图类型不被后续 tileop 的模板匹配接受。问题在视图类型与 tileop 入参兼容性。

### 15. [契约] range::Subview / range::Assemble

range-modifiers.md 给出完整模板签名 + 示例 + 合法值域 + 发射 asm。
- **range::Assemble**:照示例参数(ParentSizeCode=12)编译即被 static_assert 拒 `B.ASSEMBLE length
  cannot exceed the parent Tile capacity`(SizeCode 反推容量 < 声明容量)。
- **range::Subview**（第二轮根因）：parent **必须是 cube 布局 tile**——模型 `Block.cpp:1052 HandleBSubview`
  断言 `IsCubeLayout(parent)` 并走 `CubeCellDescribeSubview`;Vec/RowMajor parent 编译过(Subview 头不限制
  parent)但运行期被拒 `illegal TSTORE operand or descriptor contract`。`range-modifiers.md` 示例用 Vec tile
  且未说此要求（doc gap，同 TEXTRACT/TIMG2COL「需持久 cube/Matrix 源」族）。进一步：正确 cube demo
  (`CubeTileM32`+`TLOAD_CUBE`+`Subview`+存) 还会撞更多未文档化 cube 契约——`TSTORE_CUBE` 要求 GM/CUBE
  dtype 一致，且其 `const&` 形参与 `Subview::data()`(非 const) 不兼容——完整 cube-subview demo 需更深
  cube 支持/owner 澄清。当前留 Vec witness 精确定位 cube-parent 要求。

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
