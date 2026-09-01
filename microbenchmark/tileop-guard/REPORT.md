# TileOP-API v0.58 文档看护报告

本报告是"依据 `Linx-TileOP-API/docs/tileop-usage/*.md` 文档为每个接口写最小看护
demo"过程的**副产物**:记录文档中缺失、不足或与实际不符之处。

- **依据**:demo 仍仅依据 `docs/tileop-usage/` 写(不读 intrinsic 源码)。
- **分类**:①编译成 `.elf`;②gfrun 跑到 `Reach the End of Benchmark`;③(新增)**精度校验**——
  host 侧独立 golden 逐元素比对通过。见「精度校验(res_check)」小节。
- demo 目录:`microbenchmark/tileop-guard/{vec,sfu,tlsu,cube,fixp,misc}/`
- 跑批:`source env.sh && bash run_guard.sh <sub>`

文档问题分类:
- **[签名]** 签名/模板参数/参数个数缺失,只能靠编译器报错反推
- **[语义]** 计算语义未说明(如"这个 select 到底选什么")
- **[dtype]** dtype/shape 约束未说明,靠 gfrun assert 反推
- **[示例]** 文档示例本身不可编译/过时
- **[命名]** canonical 名与工具链实际暴露不符

## 基线指纹(结论只认此基线)

全部状态来自**一次干净重编重跑**:`make clean_all` 后全量 `run_guard.sh`(共享头改动不触发
增量重编,必须清缓存才可签字)。

```
run-date : 2026-09-01 15:34
toolchain: env_test/linx-toolchain-build/output/linx_blockisa_llvm_musl
           clang++ md5 c5a5edef0d9ca809dc368de7dbc2ad28
gfrun    : env_test/SuperScalarModel/bin/gfrun
           gfrun   md5 04ca39ece7533eb35c805a4741996ebb
```

> **env_test 二进制已较上次基线更新**(旧 clang 13660b56 / gfrun 848906bc → 新
> c5a5edef / 04ca39ec)。因此若干编译/运行分类发生漂移(如 TCMP/TCMPS 由「编译崩」变
> 「gfrun 崩」;TMATMUL_MX/TGEMV 由「gfrun 崩」变「run-only」)。本表为当前二进制下的重测结果。

**合计 120 case:45 精度PASS / 48 run-only / 14 编译失败 / 13 run-fail(崩)。**

| 域 | 精度PASS | run-only | 编译失败 | run-fail | 合计 |
|----|---------|----------|----------|----------|------|
| vec  | 30 | 1  | 0 | 4 | 35 |
| sfu  | 8  | 30 | 6 | 8 | 52 |
| tlsu | 0  | 5  | 6 | 1 | 12 |
| cube | 6  | 2  | 1 | 0 | 9  |
| fixp | 1  | 10 | 0 | 0 | 11 |
| misc | 0  | 0  | 1 | 0 | 1  |
| 合计 | **45** | **48** | **14** | **13** | **120** |

- **精度PASS**:host 独立 golden 逐元素比对通过(真「算对」)。
- **run-only**:跑到 end-of-benchmark 但本轮未写 golden(语义待定/暂缓);仅「能跑」,未验数值。
- **run-fail**:gfrun 崩(assert/fault),无输出可校。

---

## 精度校验(res_check)

原报告的「通过」只到「能编译 + 能跑完」,不含数值正确性。本轮放宽「只依赖文档、不越界」
约束,补上**独立 golden 精度校验**,让「跑通」升级为「跑通且算对」。

### 机制(res_check + host 侧独立 golden)

```
host  golden.py gen  → 按 case 语义 numpy 生成输入 compare/<sub>/<case>/in_*.bin
ELF   guard_read_bin(read() 系统调用整块读入) → TLOAD/op/TSTORE → guard_dump_bin(out.bin)
host  golden.py check → 读 in_*.bin + out.bin,numpy 独立算 ref,逐元素带容差比 → exit 0/1
```

- **独立性**:golden 用 numpy 从**接口语义**独立实现,不读 emulator 的 tile 实现,是真独立 oracle。
- **输入 host 生成 + `read()` 读入**:设备端填充会被 tile 后端错编导致输入退化(实测两 fill 调用
  被混叠、一个操作数塌成 0),故由 host 拥有输入、ELF 用 `read()` 整块读入(emulator syscall 直接
  写内存,绕过 tile-register 提升)。
- **无 printf dump**:工具链自带 `writeBinaryFile` 的尾部 `printf/fflush` 会死循环卡死 gfrun,改用
  自写 `guard_dump_bin`(open/write/close)。
- 校验代码:`golden/golden.py`(注册表 + 语义)、`common/guard_io.{h,c}`(fill/read/dump,以
  `-mlxbc -O2` 无 matrix flags 编译)、`common/guard_case.hpp`(case 宏)。

### 本轮覆盖(45 个精度PASS,均在语义无歧义域)

| 域 | 已带 golden(精度PASS) | 独立 golden 语义 |
|----|----------------------|------------------|
| VEC 二元 | tadd/tsub/tmul/tdiv/tmax/tmin(f32)、tand/tor/txor/trem/tshl/tshr(i32) | elementwise;移位取正、rem 取正保证逻辑=算术、无溢出 |
| VEC 一元/标量 | tabs/tneg/trelu/tnot、tfma、t{add,sub,mul,div,max,min}s、t{and,or,xor,rem,shl,shr}s、texpands | abs/neg/relu/not、a*b+c、tile⊕标量、填充 |
| SFU reduce | trow{sum,max,min,prod}、tcol{sum,max,min,prod} | 沿轴 sum/max/min/prod;row 比 out[r*N+0]、col 比 out[c] |
| CUBE matmul | tmatmul、_acc、_bias、_f16、_relu、_rowmax | A@B(f16→f32 累加);+C / +bias(1×N) / relu / f16 输出 |
| FIXP | convert | A@B 后 cast f16 |

容差:f32/整数精确(整数 eps=0);f16 matmul rel-eps 2e-2~3e-2(f16 舍入)。实测 tmatmul
`max|out-A@B|=0.0`(f16 matmul 与 numpy 逐字节一致)。

### 暂缓(run-only,未写 golden 的原因)

- **语义需读 ISA 才能独立实现**:expand-arith(t{row,col}expand{add,sub,mul,div,max,min}、expdif,
  文档未定义精确语义)、tpart*、tconcat/ttrans/tfillpad(layout)、tci。
- **量化需 RNE+饱和 spec golden**:tquant/tdequant、fixp s8/vector quant/lrelu/prelu/rowmax_acc/
  group_max/cscale/chain。
- **超越函数需容差建模**:texp/tlog/trecip/tsqrt/trsqrt。
- **舍入模式未定**:tcvt(f32→i32,trunc vs RNE 未由文档钉死)。
- **本就崩/未实现**:argmax/argmin、sort/mrgsort、img2col、gather/scatter 族、MX/TGEMV、reinterpret_tile 等
  (compile-fail / run-fail,无输出可校)。

这些留待后续轮次:先按 ISA/refmodel 钉定语义,再补 golden。

### 文档更新跟踪(远程 linx 分支)

对比本地 `Linx-TileOP-API` d6a52b8 与远程 `linx` 顶端 fa24eae(2026-09-01),
`docs/tileop-usage/` 有 2 个 commit、3 个文件更新:

| 文件 | 变更 | 是否新接口 / demo 缺口 |
|------|------|----------------------|
| `options.md` | +420/−26 大改 | 否。是 fixp 后处理 options 的**文档扩写**,builder 集合(keep_acc/f16/bf16/convert/scalar/vector/s8/lrelu/row_max/group_max/max_abs/cscale)均为已知,fixp demo 已覆盖 |
| `range-modifiers.md` (+33)、`range-modifiers-developer-guide.md` (+85) | 新增 **TileArray region API** | **是**。`TPARTVIEW` / `TileArray` / `TASSEMBLY` 为新接口,原 demo 只有 `range::Subview`/`range::Assemble`,**缺口** |

**已补 demo**:`tlsu/src/region_tilearray.cpp`(看护 TileArray/TASSEMBLY + TCVT slot producer)。
guard 结论 = **run-fail**,并暴露文档/实现缺口:
1. 文档示例 `TCVT(destinations[0][2], source_tile)` 用临时量,而 region `TCVT` 签名是
   `TCVT(TileArrayOutputRef& dst, In& src)`(非 const 左值引用),临时量无法绑定;且示例源
   变量名 `source`(=TPARTVIEW 视图)与 `source_tile` 不一致——**照抄不可编译**,须绑具名左值。
2. 用 TPARTVIEW 父 strided 子视图作源 → gfrun `raw tile spill source does not fit the carrier
   shape`;改独立紧凑 Fragment 作源仍触发同一断言。结合文档自身 "until the … path is
   implemented and validated" 告警,且该特性是远程顶端 commit,判定**region producer 路径在
   当前 env_test 模型上尚不可运行**,待模型侧补齐后再验证组装布局。

---

## VEC 族(elementwise / scalar / compare)

| 接口 | 编译 | gfrun | 文档问题 |
|------|------|-------|----------|
| TADD/TSUB/TMUL/TDIV | ✅ | ✅ | [签名] 基础算术无签名/示例,`(dst,s0,s1)` 靠惯例推断 |
| TREM/TAND/TOR/TXOR/TSHL/TSHR | ✅ | ✅ | [签名] 同上;整数域 |
| TMAX/TMIN | ✅ | ✅ | [签名] 同上 |
| TABS/TNEG/TRELU | ✅ | ✅ | [签名] 一元,无签名;`(dst,src)` 推断 |
| TNOT | ✅ | ✅ | [签名] 同上;整数 |
| TCVT | ✅ | ✅ | [签名] 无签名;异 dtype 转换,`(dst,src)` 推断 |
| TFMA | ✅ | ✅ | [签名] 无签名;当作三元 `(dst,a,b,c)` |
| TADDS/TSUBS/TMULS/TDIVS | ✅ | ✅ | [签名] `*S` 标量类无签名;`(dst,src,scalar)` 推断 |
| TREMS/TANDS/TORS/TXORS/TSHLS/TSHRS | ✅ | ✅ | [签名] 同上;整数 |
| TMAXS/TMINS | ✅ | ✅ | [签名] 同上 |
| TSELS | ✅ | ✅ | **[签名]** 真实 `(dst,src0,scalar,src1)`,标量夹中间,极反直觉(详述 2) |
| TEXPANDS | ✅ | ✅ | **[签名]** 真实 `(dst,scalar)` 仅 2 参,无 src tile(详述 3) |
| TSEL | ✅ | ❌ | **[签名][语义][dtype]** 3 参 `(dst,src0,src1)`;gfrun 拒源元组(详述 1) |
| TCMP | ❌ | - | **[签名 OK / 后端拒]** cmp.md 签名完整;后端 `Match Instruction Error`(详述 5) |
| TCMPS | ❌ | - | 同上 |

VEC 小结:35 个 case,**32 通过 / 2 编译失败 / 1 gfrun 失败**。除 cmp.md(TCMP/TCMPS)
外,VEC 全族文档**不给 C++ 签名**——engines.md 只有"名字 + canonical 汇编 + 分类"
三列(详述 4)。而 cmp.md 虽给出签名,demo 照签名写却在后端被拒(详述 5)。

## SFU 族(reduce / expand / layout / irregular)

| 接口 | 编译 | gfrun | 文档问题 |
|------|------|-------|----------|
| TEXP/TLOG/TRECIP/TSQRT/TRSQRT | ✅ | ✅ | [签名] 一元 `(dst,src)` 推断;fp32 |
| TROWSUM/TROWMAX/TROWMIN/TROWPROD | ✅ | ✅ | **[签名][dtype]** 无签名;dst 需物理 M×N + ValidCol=1(详述 6) |
| TCOLSUM/TCOLMAX/TCOLMIN/TCOLPROD | ✅ | ✅ | **[签名][dtype]** 无签名;dst 需物理 M×N + ValidRow=1(详述 6) |
| TCOLEXPANDADD/SUB/MUL/DIV/MAX/MIN/EXPDIF | ✅ | ✅ | **[签名][dtype]** 二元;src1 需 1×N 行广播源(详述 7) |
| TROWEXPANDADD/SUB/MUL/DIV/MAX/MIN/EXPDIF | ✅ | ✅ | **[签名][dtype]** 二元;src1 需物理 M×1 列广播源(详述 7) |
| TROWEXPAND | ✅ | ❌ | **[签名][语义]** 纯 copy-expand;gfrun 要 one broadcast source,形状不明(详述 7) |
| TCOLEXPAND | ✅ | ❌ | 同上 |
| TROWARGMAX/TROWARGMIN/TCOLARGMIN | ✅ | ❌ | **[签名][语义]** gfrun `Undefined TEPL TileOp function`(详述 8) |
| TCOLARGMAX | ✅ | ❌ | **[签名]** gfrun `TCONCAT requires 2 source tiles`(一元写法被 lower 成 TCONCAT) |
| TCONCAT | ✅ | ✅ | **[签名]** layout.md 无签名;`(dst,src0,src1)` 二元恰好通过 |
| TFILLPAD/TTRANS | ✅ | ✅ | **[签名]** layout.md 只有散文;`(dst,src)` 一元恰好通过 |
| TPARTADD/TPARTMUL/TPARTMAX/TPARTMIN | ✅ | ✅ | **[签名]** 无任何文档;`(dst,src0,src1)` 二元恰好通过 |
| TEXTRACT/TINSERT | ❌ | - | **[签名]** 无签名;`no matching function`,无从推断(详述 10) |
| TTRI/THISTOGRAM | ❌ | - | **[签名]** 无任何文档;`no matching function`(详述 10) |
| TGATHER/TSCATTER | ❌ | - | **[签名]** 无任何文档;后端 `Match Instruction Error`(详述 10) |
| TCI | ✅ | ✅ | 文档签名完整(tci.md);S32 单行升序 |
| TQUANT | ✅ | ✅ | 文档签名完整(quant-and-im2col.md);FP32→S8,RNE+sat |
| TDEQUANT | ✅ | ❌ | 文档签名完整;src=int8→gfrun `source must be S8/U8/S16/U16`(详述 11) |
| TIMG2COL | ✅ | ❌ | 文档签名完整;gfrun `TIMG2COL not yet fully implemented`(详述 11) |
| TSORT | ✅ | ❌ | 文档签名完整(sort.md);gfrun `Undefined TEPL TileOp function`(详述 11) |
| TMRGSORT | ✅ | ❌ | **[示例]** 文档示例 shape 不可编译;修正后 gfrun `not yet fully implemented`(详述 9) |

SFU 小结:52 个 case,**36 通过 / 6 编译失败 / 10 gfrun 失败**。
- **通过 (36)**:5 一元 elementwise + 8 reduce + 14 expand-arith(col/row 各 7)+
  TCI/TQUANT + TCONCAT/TFILLPAD/TTRANS + 4×TPART。后 7 个"无签名却通过"靠
  dst-first 惯例命中。
- **编译失败 (6)**:TEXTRACT/TINSERT/TTRI/THISTOGRAM/TGATHER/TSCATTER——**文档无签名,
  从文档无法写出可编译调用**,最硬的文档缺口。
- **gfrun 失败 (10)**:copy-expand(TROWEXPAND/TCOLEXPAND,2)、argmax/argmin(4)、
  TDEQUANT/TIMG2COL/TSORT/TMRGSORT(4)。多为 `not yet fully implemented` /
  `Undefined TEPL TileOp function` / dtype 断言,判为工具链/仿真器支持缺口,
  按约束标"待判断"交用户核对,不深挖源码。

## TLSU 族(load / store / move / gather / scatter)

| 接口 | 编译 | gfrun | 文档问题 |
|------|------|-------|----------|
| TLOAD | ✅ | ✅ | 文档充分(tlsu.md);全套 case 均依赖,通路正常 |
| TSTORE | ✅ | ✅ | 文档充分;通路正常 |
| range::Subview | ✅ | ❌ | **[契约]** range-modifiers.md 签名+示例完整;照写编译过,gfrun `illegal TSTORE operand or descriptor contract`(详述 16) |
| range::Assemble | ✅ | ❌ | **[契约]** 同上;gfrun `illegal B.ASSEMBLE generation or descriptor contract`(详述 16) |
| TPREFETCH | ✅ | ✅ | **[签名]** src 必须是 static RowMajor `global_tensor` 本体,非 iterator 视图(详述 12) |
| MGATHER_CAS | ✅ | ✅ | 文档签名完整(tlsu.md);base+offset+expected+replacement |
| TMOV | ❌ | - | **[签名]** 基础 Local 形式无签名;后端 `Match Instruction Error`(详述 12) |
| GMOV | ❌ | - | **[示例]** 照抄文档示例 `GMOV<15>(dst,peer_tid,src)` 仍后端 `Match Instruction Error`(详述 12) |
| MGATHER/MSCATTER | ❌ | - | **[签名]** 仅散文,无签名;`no matching function`(详述 13) |
| MGATHER_MASK/MSCATTER_MASK | ❌ | - | **[签名]** 同上 |

TLSU 小结:12 个 case,**4 通过 / 6 编译失败 / 2 gfrun 失败**。
- **通过 (4)**:TLOAD/TSTORE/TPREFETCH/MGATHER_CAS——签名完整或可从散文补全。
- **编译失败 (6)**:MGATHER/MSCATTER/掩码变体(4)文档只有散文无签名;TMOV/GMOV(2)
  后端 `Match Instruction Error`。均为硬文档缺口或文档-后端不一致。
- **gfrun 失败 (2)**:range::Subview/range::Assemble——range-modifiers.md 签名+示例
  完整,严格照示例参数(SubviewSizeCode=1/ParentSizeCode=12/RegSrc=0/Offset=0)写,
  wrapper static_assert 通过,gfrun 却拒 descriptor 契约(详述 16),按约束标"待判断"。
- 注:Shared TMOV 四变体(TMOV_L2S_INSERT/PUBLISH、TMOV_S2L_BROADCAST/EXTRACT)与
  TSTORE_PART 依赖 opaque Shared handle,文档未给可复现的 Shared tile 构造示例,
  单列后续处理(详述 13)。

## CUBE 族(matmul / gemv)

| 接口 | 编译 | gfrun | 文档问题 |
|------|------|-------|----------|
| TMATMUL | ✅ | ✅ | cube.md 给示例;`(out,a,b)` / `(out,a,b,options)` 均通过 |
| TMATMUL_ACC | ✅ | ✅ | matrix-postprocess.md 签名完整 `(Dst,C,A,B,options)` |
| TMATMUL_BIAS | ✅ | ✅ | **[dtype][签名]** Bias 须 1×N + RowMajor + 派生 AccType + 普通 TLOAD(详述 14) |
| TMATMUL(f16) | ✅ | ✅ | `fixp::f16()` 通过 |
| TMATMUL(relu) | ✅ | ✅ | `fixp::f16().relu()` 链式通过 |
| TMATMUL(row_max) | ✅ | ✅ | `fixp::keep_acc().row_max(out)` 通过;示例完整 |
| TMATMUL(bf16) | ❌ | - | **[示例]** 照 `fixp::bf16()` 写→clang frontend abort(exit 134)(详述 14) |
| TMATMUL_MX | ✅ | ❌ | 照无 scale 便捷式写;gfrun `source stream does not match ASL contract`(详述 14) |
| TGEMV | ✅ | ❌ | 照文档写;gfrun `CUBE D must hold M x N output region`(dst 形状)(详述 14) |

CUBE 小结:9 个代表 case,**6 通过 / 1 编译失败 / 2 gfrun 失败**。
- **通过 (6)**:TMATMUL 基础 + ACC + BIAS + f16/relu/row_max postprocess。核心
  matmul 通路 + B.FPATR options 链式文档足够,可用。
- **失败 (3)**:bf16 options 触发 clang 崩溃、MX 无 scale 便捷式 binder 契约不符、
  TGEMV dst 形状断言——均照文档写,非明显写法错误,按约束交用户核对。
- 未覆盖项集中见文末「未覆盖清单」。

## FIXP 族(matrix postprocess options,matrix-postprocess.md)

`fixp::` options 挂在 TMATMUL/TGEMV 上配置 `B.FPATR`,承载全部 PostProcess
(scalar/vector quant、LReLU/PReLU、RowMax、GroupMax、MaxAbs、CScale)。
matrix-postprocess.md 是 docs/tileop-usage 里**签名/示例最完整**的文档,demo 均严格
照其示例写。

| 接口 | 编译 | gfrun | 文档问题 |
|------|------|-------|----------|
| fixp::convert\<Mode\>() | ✅ | ✅ | 文档示例完整;`convert<F322F16>()` → dst FP16 |
| fixp::s8(desc)(scalar) | ✅ | ✅ | 文档示例完整;`make_s8_quant` descriptor builder 照抄可用 |
| fixp::scalar\<Mode\>(desc) | ✅ | ✅ | generic 写法;用 QF322S8Pre 通过(详述 17 dtype 约束) |
| fixp::s8(tile)(vector) | ✅ | ✅ | vector-quant 快捷式 VQF322S8Pre;quant Tile 物理 2×32 valid 1×32 |
| fixp::vector\<Mode\>(tile) | ✅ | ✅ | generic vector-quant;VQF322F16Pre → dst FP16 |
| fixp::s8(desc).lrelu(fp19) | ✅ | ✅ | scalar quant(SrcReg0)+ LReLU(SrcReg1)链式 |
| fixp::f16().prelu(tile) | ✅ | ✅ | convert + PReLU;FP19 Tile 物理 2×32 valid 1×32 |
| fixp::keep_acc().row_max(in,out) | ✅ | ✅ | 累加式 RowMax(RowMaxInit=1);RM tile valid M×1 |
| fixp::keep_acc().group_max\<8\>(out) | ✅ | ✅ | GroupMax;out valid M×ceil(N/GroupN)=32×4 |
| fixp::keep_acc().cscale(scale) | ✅ | ❌ | **[契约]** TMATMUL_ACC + CScale(U8 CUBE_M32);gfrun `m_handlers.find(grp)` 崩(详述 17) |
| .row_max().group_max().max_abs() | ✅ | ❌ | **[契约]** 链式 max_abs 组合;gfrun `requires one ASL B.FPATR descriptor`(详述 17) |

FIXP 小结:11 个 case,**9 通过 / 0 编译失败 / 2 gfrun 失败**。
- **通过 (9)**:convert、scalar/vector quant(s8 快捷式 + generic scalar/vector)、
  LReLU、PReLU、累加式 RowMax、GroupMax。核心 postprocess 链(quant / LReLU-PReLU /
  RowMax / GroupMax)文档足够,可用。
- **gfrun 失败 (2)**:cscale(ACC 路径)、max_abs 链式组合——均照文档写、编译通过,
  gfrun 拒(详述 17),按约束标"待判断"。
- **编译期约束发现**:B.FPATR 表列 QF322S16Pre→S16,但 fp16 矩阵输入 + S16 dst 被
  static_assert 拒;表未给 PreQuantMode↔输入矩阵 dtype 兼容矩阵(详述 17)。

## misc

| 接口 | 编译 | gfrun | 文档问题 |
|------|------|-------|----------|
| reinterpret_tile | ❌ | - | **[实现]** reinterpret-tile.md 签名/示例完整;照写后 `reinterpret_tile<int32_t>(src)` 得到的视图被后续 `TABS` 拒(`no matching function`),视图类型不被 tileop 接受(详述 15) |

- **misc**:1 case,编译失败。文档侧签名完整,问题在返回视图无法被下游 tileop 消费。

---

## 未覆盖清单

matrix-postprocess.md 列出的部分 CUBE 操作族尚未写 demo。它们不是遗漏,而是被
上游的待判断项或文档缺口卡住;集中登记于此,待前置项澄清后批量补。

| 未覆盖项 | 域 | 文档位置 | 暂缓原因 |
|----------|----|----------|----------|
| TMATMUL_MX 带 scale 全组(单/双 scale + `_BIAS`/`_ACC`) | cube | matrix-postprocess.md L44-49 | 基础形 `tmatmul_mx`(无 scale)gfrun 已崩(ASL contract,详述 14);带 scale 变体几乎必然撞同一运行期契约,成批复现同一待判断项 |
| TGEMV 全族(`_BIAS`/`_ACC`/`_MX`/`_MX_BIAS`/`_MX_ACC`) | cube | matrix-postprocess.md L50-58 | 基础形 `tgemv` gfrun 已崩(dst 形状,详述 14);同上,变体会成批复现 |
| Shared Right(`SharedTile<RightTile>` 作 B) | cube | matrix-postprocess.md L377-398 | 依赖 Shared tile 构造,而 docs/tileop-usage 全目录无 Shared tile 构造示例(详述 12/13);严格按"只依赖文档"约束无法写出可编译 demo |

前置澄清后的补齐顺序建议:先定性 MX/TGEMV 基础形的 gfrun 崩(工具链版本 or 仿真器
未实现)→ 若可用则批量补 MX-scale/TGEMV 变体;Shared Right 待文档补 Shared 构造示例。

---

## 重点文档问题详述

### 1. [签名][语义][dtype] TSEL — 三重缺口

engines.md 仅列 `TSEL | BSTART.VEC TSEL | 26 | elementwise-tile-tile`。

- **[签名]** 经典 select 的 `(dst, cond, a, b)` 4 参形式编译失败;真实是
  3 参 `TSEL(dst, src0, src1)`(编译器报错反推)。
- **[语义]** 只有两个源的 select "按什么条件选哪个",文档完全没写。
- **[dtype]** fp32 编译即被拒;int32 编译通过,但 gfrun 断言
  `srcs.size()==3 && ... "select first B.IOT requires mask then true/source Tile"`——
  运行期要求 mask + true/source 三源元组,与文档可推得的两源调用形不符。
  签名、语义、dtype/源元组三层文档信息均缺,按约束标"待判断"。

### 2. [签名] TSELS — 参数顺序反直觉

真实签名 `TSELS(dst, src0, scalar, src1)`:标量是**第 3 个**参数,夹在两个
tile 源中间。无文档、无从推断,只能靠编译器报错。

### 3. [签名] TEXPANDS — 无 src 的纯标量填充

真实签名 `TEXPANDS(dst, scalar)` 仅 2 参,没有 src tile 输入——是"标量广播填充
整个 tile"。分类栏 `tile-scalar-and-immediate` 有误导性(看起来像 tile⊕scalar)。

### 4. [签名] 系统性缺口:VEC 全族缺 C++ 签名

除 cmp.md 覆盖的 TCMP/TCMPS 外,VEC 的其余算子在文档里**没有一个**给出 C++
调用签名或示例。engines.md 是"汇编投影表",不是 C++ API 文档。使用者只能靠
dst-first 惯例猜,遇到 TSEL/TSELS/TEXPANDS 这类非常规签名时必然猜错。
**建议**:engines.md 或配套文档为每个算子补一行 C++ 签名。

### 5. [后端] TCMP/TCMPS — 文档签名完整,后端 Match Instruction Error

cmp.md 给了完整 `TCMP<Mode>(dst,src0,src1)` / `TCMPS<Mode>(dst,src,scalar)` 签名 +
dtype 表(int32 支持全部 6 个 CmpMode)+ 示例。demo 严格照签名写(int32 in/out,
CmpMode::GT / GE),编译在工具链头 `template_asm.hpp:6765`(TCMP)/`:7485`(TCMPS)
报 `Match Instruction Error!`——后端 lower 层拒绝该 tile 形状/dtype 组合。
文档签名侧无缺口,失败在工具链后端,按约束标"待判断",交用户核对是否
工具链版本问题。**这是 VEC 族唯一有完整签名却仍不可用的接口对。**

### 6. [签名][dtype] TROW*/TCOL* reduce — 输出 tile 形状规则缺失

engines.md 只列名字,TROWSUM/TROWMAX/... 一族无签名。实测:

- 一元 `(dst, src)`,src 为 M×N。
- **输出 tile 形状**:row-reduce(收缩列)dst 必须是**物理 M×N + ValidCol=1**;
  col-reduce(收缩行)dst 必须是**物理 M×N + ValidRow=1**。文档完全没写这个
  "物理满宽 + valid 轴=1"规则,只能靠 gfrun 断言反推(物理 M×1 非法:fp32 需
  Cols×bits % 256 == 0)。
- **建议**:补充 reduce 族的输出形状契约。

### 7. [签名][dtype] TROWEXPAND*/TCOLEXPAND* — 广播源形状不对称

engines.md 分类为 `reduce-and-expand`,无签名、无形状规则。实测:

- **arith 变体(各 7,全部通过)**:二元 `(dst M×N, src0 M×N, src1_broadcast)`。
  - col-expand:src1 用物理 M×N + ValidRow=1(1×N 行广播源)。
  - row-expand:src1 用物理 M×1 列广播源(gfrun 断言
    `IsCompatibleDataTile(srcs[2],...,validRow,1,1,...)`,即要求物理单列)。
  - **不对称**:col-expand 收物理满宽 + ValidRow=1,row-expand 收物理 M×1——
    两侧广播源形状规则不同,文档均未给,靠 gfrun 反推。
- **copy-expand(TROWEXPAND/TCOLEXPAND,均 gfrun 失败)**:纯 `(dst, src)` 一元,
  gfrun 断言 `"PTO v0.58 COPY expansion requires one broadcast source"`。广播源
  如何构造文档无说明,按约束标"待判断"。
- **建议**:补 expand 族签名 + col/row 广播源形状规则(含 copy-expand 的构造)。

### 8. [签名][语义] TROWARGMAX/TROWARGMIN/TCOLARGMAX/TCOLARGMIN — 接口疑似未启用

engines.md 列出这些名字,编译均通过,但 gfrun:

- TROWARGMAX/TROWARGMIN/TCOLARGMIN 报 `false && "Undefined TEPL TileOp function"`;
- TCOLARGMAX 报 `TCONCAT requires 2 source tiles`(一元写法被 lower 成 TCONCAT)。

签名/语义文档均缺失,运行期提示"未定义的 TEPL 函数"。疑似该组在此工具链/
仿真器版本尚未落地,按约束标"待判断"。

### 9. [示例] TMRGSORT — sort.md 示例本身不可编译

sort.md 给了完整 TMRGSORT 签名,但配套示例:

```cpp
using Row = Tile<Location::Vec, float, 1, 256, BLayout::RowMajor>;
Row a, b, out;
TMRGSORT(out, a, b);          // ascending merge
```

**直接照抄编译失败**:`template_asm.hpp:8120` static_assert
`TMRGSORT destination must contain the combined source columns`,即要求
`dst.ValidCol == left.ValidCol + right.ValidCol`。示例三个 tile 全用 1×256,
`256 != 256 + 256`。修正为两源各 1×128、dst 1×256 后编译通过。
- **建议**:改示例为 `left/right = 1×128, dst = 1×256`。
- 修正后 gfrun 报 `"TMRGSORT not yet fully implemented"`(仿真器未实现),
  运行结果待判断;"示例不可编译"是纯文档问题,独立成立。

### 10. [签名] layout / irregular 无签名族 — 文档零覆盖,靠猜

layout.md 只有散文,**不给 TCONCAT/TEXTRACT/TINSERT/TFILLPAD/TTRANS 任何 C++
签名**;TTRI/THISTOGRAM/TGATHER/TSCATTER/TPART* 连专题文档都没有,仅 engines.md
一行分类。全部靠 dst-first 惯例猜:

- **猜中通过(9)**:TCONCAT(dst,s0,s1)、TFILLPAD(dst,src)、TTRANS(dst,src)、
  TPARTADD/MUL/MAX/MIN(dst,s0,s1)。能过纯属惯例命中。
- **猜错编译失败(6)**:
  - TEXTRACT/TINSERT/TTRI/THISTOGRAM:`no matching function`——参数个数/顺序无从得知。
  - TGATHER/TSCATTER:后端 `Match Instruction Error`(即使参数蒙对,索引/掩码语义仍无文档)。
- **建议**:layout / irregular 每个算子补签名 + 示例(达到 tci.md / sort.md /
  quant-and-im2col.md 的详尽度)。

### 11. [dtype][语义] TQUANT 通过 / TDEQUANT/TIMG2COL/TSORT 运行受阻

quant-and-im2col.md、sort.md 给了完整签名。实测:

- **TQUANT**(FP32→S8)✅ 编译 + gfrun 通过。
- **TDEQUANT**(照文档 src=int8)编译通过,gfrun 断言
  `"TDEQUANT source must be S8/U8/S16/U16"`——传入的正是 int8 源,运行期却认为
  源 dtype 不符,疑似仿真器 dtype 判定口径问题,待判断。
- **TIMG2COL** 编译通过,gfrun `"TIMG2COL not yet fully implemented"`;
  **TSORT** gfrun `"Undefined TEPL TileOp function"`。两者仿真器未实现,待判断。
  文档签名侧无缺口,问题在运行支持。

### 12. [签名][类型] TPREFETCH 的 src 类型 / TMOV / GMOV

tlsu.md 覆盖了 TLSU 组散文语义,仍有缺口:

- **TPREFETCH**:文档签名 `TPREFETCH(src, valid_col, valid_row)`,但**未说明 src
  的确切类型**。传 iterator 解引用视图(ND layout)编译报 `no member named 'Cols'`;
  改传 static RowMajor `global_tensor` 本体后编译 + gfrun 通过。应在签名处点明
  src 必须是带 compile-time 列数的 global_tensor 本体,而非 iterator 视图。
- **TMOV**(基础 Local 形式):engines.md 列 function 2,但文档**未给 C++ 签名**。
  猜测 `(dst, src)` 后端报 `Match Instruction Error`。tlsu.md 只描述 Shared 四变体。
- **GMOV**:tlsu.md 给示例 `GMOV<15>(dst, peer_tid, src)`,照抄后端仍报
  `Match Instruction Error`。签名看似完整但后端拒,文档示例与后端不一致,待判断。

### 13. [签名][构造] MGATHER/MSCATTER/掩码变体 + Shared TMOV 四变体 + TSTORE_PART

- **MGATHER/MSCATTER/MGATHER_MASK/MSCATTER_MASK**:tlsu.md 只在散文里说
  "offset/mask 是 Local 操作数,base+stride 是标量输入",**未给任一 C++ 签名**,
  参数个数猜测均 `no matching function`。对比同文档 **MGATHER_CAS 给了完整模板
  签名 → 编译 + gfrun 一次通过**,可见"给签名就能用,不给就废"。
- **Shared TMOV 四变体 + TSTORE_PART**:tlsu.md 提到这些 wrapper 名字,但**既没给
  签名,也没给 Shared-location tile 的 C++ 声明/构造方式**。docs/tileop-usage 全
  目录无一处 Shared tile 构造示例,仅凭文档**无法写出可编译的 demo**,不强行猜测。
  这是 TLSU Shared 面的系统性文档缺口。

### 14. [签名][示例][dtype] CUBE:Bias 形状缺口;bf16/MX/TGEMV 受阻

cube.md + matrix-postprocess.md 是 docs/tileop-usage 里**签名最完整**的一组。
实测通过 6 个(tmatmul / _acc / _bias / _f16 / _relu / _rowmax)。仍有缺口:

- **[签名] Bias 的 tile 形状**:matrix-postprocess.md 只说"辅助参数仍是普通 Local
  Tile",没说 Bias 的 dtype / valid shape。实测靠 static_assert 反推:Bias 必须是
  **普通 RowMajor + 派生 AccType(fp16 输入→FP32)+ valid shape `1 x N`**,且用普通
  `TLOAD` 而非 `TLOAD_CUBE`。应在 _BIAS 签名旁点明这三条。
- **[签名] TLOAD_CUBE / TSTORE_CUBE 无 C++ 签名**:cube.md 只说"用它们做 GM 转换
  边界",没给参数。猜测 `(cube_tile, global_tensor)` 二参恰好可用(tmatmul 等 6 例
  据此通过),属运气,应补签名。
- **[示例] tmatmul_bf16**:严格照文档 `TMATMUL(dst_bf16, a, b, fixp::bf16())` 写,
  `__bf16` 累加器,**clang frontend abort(exit 134)**——编译器直接崩溃,疑似工具链
  问题,待判断。
- **[待判断] tmatmul_mx**(无 scale 便捷式):照 `TMATMUL_MX(Dst, A, B, options)` 写,
  gfrun `CUBE local source stream does not match the current ASL contract`(source
  数量不符),待判断。
- **[待判断] tgemv**:照文档 `CubeAccumulatorM16<AccT,1,N>` + `TGEMV(Dst,Mtx,Vec,
  options)` 写,gfrun `CUBE D must hold the M x N FPATR output region`(dst 物理形状
  约束文档未明),待判断。

### 15. [实现] reinterpret_tile — 签名完整,返回视图不被下游 tileop 接受

reinterpret-tile.md 给了完整签名 + 示例:`auto v = reinterpret_tile<int32_t>(src)`
(同位宽、Local only、无 TCVT/copy、视图非拥有需保持 src 存活)。demo 照文档写
`fp32 tile → reinterpret_tile<int32_t>` 得到视图,随后用 `TABS` 消费该视图(选 TABS
是因它发 lb2、可直接落 TSTORE),编译报 `no matching function for call to 'TABS'`。
即 reinterpret_tile 返回的视图类型不被后续 tileop 的模板匹配接受。文档签名侧无缺口,
问题在视图类型与 tileop 入参的兼容性,按约束标"待判断",交用户核对。

### 16. [契约] range::Subview / range::Assemble — 编译过,gfrun 拒 descriptor 契约

range-modifiers.md 给出完整模板签名 + 示例 + 合法值域 + 发射 asm,是难得详尽的一篇。
demo 严格照示例写:

```cpp
auto sv = range::Subview<Src, 1, /*Off*/0, /*RegSrc*/0>(s, 0);  TSTORE(gm, sv);
auto as = range::Assemble<Dst, 12, /*INIT*/true, /*LAST*/false, 0, 0>(d, 0); TLOAD(as, gm);
```

- **编译通过**:wrapper 的 `static_assert`(SizeCode/INIT 契约/RegSrc 范围)全部满足。
- **gfrun 拒**:
  - Subview over TSTORE → `illegal TSTORE operand or descriptor contract`;
  - Assemble over TLOAD → `illegal B.ASSEMBLE generation or descriptor contract`。
- 文档"Lineage and status"称 Local Subview/Assemble 已 implemented 且 LLVM MC
  round-trip 全覆盖——签名与汇编层无缺口,但**运行期(gfrun)拒绝按文档示例参数
  生成的 descriptor**。疑似仿真器对 range descriptor 的支持/口径问题,按约束标
  "待判断",交用户核对是否工具链/仿真器版本问题。

### 17. [契约][dtype] FIXP postprocess — cscale/max_abs gfrun 拒;S16 mode 编译期 dtype 约束

matrix-postprocess.md 示例完整,11 个 postprocess demo 中 9 个编译+gfrun 通过。
三处缺口:

- **cscale(FP32 accumulator C scaling)**:照文档
  `TMATMUL_ACC(d, c, a, b, fixp::keep_acc().cscale(scale))` 写(CScale 为 Local U8
  CUBE_M32,valid M×1),编译通过,gfrun `m_handlers.find(grp) != m_handlers.cend()`
  断言崩。文档未给 CUBE_M32 layout scale 的加载方式(本 demo 按 CUBE 惯例用
  TLOAD_CUBE),cscale ACC 路径运行期不被支持,待判断。
- **max_abs 链式**:照文档
  `fixp::keep_acc().row_max(in,out).group_max<8>(gout).max_abs()` 写,编译通过,
  gfrun `hasFixpAttr && "PTO v0.58 Matrix requires one ASL B.FPATR descriptor"` 崩。
  单独的 row_max / group_max 均通过,叠加 max_abs 的组合运行期不被支持,待判断。
- **[dtype] QF322S16Pre 的输入约束**:B.FPATR 表(matrix-postprocess.md)列
  QF322S16Pre→S16,但用 fp16 矩阵输入 + S16 dst 编译即被 static_assert 拒
  (`PreQuantMode incompatible with the derived matrix accumulator type` /
  `D dtype must match the derived accumulator/output type`)。表只给了"mode→dst
  dtype",**未给 mode 与输入矩阵 dtype 的兼容矩阵**(S16 量化疑似要求整数矩阵输入,
  派生 S32 accumulator)。generic scalar demo 遂改用 QF322S8Pre 演示 `fixp::scalar<>`
  写法。建议文档补 PreQuantMode↔输入 dtype 的合法组合表。
