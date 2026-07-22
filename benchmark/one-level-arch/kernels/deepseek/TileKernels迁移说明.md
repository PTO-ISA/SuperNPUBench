# TileKernels 迁移说明

> 将 TileKernels（基于 TileLang DSL，NVIDIA SM90/SM100）的算子迁移到 SuperNPUBench
> one-level-arch（PTO Tile ISA，C++ header-only）的实现说明。迁移计划见仓库根目录
> 《TileKernels迁移计划.md》。
>
> 本目录 `kernels/deepseek/` 收录全部已迁移算子，按源端模块组织为
> `engram/`、`mhc/`、`moe/`、`quant/`、`transpose/` 五个子目录。每个 kernel 头文件含
> 标准注释块（功能 / 源端 / 迁移映射 / 约束 / 算法步骤）。
>
> **重要**：所有算子已 tile 版化并通过 linx 工具链（`linx_blockisa_llvm_musl`）编译+链接验证。
> 由于该工具链实际暴露的指令集**窄于** Linx-TileOP-API 文档全集，下文"主指令"列给出的是
> **实际编译版**的指令（不可用指令用可用指令模拟，详见各 `.hpp` 头部注释）。

---

## 一、已迁移算子总表（19 个 kernel 头文件，tile 版编译通过）

| 阶段 | 文件 | 对应 TileKernels | 功能 | 主指令（实际编译版） |
|:---:|------|------------------|------|----------------------|
| 1 | `engram/fused_weight_pto.hpp` | `engram_fused_weight_kernel.py` | bf16×bf16→fp32 权重融合 | `TCVT`+`TMUL` |
| 1 | `quant/cast_back_pto.hpp` | `cast_back_kernel.py` | FP8/FP4 反量化(per-token/per-channel) | `TCVT`+`TROWEXPANDMUL`/`TCOLEXPANDMUL`（TDEQUANT 未提供→显式） |
| 1 | `moe/normalize_weight_pto.hpp` | `normalize_weight_kernel.py` | top-k 权重行归一化 | `TROWSUM`+`TADDS`+`TRECIP`+`TROWEXPANDMUL`（TROWEXPANDDIV 未提供→recip+mul） |
| 1 | `mhc/expand_to_mhc_pto.hpp` | `expand_kernel.py`(fwd) | token 沿 mhc 维扩展 | `TLOAD`+循环`TSTORE` |
| 1 | `mhc/sinkhorn_pto.hpp` | `sinkhorn_kernel.py`(fwd) | 行列交替归一化 | `TROWMAX`+`TROWEXPANDSUB`+`TEXP`+`TROWSUM`+`TRECIP`+`TROWEXPANDMUL`+`TCOLSUM`/`TCOLEXPANDMUL` |
| 1 | `transpose/batched_transpose_pto.hpp` | `batched_transpose_kernel.py` | (B,M,N)→(B,N,M) 批量转置 | `TLOAD`→`TTRANS`→`TSTORE` |
| 2 | `quant/per_token_cast_pto.hpp` | `per_token_cast`/`per_channel_cast` | 行/列级 FP8 量化 | `TMAX(TROWMAX(x),TROWMAX(TMULS(x,-1)))`(absmax) + `TRECIP`+`TROW/TCOLEXPANDMUL`+`TCVT`（TABS 未提供→max(x,-x)） |
| 2 | `quant/swiglu_fused_cast_pto.hpp` | `swiglu_forward_and_per_token_cast` | 融合 SwiGLU+per-token 量化 | `TMULS(x,-1)`(TNEG 未提供)+`TEXP`+`TADDS`+`TRECIP`+`TMUL`(silu)→量化链 |
| 3 | `moe/topk_gate_pto.hpp` | `topk_gate_kernel.py` | top-k 专家选择 | `TROWMAX`+`TROWEXPANDSUB`+`TCMP`+逆索引`TSEL`+`TROWEXPAND`（TROWARGMAX 未提供→模拟） |
| 3 | `moe/group_count_aux_fi_pto.hpp` | `group_count`/`aux_fi` | 负载均衡统计 | `TEXPANDS`+`TCMP`+`TSEL`+`TROWSUM`+`TINSERT` 直方图tile（THISTOGRAM 未提供→模拟）+`TCVT`/`TMULS` |
| 4 | `moe/expand_to_fused_pto.hpp` | `expand_to_fused_kernel.py` | token 扩展到 fused 布局 | `TLOAD`+`TSTORE`(pos 偏移)+`TEXPANDS` 清零 |
| 4 | `moe/reduce_fused_pto.hpp` | `reduce_fused_kernel.py` | 加权归约回 token 级 | `TMULS`+`TADD`（TFMA 未提供→拆乘加）+`TCVT` |
| 4 | `moe/inplace_unique_group_indices_pto.hpp` | `inplace_unique_group_indices_kernel.py` | 组索引原地去重 | `TEXTRACT`+`TCMP`+`TSEL`+`TINSERT` pairwise（位图需 cumsum→改 O(k²)） |
| 4 | `moe/mask_indices_by_tp_pto.hpp` | `mask_indices_by_tp_kernel.py` | 按 TP rank 屏蔽索引 | `TDIVS`/`TREMS`/`TSUBS`+`TEXPANDS`+`TCMP`+`TAND`(符号位)+`TSEL`（TCMPS 未提供→广播+TCMP） |
| 4 | `moe/get_fused_mapping_pto.hpp` | `get_fused_mapping_kernel.py` | token-topk↔expert-major 映射 | 直方图 `TCMP`+`TSEL`+`TROWSUM` + `TEXPANDS` 区间填充；位置映射需 scan（无→标量 cursor） |
| 4 | `engram/engram_hash_pto.hpp` | `engram_hash_kernel.py` | n-gram 哈希嵌入索引 | `TMULS`+`TXOR`+`TREMS`+`TADDS`（int32 hash） |
| 6 | `mhc/expand_to_mhc_bwd_pto.hpp` | `expand_kernel.py`(bwd) | 沿 mhc 维求和归约 | `TCVT`+`TCOLSUM` |
| 6 | `mhc/norm_fn_pto.hpp` | `norm_fn_kernel.py` | RMSNorm+normw 合并 | `TMUL`+`TROWSUM`+牛顿 rsqrt(`TRECIP`+`TMUL`/`TMULS`/`TADDS`)+`TROWEXPANDMUL`/`TCOLEXPANDMUL`（TRSQRT 未提供→牛顿） |
| 6 | `mhc/multilayer_recompute_pto.hpp` | `multilayer_recompute_kernel.py` | 多层重算 GEMM 累加链 | `TMATMUL_ACC`(链式累加)+`ACCCVT` |

---

## 二、阶段 1：一档算子（直接对应）

### 2.1 batched_transpose（批量 3D 转置）
**源端**：`transpose/batched_transpose_kernel.py` | **目标**：`transpose/batched_transpose_pto.hpp`
（2D 已由 `kernels/transpose/transpose_pto.hpp` 预存实现）
**功能**：`(B,M,N)→(B,N,M)`。
**如何迁移**：源端用寄存器 4×4 转置 + swizzle shared memory 消 bank conflict（GPU SIMT 优化）。
PTO 有硬件原生 `TTRANS(dst,src)` 单指令完成 tile 转置，无需 swizzle，走 `TLOAD→TTRANS→TSTORE`。
3D 展平为 2D（global_iterator 仅 2 参）。**差异**：代码量大幅减少，无 bank-conflict 处理。

### 2.2 fused_weight（权重融合 bf16×bf16→fp32）
**源端**：`engram/engram_fused_weight_kernel.py` | **目标**：`engram/fused_weight_pto.hpp`
**功能**：`weight_hidden × weight_embed → fp32`。
**如何迁移**：`TCVT` 升精度 + `TMUL` 逐元素乘。tile 级并行替代 SIMT warp。1 行 tile 须 ≥128B（TileW≥64 bf16）。

### 2.3 cast_back（FP8/FP4 反量化）
**源端**：`quant/cast_back_kernel.py` | **目标**：`quant/cast_back_pto.hpp`
**功能**：`out[i,j]=data[i,j]*sf[i]`（per-token）/ `*sf[j]`（per-channel）。
**如何迁移**：`TDEQUANT` 工具链未提供 → 显式 `TCVT(fp8→fp32)` + `TROWEXPANDMUL`/`TCOLEXPANDMUL`（行/列广播乘）。
行 scale 在行循环外只 load 一次。E5M6 走单独位解包路径（未实现）。

### 2.4 normalize_weight（MoE 权重行归一化）
**源端**：`moe/normalize_weight_kernel.py` | **目标**：`moe/normalize_weight_pto.hpp`
**功能**：`normalized = row / (sum(row)+eps)`。
**如何迁移**：`TROWSUM` 行求和（输出物理 8 宽 valid1）+`TADDS(+eps)`+`TRECIP`+`TROWEXPANDMUL`。
**注**：`TROWEXPANDDIV` 未提供 → 用 `TRECIP`+`TROWEXPANDMUL` 替代行广播除。

### 2.5 sinkhorn_fwd（Sinkhorn 行列交替归一化）
**源端**：`mhc/sinkhorn_kernel.py` | **目标**：`mhc/sinkhorn_pto.hpp`
**功能**：行列交替归一化使矩阵双随机近似。
**如何迁移**：归约+广播指令族。行 softmax：`TROWMAX→TROWEXPANDSUB→TEXP→TROWSUM→TADDS(+eps)→TRECIP→TROWEXPANDMUL→TADDS(+eps)`；
列归一化：`TCOLSUM→TADDS→TRECIP→TCOLEXPANDMUL`。`TROWEXPANDDIV`/`TCOLEXPANDDIV` 未提供→recip+mul。
eps 用 constexpr 局部量（float 非类型模板参数不支持）。3D 展平 2D。

### 2.6 expand_to_mhc_fwd（MHC token 扩展）
**源端**：`mhc/expand_kernel.py` | **目标**：`mhc/expand_to_mhc_pto.hpp`
**功能**：`(N,H)→(N,MHC,H)` 沿 mhc 维复制。
**如何迁移**："广播写"——输入 tile 循环外 `TLOAD` 一次，m 维标量循环内 `TSTORE` 到 m 个位置。
逐 token（1×TileH tile，TileH≥64 bf16 满足 128B 最小）。3D 输出展平 2D。

---

## 三、阶段 2：量化全集

### 3.1 per_token_cast / per_channel_cast（行/列级 FP8 量化）
**源端**：`quant/per_token_cast_kernel.py`、`per_channel_cast_kernel.py` | **目标**：`quant/per_token_cast_pto.hpp`
**功能**：按 block 求 amax → sf → 缩放。
**如何迁移**：关键洞察——"按 Npc 分组求 amax" 等价于"tile 列宽=Npc，`TROWMAX` 行归约"。
`TABS` 工具链未提供 → absmax = `TMAX(TROWMAX(x), TROWMAX(TMULS(x,-1)))`（max(x,-x)）。
指令链：`TCVT→TMULS(-1)→TROWMAX×2→TMAX→TMAXS(clamp)→TMULS(sf)→TRECIP→TMULS(sfinv)→TROWEXPANDMUL→TCVT`。
per_channel 对称换 `TCOLMAX`+`TCOLEXPANDMUL`。bf16 列宽须 16 的倍数。

### 3.2 swiglu_forward_and_per_token_cast（融合 SwiGLU+量化）
**源端**：`quant/swiglu_forward_and_per_token_cast_kernel.py` | **目标**：`quant/swiglu_fused_cast_pto.hpp`
**功能**：`silu(gate)*up` 后接 per-token 量化。
**如何迁移**：`silu(g)=g*sigmoid(g)=g/(1+exp(-g))` 用 `TMULS(g,-1)`（`TNEG` 未提供）→`TEXP`→`TADDS(+1)`→`TRECIP`→`TMUL(g*sig)`。
`TMUL(swiglu, silu, up)` 后接量化链（同 3.1，含 absmax 模拟）。同一 tile 内先 SwiGLU 后量化省 GM 读写。

---

## 四、阶段 3：MoE 路由归约/统计

### 4.1 topk_gate（top-k 专家选择）
**源端**：`moe/topk_gate_kernel.py` | **目标**：`moe/topk_gate_pto.hpp`
**功能**：每 token 取 top-k 专家索引，并列取较小索引。
**如何迁移**：`TROWARGMAX` 工具链未提供 → argmax 模拟：`TROWMAX` 取最大值 → `TROWEXPANDSUB(差)` →
`TCMP(==0)` 掩码 → 逆索引 `rev=N-1-I`（实现并列取小：`TROWMAX(rev)`=min(idx)）→ `TSEL` 取掩码处 rev →
`best_idx=N-1-best_rev` → `TROWEXPAND` 广播 + `TCMP(I==)` + `TSEL(-inf)` 掩码置 -inf；重复 NumTopk 轮。
**注**：`TCMP`/`TSEL` 三参须同 dtype → 索引运算全用 float（`TCI`+`TREMS` int→`TCVT` float），存储时 `TCVT` 回 int32。
多行 tile（TileN×NumExperts，TileN≥4 使 valid1 输出≥128B）。

### 4.2 group_count / aux_fi（负载均衡统计）
**源端**：`moe/group_count_kernel.py`、`aux_fi_kernel.py` | **目标**：`moe/group_count_aux_fi_pto.hpp`
**功能**：每 group/expert 计数直方图 + aux 浮点归一化。
**如何迁移**：`THISTOGRAM` 工具链未提供 → 直方图 tile：每专家 e 用 `TEXPANDS(e)`+`TCMP(==)`+`TSEL(命中→1)`+`TROWSUM(每行命中)`
+`TINSERT` 进 (TileM×NumGroups) 直方图列 e；跨 tile `TADD` 累加；横向铺所有专家列避免单标量 tile（1×8<128B）。
aux_fi 加 `TCVT(int→fp32)`+`TMULS(*N/(T*K))`。

---

## 五、阶段 4：MoE 映射/扩展类

### 5.1 expand_to_fused（token 扩展到 fused 布局）
**源端**：`moe/expand_to_fused_kernel.py` | **目标**：`moe/expand_to_fused_pto.hpp`
**功能**：按 `token_topk_to_pos` 复制到扩展位置；无效位置清零。
**如何迁移**："scatter 写"——输入 tile 循环外 `TLOAD` 一次，k 循环内 `TSTORE` 到 pos 偏移；`TEXPANDS(0)`+`TSTORE` 清零。

### 5.2 reduce_fused（加权归约回 token 级）
**源端**：`moe/reduce_fused_kernel.py` | **目标**：`moe/reduce_fused_pto.hpp`
**功能**：`out[token]=Σ_k weight[token,k]*x[pos]`。
**如何迁移**：`TFMA` 工具链未提供 → `TMULS(x,weight)`+`TADD(累加)`（融合乘加拆两步）。`TCVT` 升精度累加，`TEXPANDS(0)` 初始化 acc。

### 5.3 inplace_unique_group_indices（组索引原地去重）
**源端**：`moe/inplace_unique_group_indices_kernel.py` | **目标**：`moe/inplace_unique_group_indices_pto.hpp`
**功能**：同 token 内重复组索引除首次外置 -1。
**如何迁移**：源端用 2×uint64 位图+串行扫描；位图前缀（cumsum）无对应 → 改 O(NumTopk²) pairwise：
每 (a<b) 对 `TEXTRACT` 列 a/b → `TCMP(==)` → `TSEL(列 b=-1 命中)` → `TINSERT` 回写。跨 token tile 并行。
-1 处理幂等（-1 vs 真值不匹配，-1 vs -1 标记 -1 无害）。

### 5.4 mask_indices_by_tp（按 TP rank 屏蔽索引）
**源端**：`moe/mask_indices_by_tp_kernel.py` | **目标**：`moe/mask_indices_by_tp_pto.hpp`
**功能**：全局→本 rank 局部索引，不属本 rank 或 <0 置 -1。
**如何迁移**：`TDIVS`/`TREMS`/`TSUBS` 索引算术；`TCMPS`（标量比较）未提供 → `TEXPANDS` 广播+`TCMP`；
`>=0` 无 GE 模式 → 符号位技巧 `TAND(local, 0x80000000)`+`TCMP(==0)`；结果 `TSEL(dst, is_ok, local)`（dst 预置 -1）。

### 5.5 get_fused_mapping（token-topk↔expert-major 映射）
**源端**：`moe/get_fused_mapping_kernel.py` | **目标**：`moe/get_fused_mapping_pto.hpp`
**功能**：重排为 expert-major 连续区间。
**如何迁移**（最难）：源端用 warp ballot `__match_any_sync`+popcount；`TSORT32`/`TMRGSORT`/`TCUMSUM` 工具链均未暴露。
直方图统计用 tile（`TEXPANDS`+`TCMP`+`TSEL`+`TROWSUM`）→ expert_start/end（标量 prefix）；
pos_to_expert 区间用 `TEXPANDS(e)`+`TSTORE`(1×32 tile) 填充。
**位置映射**（token_topk_to_pos/pos_to_token 等）需 intra-tile 前缀和(scan)，工具链暂无 cumsum/sort，
该步用标量 cursor 推进（直方图与区间填充仍为 tile），待 scan 指令后可全 tile 化。

### 5.6 engram_hash（n-gram 哈希嵌入索引）
**源端**：`engram/engram_hash_kernel.py` | **目标**：`engram/engram_hash_pto.hpp`
**功能**：`hash = XOR_n(id[n]*mult[n])`；`out[col]=(hash%vocab)+offset`。
**如何迁移**：哈希累加跨 ngram 步是每 token 串行，但每步运算跨 token tile 并行：
`TMULS(id*mult)`+`TXOR`（累加）+`TREMS`(%vocab)+`TADDS`(+offset)。取列 n 的 id 用 (TileN×8 valid1) tile。
int32 hash（真实 int64 需 4 宽 tile，编译可用 int32）。

---

## 六、阶段 6：engram / mhc matmul 基座类

### 6.1 expand_to_mhc_bwd（MHC 扩展反向）
**源端**：`mhc/expand_kernel.py`(bwd) | **目标**：`mhc/expand_to_mhc_bwd_pto.hpp`
**功能**：`x_grad = Σ_m o_grad` 沿 mhc 维归约。
**如何迁移**：(MhcMult×TileH) tile，`TCVT`(bf16→fp32) 升精度 + `TCOLSUM` 沿行(mhc)归约 → (1×TileH)。与 fwd 正反对称。

### 6.2 norm_fn（RMSNorm+normw 合并）
**源端**：`mhc/norm_fn_kernel.py` | **目标**：`mhc/norm_fn_pto.hpp`
**功能**：`out = x*rsqrt(mean(x²)+eps)`；`out_fn = fn*normw`（列广播乘）。
**如何迁移**：`TMUL`(x²)+`TROWSUM`(平方和)+`TMULS`(/N)+`TADDS`(+eps) + 牛顿 rsqrt + `TROWEXPANDMUL`。
`TRSQRT` 工具链未提供 → 牛顿迭代：初值 `TRECIP(a)`，迭代 `x = x*(1.5 - 0.5*a*x*x)` 用 `TMUL`/`TMULS`/`TADDS` 4 轮。
normw_merge 用 `TCOLEXPANDMUL` 列广播乘。

### 6.3 multilayer_recompute（多层重算 GEMM 累加链）
**源端**：`mhc/multilayer_recompute_kernel.py` | **目标**：`mhc/multilayer_recompute_pto.hpp`
**功能**：逐层 `acc += comb_mix*residual` 重算前向中间激活。
**如何迁移**：`TMATMUL_ACC`(C+=A×B) 链式累加复用 `TileAcc`；首层 `TMATMUL` 清零起算；末尾 `ACCCVT` 导出。
A=TileLeft/B=TileRight/C=TileAcc。warp spec 软流水无对应 → 显式 serial 层循环+ACC 链式复用。

---

## 七、核心迁移模式总结（实际编译版）

1. **归约+广播指令族是收益点**：`TROWSUM/TROWMAX/TCOLSUM/TCOLMAX` + `TROWEXPANDMUL/SUB`/`TCOLEXPANDMUL`
   把归一化、sinkhorn、cast、量化的"逐元素归约+广播"降到 O(tile 数)。
2. **不可用指令普遍用可用指令模拟**（非"直接覆盖"，因工具链暴露面窄于 API 文档）：
   - `TROWEXPANDDIV` → `TRECIP`+`TROWEXPANDMUL`
   - `TABS` → `TMAX(TROWMAX(x), TROWMAX(TMULS(x,-1)))`
   - `TNEG` → `TMULS(x,-1)`
   - `TRSQRT` → 牛顿迭代（`TRECIP`+`TMUL`/`TMULS`/`TADDS`）
   - `TFMA` → `TMUL`+`TADD`
   - `TROWARGMAX` → `TROWMAX`+`TCMP`+逆索引`TSEL`+`TROWEXPAND`
   - `THISTOGRAM` → per-expert `TCMP`+`TSEL`+`TROWSUM`+`TINSERT` 直方图 tile
   - `TCMPS`(标量比较) → `TEXPANDS`广播+`TCMP`；`>=0` → 符号位 `TAND`+`TCMP(==0)`
3. **硬件原生指令替代手写优化**：`TTRANS` 替代 swizzle 转置、`TMATMUL_ACC`+`ACCCVT` 替代手写 GEMM 累加链。
4. **工具链约束适配**：`TLOAD/TSTORE` 形参左值引用→先 `auto ref=iter()` 绑定；`global_iterator` 仅 2 参→3D 展平 2D；
   tile 最小 128B、列宽 32B 对齐→归约输出用物理 8 宽 valid1、1 行 tile 取足宽度；
   `TCMP/TSEL` 三参同 dtype→索引运算全 float 存储时 `TCVT` 回 int；float 非类型模板参数不支持→eps 用 constexpr 局部量。
5. **SIMT→tile 范式断层**：warp spec/软流水/`__match_any_sync` ballot 无对应——
   `multilayer_recompute` 改显式 serial 层循环+ACC 链复用；`get_fused_mapping` ballot 改直方图+标量 cursor（位置映射需 scan，待 cumsum 指令）。

---

## 八、工具链约束速查（影响所有算子的通用规则）

- `TLOAD(dst, gm)/TSTORE(gm, src)` 形参为左值引用 → `iter(args)` 返回右值须先 `auto ref = iter(args);`。
- `global_iterator::operator()(int i, int j)` 仅 2 参 → 3D 张量展平为 2D。
- tile 物理大小须 128B..8KB；RowMajor 列宽须 32B 对齐（float/int32:8 的倍数，bf16:16 的倍数）。
  → 归约输出 1 宽 tile 用物理 8 宽 valid1；1 行 tile 取足宽（float≥32、bf16≥64）满足 128B。
- `TCMP(dst,s0,s1)`、`TSEL(dst,mask,src)` 三参须同 tile_shape（同 dtype 同形状）。
- float 不可作非类型模板参数 → 常数（eps 等）用 `constexpr` 局部量或函数参数。
- 注释中勿写 `TROW*/TCOL*` 等含 `*/` 的通配（会提前关闭块注释）。

## 九、验证

`deepseek/_compile_test.cpp` 实例化全部 19 个算子（含 norm_fn 的 rms_norm + fn_normw_merge 两变体），
经 `linx_blockisa_llvm_musl` 工具链（`-mlxbc -fenable-matrix -O2 -std=c++20 -D__linx`）编译+链接为 ELF，零错误。

## 十、待完成（见计划 §5）

- 阶段 2 剩余：per_block_cast、per_channel_cast_fused/and_transpose、cast_back_e5m6、per_token_cast_to_e5m6、
  per_block_cast_lossless（位运算路径）
- 阶段 3 剩余：top2_sum_gate、topk_sum_and_topk_group_idx（pairwise/归并）
- 阶段 5：MoE 门控端到端串联（compile.all）
- 阶段 6 剩余：engram_gate_fwd/bwd、grad_w_reduce、mhc 的 post/pre_apply_mix/pre_split_mixes/head_compute_mix/pre_big_fuse、sinkhorn_bwd
- 待 toolchain 暴露 scan/cumsum/sort 后，get_fused_mapping 位置映射可全 tile 化
