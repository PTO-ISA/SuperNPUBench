# QSMLA — Quant Sparse Flash MLA（单层 PTO 实现）

> Quant Sparse Flash MLA：面向推理的**全量化稀疏 MLA 注意力**算子的
> SuperScalar（PTO 一层架构）参考实现。语义对标 ops-transformer 的
> `aclnnQuantSparseFlashMla`（Ascend 950PR/950DT，arch35）。

---

## 1. 算子原理

### 1.1 核心公式

MLA（Multi-head Latent Attention）中 K 与 V **共享同一份 latent 数据**，
配合 GQA（`G = N1/N2` 个 query 头共享一份 KV）：

```
O = softmax(Q · Kᵀ × score_scale + mask) · V，  K = V（共享 latent）

score_scale = softmax_scale × q_descale × kv_descale
  - FP16 路径：descale 全 1（纯 FP16 模拟）
  - HIF8 路径：per-tensor 反量化因子（Q/KV 各自持有）
```

注意力采用 **online softmax（两遍式，FlashAttention V2 风格）**：

```
Pass 1: 遍历全部 KV 块，归约行最大值 m 与 exp 和 l
Pass 2: 重访同一批块，P = exp(S − m)/l，O = Σ P·V（逐 D 块累加）
```

### 1.2 五种计算模式（按输入组合由 tiling 路由）

| 模式 | 输入组合 | KV 参与方式 |
|---|---|---|
| **SWA** | 仅 oriKv | 滑动窗口连续 KV + 窗口 mask |
| **HCA** | oriKv + cmpKv | ORI 窗口 + 压缩段连续取 `[0, cmpThreshold)` |
| **CSA** | oriKv + cmpKv + cmpSparseIndices | ORI 窗口 + 压缩段 **TopK** 选中行 |
| **ORI_SPARSE** | oriKv + oriSparseIndices | 全稀疏：ORI 侧 TopK |
| **ORI_CMP_SPARSE** | 两侧都带 SparseIndices | 两侧都 TopK |

关键语义：

- **双源逻辑拼接**：物理上从不拼出大 KV，kernel 先 ORI 块后 CMP 块遍历，
  靠 online softmax 共享 (m, l, O) 状态统一归一化
- **ORI 窗口**：`begin/end = clip(diagonal ∓ w, 0, S2)`，其中
  `diagonal = S2 − S1 + q`（因果对角线）；块对齐裁剪 + 边缘 mask（-1e30）
- **CMP 因果边界**：`cmp_valid_end = clamp(⌊(CmpS2·ratio − S1 + q + 1)/ratio⌋, 0, CmpS2)`
- **TopK 索引语义**：`-1` 终止、负值/≥valid_end 剔除、`topk_length` 截断、
  重复索引保留
- **HIF8 量化契约**（对齐官方算子）：P×16 → HIF8 参与 BMM2，PV 结果
  ×kv_descale，输出统一 ×1/16 还原，BF16 输出（`hifp8ScaleValue=16`）

### 1.3 实现形态（三条 kernel 路径）
当前主要迭代 `4-PE 动态 shape` 和 `单 PE CSA` 场景的gfrun和gfsim
| 路径 | kernel | CUBE 机制 | 转置方案 |
|---|---|---|---|
| **4-PE 统一**（五模式） | `quant_sparse_flash_mla.hpp` | Shared CUBE + `transpose_b()`（B 物理存储形状声明，pto-spec #257） | 数据不动，视角在 TMATMUL 指令里 |
| **4-PE 动态形状** | `quant_sparse_flash_mla_v2.hpp` | 同上 + Vec tile 用 DYNAMIC ValidRow/ValidCol | 同上 |
| **单 PE CSA** | `quant_sparse_flash_mla_csa_1pe.hpp` | Local CUBE（CubeTileM32/N8） | **qli 式输入预转置** + MGATHER staging（见下） |
| **单 PE SWA** | `quant_sparse_flash_mla_tadd.hpp` | Local CUBE | 输入预转置（batch 共享，经 `ori_kv_t_ptr` 参数） |

单 PE 路径为什么不能像 4-PE 那样"免转置"：TTRANS 已退役（PTO-ISA
0.58.5，TEPL 0x6E 保留为 reserved-illegal，无替代指令）；`TLOAD_CUBE`
的布局码只有 ND 源（`ND2M32/M16/N8`）；TMATMUL 的 `transpose_b()`
属性仅对 Shared B 合法（Local B 编译期拒绝）。因此单 PE 的 Kᵀ 布局由
**输入侧提供**（test 预转置出 `ori_kv_t [D, S2]`），CMP 侧 gather 用
**MGATHER 索引搬运**（TCI 线性索引 → 坐标算术 → byte offset → gather）
一次直出行序/转置序两个视图；mask 构造用 **U32 位模式 tile 链**
（TCI → TCMPS → TSEL → TSTORE）。

---

## 2. 文件说明

### 2.1 kernel 侧（`kernels/solution/quant_sparse_flash_mla/`）

| 文件 | 作用 |
|---|---|
| `quant_sparse_flash_mla.hpp` | **4-PE 统一五模式 kernel**（Shared CUBE + TransB）。`IMPL=swa/hca/csa/ori_sparse/ori_cmp_sparse_tadd_4pe` 的实现体 |
| `quant_sparse_flash_mla_v2.hpp` | 4-PE **动态形状变体**（`QSMLA_DYNAMIC_SHAPE=on`），Vec tile DYNAMIC ValidRow/ValidCol；八分节结构 |
| `quant_sparse_flash_mla_csa_1pe.hpp` | **单 PE CSA kernel**（FP16/HIF8 双态，六分节）：`CsaTiles` 类型目录 / `csa_stage_cmp` MGATHER 双链 / `csa_build_masks` 位模式 mask / `csa_mm1_score`（4 处去重）/ `csa_pass2_pv`（2 处去重）/ 主循环。`IMPL=csa_tadd_1pe` |
| `quant_sparse_flash_mla_tadd.hpp` | 单 PE SWA kernel（单 PE Local CUBE 机制参考源；方阵 Tk==Td 约束 + 行归约 ≤2048B 断言；BSND 层 batch 共享 Kᵀ 转置）。`IMPL=tadd` |
| `qsmla_config.hpp` | 共享形状/地址/窗口辅助：`QsmlaConfig`（work item 编解码）、`qsmla_swa_range/block_range`（窗口与块裁剪）、`QsmlaSwaRange` 等 |
| `qsmla_mode.hpp` | 五模式配置：`QsmlaModeConfig`、`qsmla_csa_cmp_valid_end`（CMP 因果边界）、`qsmla_sparse_collect_indices`（TopK 索引收集） |

### 2.2 测试侧（本目录）

| 文件 | 作用 |
|---|---|
| `Makefile` | 构建入口：`IMPL` / `QSMLA_DTYPE` / 形状参数 → ELF；嵌入输入数据对象打包；`QSMLA_DYNAMIC_SHAPE` 开关 |
| `src/quant_sparse_flash_mla.cpp` | 测试 main：dtype/模式选择、嵌入输入符号接线、按 `QSMLA_USE_*` 宏分发到对应 kernel |
| `qsmla_cases.py` | **用例定义与数据生成**：`QSMLA_CASES`（csa_small 等五模式用例）→ 生成 Q/KV/索引 bin + FP32 golden + **预转置 `ori_kv_t.bin`**（单 PE 路径输入） |
| `qsmla_reference.py` | CPU golden 参考实现（五模式统一语义，Python） |
| `run_qsmla_cpu_reference.py` | 用例驱动前端（`--case/--list/--all-reference`，`--dtype HIF8/FP16`） |
| `src/qsmla_compare.py` | **精度比对工具**（`--actual-dtype fp16/bf16`，atol/rtol 可调；HIF8 用 2e-2，FP16 用 1e-3） |
| `src/qsmla_cpu_reference.cpp` | C++ CPU golden 生成器（读已有 bin 出 golden，或 `--generate-deterministic` 自产数据） |
| `build_qsmla_input_object.py` | bin → 可链接 ELF 数据对象（`_binary_<sym>_start` 符号） |

### 2.3 kernel 结构约定（单 PE 路径的硬约束）

- **tile 必须函数局部**：跨函数的 tile 引用会触发编译器 raw TSTORE
  栈 spill（模型 `RecordRawTileTransport` 断言）——跨块状态只经 GM
  scratch（score/prob/pv 缓冲）传递
- **方阵 B tile（kTk == kTd）**：未打补丁 TileOP 头的 Local-B 双约定
  （编译期按 ValidCol / 运行期按 validRow）仅在方阵上一致
- **行归约源 ≤ 2048B**（pto-spec 0.58.6 的 TROWMAX/TROWSUM 契约）：
  tW [32,16] FP32 = 2048B 恰好合规
- **主段 Db 循环 `unroll_count(4)`**：unroll(full) 的 32 组在飞 tile
  峰值会打满 256KB tile 池（gfsim HIF8 实测）
- **不 include bench 的 `test/common/template_asm.h`**：其旧式 MGATHER
  （`BSTART.TMA 4`）会遮蔽 TileOP-API 活跃版（`BSTART.TLSU MGATHER`）

---

## 3. 编译与运行

### 3.0 环境准备

```bash
export COMPILER_DIR=<linx_blockisa_llvm_musl>/bin   # linx 工具链
# 模型（gfrun/gfsim）与数据路径示例：
#   GFRUN/GFSIM = <SuperScalarModel>/bin/{gfrun,gfsim}
#   数据根       = /tmp/qsmla-sparse-hif8（或 -fp16）
```

### 3.1 生成数据与 golden

```bash
cd benchmark/one-level-arch/test/solution/quant_sparse_flash_mla

# 列出全部注册用例
python3 run_qsmla_cpu_reference.py --list

# 生成 csa_small（含 ori_kv_t 预转置视图，单 PE CSA 必需）
python3 run_qsmla_cpu_reference.py --case csa_small --dtype HIF8 \
  --output-root /tmp/qsmla-sparse-hif8
python3 run_qsmla_cpu_reference.py --case csa_small --dtype FP16 \
  --output-root /tmp/qsmla-sparse-fp16
```

产物（`<root>/csa_small/`）：`q/ori_kv/ori_kv_t/cmp_kv.<ext>.bin`、
`cmp_sparse_indices.int32.bin`、`golden.fp32.bin`、`manifest.json`。

### 3.2 编译（常用四例）

```bash
# 必带：TESTCASE=quant_sparse_flash_mla
COMMON="TESTCASE=quant_sparse_flash_mla QSMLA_SPARSE_EMBED_INPUT=on \
  B=1 s1=1 s2=128 N1=64 N2=1 D=512 wleft=127 wright=0 \
  softmax_scale=0.04419417 cmp_s2=64 ori_topk=40 cmp_topk=40 cmp_ratio=4"

# (a) 单 PE CSA（FP16）—— 默认 Tm=32/Tk=16/Td_block=16
make $COMMON IMPL=csa_tadd_1pe QSMLA_DTYPE=FP16 \
  QSMLA_SPARSE_DATA_ROOT=/tmp/qsmla-sparse-fp16 Tm=32 Tk=16 Td_block=16

# (b) 单 PE CSA（HIF8）—— 注意数据 root 用 HIF8 的
make $COMMON IMPL=csa_tadd_1pe QSMLA_DTYPE=HIF8 \
  QSMLA_SPARSE_DATA_ROOT=/tmp/qsmla-sparse-hif8 Tm=32 Tk=16 Td_block=16

# (c) 4-PE CSA（静态 shape，HIF8）
make $COMMON IMPL=csa_tadd_4pe QSMLA_DTYPE=HIF8 \
  QSMLA_SPARSE_DATA_ROOT=/tmp/qsmla-sparse-hif8 Tm=64 Tk=32 Td_block=64

# (d) 4-PE CSA（动态 shape v2）—— (c) 再加 QSMLA_DYNAMIC_SHAPE=on
make $COMMON IMPL=csa_tadd_4pe QSMLA_DYNAMIC_SHAPE=on QSMLA_DTYPE=HIF8 \
  QSMLA_SPARSE_DATA_ROOT=/tmp/qsmla-sparse-hif8 Tm=64 Tk=32 Td_block=64

# (e) 单 PE SWA（tadd）—— 默认参数即 Tk16 方阵（N1=1 的 SWA 形状）
make TESTCASE=quant_sparse_flash_mla IMPL=tadd QSMLA_DTYPE=FP16
```

产物 ELF：`benchmark/one-level-arch/output/solution/quant_sparse_flash_mla/
elf/solution_quant_sparse_flash_mla/quant_sparse_flash_mla_B*_..._IMPL*_DTYPE*_INPUTembedded.elf`

> 注意：Makefile **不追踪 .hpp 依赖**，改 kernel 后须
> `touch src/quant_sparse_flash_mla.cpp` 强制重编。

### 3.3 运行 gfrun（精度）

```bash
ELF=<...>/quant_sparse_flash_mla_B1_s11_oriS2128_..._IMPLcsa_tadd_1pe_DTYPEHIF8_INPUTembedded.elf
TS=$(date +%Y%m%d_%H%M)

# 4-PE 需要 4 线程；单 PE 自动检测
$GFRUN -f $ELF -s softcore.multiThreadNum=4 \
  --dump-memory 0x4000802000:0x10000:out.bf16.bin > out.log 2>&1
tail -3 out.log          # 期望 "Suaccelss to Reach the End ... R2 = 0"
```

### 3.4 精度比对

```bash
# FP16：atol=rtol=1e-3；HIF8/BF16：atol=rtol=2e-2（不要沿用 1e-3）
python3 src/qsmla_compare.py --actual out.bf16.bin --actual-dtype bf16 \
  --golden /tmp/qsmla-sparse-hif8/csa_small/golden.fp32.bin \
  --b 1 --s1 1 --n1 64 --d 512 --atol 2e-2 --rtol 2e-2
```

### 3.5 运行 gfsim（时序/性能）

```bash
$GFSIM -f $ELF --dump-memory 0x4000802000:0x10000:sim.bf16.bin > sim.log 2>&1
grep -E "Total Cycles|Tileop Counter" sim.log
# gfsim dump 数值无意义（TimingSim 已知问题），数值以 gfrun 为准
# 单 PE CSA HIF8 参考：Total Cycles ≈ 2.98M（csa_small，TLSU 占 64%）
```

### 3.6 结果归档

按 `res/<几pe>_<场景>_<case名称>_<时间>/` 命名归档（见
`SuperScalar/res/README.md`），例：`res/1pe_csa_csa_small_20260917/`。

---

## 4. 已验证状态（2026-09-17）

| 场景 | gfrun | gfsim |
|---|---|---|
| 单 PE CSA FP16 | 100% @1e-3（max_abs 9.07e-06） | 无死锁，可完整推进 |
| 单 PE CSA HIF8 | 100% @2e-2（max_abs 4.68e-03） | **完整跑通 2,982,900 cycles**（finisher pass） |
| 4-PE CSA 静态/动态（HIF8） | 100% @2e-2 | — |
| 单 PE SWA（tadd FP16） | 100% @1e-3 | 历史通过（340 万 cycles） |
