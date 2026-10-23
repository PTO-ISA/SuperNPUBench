# 多 PE（4-PE）算子编程指南

> 适用范围：`benchmark/one-level-arch/kernels/basic_op/` 与
> `test/kernel/` 下的四 PE 算子。
> 依据：PTO-ISA v0.58.x（TileOP-API `linx` 分支 `bffad5b3`）、
> SuperScalarModel `587ec8cbe` + llvm `7675921e`（gfrun/gfsim 实测，
> 含 fp16 golden 位级验证）。
> 实例来源：本仓 basic_op 算子（matmul/fa/element_wise/...）与
> `solution/matmul_test/` 统一内核的现有实现。

## 1. 硬件与执行模型

| 概念 | 说明 |
|---|---|
| SMT4 分区 | 4 个硬件线程，`thread i ↔ PE i`（`get_thread_idx()` 返回 0..3）；各 PE 有独立的 TLSU/VEC/CUBE 流水与 Local Tile 寄存器 |
| 栈 | 每个 PE 独立 128 MB 栈 bank（gfrun 物理间隔 0x8010000） |
| 静态数据 | ELF `.bss/.data` 为**单一共享地址空间**——4 个 PE 读写同一份静态缓冲（见 §6 res_check 约定） |
| Local Tile | PE 私有，`B.IOT` 绑定，SizeCode 1..10（128 B..64 KiB / PE） |
| Shared Tile | 4 PE 共享数据面，`B.IOS` 绑定绝对寄存器 S0..S255，SizeCode 1..12（128 B..256 KiB / PE），**总池 256 KiB**；物理容量 = `popcount(PE_MASK) × per-PE` |
| PEMode | 0.58.3 起掩码仅限 `0000/1000/0100/0010/0001/1100/1110/1111`（参与型操作拒绝 0） |

运行方式（gfrun）：

```bash
gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>
```

不加 `-s softcore.multiThreadNum=4` 时 cooperative 程序会报 `Block BARG target`。

## 2. 四种执行模型与选型

**按算子特征选模型，不是风格偏好：**

| 模型 | 代表实现 | 何时用 |
|---|---|---|
| ① 连续 SPMD 切分 | `element_wise/tadd_multithread.hpp` | elementwise / 逐 tile 独立算子：无跨 PE 数据复用，`tid` 只切数据区间 |
| ② Tile 块 SPMD 轮转 | `conv2d`、`transpose` | 输出空间块可独立分配、需保持全局 layout |
| ③ Cooperative Group TMATMUL（Shared A+B） | `matmul/matmul_shared.hpp`、`solution/matmul_test/matmul_test_mt.hpp`、`fa/` | **单个大 GEMM 沿 M 切分**：B 只装一次 4 PE 共读，GM 带宽省 4× |
| ③ + B 跨 M 驻留（reuseB） | `matmul/matmul_shared_reuseB.hpp`、统一内核 `matmul_test.hpp`（`kMode=2`） | M 条带 ≥ 2 的瘦高 GEMM：B 每 k 装一次跨 R 个 M 条带复用 |
| ③ + A 跨 N 驻留（reuseA） | 统一内核 `matmul_test.hpp`（`kMode=1`） | A 紧、N 块多：A 每 k 装一次跨 R 个 N 块驻留累加器复用 |

决策要点：
- 算子内**没有矩阵乘** → ①/②（纯 Local tile，无集合点，最简单）。
- 算子是 GEMM/FA 且 B（或 K/V）被 4 个 PE 重复消费 → ③。
- ② 的"4 个独立 GEMM 共享 B"（batch 并行）历史上用 Local A + `TMOV_L2S_PUBLISH` 广播 B（`matmul_multithread.hpp`，**已废弃**），新代码直接用 ③。
- A 或 B 的重复装载成为 TLSU 墙时 → ③ + 对应方向的驻留复用（reuseA/reuseB，
  见 §4.6）；先按"重复装载方向"选：N 块多选 reuseA，M 条带多选 reuseB。

## 3. 模型①：连续 SPMD 切分

```cpp
template <int Rows, int Cols>
void vec_multithread(float* out, float* a, float* b) {
    using tileT = Tile<Location::Vec, float, Rows, Cols, BLayout::RowMajor>;
    const uint32_t tid = get_thread_idx();
    // 每 PE 负责连续一段：要求 Rows 可被 4 整除（编译期断言）
    itIn a_iter(a + tid * Rows * Cols);
    ...
    TLOAD(tA, src_a); TLOAD(tB, src_b); TADD(tC, tA, tB); TSTORE(dst, tC);
}
```

- 无任何跨 PE 操作：无 Shared、无集合点、无 barrier。
- 约束：切分维度整除 4（`static_assert`）；单 PE tile ≤ 8 KiB（vec 路径惯例）。

## 4. 模型③：Cooperative Group TMATMUL（重点）

### 4.1 标准骨架（对照 `matmul_test_mt.hpp`）

```cpp
constexpr int kPeNum = 4;
constexpr int kPeM   = tM / kPeNum;            // 每 PE 行数，见 4.2 契约
const uint32_t tid = get_thread_idx();

using tileAMatrix = SharedMatrixLeft<dtype, tM, tK>;   // 共享左操作数：普通 RowMajor 矩形
using tileBMatrix = SharedMatrixRight<dtype, tK, tN>;  // 共享右操作数
using tileAShared = SharedTile<tileAMatrix>;
using tileBShared = SharedTile<tileBMatrix>;
using tileAcc = CubeAccumulatorM16/M32<float, kPeM, tN>;  // PE 私有累加器（CUBE 布局）

for (/* 每个 [tM, tN] 输出块 */) {
    tileAcc tACC;
    for (k = 0; k < Kb; ++k) {
        tileAShared tAS; tileBShared tBS;
        TLOAD<tileAMatrix, 1>(tAS, gA);        // PEMask=1：PE0 单发整块（见 4.3）
        TLOAD<tileBMatrix, 1>(tBS, gB);
        k == 0 ? TMATMUL(tACC, tAS, tBS) : TMATMUL_ACC(tACC, tACC, tAS, tBS);
    }
    TSTORE_CUBE(gC, tACC);                     // 每 PE 写自己的行切片（见 4.4）
}
```

### 4.2 硬性契约（违反即编译期断言/运行期 abort）

| 契约 | 值/规则 | 出处 |
|---|---|---|
| group_M（共享 A 的行数） | 1..128；`SharedA.Rows == group_M`，每 PE 消费自己的 `kPeM` 行块 | `resolve_matmul_shape`、ADR-0100 |
| 每 PE 行数 kPeM | `group_M<=64 → 16`，否则 `32`（`cooperative_group_m_rows_per_pe`） | TileOP-API |
| **tM 取值** | 4-PE 行切分下只有 **tM=64（kPeM=16）或 tM=128（kPeM=32）** | 由上行推出 |
| tM>128 | 物化为多个完整 128 行 group（`kGroupM = min(tM,128)`），M 需整除 kGroupM | `matmul_shared.hpp` |
| 共享瓦片形状 | **静态有效形状**（valid 必须编译期常量，动态 valid 直接 static_assert 拒绝）→ 动态 shape 内核必须 M/N/K 整除 tile，无运行时尾瓦片 | `resolve_matmul_shape` |
| 共享主操作数布局 | 普通 RowMajor 矩形（`SharedMatrixLeft/Right`），**不得**用 CubeTile 布局 | `validate_matrix_contract` |
| PE 私有累加器 | `CubeAccumulatorM16/M32`（CUBE CELL 布局），fp32/整型输入派生 Acc 类型 | 同上 |
| B.IOS 绑定 | 集合操作要求 `mask=1111`（4 PE 到齐 rendezvous 才推进，任一 PE 缺席即死锁） | gfrun collective |
| Shared 容量 | 活跃操作数 + 驻留 tile 总计 ≤ 256 KiB | `matmul_shared_reuseB.hpp` |

### 4.3 GM→Shared 加载的两种 PEMask 语义

```cpp
TLOAD<tileAMatrix, 1>(tAS, gA);   // PEMask=0001
TLOAD<tileAMatrix, 15>(tAS, gA);  // PEMask=1111（可省略，默认 15）
```

| PEMask | 语义 | 供数地址 |
|---|---|---|
| `1`（0001） | **PE0 单发整块**（single-issuer full-object）：PE0 从自己的 GM 视图装入完整共享对象；PE1-3 执行同 PC 但掩码不命中，等效 no-op | 仅 PE0 的 `gA` 基址 |
| `1111` | **每 PE 填 1/4 字节区**（quarter）：共享对象按字节均分 4 区，各 PE 用**自己的**基址/stride 供数，各写各区 | 每 PE 独立 |

选型：SPMD 下 4 线程数据副本天然一致 → 用 `1`（当前仓库 shared/lowp/FA/test_mt 统一用法）；
各 PE 持有不同行切片想一次性拼成共享大 tile 时才用 `1111`。
（注意：`1` 的整块语义有精确守卫——仅 one-hot + 非分块 layout + 整尺寸时生效；部分更新自动退回 quarter 语义。）

### 4.4 输出写回（每 PE 行切片）

```cpp
// 逻辑输出块 [tM, tN]，本 PE 持第 tid*kPeM .. (tid+1)*kPeM 行
auto gC = gIterC(i * kPeNum + tid, j);   // 或按 gN 手工寻址
TSTORE_CUBE(gC, tACC);
```

**fp32 CUBE 累加器 → fp16 输出（TCVT，推荐）**：K 链 CCTRL 用四变体——
`Kb==1 → CCTRL=0`（单 op 直接写普通 tile）；`Kb>1 → head=1 / mid=3 / tail=2`。
末条 `TMATMUL_ACC` 用 CCTRL=2（`CubeCtrlInternalAccHint`）把累加结果
**物化到 CellReg tile**，`TCVT` 直接读该 tile 转 fp16 后 `TSTORE_CUBE`：

```cpp
constexpr FixpAttr kOptSingle = FixpAttr{}.transpose_b();                       // CCTRL=0
constexpr FixpAttr kOptHead   = kOptSingle.with_cube_ctrl(CubeCtrlRawAccumulator);   // 1
constexpr FixpAttr kOptMid    = kOptSingle.with_cube_ctrl(CubeCtrlRawAccAndHint);    // 3
constexpr FixpAttr kOptTail   = kOptSingle.with_cube_ctrl(CubeCtrlInternalAccHint);  // 2
...
Kb == 1 ? TMATMUL<kOptSingle>(tACC, tA, tB, groupM)
        : (k == 0       ? TMATMUL<kOptHead>(tACC, tA, tB, groupM)
        : (k == Kb - 1  ? TMATMUL_ACC<kOptTail>(tACC, tACC, tA, tB, groupM)
                        : TMATMUL_ACC<kOptMid>(tACC, tACC, tA, tB, groupM)));
TCVT(tCF16, tACC);        // tCF16: CubeAccumulatorM16/M32<__half, kPeM, tN>
TSTORE_CUBE(gC, tCF16);
```

- 直接在 CUBE 累加器上做 TCVT（不经 GM 往返）在新模型下可用且位级正确；
- **必须保留显式 tail（CCTRL=2）**：若末条仍用 CCTRL=3，新模型把该 D
  归类为 InternalAcc，TCVT 读不到合法 CellReg 描述符 →
  `PTO 0.58 CUBE M-format TCVT requires independently legal CELL descriptors`
  断言（此前的 GM 往返规避正是绕开这一路径）；
- 历史规避（仍可用的后备）：`TSTORE_CUBE(scratch_fp32) → TLOAD(Vec) →
  TCVT(fp16) → TSTORE`，scratch 每 PE 一片（≥ kPeM×tN float）。

### 4.5 B 驻留复用（reuseB）

N 外 M 内循环序；B tiles 数组常驻 SharedTReg 跨 M 块复用，容量按
`(256 KiB − 活跃 A/B) / tileB 字节数` 编译期计算驻留数（`matmul_shared_reuseB.hpp`）。
统一内核 `matmul_test.hpp` 的 `kMode=2` 是同族实现：每 k 步装 1 次 B 跨 R
个驻留累加器（M 条带方向），M 尾条带回退单链处理。

### 4.6 统一内核 matmul_test.hpp 的执行模型（reuseA/reuseB + TCVT）

`solution/matmul_test/matmul_test.hpp` 是动态 shape cooperative GEMM 的统一
实现（替代 `matmul_test_mt.hpp` 的 GM 往返路径）：

```cpp
template <int tM, int tN, int tK, int kTailM, int kMode, int kReuse,
          int gM, int gN, int gK>
void matmul_test(__half *c, __half *a, __half *b, float *scratch);
```

| 参数 | 含义 |
|---|---|
| `tM/tN/tK` | 基本块；tM∈{64,128}，tN/tK 为 2 的幂 |
| `kTailM` | M 尾组行数 ∈ {0,16,32,64}（pow2 group_M，shard 16）；0=传统整组 padding |
| `kMode` | 0=不复用 / 1=reuseA / 2=reuseB（宏 `REUSE_MODE`） |
| `kReuse` | 驻留累加器个数 1..4（宏 `REUSE_COUNT`，默认 4=上限） |
| `gM/gN/gK` | **编译期** shape（static RowMajor global_tensor） |

**三种执行模式**（每条带内）：

```
kMode=0：for j { for k { 装B; 装A; TMATMUL[_ACC] } } TCVT/TSTORE      （A/B 每 k 重装）
kMode=1：for jBase step R { acc[R]; for k { 装A 一次; for n<R { 装B; TMATMUL_ACC(acc[n]) } }
                            for n<R 物化+TCVT+TSTORE }               （A 跨 N 块复用 R 次）
kMode=2：for j { for g step R { acc[R]; for k { 装B 一次; for ms<R { 装A; TMATMUL_ACC(acc[ms]) } }
                                for ms<R 物化+TCVT+TSTORE } }         （B 跨 M 条带复用 R 次）
```

**M 尾组**（`kTailM>0`）：M 非 tM 整数倍时，尾行用独立 pow2 组
（group_M=kTailM，shard 16，零行 PE 读 A 零填充行、写 C 填充行），
避免整组 padding 的计算浪费；`compile.all` 用宏 `NO_TAIL_GROUP` 可回退。

**硬性实现约束（违反则模型断言/gfsim 挂起）**：

1. 驻留累加器**数组访问必须 `#pragma clang loop unroll(full)`**——运行时
   索引 `cs[n]` 会迫使编译器把累加器降到内存逐次重载，产生带 LB2 的
   `CUBE TLOAD requires omitted LB2/physical Col` 非法编码。reuseB 的条带
   循环须以编译期 R 为界（`for ms<R: if (ms<stripCount)`）。
2. **shape 必须编译期**（gM/gN/gK 模板参数）：运行时维度使寄存器分配器
   把 A tile 与某累加器 dst 放同一 tile 手（N/T/U/M 四手），dst 改名后
   cooperative 左操作数版本解析混淆 →
   `requestLeftInfo->validCol == leftPhysicalCols` 断言（R=4 必现）。
3. `kReuse ≤ 4`：驻留累加器占用 tile 手（4 hit 手 U/N/T/M 上限）；
   同手多代（T#1..T#4）由编译器处理，A 需独占一手。
4. 驻留容量编译期断言：`A tile + R×acc slice ≤ 256 KiB / PE`。

**选 tile 经验（M120–M256 / K=1536–7168 实测）**：tM=128、tN=32/64、
tK=128 起步；加速比与模式选择规律——Mb=1 用 reuseA、Mb≥2 用 reuseB；
大 K（Kb 大）时模型可回收重命名条目（无需担心 Kb 上限），
如 M256N512K7168 tN=64 reuseA 达 **99.6% 逻辑 MFU**。

## 5. 测试 harness（test/kernel/）

当前 harness 同一份源码支持两条执行路径（见 `test/common/LIGHTWEIGHT_GROUP_RUNTIME.md`）：

```cpp
extern "C" int __linx_group_worker_main(uint32_t peId, void *opaque) {
    // 算子体：peId 不用（内核内部用 get_thread_idx()）
}

int main() {
    // 静态缓冲（.bss 共享）+ 输入准备
#ifdef LINX_GROUP_RUNTIME
    status = linx_group_run(&context);       // hosted：main 仅 PE0 跑
#else
    status = __linx_group_worker_main(0, &context);  // SPMD：4 PE 各跑一遍 main
#endif
}
```

- **默认构建 = SPMD**：4 个 PE 从同一入口各跑一遍 `main()` + worker。静态缓冲共享同一地址；
  非 res_check 的输入填充若 4 份相同则幂等安全。
- **res_check（SPMD 路径）**：文件 I/O 只允许 PE0（`tid==0`），用
  `MultiThreadResCheckSync`（`test/common/multi_thread_res_check.h`）做
  输入发布/完成同步；写 res 也仅 PE0。
- worker **不得**调用 `_exit/exit_group`（会杀整个组），返回后由 PE0 的正常退出路径收尾。
- Makefile 约定：`group_runtime := on`、`TESTCASE` 白名单（未知值 `$(error)`）、
  非 float dtype 在 ELF 名加 `_DType<dtype>` 后缀防冲突。

## 6. 验证方法

- **功能**：`gfrun -t 1 -s softcore.multiThreadNum=4 -f <elf>`；PASS = 退出码 0 +
  `Reach the End of Benchmark` + `R2 = 0`。
- **精度（res_check 受限时）**：`gfrun --dump-memory <addr>:<size>:<file>` 导出输出缓冲
  （一次调用只支持一个 dump）。地址从 trace 提取：`BSTART.TMA TSTORE FP16` 后最近的
  `B.IOR` 基址。注意新旧 harness 内存布局差异：
  - 静态缓冲（新 harness）：4 线程写**同一**连续输出，单次 dump 即全量 C；
  - 栈缓冲（旧）：每线程独立 bank 视图，需 dump 4 次再按 `i*tM + tid*kPeM` 拼接。
- **golden 语义注意**：cooperative TMATMUL_ACC 是"从 C 出发的全序平坦累加"
  （`value = C; for k: value += a_k*b_k`），与单线程 Local 路径的
  block-then-add 结合律**不同**（见 Test/ISSUE-007）；bit 级比对需按路径选 golden。
- 输入用确定性 LCG 位模式（hard 随机 fp16）可让 host 侧位级复现，
  避免"对齐 pattern 不触发舍入"的假阳性（见 `test/solution/matmul_test/src/verify_matmul_test.py`）。

## 7. 常见陷阱

1. **忘带 `-s softcore.multiThreadNum=4`** → `Block BARG target` 断言。
2. **tM 取 96/160 等非 64/128 值** → 违反 kPeM 契约，编译期拒绝。
3. **动态 shape 想做运行时尾瓦片** → 共享瓦片静态形状断言；改用零填充（调用方
   pad 到 tile 整数倍）或约束整除。
4. **任一 PE 提前退出/走不同循环次数** → 集合 rendezvous 永远到不齐 → 死锁
   （gfsim/gfrun 表现为 stall）。4 PE 必须执行**完全相同**的 TMATMUL 序列。
5. **对 CUBE 累加器直接 TCVT**：K 链末条用 CCTRL=2 显式物化到 CellReg 后
   直接 TCVT 已支持（§4.4）；末条仍用 CCTRL=3 会被新模型归为 InternalAcc
   而断言。历史 GM 往返仍可作后备。
6. **TLOAD 的 PEMask 当成"每 PE 冗余加载"** → 语义是 PE0 单发或 quarter 分区
   （§4.3），供数地址跟线程走，理解错会导致静态布局下读错 bank。
7. **栈越界症状怪异**：guest 栈被踩后 PC 乱飞、trace 刷 `invalid_block_type`、
   超时——先怀疑内核侧越界（如缓冲尺寸/leading dimension 算错），不是模型慢。
8. **worker 里调 exit** → 杀整组；worker 只能返回。
9. **驻留累加器数组未 unroll(full)** → 编译器把累加器降内存重载，产生
   `CUBE TLOAD requires omitted LB2` 非法编码（§4.6 约束 1）。
10. **运行时 shape 做 R≥2 驻留** → tile 手分配漂移（A 与累加器 dst 同手），
    cooperative 左操作数版本断言（§4.6 约束 2）；shape 必须编译期。

## 8. 参考实现索引

| 需求 | 看 |
|---|---|
| 最简 SPMD elementwise | `kernels/basic_op/element_wise/tadd_multithread.hpp` |
| 标准 cooperative GEMM | `kernels/basic_op/matmul/matmul_shared.hpp` |
| **动态 shape cooperative GEMM 统一内核（reuseA/reuseB + 直接 TCVT）** | `kernels/solution/matmul_test/matmul_test.hpp`（§4.6） |
| 动态 shape cooperative GEMM（旧：GM 往返 TCVT） | `kernels/solution/matmul_test/matmul_test_mt.hpp` |
| B 驻留复用 | `kernels/basic_op/matmul/matmul_shared_reuseB.hpp` |
| 低精度/MX cooperative | `kernels/basic_op/matmul/matmul_shared_lowp.hpp` |
| 多阶段共享瓦物（Q/K/V staging） | `kernels/basic_op/fa/fa_2d_unroll_gmma.hpp` |
| 测试 harness（SPMD/group runtime 双路径） | `test/kernel/matmul/src/matmul_shared.cpp`、`test/solution/matmul_test/src/matmul_test_mt.cpp` |
| **统一内核测试入口（REUSE_MODE/REUSE_COUNT 宏）** | `test/solution/matmul_test/src/matmul_test.cpp` |
| 精度验证工具 | `test/solution/matmul_test/src/verify_matmul_test.py`（仓内） |
