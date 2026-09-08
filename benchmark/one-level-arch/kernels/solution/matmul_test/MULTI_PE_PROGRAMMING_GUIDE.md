# 多 PE（4-PE）算子编程指南

> 适用范围：`benchmark/one-level-arch/kernels/multi_thread/` 与
> `test/kernel/multi_thread/` 下的四 PE 算子。
> 依据：PTO-ISA v0.58.3/0.58.4（TileOP-API `linx` 分支 `docs/tileop-usage/`）、
> SuperScalarModel `codex/pr-0.58.4-shared-model`（gfrun d8903938 实测）。
> 实例来源：本仓 multi_thread 算子（matmul/fa/element_wise/...）的现有实现。

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
| ③ Cooperative Group TMATMUL（Shared A+B） | `matmul/matmul_shared.hpp`、`matmul/matmul_test_mt.hpp`、`fa/` | **单个大 GEMM 沿 M 切分**：B 只装一次 4 PE 共读，GM 带宽省 4× |
| ③ + B 跨 M 驻留 | `matmul/matmul_shared_reuseB.hpp` | M 很大、K 块集可装进 Shared 256 KiB 的瘦高 GEMM，B 进一步免重装 |

决策要点：
- 算子内**没有矩阵乘** → ①/②（纯 Local tile，无集合点，最简单）。
- 算子是 GEMM/FA 且 B（或 K/V）被 4 个 PE 重复消费 → ③。
- ② 的"4 个独立 GEMM 共享 B"（batch 并行）历史上用 Local A + `TMOV_L2S_PUBLISH` 广播 B（`matmul_multithread.hpp`，**已废弃**），新代码直接用 ③。

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

fp32 CUBE 累加器 → fp16 输出：模型要求 TCVT 源/目的 layout 一致，CUBE 源直接 TCVT 会被断言拒绝；
用 **GM 往返**（FA 同款）：`TSTORE_CUBE(scratch_fp32) → TLOAD(Vec fp32) → TCVT(fp16) → TSTORE`，
scratch 每 PE 一片（≥ kPeM×tN float），由调用方提供。

### 4.5 B 驻留复用（③ 的扩展）

N 外 M 内循环序；B tiles 数组常驻 SharedTReg 跨 M 块复用，容量按
`(256 KiB − 活跃 A/B) / tileB 字节数` 编译期计算驻留数（`matmul_shared_reuseB.hpp`）。

## 5. 测试 harness（test/kernel/multi_thread/）

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
  避免"对齐 pattern 不触发舍入"的假阳性（见 `Test/verify_matmul_test_hard.py`）。

## 7. 常见陷阱

1. **忘带 `-s softcore.multiThreadNum=4`** → `Block BARG target` 断言。
2. **tM 取 96/160 等非 64/128 值** → 违反 kPeM 契约，编译期拒绝。
3. **动态 shape 想做运行时尾瓦片** → 共享瓦片静态形状断言；改用零填充（调用方
   pad 到 tile 整数倍）或约束整除。
4. **任一 PE 提前退出/走不同循环次数** → 集合 rendezvous 永远到不齐 → 死锁
   （gfsim/gfrun 表现为 stall）。4 PE 必须执行**完全相同**的 TMATMUL 序列。
5. **对 CUBE 累加器直接 TCVT** → "TCVT requires matching source/destination
   logical shapes" 断言；走 GM 往返。
6. **TLOAD 的 PEMask 当成"每 PE 冗余加载"** → 语义是 PE0 单发或 quarter 分区
   （§4.3），供数地址跟线程走，理解错会导致静态布局下读错 bank。
7. **栈越界症状怪异**：guest 栈被踩后 PC 乱飞、trace 刷 `invalid_block_type`、
   超时——先怀疑内核侧越界（如缓冲尺寸/leading dimension 算错），不是模型慢。
8. **worker 里调 exit** → 杀整组；worker 只能返回。

## 8. 参考实现索引

| 需求 | 看 |
|---|---|
| 最简 SPMD elementwise | `kernels/multi_thread/element_wise/tadd_multithread.hpp` |
| 标准 cooperative GEMM | `kernels/multi_thread/matmul/matmul_shared.hpp` |
| 动态 shape cooperative GEMM（零填充 + scratch 往返） | `kernels/multi_thread/matmul/matmul_test_mt.hpp` |
| B 驻留复用 | `kernels/multi_thread/matmul/matmul_shared_reuseB.hpp` |
| 低精度/MX cooperative | `kernels/multi_thread/matmul/matmul_shared_lowp.hpp` |
| 多阶段共享瓦物（Q/K/V staging） | `kernels/multi_thread/fa/fa_2d_unroll_gmma.hpp` |
| 测试 harness（SPMD/group runtime 双路径） | `test/kernel/multi_thread/matmul/src/matmul_shared.cpp`、`matmul_test_mt.cpp` |
| 精度验证工具 | `Test/verify_matmul_test_hard.py`（fork 侧，非仓内） |
