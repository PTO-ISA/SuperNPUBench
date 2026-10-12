# SuperNPUBench 数值校验流程指南

> 本文描述 SuperNPUBench 的数值校验（金标准 / golden 比对）体系：原理、操作流程、
> 新算子接入方法、容差制定与结果解读。
> 历史校验结果的归因分析见 [`res_check_issues_2026-09-01.md`](../../docs/res_check_issues_2026-09-01.md)。

---

## 1. 概述：三层验证体系

| 层 | 名称 | 校验什么 | 工具 |
|---|---|---|---|
| 1 | gfrun 功能回归 | 只验"模型没崩、跑到终点"（不查数值） | `regression-gfrun` skill + `scripts/run_all.sh` |
| 2a | microbenchmark 数值校验 | kernel 内与 host C 参考逐元素比对 | `make res_check=on` + `bench_utils.hpp::verify()` |
| 2b | one-level-arch 数值校验 | kernel 写结果文件，host 侧 numpy golden 比对 | `make res_check=on` + `res_check_all.py` |
| 3 | 专用验证脚本 | 特定算子的多输出 / 格式化解码比对 | `golden_cmp.py`、`verify_*.py` 等 per-operator 脚本 |

**核心区别**：常规回归里 "PASS" 只代表模型跑到终点、无断言中止；数值校验才回答
**"结果对不对"**。两者通过率不可直接对比（见 §6）。

---

## 2. 原理

### 2.1 R2 机制 — 数值失败的传导链

gfrun（功能模型）跑完程序后打印：

```
Suaccelss to Reach the End of Benchmark! R2 = <rc>
```

其中 `R2` 是 ELF `main()` 的**返回值**：

- 普通构建：`main` 返回 0 → `R2=0` 只表示"跑完"。
- microbenchmark 的 `RES_CHECK` 构建：`main` 返回 `g_numeric_failure`
  （数值比对失败时置 1）→ **`R2=1` 即数值失配**。
- one-level-arch 的 host-golden 构建：kernel 只把结果写到文件，`R2` 仍只表示
  "跑完"；**真正的数值比对在 host 侧**（`np.allclose`），失败判为 FAIL。

### 2.2 RES_CHECK 编译开关

`make res_check=on` 时，`test/common/Makefile.common` 注入：

```makefile
ifeq ($(res_check), on)
DEFINES += -DRES_CHECK -DENABLE_BINARY_OUTPUT
DEFINES += -DCHK_DIR=\"$(ROOT)/compare/$(notdir $(basename $(TARGET)))\"
CC_LINK =
EXTRA_OBJ_FILES += $(OBJ_ROOT)/$(COMM_SRC_DIR)/group_worker_runtime.o
endif
```

- `-DRES_CHECK`：测试 cpp 中的 `#ifdef RES_CHECK` 块生效（读输入 / 写结果）。
- `-DCHK_DIR`：输入/输出文件的宿主目录，固定为
  `benchmark/one-level-arch/compare/<ELF-stem>/`。
- 链接 `group_worker_runtime.o`：4-PE hosted 执行的 worker 入口（见 §2.3）。

### 2.3 4-PE hosted 执行 — group_worker_runtime

`test/common/src/group_worker_runtime.c`：

- gfrun（commit accc09b9 起）把 **PE0 的 PC 设为 ELF 入口**（musl `_start`），
  **PE1..PE3 的 PC 设为 `__linx_group_worker_start`**，每个 PE 有独立 128 MB 栈。
- worker 调一次 `main()` 后进入死循环（不能 `exit_group`，否则会杀掉还没写完
  结果文件的 PE0）；PE0 的 libc exit（`SYS_exit_group`）最终释放整组。
- 效果：**四个 PE 跑的是同一个 `main()`（SPMD）**，靠 `tid = get_thread_idx()`
  区分角色。

### 2.4 SPMD 同步 barrier — multi_thread_res_check.h

`test/common/multi_thread_res_check.h` 提供两个自旋 barrier：

```cpp
struct MultiThreadResCheckSync {
    volatile std::uint32_t input_ready;   // PE0 写完输入后置 1
    volatile std::uint32_t done[4];       // 各 PE 算完后各置 1
};
```

- `res_check_publish_inputs()`：PE0 发布输入（其余 PE 自旋等 `input_ready`），
  防止 PE1..3 在输入落盘前开跑。
- `res_check_wait_for_all()`：各 PE 报完成，PE0 等 PE1..3 都 `done`，防止 PE0
  在部分 PE 未完成时就导出结果。

### 2.5 microbenchmark：kernel 内自比对

`microbenchmark/common/bench_utils.hpp` 提供：

```cpp
template <typename T>
bool verify(const T *got, const T *ref, int n, T eps = (T)1e-3,
            T rel_eps = (T)1e-3) {
    for (int i = 0; i < n; ++i) {
        if (got[i] == ref[i]) continue;        // bit-exact 直接过
        if (got != got || ref != ref) return false;  // NaN 失败
        const double limit = eps + rel_eps * |ref|;
        if (|got - ref| > limit) return false;
    }
    return true;
}
```

判定公式即 `|got - ref| ≤ eps + rel_eps·|ref|`（绝对 + 相对混合容差）。
测试 cpp 在 `#ifdef RES_CHECK` 下把算子输出与 host C 参考比对，失败置
`static volatile int g_numeric_failure = 1`，`main()` 末尾 `return g_numeric_failure`
→ gfrun 报 `R2=1`。

### 2.6 one-level-arch：host-golden 范式

**kernel 侧**（以 `fa_gmma_kchains.cpp` / `matmul_shared.cpp` 为模板）：

```cpp
#ifdef RES_CHECK
    static MultiThreadResCheckSync resCheckSync{};
#endif
    // ... 静态对齐缓冲区 q/k/v/out ...
#ifdef RES_CHECK
    if (tid == kIoTid) {                       // 只有 PE0 做 I/O
        readBinaryFile(CHK_DIR "/srcq.bin", q, ...);
        readBinaryFile(CHK_DIR "/srck.bin", k, ...);
        readBinaryFile(CHK_DIR "/srcv.bin", v, ...);
    }
    res_check_publish_inputs(resCheckSync, tid);
#endif
    BENCHSTART;
    flash_attention_..._impl(out, q, k, v);    // 四个 PE 一起算
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(resCheckSync, tid);
    if (tid == kIoTid) {
        writeBinaryFile(CHK_DIR "/res.bin", out, ...);
    }
#endif
```

**host 侧**（`test/kernel/res_check_all.py`，16 个 Case）：

```
prep_<op>()          # numpy 生成 input.bin + 返回 golden（参考实现）
    ↓
gfrun -s softcore.multiThreadNum=4 -f <ELF>     # 跑 res_check=on 编译的 ELF
    ↓
np.fromfile(compare/<ELF-stem>/res.bin)         # 读 kernel 写出的结果
    ↓
np.allclose(actual, golden, atol, rtol)         # PASS / FAIL
```

要点：
- 输出文件会先被 runner 预写为全零占位（防止上次残留 + 保证尺寸可判）。
- `Case` dataclass 决定 `elf / output_name / output_dtype / atol / rtol`。
- ELF 缺失判 SKIP；gfrun 超时判 TIMEOUT；rc≠0 判 FAIL。
- 退出码：有 FAIL 返回 1，可接入 CI。

solution 树有对称版本 `test/solution/common/res_check_all.py`：同一范式，但
**自包含编译**（运行前对涉及算子注入 `res_check=on` 重编），并支持多输出 /
专用 golden 的 `verify` 钩子（如 `dynamic_mx_quant` 的 data + scale 双输出解码比对）。

### 2.7 专用脚本层（第 3 层）

| 脚本 | 用途 |
|---|---|
| `test/kernel/matmul/src/golden_cmp.py` | matmul 通用 golden 比对（`make golden-check` 集成） |
| `test/kernel/matmul/src/gfrun_matmul.py` | matmul_lowp FP8/FP4 解码比对 |
| `test/kernel/mxquant/src/run_mxquant_check.py` | MX 量化检查 |
| `test/solution/matmul_test/src/verify_matmul_test.py` | solution matmul |
| `test/solution/quant_batch_matmul_test/src/verify_quant_batch_matmul_test.py`、`verify_hif4.py` | 量化 batch matmul（含 HIF4） |
| `test/solution/quant/dynamic_mx_quant/src/*.py` | 动态 MX 量化 gen + compare + precision |
| `test/solution/quant_sparse_flash_mla/*.py` | 稀疏 FA（qsmla_cases / reference / cpu_reference / compare） |
| `test/solution/qli/src/gen_qli_golden.py` | QLI golden 生成 |

golden 数据入库于 `benchmark/one-level-arch/compare/<test>/golden*.bin`。

---

## 3. 操作流程

环境约束（`AGENTS.md`）：
```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
GFRUN=/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl/bin/gfrun   # asl worktree
```

### 3.1 第 1 层：功能回归（对照）

```bash
# 见 .agents/skills/regression-gfrun/SKILL.md 完整流程
JOBS=6 bash .agents/skills/regression-gfrun/scripts/run_all.sh /tmp/regress_$DATE/run
```
PASS 判据：gfrun 输出（末 64KB）含 `Reach the End of Benchmark` 且 `R2 = 0`。

### 3.2 microbenchmark 数值校验

`microbenchmark/Makefile.common` 在 `res_check=on` 时把 `OBJ_ROOT` 重定向到
**独立的 `output/res_check/`**（与普通产物分开），并加 `-DRES_CHECK`：

```bash
# 全量（在 microbenchmark/ 目录内运行，compile_all.sh 用相对 cd）：
# 环境变量 res_check=on 经 make 的 ?= 默认值链路传入各 category
cd microbenchmark && res_check=on bash compile_all.sh all   # 产物落 output/res_check/

# 单个：
cd microbenchmark/<cat> && make res_check=on TESTCASE=<name> COMPILER_DIR="$COMPILER_DIR"
$GFRUN -f output/res_check/<cat>/<elf>          # 看 R2
```
`R2=0` → 数值一致；`R2=1` → kernel 内 verify 失败（数值失配）。

### 3.3 one-level-arch 数值校验

```bash
cd benchmark/one-level-arch/test/kernel/<op>

# 1) 编 res_check 版 ELF（同目录 Makefile）
make res_check=on TESTCASE=<name> COMPILER_DIR="$COMPILER_DIR" <params>

# 2) host 侧跑比对（单 case 或全量）
python3 benchmark/one-level-arch/test/kernel/res_check_all.py [case名...] \
    --gfrun $GFRUN
```

输出格式：
```
PASS    fa                   max_abs=1.2e-06
FAIL    matmul_shared        max_abs=0.5
SKIP    conv2d               missing ELF: ...
summary: PASS=14 FAIL=1 SKIP=1
```

### 3.4 专用 golden-check（matmul 例）

```bash
cd benchmark/one-level-arch/test/kernel/matmul
make golden-check TESTCASE=matmul_shared COMPILER_DIR="$COMPILER_DIR" \
     B=1 M=256 N=256 K=256 tM=128 tN=256 tK=128
# 内部：res_check=on 强制重编 + 调 golden_cmp.py / gfrun_matmul.py
```

---

## 4. 新算子接入指南（以 FA 类算子为例）

**第 1 步：kernel 测试 cpp 加 RES_CHECK 块**（模板照抄 `fa_gmma_kchains.cpp`）：

```cpp
#include "multi_thread_res_check.h"

int main() {
    const uint32_t tid = get_thread_idx();
    constexpr int kIoTid = 0;
    // 静态缓冲（4KB 对齐）
    static Dtype qp[Sq * QD + 2 * ALIGN]; /* ... k, v, out 同理 */
    auto *q = align_ptr(qp);              // (ptr & ~0xfff) + 0x1000
#ifdef RES_CHECK
    static MultiThreadResCheckSync resCheckSync{};
#endif
#ifdef RES_CHECK
    if (tid == kIoTid) {
        readBinaryFile(CHK_DIR "/srcq.bin", q, ...);   // 命名自定义，与 prep_* 对应
        readBinaryFile(CHK_DIR "/srck.bin", k, ...);
        readBinaryFile(CHK_DIR "/srcv.bin", v, ...);
    }
    res_check_publish_inputs(resCheckSync, tid);
#endif
    BENCHSTART;
    my_operator_impl(out, q, k, v);
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(resCheckSync, tid);
    if (tid == kIoTid)
        writeBinaryFile(CHK_DIR "/res.bin", out, ...);
#endif
    return 0;
}
```

注意：
- 文件 I/O 只在 PE0；输入/输出缓冲在共享静态存储（四个 PE 可见）。
- `CHK_DIR` 由 Makefile 注入，指向 `compare/<ELF-stem>/`。

**第 2 步：编译验证**

```bash
make res_check=on TESTCASE=<name> COMPILER_DIR="$COMPILER_DIR" <params>
```

**第 3 步：res_check_all.py 加 Case**

```python
def prep_myop(case_dir: Path) -> np.ndarray:
    rng = np.random.default_rng(<固定种子>)
    q = rng.uniform(-0.1, 0.1, (Sq, QD)).astype(np.float32)
    # ... k, v
    write(case_dir, "srcq.bin", q); write(case_dir, "srck.bin", k); write(case_dir, "srcv.bin", v)
    # numpy 参考实现（golden）
    score = q @ k.T / np.sqrt(QD)
    score -= score.max(axis=1, keepdims=True)
    p = np.exp(score); p /= p.sum(axis=1, keepdims=True)
    return (p @ v).astype(np.float32).reshape(-1)

CASES += [Case("myop", "fa/elf/kernel_fa_<配置>.elf", prep_myop,
               output_name="res.bin", atol=3e-2, rtol=3e-2)]
```

**第 4 步：跑**

```bash
python3 benchmark/one-level-arch/test/kernel/res_check_all.py myop
```

**solution 树算子**：接 `test/solution/common/res_check_all.py`，范式见其文件头
docstring（单输出 numpy golden 用 `prepare`；多输出/专用 golden 用 `verify` 钩子，
golden 可由算子自带 gen 脚本在 compile.all 阶段生成）。

---

## 5. 容差制定

| 算子/类型 | atol / rtol | 依据 |
|---|---|---|
| transpose（int32） | 0 / 0 | bit-exact 要求 |
| matmul（FP32） | 1e-3 | FP32 累加序差异 |
| conv2d | 2e-3 | 累加 + 布局 |
| FA（softmax+exp） | 3e-2 | exp/除法非线性放大舍入差 |
| gelu（FP16 出） | 2e-2（dtype=fp16） | 半精度 |
| microbench `verify()` | eps=1e-3, rel=1e-3（默认） | 通用 |
| microbench `verify_scalar` | ≤2B 类型 2e-2，否则 1e-4 | `verify_epsilon<T>()` |

制定原则：
1. **从参考实现的数值路径推**：含 exp/log/div 的链路（softmax）容差放大一个量级。
2. **从输出 dtype 推**：FP16 出口 ≥2e-2；FP32 出口 1e-3~1e-4。
3. **先收紧再放宽**：新算子先用紧容差跑，看 max_abs 分布再定档，避免一开始
   就放宽掩盖真实问题。
4. runner 会输出 `max_abs`，可直接用于回归监控（数值漂移告警）。

---

## 6. 结果解读与历史数据

### 判定语义速查

| 层 | 信号 | 含义 |
|---|---|---|
| 功能回归 | `Reach the End` + `R2=0` | 跑完、无断言、（microbench res_check 版还含数值一致） |
| microbench res_check | `R2=1`，rc=0 | kernel 内 verify 数值失配 |
| one-level res_check | runner 打印 `FAIL` + `max_abs` | host 侧 golden 比对失配 |
| 任一层 | rc≠0 / `ASSERTION FAILED` / `illegal` | 模型断言中止（功能问题，先于数值） |
| runner | `SKIP` | ELF 未编译（先查 compile） |

### 全量数值校验结果（2026-09-01 基线）

- 范围：**492 个 res_check ELF**（microbench 398 + one-level 94），全量重编 +
  gfrun 复跑 + golden 比对。
- 结果：**271 PASS / 221 FAIL（55.1%）**；同期常规功能回归通过率 ~81.7%。
  **差距 ≈ 26.6 个百分点全部是"跑到终点但数值不对"的假绿**。
- 归因（运行阶段 221 FAIL）：编译器 0、算子 0、**gfrun 模型 219**、待定 2；
  另有编译阶段 clang 前端 SIGABRT 2（`trem/trems_fp16`，仅 `-DRES_CHECK` 触发）。
- 最大单项：`micro/vector` 114 个 binary TEPL `srcTile[0]` 读零缺陷
  （`validRow/validCol` 未置位 → 读出全零）。
- 详细归因矩阵、断言聚类与处置建议：[`res_check_issues_2026-09-01.md`](../../docs/res_check_issues_2026-09-01.md)。

### 与功能回归的关系

```
功能回归 PASS（~82-96%）
    └── 其中相当一部分在数值校验下 FAIL —— "假绿"
数值校验 PASS（~55%，09-01 基线）
    └── 真正的端到端数值正确
```

因此：**功能回归是冒烟，数值校验是准入**。模型侧修复后应重跑数值校验确认
真实收益，而不是只看功能回归通过率。

---

## 7. 文件索引

| 文件 | 角色 |
|---|---|
| `benchmark/one-level-arch/test/common/Makefile.common` | `res_check=on` 注入 `-DRES_CHECK`/`CHK_DIR` |
| `benchmark/one-level-arch/test/common/multi_thread_res_check.h` | 4-PE SPMD barrier |
| `benchmark/one-level-arch/test/common/src/group_worker_runtime.c` | PE1..3 worker 入口 |
| `benchmark/one-level-arch/test/kernel/res_check_all.py` | kernel 树 host-golden runner（16 Case） |
| `benchmark/one-level-arch/test/solution/common/res_check_all.py` | solution 树 runner（自包含编译 + verify 钩子） |
| `microbenchmark/common/bench_utils.hpp` | `verify()/verify_scalar()/verify_zero()` |
| `microbenchmark/fixp/src/fixp_tmatmul.cpp` 等 | `g_numeric_failure` → `main` 返回值 → R2 |
| `benchmark/one-level-arch/compare/<ELF-stem>/` | 输入 src*.bin、结果 res.bin、golden*.bin |
| `.agents/skills/regression-gfrun/SKILL.md` | 第 1 层功能回归流程 |
| `docs/res_check_issues_2026-09-01.md` | 09-01 全量校验归因分析 |

## 8. 已知问题与注意事项

1. **gfrun 输出切勿重定向到文件**：失控算子会无限打印 counter；用滚动窗口
   （`tail -c 65536`）或 runner 的内存窗口。
2. **旧结果残留**：`compare/<case>/` 下上轮的 `res.bin` 会被 runner 预写零覆盖，
   但手工调试时要自己清理，避免读到陈旧文件误判。
3. **clang-15 前端 bug**：`trem/trems_fp16` 在 `-DRES_CHECK` + `-fenable-matrix`
   下 SIGABRT（普通构建通过），属编译器侧，已建档。
4. **多输出算子**不要硬塞进单输出 `Case`，用 solution 树的 `verify` 钩子范式。
5. **R2 的语义取决于构建方式**：看到 `R2=1` 先确认是 microbench res_check 版
   （数值失配）还是普通版（不太可能非零返回）。
