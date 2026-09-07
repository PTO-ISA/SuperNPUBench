# [linx-toolchain][ops-20260904] TGPR2T 后端 Match Instruction Error

> **已提交**（2026-09-05）：`LinxISA/llvm-project` **#87** — https://github.com/LinxISA/llvm-project/issues/87

接口按文档正确写，但在 clang++ 内联汇编 → LLVM 后端指令匹配阶段编译失败。已排除 demo 写法：崩发生在
header 自身展开的内联汇编、后端匹配阶段。

## 组件版本清单

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | ziyang-cheng/SuperNPUBench | `tileop-guard-batch1` | `7395fec` |
| **llvm-project(本 issue 目标)** | LinxISA/llvm-project | `dev-llvm15_56` | `67d3ac9` |
| **Linx-TileOP-API(相关)** | LinxISA/Linx-TileOP-API | `linx` | `f8fb894` |
| linx clang/lld(产物) | — | — | md5 `e427d1429c0e` |
| musl | LinxISA/linx-musl | `linx` | `af0dfc2` |
| jemalloc | LinxISA/jemalloc | `linx` | `4495309` |
| linux-linxisa | LinxISA/linux | `main` | `1055a74` |

> 看护 demo 代码见 **SuperNPUBench PR #96**：https://github.com/PTO-ISA/SuperNPUBench/pull/96 （分支 `tileop-guard-batch1` @ `7395fec`）

## 通用复现步骤

```bash
git fetch origin tileop-guard-batch1 && git checkout 7395fec
source microbenchmark/tileop-guard/env.sh
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

---

## linx-1 · TGPR2T：header 自身 bundle 无法被后端汇编

**涉及接口**：TGPR2T。

**问题**：TileOP-API header（f8fb894）已声明 `TGPR2T`（`template_asm.hpp` 内有完整模板 + static_assert，属
0.58.5 layout-and-rearrangement 算子），其展开的内联汇编 `GPR2T` bundle 无法被配套 LLVM 后端（LinxV5）匹配
——header 暴露的 intrinsic 与后端指令定义不同步（header↔backend skew）。

**复现**：
```bash
bash run_guard.sh sfu tgpr2t
```

**错误信息**：
```
.../tileop-api/jcore/template_asm.hpp:11077:8: error: Match Instruction Error!
  <inline asm>:4:1: note: instantiated into assembly here
clang-15: 编译失败
```

**附加说明（自证非 demo）**：demo 为最小单-intrinsic 调用，严格按 doc 签名 `TGPR2T(dst, gpr0..3)` 写；崩发生在
**header 自身**展开的内联汇编、**LLVM 后端**指令匹配阶段（`template_asm.hpp:11077`），与 demo 的 tile 组织无关
——即 header 自己的 GPR2T 助记符就无法汇编，任何调用路径都会失败。**建议**：LinxV5 后端补齐 GPR2T 指令定义，
或 header 暂收起未被后端支持的 TGPR2T 声明。

---

> **更正（2026-09-07）**：`tmatmul_bf16` 经"唯 pto-spec 合规"判据复核 → 实为 **clang codegen bug**（见下方 linx-2）：
> demo 内联版符合 spec，clang 却 SIGABRT；noinline 能编证明代码合法（编译器对任何输入都不应 abort），
> 不能以"release 用 noinline 结构能编"为由判 demo 侧。`thistogram`（THISTOGRAM 已退休，见 `retired/`）作退休 selector 处理，不入本 issue。

---

## linx-2 · clang-15 对内联 bf16 CUBE matmul codegen SIGABRT

**涉及接口**：TMATMUL + `fixp::bf16()`（F322BF16，bf16 输出）。

**问题**：把 CUBE bf16 matmul 直接写在 `main()` 内联时，clang-15 在中端 "Function Pass Manager" 阶段
`abort()`（exit 134），非诊断错误：
```
#8 ... abort ./stdlib/abort.c:81:7
clang-15: error: clang frontend command failed with exit code 134
```

**复现**：`bash run_guard.sh cube tmatmul_bf16`（内联版）。

**自证 demo 合规 + 是编译器 bug**：
- demo 符合 spec：`fixp::bf16()`=F322BF16=码 16，在 B.FPATR 合法 PreQuantMode（16..20）内；matrix-postprocess.asl
  把 BF16 作合法目标转换。用法与 release `microbenchmark/fixp` BF16 模式一致（`run_single<__half,__bf16>`）。
- **编译器对任何输入都不应 SIGABRT**——把同一段逻辑包进 `noinline` 函数即正常编译（release fixp 正是 noinline 结构），
  证明代码语义合法，是 **clang 在 `__bf16 CubeAccumulatorM32` 内联实例化下的 codegen 健壮性 bug**。
- **不用 noinline 绕过**（会失去看护意义）；demo 保持内联以暴露该 bug。

**建议**：修 LinxV5 后端/clang 对内联 bf16 CUBE 累加器的 codegen（不应 abort）。
