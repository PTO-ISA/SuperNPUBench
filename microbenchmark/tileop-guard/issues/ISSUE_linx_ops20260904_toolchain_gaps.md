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

> 本轮另有 `tmatmul_bf16`（clang frontend abort）与 `thistogram`（后端 Match Error）两个编译失败：经核实
> `tmatmul_bf16` 属 **demo 侧**（release `tmatmul_bf16_32x64x64` 现编通过），`thistogram` 因 release kernel
> `group_count_aux_fi.hpp` 亦使用 THISTOGRAM、我方调用配置是否不同**尚未隔离**；二者按「先自证非 demo 写法错
> 再归因」纪律**暂不纳入本 issue**，详见 `REPORT_20260905.md`。
