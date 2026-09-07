# [linx-toolchain][ops-20260904] clang-15 对内联 bf16 CUBE matmul codegen SIGABRT

> **已提交**（2026-09-07）：`LinxISA/llvm-project` **#89** — https://github.com/LinxISA/llvm-project/issues/89

## 组件版本清单

| 组件 | 仓库 | 分支 / tag | commit |
|---|---|---|---|
| SuperNPUBench(看护 demo) | ziyang-cheng/SuperNPUBench | `tileop-guard-batch1`(PR #96) | `7163e60` |
| SuperScalarModel | LinxISA/SuperScalarModel | `codex/consolidate-post-main-fixes-20260903` | `49547742` |
| Linx-TileOP-API | LinxISA/Linx-TileOP-API | `linx` | `0566283` |
| llvm-project | LinxISA/llvm-project | `dev-llvm15_56` | `67d3ac9` |
| SuperNPUBench(release 参考) | PTO-ISA/SuperNPUBench | `ops-20260904` | `a0ddcc3` |

> 看护 demo 见 **SuperNPUBench PR #96**：https://github.com/PTO-ISA/SuperNPUBench/pull/96
> 工具链指纹：clang++ md5 `e427d1429c0e`、gfrun md5 `0c433cd11c00`。

## 通用复现步骤
```bash
git fetch origin tileop-guard-batch1 && git checkout 7163e60
source microbenchmark/tileop-guard/env.sh   # COMPILER_DIR / GFRUN
bash microbenchmark/tileop-guard/run_guard.sh <domain> <case>
```

以下每条均已用「唯 pto-spec 合规」判据自证 demo 写法无误，缺口在模型/工具链侧。

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
