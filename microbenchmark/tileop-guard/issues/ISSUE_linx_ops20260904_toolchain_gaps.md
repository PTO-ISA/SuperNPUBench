# [linx-toolchain][ops-20260904] tgpr2t 后端 Match Instruction Error

## 验证环境

| 组件 | 分支 | commit / md5 |
|---|---|---|
| linx clang++ / lld | dev-llvm15_56 | md5 `e427d1429c0e` |
| Linx-TileOP-API | linx | `f8fb894` |
| SuperNPUBench | ops-20260904 | `a0ddcc3` |

---

## §1 TGPR2T：header 自身 bundle 无法被后端汇编

### 现象
```
.../tileop-api/jcore/template_asm.hpp:11077:8: error: Match Instruction Error!
  <inline asm>:4:1: note: instantiated into assembly here
```
`TGPR2T(dst, 0ull, 0ull, 0ull, 0ull)` 编译失败（clang++ 内联汇编 → LLVM 后端指令匹配阶段）。

### 根因
TileOP-API header（f8fb894）已声明 `TGPR2T`（`template_asm.hpp` 内有完整模板 + static_assert，属 0.58.5
layout-and-rearrangement 算子），其展开的内联汇编 `GPR2T` bundle 无法被配套 LLVM 后端（LinxV5）匹配——
即 header 暴露的 intrinsic 与后端的指令定义不同步（header↔backend skew）。

### 自证「非 demo 写法错」
demo 为最小单-intrinsic 调用，严格按 doc 签名 `TGPR2T(dst, gpr0..3)` 写；崩发生在 **header 自身**展开的
内联汇编、**LLVM 后端**指令匹配阶段（`template_asm.hpp:11077`），与 demo 的 tile 组织无关。即 header 自己的
GPR2T 助记符就无法汇编，任何调用路径都会失败。

### 建议
LinxV5 后端补齐 GPR2T 指令定义（或 header 暂时收起未被后端支持的 TGPR2T 声明，避免暴露无法汇编的 intrinsic）。

---

> 说明：本轮另有 `tmatmul_bf16`（clang frontend abort）与 `thistogram`（后端 Match Error）两个编译失败，
> 经核实分别属 **demo 侧**（release bf16 matmul 现编通过）与**尚未隔离**（release kernel 使用了 THISTOGRAM），
> 按「先自证非 demo 写法错再归因」纪律**不纳入本 issue**，详见 `REPORT_20260905.md`。
