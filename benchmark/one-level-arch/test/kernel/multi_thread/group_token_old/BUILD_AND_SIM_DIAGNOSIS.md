# Multi-Thread Group Token Old — 编译与仿真诊断记录

> 目标：编译并仿真 `test/kernel/multi_thread/group_token_old`（4 PE SPMD 版三阶段
> MoE Token 分组算子）。结论：**当前工具链 + 仿真器组合下无法端到端跑通**，
> 本文记录全部执行路径、命令、现象与逐层定位证据。
> 验证日期：2026-08-13。

## 1. 执行环境路径

| 项 | 路径 |
|---|---|
| 测试源码 | `benchmark/one-level-arch/test/kernel/multi_thread/group_token_old/src/group_token_old.cpp` |
| 内核头文件 | `benchmark/one-level-arch/kernels/group_token_old/group_token_old.hpp` |
| Linx 工具链 | `<workspace>/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin`（clang 15.0.4，`linx64v5-unknown-linux-musl`） |
| 生成 ELF | `benchmark/one-level-arch/output/kernel/multi_thread/group_token_old/elf/kernel_multi_thread_group_token_old_group_token_old.elf` |
| 仿 真 器 | `<workspace>/SuperScalarModel/bin/`（`gfrun` 功能模型 / `gfsim` 时序模型） |
| 运行线程 | gfrun：`-s softcore.multiThreadNum=4`（PEID=0x802 返回 0..3，`SoftCore.cpp:28`） |

## 2. 编译

### 2.1 命令与结果

```bash
export COMPILER_DIR=<workspace>/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
cd benchmark/one-level-arch/test/kernel/multi_thread/group_token_old
make TESTCASE=group_token_old COMPILER_DIR=$COMPILER_DIR
```

**直接编译失败**（所有 multi_thread 测试均如此，含 vec/matmul/fa）：

```text
error: inline-asm should start from BSTART or Tile Call.
  asm volatile("SSRGET 0x802, ->%0" : "=r"(thread_id));   // pto_tileop.hpp:15
```

- 根因：`linx-toolchain-build/src/llvm-project/llvm/lib/Target/LinxV5/AsmParser/LinxV5AsmParser.cpp` 的
  `maybeValidateInlineAsm()` —— 内联汇编必须以 tile 指令、BSTART 或 header-only 指令开头；
  标量指令 `ssrget` 不满足（`SSR_GET` 为 E 类核心指令，见 `LinxV5InstrInfo.td:1965`）。
- 工具链自带逃生口：汇编指令 `.unsafeasm`（`ParseDirective` 置 `IAVAS=IA_UNSAFE` 跳过校验）。
- 规避：在测试文件内宏改名工具链的 `get_thread_idx`，用 `.unsafeasm` 前缀自实现同名函数后
  **可编译通过**（`CC_OPT=default` 亦可）。测试源码已还原，规避补丁未落盘。

### 2.2 编译产物

`kernel_multi_thread_group_token_old_group_token_old.elf`（.text ~1.6KB，.bss 36KB）。

## 3. 仿真过程与结果

统一命令：

```bash
cd <workspace>/SuperScalarModel
./bin/gfrun -f <elf> -s softcore.multiThreadNum=4 [-t 1] [-m <块上限>]
```

### 3.1 现象矩阵（不同代码形态 × 4 线程）

| # | 代码形态 | 编译 | 运行 | 定位 |
|---|---|---|---|---|
| A | 原样（官方 -O2 flags） | ✗ 汇编校验拒绝 | — | AsmParser 校验 |
| B | `.unsafeasm` 内联版 | ✓ | **无限循环挂起** | ssrget 被 remat 进循环，覆盖归纳变量 a0 |
| C | `.unsafeasm` + `noinline` | ✓ | **仿真器 SIGSEGV** | 块调用路径 FENTRY 处崩溃 |
| D | `.unsafeasm` + volatile 全局中转 | ✓ | **确定性 SIGSEGV**（B4689） | 循环边界寄存器 a7 被破坏为负值 |
| E | `.unsafeasm` + tid 作参数（只取一次） | ✓ | **无限循环挂起** | 同 D：a7 巨型负值 → 内层循环死转 |

对照组：单线程 `test/kernel/group_token_old`（无 `ssrget`）编译、gfrun（R2=0）、
gfsim（255598 cycles）**全部正常** —— 差异仅在 multi_thread 版线程 ID 的读取路径。

## 4. 日志分析（关键证据）

### 4.1 现象 B：ssrget 重载入循环（-O2 官方 flags）

```asm
；  phase2 外层循环（groupToken_multithread 的 for(i=tid; i<512; i+=4)）
1148e:  ssrget 2050, ->a0        ; a0 = tid（循环预头）
11506:  addi   a0, 4, ->a0       ; i += 4  ← a0 兼作归纳变量
1150c:  setc.ltui t#1, 512        ; 基于 i 的退出条件
11518:  ssrget 2050, ->a0        ; ★ 循环头部再次 ssrget，a0 = tid（重载！）
1151c:  C.BSTART.STD COND, ...
1151e:  setc.ltu a2, a0.sw        ; 退出测试改用 tid：511 < tid 恒为假 → 死循环
```

即编译器把 volatile 内联汇编的结果 rematerialize 到循环体内，与循环归纳变量
共用一个寄存器 a0，退出条件被 tid 覆盖 ⇒ **代码生成缺陷**。

### 4.2 现象 C：块调用崩溃（gfrun backtrace）

```text
Program received signal SIGSEGV
#0 superScalar::SoftCore::ExecuteSTDMinst(...)
#1 ExecuteMinst #2 EmulatorBlock #3 Step #4 main
```

trace 显示崩溃前最后执行块为 `C.BSTART.STD DIRECT, 0x112da <get_thread_idx>`
（`c.setret ... -> ra` 后在函数 FENTRY 处崩溃）—— 仿真器对"普通函数块调用"路径支持不完整。

### 4.3 现象 D/E：循环边界寄存器被破坏（决定性证据）

```asm
11426: ssrget 2050, ->a0      ; tid（单线程时 a0=0，trace 确认正确）
1142a: C.BSTART.STD DIRECT, 0x11630 <memset>   ; ★ 调用夹在中间
11432: swi.u   a0, [t#1, 690] ; tid 存储被调度到调用之后（a0 语义已悬垂）
11436: lwui.u  [t#1, 690], -> s1               ; s1 = 栈槽装入
```

内层 topk 循环死转时的实测状态（`-t 1` trace，单线程亦复现）：

```text
M1332616|0x114fc: addi a5, 1 -> a5        ; a5: 0x1c00a → 0x1c00c 持续增长
M1332617|0x11500: setc.ltu a5, [a7.sw]    ; [a7.sw] = 0xffffffffffffffef(=-17)
```

循环上界 a7（应为 16）变为巨型负值 ⇒ `for(j=i*topk; j<stop; j++)` 永不退出。
tid 值经全局/栈槽中转后，在下游 `s1<<4 / s1<<9 / or zero, t#1.uw` 等操作中寄存器
复用出错，破坏循环不变量 —— 属 **LinxV5 后端调度/liveness 建模缺陷**，非算子逻辑错误。

## 5. 结论与建议

1. **多线程测试套件当下不可端到端运行**：`get_thread_idx()`（内联 `ssrget`）是唯一触发点，
   编译期被 AsmParser 拒绝，运行期被后端代码生成（remat/调度）与仿真器（块调用）
   多重缺陷击中；未受影响的单线程姊妹算子完整通过。
2. 该问题需要**上游修复**（benchmark 仓无法在不动工具链/仿真器的前提下绕过）：
   - 编译器 `linx-blockisa` 后端：inline-asm 输出寄存器不得与循环归纳变量/条件寄存器共址；
     `SSRGET` 应具备合法 asm 起始指令资格（或 header 增加 `.unsafeasm`/标记）。
   - AsmParser：`ssrget` 作为标准标量指令应放行（同步 `maybeValidateInlineAsm` 白名单）。
   - Simulator：支持普通函数块调用（FENTRY 路径的 SIGSEGV）。
   - 复现物料：`make TESTCASE=group_token_old` + `.unsafeasm` 补丁（见 §2.1）+ 命令
     `bin/gfrun -f <elf> -s softcore.multiThreadNum=4`。
3. 后续工具链升级后，建议先跑通 single-thread 回归（绿），再启用 multi_thread 套件；
   届时 `get_thread_idx` 若被正式支持，测试源码可持续保持原样（本次已还原）。

## 6. 附录：本次实验的临时补丁形态（未落盘）

```cpp
#define get_thread_idx get_thread_idx_toolchain
#include <common/pto_tileop.hpp>
#undef get_thread_idx
inline uint32_t get_thread_idx() {
    uint32_t id;                       // + 可选 volatile 全局中转 / noinline
    asm volatile(".unsafeasm\n\tSSRGET 0x802, ->%0" : "=r"(id));
    return id;
}
```