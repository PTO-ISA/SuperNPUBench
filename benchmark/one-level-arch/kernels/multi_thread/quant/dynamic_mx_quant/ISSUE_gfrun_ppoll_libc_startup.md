# ISSUE: gfrun 缺 `ppoll` syscall handler → hosted musl 启动即 `abort()`(res_check ELF 全崩)

- **归属仓库**: SuperScalarModel(gfrun / emulator）
- **状态**: 已定位根因 + 已验证修复（本 issue 附最小 patch，补上后端到端逐字节 pass）
- **严重度**: 阻塞级 —— 所有 hosted-musl ELF（含全部 res_check 精度 harness）启动即崩，无法跑到计算路径

## 组件清单（暴露此问题的版本组合）

| 组件 | 版本 | 备注 |
|---|---|---|
| SuperScalarModel（gfrun） | 分支 `codex/consolidate-post-main-fixes-20260903`，tip `49547742` | 缺 ppoll handler |
| 工具链 llvm-project | `dev-llvm15_56` @ `1ae4ee39` | clang 15.0.4 |
| 工具链 Linx-TileOP-API | `linx` @ `804eb03` | — |
| 工具链 musl | `af0dfc20`（分支头重编） | **新 musl 才引入启动期 ppoll** |
| Bench 算子代码 | PR: https://github.com/PTO-ISA/SuperNPUBench/pull/111 （fork 分支 `ziyang-cheng/SuperNPUBench:dmxq-ops-20260904` @ `0ad545e`） | 触发用 kernel/驱动/金标脚本 |

## 一、现象

任意用当前工具链编出的 **hosted-musl ELF**（statically-linked，`HasHostedRuntime()==true`），在 gfrun 上启动即 `abort()`：

```
At .../emulator/SysCall.h 29e EcallAgent:
Bad Syscall Request: syscall(49, 808d420, 3, 808d410, 0, 8, 0);
```

- 打印为 hex：`49` = 十进制 **73** = `ppoll`。
- 发生在 **`__init_libc` / `__libc_start_main`**（block B66，程序最开头），远早于任何计算或 res_check I/O。

## 二、复现步骤

```bash
# 0) 环境（用上表工具链）
export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot/usr

# 1) 取 Bench 算子代码（PR 分支）
#    cd SuperNPUBench/benchmark/one-level-arch
cd test/kernel/multi_thread/quant/dynamic_mx_quant

# 2) 生成金标输入（tail / OCP / FP8 / fp16 in / compact scale）
python3 src/gen_dynamic_mx_quant_data.py \
    --M 512 --K 256 --block-size 32 --algo OCP --kernel tail --dtype FP8 \
    --in-dtype fp16 --scale-layout compact --seed 42 \
    -o ../../../../compare/dynamic_mx_quant_tail_ocp_fp8

# 3) 编 res_check ELF（4-PE SPMD kernel）
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP8 res_check=on diss

# 4) 4 线程跑 gfrun —— 未打补丁时在此 abort（Bad Syscall 73）
bin/gfrun -f .../dynamic_mx_quant_tail_ocp_fp8.elf -s softcore.multiThreadNum=4 -t 1
```

> 说明：老流程曾用 `GFRUN_FORCE_DIRECTBOOT_ABI=1` 绕开——那是让 ELF 走 **direct-boot**、**整段跳过 musl libc init**，因而躲过了 ppoll。新 model 把该 flag 在 `EcallAgent` 里 `(void)` 忽略（始终 X1-then-A7 选号），且 ELF 按 hosted 加载 → `__init_libc` 必被执行 → 必发 ppoll。故该 env **不是**成因，也不再是可行绕过（res_check 需要 hosted 的 openat/read/write 落盘）。

## 三、根因

新 musl（`src/env/__libc_start_main.c:45-50`）在启动时对 fd 0/1/2 发 ppoll 探测 stdio 是否打开：

```c
struct pollfd pfd[3] = { {.fd=0}, {.fd=1}, {.fd=2} };
#ifdef SYS_poll
    __syscall(SYS_poll, pfd, 3, 0);
#else
    __syscall(SYS_ppoll, pfd, 3, &(struct timespec){0}, 0, _NSIG/8);
#endif
// 之后：for i in 0..2: if (pfd[i].revents & POLLNVAL) 把 /dev/null open 到该 fd
```

LinxISA 无 `SYS_poll` → 走 `#else` 分支发 `ppoll`（号 73）。而 gfrun 的 `HandlerTable`（`emulator/SysCall.h:1254`）白名单里**没有 ppoll**（仅 read/write/readv/writev/openat/... 等 26 个）。命中未注册号 → `EcallAgent` 构造函数走到 `abort()`（`SysCall.h:668`），整个仿真终止。

- 老 musl 无此启动 poll，故历史上未暴露；本问题随 musl 升级到 `af0dfc20` 才出现。
- 三个既有 backup model 分支均无 ppoll handler → 非回归，是新增覆盖缺口。

## 四、建议修复（已验证）

补一个最小 `do_ppoll`：stdio 0/1/2 在仿真里恒有效，清各 pollfd 的 `revents`（struct 偏移 6 的 short）为 0、返回 0（timeout、无就绪 fd）。musl 检 `revents & POLLNVAL == 0` → 判定全部有效 → 不再 open /dev/null，正常继续。纯增量，不触碰任何既有 handler / pass-list。

```diff
--- a/emulator/SysCall.h
+++ b/emulator/SysCall.h
@@ HANDLER(mprotect) ... 之后新增：
+    HANDLER(ppoll)
+    {
+        // musl __libc_start_main polls fds 0/1/2 at startup to detect closed
+        // descriptors (checks revents & POLLNVAL, then re-opens /dev/null onto
+        // any invalid one). In the hosted simulator stdio is always valid, so
+        // report no invalid/ready fds: clear each pollfd's revents (short at
+        // struct offset 6) and return 0 (timeout, nothing ready).
+        //   struct pollfd { int fd; short events; short revents; }  // 8 bytes
+        const uint64_t pfdAddr = arguments[0];
+        const uint64_t nfds    = arguments[1];
+        for (uint64_t i = 0; i < nfds; ++i) {
+            aaccelssor.Store(pfdAddr + i * 8 + 6, static_cast<uint16_t>(0));
+        }
+        returnValue = 0;
+    }
@@ HandlerTable 注册表：
         REGISTER_HANDLER(openat),
+        REGISTER_HANDLER(ppoll),
         REGISTER_HANDLER(lseek),
```

## 五、验证

补丁重编 gfrun 后，`dynamic_mx_quant_tail_ocp_fp8`（512×256，4-PE res_check）端到端跑通并 `exit_group` 正常收尾，逐字节对金标：

```
dynamic_mx_quant_tail_ocp_fp8: output=pass (MSE=0.000000, MaxAE=0.011719), scale=pass (MSE=0.000000, MaxAE=0.000000)
```
（output MaxAE=0.0117 为 fp8 e4m3 LSB 舍入，MSE=0；scale 逐字节精确。）

## 六、影响范围

- **所有 hosted-musl ELF** 启动即崩，包括但不限于 dynamic_mx_quant 全部 res_check 精度 harness。非 dmxq 专属。
- kernel 计算逻辑不受影响（问题在 libc 启动路径，未到计算）。
- 补 ppoll 是全 res_check 精度流程的共用前置，一处修复解锁全部。
