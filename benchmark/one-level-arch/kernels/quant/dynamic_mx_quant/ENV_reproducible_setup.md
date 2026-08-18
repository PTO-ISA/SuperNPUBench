# 可执行环境重建说明（dynamic_mx_quant / e4m3 复现基线）

> 目的：按本说明可从零重建「当前能跑通 `gfrun` 的可执行环境」。所有仓库**一律以 commit id 为准**，
> 分支名仅供参考——其中 `feat/pto-v058-adaptation` 会被反复 force-push 重写（见文末告警），
> 只认分支名会对不上历史。
>
> 基线快照时间：2026-08-17。

## 0. 顶层布局

四个仓库平级放在同一工作根目录下：

```
<root>/
├── SuperNPUBench/          # kernel + 探针（编译端）
├── SuperScalarModel/       # gfrun / gfsim（执行端）
├── linx-toolchain-build/   # 工具链构建编排（其 src/ 下再拉 5 个组件）
└── SuperNPU/          # 工作区文档（本仓）
```

## 1. 组件仓库与 commit id（权威锚点）

### 1.1 三个顶层仓库

| 仓库 | 远程 URL | 分支 | commit id |
|---|---|---|---|
| SuperScalarModel（执行端） | `github.com/LinxISA/SuperScalarModel.git`（origin）<br>`github.com/ziyang-cheng/SuperScalarModel.git`（personal，备份） | `local_test` | `0e213a2c1341668fb250b7a56f1892dd4a1858c0` |
| SuperNPUBench（编译端，含探针） | `github.com/ziyang-cheng/SuperNPUBench.git`（origin，fork）<br>`github.com/PTO-ISA/SuperNPUBench.git`（upstream） | `feat/dmxq-rel0812` | `71203cd3861512fbb355a64941370a793667fd6a` |
| linx-toolchain-build（工具链编排） | `github.com/LinxISA/linx-toolchain-build.git` | `main` | `e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156` |

### 1.2 工具链 5 个组件（位于 `linx-toolchain-build/src/`）

| 组件 | 远程 URL | 分支 | commit id |
|---|---|---|---|
| llvm-project | `github.com/LinxISA/llvm-project.git` | `temp/shared-tload-integration-20260811` | `eb64de8afcbda043aec7e56dae346905dc982039` |
| musl | `github.com/LinxISA/linx-musl.git` | `linx` | `af0dfc2066272563fa5607cb6ae8cf974baaa415` |
| jemalloc | `github.com/LinxISA/jemalloc.git` | `linx` | `4495309cd11cae1a0a2d008c65acd9c410076f06` |
| linux-linxisa | `github.com/LinxISA/linux.git` | `main` | `7c37630555f8a9f25258ff9f9dc4f37c50271a60` |
| Linx-TileOP-API | `github.com/LinxISA/Linx-TileOP-API.git` | `feat/v058-reinterpret-cmpmode-backfill` | `cb47f6d548e0bff357122a4cf97351f0aca27c7e` |

## 2. SuperScalarModel `local_test` 的含义

`local_test @ 0e213a2c` = 上游 `feat/pto-v058-adaptation` 某个 tip 之上叠加两个已提交上游 issue 修复：

- **#253** `fix(emulator): implement bf16->e8m0 (SF8) TCVT conversion`（`0e213a2c`，即当前 HEAD）
- **#254** `fix(emulator): accept zero-instruction reinterpret_tile in logical TEPL`（父提交 `f60ca82f`）

这两个补丁是 e4m3 探针执行流抵达末端 `TCVT fp32→e4m3` 的**前置条件**（否则在 scale pass /
data pass 就被拦截）。缺陷本身与这两补丁无关，只是它们让执行流跑到数据路径末端。

**当前 `bin/gfrun`（2026-08-17 19:34 编出）即由此分支编译，含 #253/#254。** 这是目前唯一能跑通探针的
执行端二进制，已推送到 personal fork 妥善保存。

## 3. 重建步骤

### 3.1 拉取源码（按 commit id 固定）

```bash
# 顶层三仓
git clone https://github.com/LinxISA/SuperScalarModel.git
git -C SuperScalarModel fetch https://github.com/ziyang-cheng/SuperScalarModel.git
git -C SuperScalarModel checkout 0e213a2c1341668fb250b7a56f1892dd4a1858c0

git clone https://github.com/ziyang-cheng/SuperNPUBench.git
git -C SuperNPUBench checkout 71203cd3861512fbb355a64941370a793667fd6a

git clone https://github.com/LinxISA/linx-toolchain-build.git
git -C linx-toolchain-build checkout e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156

# 工具链 5 组件（linx-toolchain-build 的 make init-src 会拉取到 src/；
# 拉取后逐个 checkout 到下表 commit）
cd linx-toolchain-build && make init-src && cd ..
git -C linx-toolchain-build/src/llvm-project    checkout eb64de8afcbda043aec7e56dae346905dc982039
git -C linx-toolchain-build/src/musl            checkout af0dfc2066272563fa5607cb6ae8cf974baaa415
git -C linx-toolchain-build/src/jemalloc        checkout 4495309cd11cae1a0a2d008c65acd9c410076f06
git -C linx-toolchain-build/src/linux-linxisa   checkout 7c37630555f8a9f25258ff9f9dc4f37c50271a60
git -C linx-toolchain-build/src/Linx-TileOP-API checkout cb47f6d548e0bff357122a4cf97351f0aca27c7e
```

### 3.2 构建工具链

```bash
cd linx-toolchain-build
make WITH_TARGET=linx64v5-linux-musl        # 全量构建，产出 output/linx_blockisa_llvm_musl/
```

产出：`output/linx_blockisa_llvm_musl/bin/clang++` + `sysroot/usr/`。

### 3.3 构建执行端 gfrun / gfsim

```bash
cd SuperScalarModel
python3 build.py all -j8                     # 产出 bin/gfrun + bin/gfsim
```

### 3.4 设置环境变量

```bash
export COMPILER_DIR=<root>/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=<root>/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot/usr
test -x "$COMPILER_DIR/clang++" && test -d "$LINX_SYSROOT" && echo OK
```

### 3.5 编译并运行 e4m3 探针

```bash
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/quant/dynamic_mx_quant
make TESTCASE=dynamic_mx_quant TYPE=PROBE_OCP_FP8_NEWCALC
<root>/SuperScalarModel/bin/gfrun \
    -f output/.../dynamic_mx_quant_probe_ocp_fp8_newcalc.elf \
    --dump-memory 0x15000:256:/tmp/y.bin
```

## 4. 重要告警（否则重建会对不上）

1. **`feat/pto-v058-adaptation` 是易变分支，会被反复 force-push / rebase 重写。**
   实测 2026-08-17 10:37、2026-08-18 09:56 两次强推。基于其某个 tip 建的本地分支会在下次 `fetch`
   （forced-update）后变成 orphan，看似「本地凭空分叉」实为远程被重写。
   **所以务必用上表 commit id 固定，不要只写分支名。** 判断某本地分支「是否真远程」用
   `git reflog origin/<branch>`（能看到 fast-forward vs forced-update），别只看当前 `origin/*` 指向或 `git cherry`。

2. **SuperScalarModel 无个人 fork 上游；`local_test` 是目前唯一能跑通的分支，已备份到
   personal（`github.com/ziyang-cheng/SuperScalarModel.git`）。** 上游一旦重写，从 personal 恢复
   `0e213a2c` 即可，不要指望上游 `feat/pto-v058-adaptation` 还停在原处。

3. **不要复用旧的 `output/` 工具链目录做「权威重建」。** 现存 `output/` 是**混合日期产物**
   （clang 二进制、llvm 源码分支、tileop-api 头文件三者构建时间不一致）。可靠重建应按 §3.2 从固定
   commit **重新构建工具链**，而非直接沿用旧 output。

4. **工具链 ↔ 仿真器可能存在版本 skew**：全新编译的 ELF 可能触发 gfrun/gfsim 崩溃（已知临时性问题）。
   若遇到，优先用与 `local_test @ 0e213a2c` 配套的这套 commit 组合，不要单独升级某一端。
