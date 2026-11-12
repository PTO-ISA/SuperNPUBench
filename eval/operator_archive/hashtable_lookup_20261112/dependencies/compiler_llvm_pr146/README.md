# 依赖：编译器（llvm-project PR146 本地构建）

| 项 | 值 |
|---|---|
| 仓库 | `git@github.com:LinxISA/llvm-project.git` |
| 基线 commit | `a75987bcc1d9cf5fcd98c1c64c6ca03badbb6919`（2026-11-05 01:41，`LinxV5: specialize element region ownership per loop instance`） |
| 本地状态 | 6 个 tracked 修改（`compiler.patch`，46,475 B，+636/−31）+ 8 个 untracked 测试文件（`untracked-files.txt`，不参与 kernel 构建） |
| 实际工具 | `build-hash/bin/clang++` sha256 `35898d127974570667e06f828dbd3fbffcef6b42bc5acd82d7c0b1686222a2e4`；`ld.lld` sha256 `d57959a1fe1941fa0bf20060398bac74012cf14a5a42bcc262b987f44f8ae0bb` |
| 版本串 | `clang version 15.0.4 (linx64v5-musl-local 3529f346745052582e6e3885e5de43f56a75616b)` |
| 构建配置 | `CMAKE_BUILD_TYPE=Release`、`LLVM_ENABLE_PROJECTS=clang;lld`、`LLVM_TARGETS_TO_BUILD=LinxV5` |
| sysroot | 主 `linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot`（只用其头/库） |

## 为什么需要本地 patch

主 `linx-toolchain-build` 的 clang 不认 `-linxv5-force-element-tile-mask=true`，且宽 lane
元素区域（256-lane `TPARTELEMENT`）需要 PR146/PR303 期 API 与 patch 行为；否则 SoA 宽
lane kernel 无法编译。classification：`correctness_fix` + `performance_mechanism`。

## 恢复方式

```sh
git clone https://github.com/LinxISA/llvm-project.git
git checkout a75987bcc
git apply /path/to/archive/dependencies/compiler_llvm_pr146/compiler.patch
# 另按 untracked-files.txt 补齐 8 个测试文件（仅测试用，可不补）
cmake -G Ninja -S llvm -B build-hash \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD=LinxV5
ninja -C build-hash clang lld
```

## 缺口

- `GAP-LOG-1`：完整构建日志未归档（仅二进制 hash + 选项）。
- patch 与基线 commit 的组合是否能逐字节重建 `clang++` 未核对（`GAP-BIND-2`）。
