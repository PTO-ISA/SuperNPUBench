# 依赖：计时模型（SuperScalarModel-pr1177）

| 项 | 值 |
|---|---|
| 仓库 | `git@github.com:LinxISA/SuperScalarModel.git` |
| commit | `18b24f6ca7f71369aee9245426daa68cc741dad5`（2026-11-03，clean，0 dirty） |
| 二进制 | `bin/gfsim` sha256 `12066cbdc29466d95cce1229129ff5fa03edbdd22880fbad57d55caec87633d8` |
| 构建入口 | `python3 build.py`（CMake + Ninja，`build/CMakeCache.txt`：`CMAKE_CXX_COMPILER=/usr/bin/clang++`） |
| 运行 cwd | 该工作树根目录 |

## 为什么不用 AGENTS 默认的主线

主线 `SuperScalarModel` 当前 `fb7db9c74`（dirty）无法跑这批 ELF：

- 早期版本：reference core 在 `Pto291::ValidateCubeElementwise` 断言
  `PTO #291 CUBE elementwise operands must share one M carrier layout`；
- 当前版本（2026-11-12 复测）：`gfrun: illegal instruction at 0x103bc: illegal TLOAD
  operand or descriptor contract`（`dtype=24` = INT64）。

因此计时使用 pr1177；它**逐拍复现**历史基线（64-lane 2+2 = 135,819、确定性 91,364、
256-lane 确定性 64,399），并且归档 `check.log` 的模型也是它（head `18b24f6ca`）。
classification：`diagnostic_bypass`（换用可运行模型），影响所有性能结论的口径。

## 恢复方式

```sh
git clone https://github.com/LinxISA/SuperScalarModel.git
git checkout 18b24f6ca
python3 build.py
# 核对 bin/gfsim sha256 = 12066cbd...
```
