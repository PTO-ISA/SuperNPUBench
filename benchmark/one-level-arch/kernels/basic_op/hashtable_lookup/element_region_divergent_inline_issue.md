# [LinxV5] 同一 PTO element helper 内联进分歧分支触发 sentinel 错误

- 状态：OPEN
- Issue：https://github.com/LinxISA/llvm-project/issues/151
- 证据 gist：https://gist.github.com/lvhao7896/db563aabda2a7b1240303d9c1f5e1202
- 归档日期：2026-11-04
- 编译器：LinxISA/llvm-project `a5f03b38a323ed797c430039bea1b3164ffc6737`（clang 15.0.4，PR #146）
- PTO API：PR #303 `decac4f`

## 现象

一个 `always_inline` helper 内含 `#pragma pto element for`，被内联到 `if/else`
的两个分支后编译失败：

```text
t4.cpp:14:32: error: PTO element region: predication: requires a dominating mandatory region sentinel
fatal error: error in backend: PTO element region compilation failed
```

`-O2` 和 `-O3` 一致。把 region 直接写在两个分支、只从一个分支调用、或拆成两个
不同 helper 都能编译。

## A/B 对照

| 源文件 | 结构 | 结果 |
|---|---|---|
| `t4_divergent_branches.cpp` | 同一 helper 内联进 if/else 两分支 | exit 1，sentinel 错误 |
| `t5_single_branch_control.cpp` | 同一 helper 只在一个分支调用 | exit 0 |
| `t6_two_distinct_helpers_control.cpp` | 两个不同 helper 顺序调用 | exit 0 |
| `t3_sequential_same_helper.cpp` | 同一 helper 顺序调用两次 | exit 1，`Tile legalization: first physical profile requires one predicated IR block`（同 #132） |

## 根因

- `clang/lib/CodeGen/CGStmt.cpp:1082-1090`：每个 pragma 生成一个 distinct token
  和一次 `llvm.linx.experimental.element_region(token)` sentinel call。
- 内联复制 sentinel 和循环，副本共享同一 token。
- `LinxV5ElementRegion.cpp:652 findRegionSentinel()` 返回函数里第一个匹配的
  sentinel。
- `planPredication()`（`:1209`，检查在 `:1223-1227`）要求 sentinel 支配 loop
  header；兄弟分支的副本不满足，报硬错误。

与 #132 同源（ownership 按源 pragma 而不是按内联实例），只是失败点不同。

## 对算子优化的影响

256-lane hashtable probe kernel 想做 probeBatch/finalizeBatch 两阶段（批次 k 的
MGATHER 与批次 k+1 的 vector hash 重叠），在这个编译器上被这条错误阻断，详见
`/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash256_db_20261104/README.md`
（Route A）。在修复前，双缓冲只能等上游解决，或采用"两个独立 helper / 直接
展开"的写法绕开。

## 本地证据

- 复现源文件与日志：`/tmp/sentinel/issue/`（`t4_divergent_branches.cpp`、
  `t5_single_branch_control.cpp`、`t6_two_distinct_helpers_control.cpp`、
  `t3_sequential_same_helper.cpp`、`compile_t4_O2.log`、`compile_t4_O3.log`、
  `compile_t3_related.log`、`reproduce.sh`、`EVIDENCE_digest.md`）
- 远程备份：上面的 gist（secret，含全部文件与 SHA256）
