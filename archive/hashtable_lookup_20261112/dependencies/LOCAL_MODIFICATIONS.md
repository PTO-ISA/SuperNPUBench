# 本地修改与硬件假设清单（NDF §3.7）

每个条目按 §3.7 字段登记：分类、目的、开关/默认值、影响范围、验证证据、副作用、
替代方案、上游去向、移除条件、关联 issue。

| id | 分类 | 目的 | 开关 / 默认值 | 影响范围 | 验证证据 | 副作用 | 替代方案 | 上游去向 | 移除条件 | 关联 issue |
|---|---|---|---|---|---|---|---|---|---|---|
| `M-COMPILER-01` | `correctness_fix` + `performance_mechanism` | 让 256-lane 元素区域（`TPARTELEMENT`）和 `-linxv5-force-element-tile-mask=true` 可用 | 构建时 `-mllvm -linxv5-force-element-tile-mask=true`；patch 无条件生效于该编译器构建 | r1/r2/r3 全部 ELF | `build_kernel.sh` 重建 r1 ELF 与归档 md5 一致；r2 ELF sha256 一致 | 仅影响本地编译器构建；主工具链行为不变 | 无（主工具链只接受 32 元区域） | `none`（未提上游 PR） | 主工具链原生支持宽元素区域与改 flag 后 | — |
| `M-MODEL-01` | `diagnostic_bypass` | 使用可运行这批 ELF 的计时模型 | 环境变量 `MODEL` 指向 `SuperScalarModel-pr1177`，commit `18b24f6ca`（clean） | r1/r2 全部计时与 trace | 逐拍复现历史基线（r1 135,819；r2 rw48 82,256；way7 76,959） | 结论只在该模型/commit 下有效；主线模型不能替代 | 等待主线修复后复跑 | 主线 `SuperScalarModel` 的 PTO #291 / INT64 TLOAD 问题（未归档为独立 issue） | 主线模型能运行全部 ELF 且周期一致后 | `LinxISA/SuperScalarModel#1210`（RWDB 语义，非本绕过本身） |
| `H-RWDB` | `hardware_assumption` | 提高 wave 授权并发以掩盖 300-cycle 随机延迟 | `sl2.RWDB_ENTRY_COUNT=48`；默认 8 | r2 `rw48` / `way7` 结果 | 同 ELF A/B：rw8 169,868 vs rw48 82,256 | 48 是假设配置，不能与默认 8 的成绩混用 | 默认 RWDB=8 时采用 r1（135,819） | `LinxISA/SuperScalarModel#1210` | 硬件/模型默认 RWDB 变更或给出官方配置后 | #1210 |
| `H-WAY` | `hardware_assumption` | 降低 L2 tag 冲突 | `sl2.TAG_WAY_COUNT=7`；默认 4 | r2 `way7` 结果（76,959） | 同 ELF A/B：way4 82,256 vs way7 76,959 | WAY=7 是假设配置，不能与 WAY=4 混用 | 保持 WAY=4，采用 82,256 | 未提 issue | 硬件/模型默认 WAY 变更后 | — |

## 未修改的依赖

- `SuperScalarModel-pr1177`：工作树 clean，无源码改动（`git status` 0 dirty）。
- 数据集、PTO API 快照、参考 ELF：内容固定（哈希见 `ASSETS.md`），未做本地修改。
- 编译器 patch 之外没有未跟踪源文件参与 kernel 构建；8 个 untracked 测试文件见
  `compiler_llvm_pr146/untracked-files.txt`，不进入 ELF。
