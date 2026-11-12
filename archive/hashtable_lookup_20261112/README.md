# hashtable_lookup：探测循环与宽 lane 优化归档

- **task_id / 负责人 / 更新日期**：`hashlk_20261112_lvhao_01` / lvhao / 2026-11-12
- **归档规范**：NDF work-archive draft.8（SuperScalarModel PR #1238，
  `docs/NDF_WORK_ARCHIVE.md`，分支 `archive/workshop-20261109`）
- **稳定要求登记表 / 覆盖映射**：`not_produced`（本归档为 draft，尚未按附录 J.4 生成 111 项
  逐条覆盖表；缺口见“缺项与风险”）
- **目标与问题**：在 random SoC（300-cycle 均值延迟）下优化 `basic_op/hashtable_lookup`
  的线性探测循环，回答“加宽投机窗口 / 加宽 lane / 加深 RWDB 各自是否有效”，并把 per-SoC
  配置下的最优实现与算子默认实现一起归档。
- **当前结论**：
  - 默认模型参数（RWDB=8、TAG_WAY=4）下，kernel 侧最优是 **64-lane 自适应 2+2 = 135,819 cycles**；
  - 把 RWDB 加深到 48（硬件假设）后，**256-lane 自适应 2+6（SoA）= 82,256 cycles**，
    再加 `TAG_WAY=7` = **76,959 cycles**，是所有实测里最好的；
  - 算子默认的 32-lane AoS 自适应 2+6 能编译，但 gfsim 在 64 位 key gather 上触发既有断言，
    **没有算子级计时**（见 r3）。
- **算法定义/前提**：`results/r2_256l_2p6_rwdb48/algorithm.md` 与
  `analysis/RWDB_BOTTLENECK.md`；核心前提是表尾 guard 可读、count 为 Lanes 整数倍、
  RWDB/TAG_WAY 按归档配置固定。
- **本地修改摘要**：
  - `M-COMPILER-01`（compiler，correctness/feature fix）：`LinxISA/llvm-project` @
    `a75987bcc` + 本地 patch（6 个 tracked 文件，`compiler.patch`；另有 8 个编译器测试
    untracked 文件，不参与 kernel 构建）。影响：SoA 宽 lane 元素区域能编译；kernel 侧
    64-lane 2+2 的基线 ELF 用同一编译器重建后字节一致（`942cd153…`）。
  - `M-MODEL-01`（model，none）：计时模型 `SuperScalarModel-pr1177` @ `18b24f6ca`，
    **clean**；但主线 `SuperScalarModel` 跑不了这批 ELF（reference 模拟器拒绝 INT64
    TLOAD / 早期版本 `PTO #291` 断言），所以计时运行使用的是 pr1177 工作树，属于
    “本地绕过/替代模型”选择，必须随结论标注（见 `dependencies/simulator_pr1177/README.md`）。
  - `H-RWDB`（hardware assumption）：`sl2.RWDB_ENTRY_COUNT` 8→48；`H-WAY`：
    `sl2.TAG_WAY_COUNT` 4→7（仅 r2 的 way7 运行）。二者都是硬件配置假设，不与默认
    配置成绩混用。
- **关键依赖 / issue**：`LinxISA/SuperScalarModel#1210`（RWDB 8-entry wave 授权全有或全无）。
  本地修改没有对应的上游 PR（compiler patch 的 PR 关系：`none` / 仅本地 diff）。
- **依赖组成**：compiler=llvm-project@a75987bcc+patch；model=SuperScalarModel@18b24f6ca
  clean；API=PR303 期快照（本地归档，来源 commit 未知，缺项）；dataset=keys/values/batch
  map/参考 ELF（哈希见 `dependencies/dataset`）。
- **外部资产交付状态**：
  - 已可获取：本归档内的日志、配置、脚本、hash、compiler patch、API 快照文件清单；
  - 仅本地：运行 ELF（r1/r2/r3 的 ELF 位于 `outputs/` 或 `/tmp`，哈希已记录；
    未随本归档提交）、完整 SwimLane/PipeView（位于 `outputs/hash64_batchmap_20261105/...`）；
  - `GAP-EXT-1`：以上大二进制尚未上传到公共 Release，读者目前只能在作者机器路径获取。
- **冻结配置**：`config/random300.json`、`config/deterministic.json`；加载脚本
  `build/run_perf.sh`（显式使用归档配置，不读个人配置）。
- **复现入口**：`build/build_kernel.sh` + `build/run_perf.sh`；验收层级见“复现性审计”。
- **分项能力**：本地重放（可，需本地 ELF）/ 共享资产重放（缺 Release）/ 源码恢复
  （kernel 源码已归档；model 与 compiler 源码在外部仓库）/ 模型与算子重建（部分：见缺口）/
  数值验证（r1/r2 有 check 构建 PASS 记录；r3 无运行验证）。

| result_id / revision | 角色与变化项 | 正确性 | 性能及口径 | 依赖/假设 | 证据与复现入口 |
|---|---|---|---|---|---|
| `r1_64l_2p2_135819` | baseline（默认模型参数） | check 构建 PASS（同结构） | 135,819 Total Cycles / random300 | model 18b24f6ca；RWDB=8、WAY=4 | `results/r1_64l_2p2_135819/` |
| `r2_256l_2p6_rwdb48` | adopted（硬件假设：RWDB=48；way7 变体 = 76,959） | check 构建 PASS（同结构） | 82,256（rw48）/ 76,959（rw48+way7） | +H-RWDB、+H-WAY | `results/r2_256l_2p6_rwdb48/` |
| `r3_operator_adaptive_aos` | operator 默认实现（诊断） | smoke/perf/verify 编译通过 | **未计时**（gfsim 断言，见 README） | 32-lane AoS、guard 112 B | `results/r3_operator_adaptive_aos/` |

## 复现性审计（问题 1：能否只按文档、不依赖本地内容构建）

| 组件 | 身份 | 能否从上游文档重建 | 缺口 / 需要的材料 |
|---|---|---|---|
| 计时模型 | `SuperScalarModel-pr1177` @ `18b24f6ca`（clean） | **可以**：该 commit 在上游 `LinxISA/SuperScalarModel`；`python3 build.py`（CMake+ninja），产物 `bin/gfsim` sha256 `12066cbd…` | 需要固定 CMake 版本/主机说明；`build/` 目录的完整 cache 未归档 |
| 编译器 | `LinxISA/llvm-project` @ `a75987bcc` + `compiler.patch`（46,475 B，6 文件） | **有条件可以**：上游 commit + 归档 patch；clang 构建选项：Release、`clang;lld`、`LLVM_TARGETS_TO_BUILD=LinxV5` | patch 之外的 8 个 untracked 测试文件未参与构建（列于 `untracked-files.txt`）；完整 LLVM 构建日志未归档 |
| PTO API 快照 | `tileop-api/`（169 文件，`MANIFEST.sha256`） | **不可以**：本地快照，来源 commit/PR 未确认 | **GAP-API-1**：需确认上游 PR/commit，或把快照发布为固定依赖包 |
| 数据集 | keys32/values/batch map/参考 ELF | **部分**：keys/values 有生成脚本与哈希；参考 ELF (`9610ba16…`) 是本地资产 | **GAP-DATA-1**：参考 ELF 未上库；生成脚本依赖它 |
| 构建产物 | 9 个矩阵 ELF + soa_wide fixture + operator ELF | 可用 `build/build_kernel.sh` 重建；已核对 md5 与历史一致（见各 result manifest） | **GAP-EXT-1**：ELF 未随归档提交，仅记录哈希与本地路径 |

结论：**目前还不能做到“完全按文档、零本地内容”复现**。最小闭合路径是：
（a）把 API 快照确认到上游 commit 或发布为依赖包；（b）把参考 ELF 与算子 ELF 发布到
Release（或纳入本归档）；（c）补 LLVM/CMake 构建日志；（d）补附录 J.4 的逐条覆盖表。
本归档已把能固定的身份/哈希/命令先固定下来（见 `dependencies/` 与 `results/*/manifest.json`）。

## 实现环境（问题 2：详细环境说明）

| 项 | 值 |
|---|---|
| 主机 | macOS darwin 25.5.0 arm64 |
| 计时模型 | `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-pr1177/bin/gfsim`，head `18b24f6ca`，sha256 `12066cbdc29466d95cce1229129ff5fa03edbdd22880fbad57d55caec87633d8` |
| 编译器 | `/Users/blacktraker/Programming/gitproj/DV4/llvm-project-pr146/build-hash/bin/clang++`，`clang version 15.0.4 (linx64v5-musl-local 3529f346…)`，sha256 `35898d127974570667e06f828dbd3fbffcef6b42bc5acd82d7c0b1686222a2e4` |
| 链接器 | 同目录 `ld.lld`，sha256 `d57959a1fe1941fa0bf20060398bac74012cf14a5a42bcc262b987f44f8ae0bb` |
| sysroot | `/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot`（仅头文件/库，不参与 clang 生成） |
| PTO API | `outputs/hash256_soa_20261105/optimization/dependencies/tileop-api`（169 文件） |
| 数据集 | `outputs/hash256_batchmap_20261105/data/{keys32,values,batch_map}.bin`；参考 ELF `eval/perf_data/hashtable_lookup/hash_6144_PE4.elf` |
| 运行配置 | `config/random300.json` / `config/deterministic.json` |
| 运行 cwd | `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-pr1177` |
| 计时口径 | gfsim 收尾打印的 `Total Cycles`；不含宿主机时间；单次确定性仿真 |

## 归档内复现入口自检（2026-11-12）

- `./build/build_kernel.sh matrix 64 2_2 /tmp/x` → `perf.elf` md5
  `f5d5a0cb935efe851504a17559fe1ba1`，与归档 ELF 一致；
- `./build/build_kernel.sh soa_operator /tmp/y` → `perf.elf` sha256
  `35218ed0…f1e2f`，与 r2 实测 ELF 一致；
- `./build/run_perf.sh /tmp/x/perf.elf random300` → `Total Cycles 135819`。

即：**只要具备本地依赖（compiler patch + API 快照 + 数据/参考 ELF）**，本归档的构建与
运行入口可以逐字节重建 ELF 并复现周期数；仍缺的是把这些依赖发布成公共可获取材料
（见上一节）。

## 缺项与风险

- `GAP-COV-1`：未生成 NDF 附录 J.4 的 111 项 requirement 覆盖表（本归档为 draft）。
- `GAP-API-1`：API 快照来源未确认。
- `GAP-DATA-1`：参考 ELF（`960ba161…`）未上库，数据生成链不闭合。
- `GAP-EXT-1`：运行 ELF / SwimLane / PipeView 未发布到公共存储。
- `GAP-LOG-1`：LLVM 与 gfsim 的完整构建日志未归档（仅记录二进制 hash 与构建入口）。
- `GAP-R3-1`：算子默认实现没有算子级计时——gfsim 在 64 位 key gather 上断言
  `responseReturnBeatId == geo.mrobEntryIdx`（2026-11-12 复现，pr1177 clean），
  因此 r3 只能声明“编译通过”，不能给性能结论。
- `GAP-MODEL-1`：计时用了 pr1177 工作树而非 AGENTS 默认的 `SuperScalarModel`；
  主线当前（`fb7db9c74`）在 reference 模拟器报
  `illegal TLOAD operand or descriptor contract`（dtype=24 INT64），无法产出周期。
