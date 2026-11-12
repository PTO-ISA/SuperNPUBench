# hashtable_lookup：探测循环与宽 lane 优化归档

- **task_id / 负责人 / 更新日期**：`hashlk_20261112_lvhao_01` / lvhao / 2026-11-12
- **归档规范**：NDF work-archive draft.8（SuperScalarModel PR #1238，
  `docs/NDF_WORK_ARCHIVE.md`，分支 `archive/workshop-20261109`）
- **稳定要求登记表 / 覆盖映射**：`requirements_coverage.md`（111 项逐条状态；revision 4 自评 pass 55 / partial 38 /
  not_checked 16 / not_applicable 2，未经独立复核）；逐条记录见 `validation/check_records.md`
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
- **本地修改摘要**：逐项字段（分类/开关/影响/验证/副作用/替代/上游/移除条件）见
  `dependencies/LOCAL_MODIFICATIONS.md`。
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
- **外部资产交付状态**：`ASSETS.md` 列 11 个资产，已发布到
  Release `hashlk-20261112-r3` 并下载验证（11/11 SHA256 匹配；见
  `validation/material_check.md`）；API 快照来源仍未确认（`GAP-API-1`）。
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
| 数据集 | keys32/values/batch map/参考 ELF | **可以**：keys/values 有生成脚本与哈希；参考 ELF 已发布（`A-REF-ELF`） | `GAP-DATA-1` 剩余：参考 ELF 的原始生成信息未恢复 |
| 构建产物 | 9 个矩阵 ELF + soa_wide fixture + operator ELF | 可用 `build/build_kernel.sh` 重建；已核对 md5/sha256 与历史一致 | ELF 已发布（`A-R1/R2/R3-ELF`），可下载验证 |

结论：**资产侧已闭合**——参考 ELF、r1/r2/r3 ELF、trace、PipeView、API 快照均已发布
并可下载验证（`ASSETS.md`）。仍未闭合的是：API 快照的上游 commit（`GAP-API-1`）、
LLVM/gfsim 的完整构建日志（`GAP-LOG-1`），以及独立复核（`run_reproduced` /
`build_reproduced`）。

## 实现环境（问题 2：详细环境说明）

| 项 | 值 |
|---|---|
| 主机 | macOS darwin 25.5.0 arm64 |
| 计时模型 | `<GFSIM_WORKTREE>/bin/gfsim`，head `18b24f6ca`，sha256 `12066cbdc29466d95cce1229129ff5fa03edbdd22880fbad57d55caec87633d8` |
| 编译器 | `<LLVM_BUILD>/bin/clang++`，`clang version 15.0.4 (linx64v5-musl-local 3529f346…)`，sha256 `35898d127974570667e06f828dbd3fbffcef6b42bc5acd82d7c0b1686222a2e4` |
| 链接器 | 同目录 `ld.lld`，sha256 `d57959a1fe1941fa0bf20060398bac74012cf14a5a42bcc262b987f44f8ae0bb` |
| sysroot | `<SYSROOT>`（仅头文件/库，不参与 clang 生成） |
| PTO API | `outputs/hash256_soa_20261105/optimization/dependencies/tileop-api`（169 文件） |
| 数据集 | `outputs/hash256_batchmap_20261105/data/{keys32,values,batch_map}.bin`；参考 ELF `eval/perf_data/hashtable_lookup/hash_6144_PE4.elf` |
| 运行配置 | `config/random300.json` / `config/deterministic.json` |
| 运行 cwd | `<GFSIM_WORKTREE>` |
| 计时口径 | gfsim 收尾打印的 `Total Cycles`；不含宿主机时间；单次确定性仿真 |

> revision 3 起本表用 `<GFSIM_WORKTREE>` / `<LLVM_BUILD>` / `<SYSROOT>` 占位；真实个人路径只保留在
> revision 1/2 的 git 历史中作为构建过程记录，不在发布材料中重复。路径与二进制身份（sha256）不变。

## 归档内复现入口自检（2026-11-12）

- `./build/build_kernel.sh matrix 64 2_2 <OUTDIR>` → `perf.elf` md5
  `f5d5a0cb935efe851504a17559fe1ba1`，与归档 ELF 一致；
- `./build/build_kernel.sh soa_operator <OUTDIR>` → `perf.elf` sha256
  `35218ed0…f1e2f`，与 r2 实测 ELF 一致；
- `./build/run_perf.sh <OUTDIR>/perf.elf random300` → `Total Cycles 135819`。

即：**只要具备本地依赖（compiler patch + API 快照 + 数据/参考 ELF）**，本归档的构建与
运行入口可以逐字节重建 ELF 并复现周期数；仍缺的是把这些依赖发布成公共可获取材料
（见上一节）。

## 缺项与风险

- `GAP-COV-1`（已关闭）：111 项覆盖映射见 `requirements_coverage.md`；状态为自评，
  仍需逐条 material check 才能关闭 WORK-ARCHIVE-010。
- `GAP-API-1`：API 快照来源未确认。
- `GAP-DATA-1`（部分关闭）：参考 ELF（`960ba161…`）已发布；其原始生成信息仍未恢复。
- `GAP-EXT-1`（已关闭）：运行 ELF / trace / PipeView 已发布到 Release 并下载验证。
- `GAP-LOG-1`：LLVM 与 gfsim 的完整构建日志未归档（仅记录二进制 hash 与构建入口）。
- `GAP-R3-1`：算子默认实现没有算子级计时——gfsim 在 64 位 key gather 上断言
  `responseReturnBeatId == geo.mrobEntryIdx`（2026-11-12 复现，pr1177 clean），
  因此 r3 只能声明“编译通过”，不能给性能结论。
- `GAP-MODEL-1`：计时用了 pr1177 工作树而非 AGENTS 默认的 `SuperScalarModel`；
  主线当前（`fb7db9c74`）在 reference 模拟器报
  `illegal TLOAD operand or descriptor contract`（dtype=24 INT64），无法产出周期。

## revision 记录

- **revision 2（2026-11-12）**：补齐 NDF draft.8 审计缺项中的可执行部分——
  - 新增 `ASSETS.md`（§3.7.4 字段表）与 `FILE_MANIFEST.json`（逐文件字节数 + SHA256）；
  - 新增 `requirements_coverage.md`（111 项覆盖映射，自评）；
  - r1/r2 各补完整 SwimLane + nocounters + 运行日志（`A-R1/R2-TRACE`，本机
    `outputs/hashtable_lookup_archive_assets/`），重跑周期数与原始 `run.log` 一致
    （135,819 / 82,256）；
  - r1/r2 README 补周期起止点、计入范围与指标分母声明；r2 `algorithm.md` 补完整
    假设/正确性/性能条件，并说明 81,317（早期 sweep）与 82,256（本归档）的来源差异；
  - `COMPLIANCE.md` 按上述证据更新（WORK-ARCHIVE-006 由 ❌ 改为部分）。
  - 仍未关闭：独立复核、负控、r2 PipeView、资产发布（`GAP-EXT-1`）。
- **revision 1（2026-11-12）**：初版（结果索引、manifest、`perf.diss`、合规审计）。

### revision 3（2026-11-12，NDF 补齐）

- 路径整改：脚本/配置/README 去个人绝对路径；新增 `build/scan_paths.py`，
  52 文件扫描个人路径 0 命中（`validation/PATH_SCAN.md`）。
- trace：补 r2 rw8 / way7 完整 SwimLane + nocounters（周期与日志一致），
  r1/r2 PipeView 原始数据（周期不变），宿主资源实测。
- 负控：r1 check ELF 翻转 1 个 golden 字节 → `val=0x3333 fail`、rc=70
  （`validation/negative_control/`）。
- 重建：参数化 `build_kernel.sh` 重建 r1/r2，ELF 与归档字节一致
  （`validation/build_repro_r3.log`）。
- 文档：`dependencies/LOCAL_MODIFICATIONS.md`、`validation/check_records.md`
  （111 项）、`requirements_coverage.md` 更新、diss 头规范化记录。
- 仍未关闭：Release 发布与下载侧 material、独立复核、`GAP-R2-CHECK`、`GAP-LOG-1`。

### revision 4（2026-11-12，发布与 material_checked）

- 发布 Release `hashlk-20261112-r3`（固定 commit `4359f297`）：4 个 ELF、
  4 组 trace、PipeView、API 快照、归档 tarball 与 `assets.tsv`。
- 从 Release 重新下载：11/11 SHA256 匹配；归档内部 56 行清单自校验通过；
  下载与解压 payload 个人路径 0 命中（`validation/material_check.md`）。
- tar 包改用中性 owner（`ndf`），旧包作废；`ASSETS.md` 更新 URL 与新 hash。
- 覆盖统计更新为 pass 55 / partial 38 / n/a 2 / not_checked 16。
- 仍未关闭：独立 `run_reproduced`、独立 `build_reproduced`、`GAP-R2-CHECK`、
  `GAP-LOG-1`、API 上游来源。

### revision 5（2026-11-12，归档路径迁移）

- 归档目录从 `archive/hashtable_lookup_20261112/` 迁移到
  `eval/operator_archive/hashtable_lookup_20261112/`，符合团队 operator archive 布局。
- 归档内部相对引用（`../dependencies`、`results/*` 等）不随迁移改变。
- 新增 `A-ARCHIVE-R5` tarball；`A-ARCHIVE-R3` 作为 revision 3 的历史快照保留，
  不覆盖已发布资产。文件清单/哈希已重算。
