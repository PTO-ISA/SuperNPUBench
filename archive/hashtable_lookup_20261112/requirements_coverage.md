# NDF 要求覆盖映射（附录 J.4 风格）

- 规范：NDF work-archive draft.8（`docs/NDF_WORK_ARCHIVE.md`，PR #1238）
- 登记表：`requirements_draft8.json`（111 项）；本表逐项给出适用性状态与证据入口。
- 归档 revision：3（2026-11-12）；**自评，未经独立复核**。
- 状态定义：`pass` = 归档内有可定位证据；`partial` = 部分满足，缺口见 note；
  `not_checked` = 未执行/未验证；`not_applicable` = 本任务不适用（已给理由）。
- 统计：pass 51 / partial 42 / not_checked 16 / not_applicable 2（共 111 项）。

| requirement_id | locator | subject | status | check_id | 证据 / 缺口 |
|---|---|---|---|---|---|
| WORK-ARCHIVE-001-R001 | §3.4 | task_id 的唯一性与任务范围 | pass | CHK-001-R001 | README/manifest.json；身份/目录/索引已建；见 README 与 manifest。 |
| WORK-ARCHIVE-001-R002 | §3.4 | result_id 绑定固定产物与执行依赖组合 | pass | CHK-001-R002 | README/manifest.json；身份/目录/索引已建；见 README 与 manifest。 |
| WORK-ARCHIVE-001-R003 | §3.4 | 每次执行具有独立 run_id | pass | CHK-001-R003 | README/manifest.json；身份/目录/索引已建；见 README 与 manifest。 |
| WORK-ARCHIVE-001-R004 | §3.4 | 负责人/代号可解析，逻辑目录及结果可定位 | pass | CHK-001-R004 | README/manifest.json；身份/目录/索引已建；见 README 与 manifest。 |
| WORK-ARCHIVE-001-R005 | §3.4 | 公共依赖有不可变 dependency_id、hash与获取位置 | pass | CHK-001-R005 | 四个依赖的 manifest.json 已建，含 dependency_id/来源/哈希/恢复；API 上游 commit 仍未知（GAP-API-1） |
| WORK-ARCHIVE-001-R006 | §1、§8 | 规范承载仓库/批准身份及本结果采用版本固定，草案不冒充已批准 | pass | CHK-001-R006 | README/manifest.json；身份/目录/索引已建；见 README 与 manifest。 |
| WORK-ARCHIVE-002-R001 | §3.3、§3.5 | T/F/P/S 选择符合实际声明且未规避条件必须项 | partial | CHK-002-R001 | README 表格；内容矩阵逐项见 README；缺项以 GAP-* 记录。 |
| WORK-ARCHIVE-002-R002 | §3.5 | 目标、负责人、日期与结果索引 | pass | CHK-002-R002 | 目标/负责人/日期/结果索引见根 README 与 results 表 |
| WORK-ARCHIVE-002-R003 | §3.5 | shape/dtype/layout/tiling/核数/工作量及不适用理由 | pass | CHK-002-R003 | shape/dtype/layout/tiling/核数/工作量在 result README 与 manifest 中 |
| WORK-ARCHIVE-002-R004 | §3.5 | 全部适用内容项的存在性与要求等级；逐项拆实例 | partial | CHK-002-R004 | README 表格；内容矩阵逐项见 README；缺项以 GAP-* 记录。 |
| WORK-ARCHIVE-002-R005 | §3.5、§5 | 冻结文件逐文件路径、字节数与SHA256清单，排除清单自身 | pass | CHK-002-R005 | FILE_MANIFEST.json 40 条目含字节数+SHA256，清单自身排除；sha256sum -c 41/41 |
| WORK-ARCHIVE-003-R001 | §3.6 | 算子源码仓库、完整commit和子模块身份 | pass | CHK-003-R001 | 仓库/完整 commit/无子模块；源码快照在 outputs 与 main 7d3503bc |
| WORK-ARCHIVE-003-R002 | §3.6 | dirty修改及实际使用源码/头文件身份 | pass | CHK-003-R002 | compiler patch（6 tracked）+ 8 untracked 测试文件清单；kernel 侧无未跟踪源文件 |
| WORK-ARCHIVE-003-R003 | §3.6 | 实际构建入口、cwd、选项、宏、架构及环境 | pass | CHK-003-R003 | 构建入口/参数/宏/架构/环境在 build 脚本与 README；脚本已参数化、无个人默认路径 |
| WORK-ARCHIVE-003-R004 | §3.6 | 实际compiler/linker/objdump身份与构建日志 | partial | CHK-003-R004 | kernel 重建日志已归档；LLVM 与 gfsim 的完整构建日志仍缺（GAP-LOG-1） |
| WORK-ARCHIVE-003-R005 | §3.6 | 最终ELF与匹配diss身份及两者对应关系 | pass | CHK-003-R005 | ELF sha256 与 perf.diss 同目录；diss 仅规范化文件头并记录前后 hash |
| WORK-ARCHIVE-003-R006 | §3.6 | 运行ELF与冻结源码/构建产物的对应依据，未知显式登记 | pass | CHK-003-R006 | 运行 ELF 与冻结构建的对应由重建字节一致证明 |
| WORK-ARCHIVE-003-R007 | §3.6 | 声称重建字节一致时的实际重建与比较证据 | pass | CHK-003-R007 | 作者本地重建 r1/r2 与归档 ELF 字节一致（非独立复核） |
| WORK-ARCHIVE-004-R001 | §3.7 | model/compiler/API来源、完整revision或已知身份与缺项 | partial | CHK-004-R001 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R002 | §3.7、§3.7.3 | patch顺序、必要untracked/生成文件/子模块及历史身份 | partial | CHK-004-R002 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R003 | §3.7 | 实际运行/构建工具二进制hash | partial | CHK-004-R003 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R004 | §3.7 | ISA/ASL固定版本及合法性、API表达、lowering、功能/时序支持分别陈述 | partial | CHK-004-R004 | ISA 目标 linx64v5、gfsim 时序模型、ASL 不适用已分别陈述；功能模型当前不可用 |
| WORK-ARCHIVE-004-R005 | §3.7.1 | 首页/清单/详情中的本地绕过启用清单；没有则none | pass | CHK-004-R005 | 本地修改/绕过/硬件假设逐项登记，含分类与移除条件 |
| WORK-ARCHIVE-004-R006 | §3.7.1 | 每项绕过的问题、来源、改前/后行为和产物身份 | pass | CHK-004-R006 | 每项含目的、改前/后行为、影响范围与产物身份 |
| WORK-ARCHIVE-004-R007 | §3.7.1 | 每项绕过的应用/撤销、验证及性能/合同边界 | pass | CHK-004-R007 | 含应用/撤销方式、验证证据、副作用与性能边界 |
| WORK-ARCHIVE-004-R008 | §3.7.2 | 可加载冻结配置入口与完整传递依赖 | partial | CHK-004-R008 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R009 | §3.7.2 | CLI/环境覆盖、加载顺序、cwd，区分未设置/空值/显式值 | partial | CHK-004-R009 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R010 | §3.7.2 | 最终有效配置/默认值快照或缺项与历史恢复依据 | partial | CHK-004-R010 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R011 | §3.7.2 | 实际加载归档配置的身份与运行证据 | partial | CHK-004-R011 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R012 | §3.7.3 | 基线/继承修复/本地增量逐项问题与源码/产物组成 | partial | CHK-004-R012 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R013 | §3.7.3 | issue/PR及合入状态快照，归档引用未冒充修复完成 | partial | CHK-004-R013 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R014 | §3.7.3 | patch可应用、tracked恢复、完整恢复、构建和运行范围分别记录 | partial | CHK-004-R014 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R015 | §3.7.4 | 跨仓库资产归属及源码commit/元数据commit/二进制身份分别固定 | partial | CHK-004-R015 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R016 | §3.7.4 | 每个外部资产用途、必需/可选、内容/排除项、格式与大小/hash | pass | CHK-004-R016 | 8 个外部资产含用途、必需/可选、字节、SHA256、格式 |
| WORK-ARCHIVE-004-R017 | §3.7.4 | 每个外部资产固定获取方式、权限、责任人及核对日期 | partial | CHK-004-R017 | 获取方式/责任人/日期已写；Release URL 与 download_verified 待发布后补 |
| WORK-ARCHIVE-004-R018 | §3.7.4 | planned/prepared/uploaded/download_verified与实际动作相符 | partial | CHK-004-R018 | prepared/uploaded/download_verified 状态已区分；当前仅 patch 在 git 内 |
| WORK-ARCHIVE-004-R019 | §3.7.4 | runtime必需依赖与历史可选trace分开，工具链按操作提供 | partial | CHK-004-R019 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R020 | §3.7.4 | Release/README/PR/索引的内容、身份、能力和限制一致 | partial | CHK-004-R020 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R021 | §3.7.4 | 跨仓库result/ELF/diss/runtime/config身份双向核对，无未来commit伪填 | partial | CHK-004-R021 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R022 | §3.7.4 | 组合附件的适用基包、必需关系、顺序、替换文件与最终身份 | partial | CHK-004-R022 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R023 | §3.7.4 | 当前组合与历史索引范围、受影响版本的命令勘误可定位 | partial | CHK-004-R023 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-005-R001 | §3.8 | 输入/生成器的获取、seed、shape、格式、大小与身份 | partial | CHK-005-R001 | 输入/生成器/seed/大小/哈希已记录；参考 ELF 未发布 |
| WORK-ARCHIVE-005-R002 | §3.8 | golden独立性、参考算法、比较范围及容差依据 | partial | CHK-005-R002 | smoke checker 用独立 oracle 扫描；dataset verify 的 golden 来自参考 ELF |
| WORK-ARCHIVE-005-R003 | §3.8 | 执行器/ELF、实际输入/读回、误差与退出状态记录 | partial | CHK-005-R003 | check 运行日志与退出状态有；无输出读回记录 |
| WORK-ARCHIVE-005-R004 | §3.8 | 新结果排除陈旧输出与缺失/错误大小输出 | partial | CHK-005-R004 | dependencies/dataset；输入哈希/check 引用已记录；checker 身份与负控缺。 |
| WORK-ARCHIVE-005-R005 | §3.8 | 数值覆盖仅限实际shape/非零/布局/边界/特殊值范围 | partial | CHK-005-R005 | dependencies/dataset；输入哈希/check 引用已记录；checker 身份与负控缺。 |
| WORK-ARCHIVE-005-R006 | §3.5、§3.8、§6.2 | 新/修改检查器拒绝错误结果的实际负控 | pass | CHK-005-R006 | 同一 ELF 翻转 1 个 golden 字节 → val=0x3333 fail、rc=70；正常 val=0x5555 pass、rc=0 |
| WORK-ARCHIVE-006-R001 | §3.9 | 原始Total Cycles与kernel/pure边界、核/线程聚合方法 | pass | CHK-006-R001 | 原始 Total Cycles + 首 TLOAD/末 TSTORE span（r1 134,326；r2 80,669/75,351/168,240） |
| WORK-ARCHIVE-006-R002 | §3.9 | 初始化、准备、同步、排空、转换、读回计入范围 | pass | CHK-006-R002 | 初始化/同步/finisher 计入，host/checker 不计入；readback 不适用 |
| WORK-ARCHIVE-006-R003 | §3.9 | 工作量、MAC/FLOP、padding、峰值和指标公式/单位 | pass | CHK-006-R003 | lookup 无 MAC/FLOP，指标为延迟 cycles，无峰值分母（已给不适用理由） |
| WORK-ARCHIVE-006-R004 | §3.9 | 运行命令、cwd、环境、seed、执行身份及重复次数/随机取样 | pass | CHK-006-R004 | 命令/cwd/env/seed/单次确定性仿真已记录 |
| WORK-ARCHIVE-006-R005 | §3.9 | 每次gfsim性能执行的日志、full及nocounters与采集入口 | pass | CHK-006-R005 | r1/r2/rw8/rw48/way7 均有日志+full+nocounters；入口 run_swimlane.py |
| WORK-ARCHIVE-006-R006 | §3.9 | 压缩trace的恢复命令、原始字节数与hash | pass | CHK-006-R006 | gzip -dc <file>.gz 可恢复；原始字节数与 hash 在 meta/ASSETS |
| WORK-ARCHIVE-006-R007 | §3.9 | PipeView原始数据/开关/解析与时序关系，未采原因和补采方式 | pass | CHK-006-R007 | r1/r2 PipeView 原始 .out.gz + 开关 -p 1 + 与性能运行周期一致；未使用解析器 |
| WORK-ARCHIVE-006-R008 | §3.9 | trace完整/截断/窗口范围及执行完成、退出证据 | pass | CHK-006-R008 | 完整运行、无截断；finisher pass、退出码 0 |
| WORK-ARCHIVE-006-R009 | §3.9 | 语义标注原字段/flows、双JSON同步、原名恢复与源/目标身份 | not_applicable | CHK-006-R009 | 未提供/引用语义标注，按条件项不适用 |
| WORK-ARCHIVE-006-R010 | §3.9.1 | 宿主OS/架构、限制/超时、磁盘需求与代表性case资源观测 | pass | CHK-006-R010 | 宿主 macOS arm64；磁盘 trace 原始/压缩字节；代表性 case 墙钟/RSS 实测 |
| WORK-ARCHIVE-006-R011 | §3.9.1 | RSS/耗时/字节的测量性质、未知项和并发资源预算 | pass | CHK-006-R011 | RSS/墙钟为 /usr/bin/time -l 实测峰值；swap=0；并发预算未测已注明 |
| WORK-ARCHIVE-006-R012 | §3.9.1 | 取消/超时/OOM等失败身份、原因、退出信号与证据保留/清理 | partial | CHK-006-R012 | r3 gfsim 断言失败身份/退出码已记录；无超时/OOM 运行 |
| WORK-ARCHIVE-006-R013 | §3.9.1、§6.1.2 | 离线trace处理与新仿真分开记录 | partial | CHK-006-R013 | trace 处理仅限 gzip/nocounters 生成；无离线重放分析记录 |
| WORK-ARCHIVE-007-R001 | §3.10、附录H | 优化过程的基线、采用方案、关键失败/对照与选择理由 | pass | CHK-007-R001 | analysis/；优化链与证据分类见 analysis/。 |
| WORK-ARCHIVE-007-R002 | §3.10、附录H | 比较双方口径、变化项及可支持的因果证据强度 | pass | CHK-007-R002 | analysis/；优化链与证据分类见 analysis/。 |
| WORK-ARCHIVE-007-R003 | §3.10、附录H | 主线模型性能与分阶段改动收益；缺同口径对照不授予主线收益 | pass | CHK-007-R003 | analysis/；优化链与证据分类见 analysis/。 |
| WORK-ARCHIVE-007-R004 | §3.10 | bound/理论下界及诊断观察与实测/整芯片验收分开 | pass | CHK-007-R004 | analysis/；优化链与证据分类见 analysis/。 |
| WORK-ARCHIVE-007-R005 | §3.10 | 理论推导、单位、假设、扫描代码和原始数据 | pass | CHK-007-R005 | analysis/；优化链与证据分类见 analysis/。 |
| WORK-ARCHIVE-007-R006 | §3.10 | 关键唯一证据保留、重复/无效中间文件清理前依赖核对 | pass | CHK-007-R006 | analysis/；优化链与证据分类见 analysis/。 |
| WORK-ARCHIVE-008-R001 | §4.1 | work/archive/numerical/performance/review各状态独立且符合证据 | partial | CHK-008-R001 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R002 | §4.1 | 本地/共享重放、模型/算子重建和数值状态独立 | partial | CHK-008-R002 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R003 | §4.2 | 必须字段有值/显式状态，未知不默认成功 | partial | CHK-008-R003 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R004 | §4.2、附录J.1 | 缺项前搜索盘点并区分已共享/本地未交付/待确认/确实缺失/不适用 | partial | CHK-008-R004 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R005 | §4.2 | 每个gap的原因、影响、负责人和下一步/恢复动作 | partial | CHK-008-R005 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R006 | §4.2、§8 | 旧规范/issue/历史验收快照保留，新事实不回写旧记录 | partial | CHK-008-R006 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-009-R001 | §3.11 | 任务/归档/共享复核与公共依赖维护责任明确 | partial | CHK-009-R001 | README；reviewer 未指派；其余协作项已记录。 |
| WORK-ARCHIVE-009-R002 | §3.11 | 冻结后不原地覆盖，执行/执行语义/文档修订按规则分配身份 | pass | CHK-009-R002 | revision 2→3 新建提交，旧 revision 保留在 git 历史；未原地覆盖发布资产 |
| WORK-ARCHIVE-009-R003 | §3.11 | supersedes/撤回原因、当前索引与历史版本保留 | pass | CHK-009-R003 | revision 记录含变更内容与未关闭项 |
| WORK-ARCHIVE-009-R004 | §3.11 | 无并行写同一冻结目标，归档后统一更新索引 | partial | CHK-009-R004 | README；reviewer 未指派；其余协作项已记录。 |
| WORK-ARCHIVE-009-R005 | §3.11 | 首页剩余工作、阻塞与关闭条件 | partial | CHK-009-R005 | README；reviewer 未指派；其余协作项已记录。 |
| WORK-ARCHIVE-010-R001 | §5、§5.1 | 最终staged、payload及实际下载解压内容的个人路径检查 | partial | CHK-010-R001 | 归档与 release payload 已扫描；实际下载解压后的 payload 待 Release 后补 |
| WORK-ARCHIVE-010-R002 | §5.1 | 二进制/压缩内容实际检查，扫描工具/文件/身份/退出码留存 | partial | CHK-010-R002 | 扫描工具 sha256/命令/文件清单/退出码已留；下载侧待补 |
| WORK-ARCHIVE-010-R003 | §5.1 | 路径命中逐项修复/误报依据，不排除目录或删改许可证换通过 | pass | CHK-010-R003 | 8 条 /tmp 命中逐项判定；无个人路径；未排除目录/改许可证 |
| WORK-ARCHIVE-010-R004 | §5.2 | prefix-map覆盖各语言/生成源码/子项目实际命令及新产物复核 | not_checked | CHK-010-R004 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R005 | §5.2 | 脱敏/派生副本的原件、变换和最终身份及可执行/审阅范围 | pass | CHK-010-R005 | 脱敏规则、原件/派生 hash 已记录（swim.log/meta） |
| WORK-ARCHIVE-010-R006 | §5.2、§6.1 | 异目录/无关cwd的实际包内执行，排除原目录/未交付链接依赖 | not_checked | CHK-010-R006 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R007 | §6.1、§6.1.2 | 逐操作/case的执行与证据类型、runner及完整执行身份 | not_checked | CHK-010-R007 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R008 | §6.1 | 独立复核者、日期、revision、命令、观察、证据和实际层级 | not_checked | CHK-010-R008 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R009 | §6.1 | material/run/build层级所需实际动作，metadata检查不越级 | not_checked | CHK-010-R009 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R010 | §6.1.1 | 每项适用性、方法/预期、实际状态/退出码和可定位证据 | pass | CHK-010-R010 | 111 项检查记录含适用性/方法/实际结果/证据/责任人 |
| WORK-ARCHIVE-010-R011 | §6.1.1 | 每项检查责任/复核及未完成项，未做不预填PASS | pass | CHK-010-R011 | 未完成项单列，未预填 PASS；复核列标未复核 |
| WORK-ARCHIVE-010-R012 | §6.1.2 | 明确提供/不提供哪些操作，依赖/权限和未完成能力分别陈述 | not_checked | CHK-010-R012 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R013 | §6.1.2 | 历史证据原记录、新旧身份/入口差异、复用依据与未覆盖变化 | not_checked | CHK-010-R013 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R014 | §6.1.3 | 新/修改的可用文档示例按原命令实测并定位证据，修订后评估影响 | not_checked | CHK-010-R014 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R015 | §6.1.3 | 最终结构化文件语法、ID/引用/大小/hash/状态及索引一致性 | pass | CHK-010-R015 | JSON 可解析、ID 唯一、引用可达、大小/hash 重算一致 |
| WORK-ARCHIVE-010-R016 | §6.1.3 | 变换后派生hash与引用清单重算，工具范围/结果/局限留存 | pass | CHK-010-R016 | diss 头规范化与日志脱敏后已重算派生 hash |
| WORK-ARCHIVE-010-R017 | §6.1.4 | 固定规范/登记表身份，稳定ID维护及新旧/retired关系 | not_checked | CHK-010-R017 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R018 | §6.1.4、附录J.4 | 全要求适用性与operation/result/case/子项覆盖映射 | pass | CHK-010-R018 | 111 项逐条覆盖映射已建 |
| WORK-ARCHIVE-010-R019 | §6.1.4 | 未映射/缺实例/未查/失败/自定义/失效引用与独立复核分别统计 | pass | CHK-010-R019 | 已映射/已执行/通过/独立复核分开统计；独立复核为 0 |
| WORK-ARCHIVE-010-R020 | §6.2 | 缺依赖/错身份/错误输出等适用正反向场景实际检查 | partial | CHK-010-R020 | 缺 ELF/MODEL/CLANG 与错误输出已测；错误身份/错配置组合未测 |
| WORK-ARCHIVE-010-R021 | §6.2 | 空case/未知case/零执行汇总的拒绝检查 | not_applicable | CHK-010-R021 | 入口是固定 6144 query kernel，无可传空 case 列表 |
| WORK-ARCHIVE-010-R022 | §3.7.2、§6.2 | 污染环境下的清理/固定/拒绝实测及继承范围 | partial | CHK-010-R022 | 主线模型拒绝该 ELF 已观察并记录；未做受控污染环境实测 |
| WORK-ARCHIVE-010-R023 | §6.2 | 链接逃逸、输出嵌入payload和冻结输入写入的拒绝检查 | not_checked | CHK-010-R023 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R024 | §6.2 | 陈旧输出、缺trace/完成证据时的失败处理；mock范围明确 | not_checked | CHK-010-R024 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R025 | §3.7.4、§6.2 | 缺overlay/不兼容基包/错误组合的拒绝检查 | not_checked | CHK-010-R025 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R026 | §6.3 | 关闭前源码/提交/staged/unstaged/untracked资产保护和校验 | not_checked | CHK-010-R026 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R027 | §6.3 | 关闭前运行/编译/协作者/实时配置依赖检查 | not_checked | CHK-010-R027 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R028 | §6.3 | 固定产物保留、恢复命令核对及可恢复关闭 | not_checked | CHK-010-R028 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R029 | §6.3 | 任务/环境登记一致及新增/关闭/仍保留入口报告 | not_checked | CHK-010-R029 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-011-R001 | §3.12 | 算法数学定义、精度及目标/实现差异 | pass | CHK-011-R001 | algorithm.md 含输入/输出、哈希、探测伪代码、数据流 |
| WORK-ARCHIVE-011-R002 | §3.12 | 数值、形状、存储、并行、流水、ISA/硬件假设逐项登记 | pass | CHK-011-R002 | 数值/形状/存储/并行/流水/ISA 维度逐项登记 |
| WORK-ARCHIVE-011-R003 | §3.12 | 每项前提的来源、影响、检查/证明依据 | pass | CHK-011-R003 | 每条前提含来源、影响、检查/保证方式 |
| WORK-ARCHIVE-011-R004 | §3.12 | 违反前提时的拒绝/回退/未支持行为及实际验证范围 | pass | CHK-011-R004 | 违反前提的拒绝/回退/未支持行为已列；验证范围注明 |
| WORK-ARCHIVE-011-R005 | §3.12 | 假设/支持范围变化新revision，不默认泛化支持 | pass | CHK-011-R005 | 变更前提/范围按 revision 处理；未泛化支持 |
