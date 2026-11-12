# 逐条检查记录（NDF §6.1.1，revision 3）

检查人：lvhao；复核人：未独立复核（见 WORK-ARCHIVE-010）。
状态列是实际结果；`not_checked` 不预填 PASS。

| requirement_id | 检查对象/范围 | 适用性 | 材料与身份 | 方法/预期 | 实际结果 | 证据 | 检查人/复核人 | 未完成项 |
|---|---|---|---|---|---|---|---|---|
| WORK-ARCHIVE-001-R001 | task_id 的唯一性与任务范围 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-001-R002 | result_id 绑定固定产物与执行依赖组合 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-001-R003 | 每次执行具有独立 run_id | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-001-R004 | 负责人/代号可解析，逻辑目录及结果可定位 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-001-R005 | 公共依赖有不可变 dependency_id、hash与获取位置 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 manifest 字段 | pass | dependencies/*/manifest.json | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-001-R006 | 规范承载仓库/批准身份及本结果采用版本固定，草案不冒充已批准 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-002-R001 | T/F/P/S 选择符合实际声明且未规避条件必须项 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | README 表格；内容矩阵逐项见 README；缺项以 GAP-* 记录。 |
| WORK-ARCHIVE-002-R002 | 目标、负责人、日期与结果索引 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对首页字段 | pass | README.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-002-R003 | shape/dtype/layout/tiling/核数/工作量及不适用理由 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 逐项核对 | pass | results/*/README.md, results/*/manifest.json | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-002-R004 | 全部适用内容项的存在性与要求等级；逐项拆实例 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | README 表格；内容矩阵逐项见 README；缺项以 GAP-* 记录。 |
| WORK-ARCHIVE-002-R005 | 冻结文件逐文件路径、字节数与SHA256清单，排除清单自身 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 运行 sha256sum -c | pass | FILE_MANIFEST.json, FILE_MANIFEST.sha256 | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-003-R001 | 算子源码仓库、完整commit和子模块身份 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 commit 与源码 | pass | results/*/manifest.json, dependencies/dataset | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-003-R002 | dirty修改及实际使用源码/头文件身份 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 patch/untracked | pass | dependencies/compiler_llvm_pr146/ | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-003-R003 | 实际构建入口、cwd、选项、宏、架构及环境 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 按脚本重建 | pass | build/build_kernel.sh, README.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-003-R004 | 实际compiler/linker/objdump身份与构建日志 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对实际工具/日志 | partial | validation/build_repro_r3.log | lvhao / 未复核 | kernel 重建日志已归档；LLVM 与 gfsim 的完整构建日志仍缺（GAP-LOG-1） |
| WORK-ARCHIVE-003-R005 | 最终ELF与匹配diss身份及两者对应关系 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 ELF/diss | pass | results/*/manifest.json, validation/diss_normalization.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-003-R006 | 运行ELF与冻结源码/构建产物的对应依据，未知显式登记 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 重建并比较 hash | pass | validation/build_repro_r3.log | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-003-R007 | 声称重建字节一致时的实际重建与比较证据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 重建+hash 比较 | pass | validation/build_repro_r3.log | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-004-R001 | model/compiler/API来源、完整revision或已知身份与缺项 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R002 | patch顺序、必要untracked/生成文件/子模块及历史身份 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R003 | 实际运行/构建工具二进制hash | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R004 | ISA/ASL固定版本及合法性、API表达、lowering、功能/时序支持分别陈述 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对陈述 | partial | results/*/README.md, results/r2.../algorithm.md | lvhao / 未复核 | ISA 目标 linx64v5、gfsim 时序模型、ASL 不适用已分别陈述；功能模型当前不可用 |
| WORK-ARCHIVE-004-R005 | 首页/清单/详情中的本地绕过启用清单；没有则none | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对清单 | pass | dependencies/LOCAL_MODIFICATIONS.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-004-R006 | 每项绕过的问题、来源、改前/后行为和产物身份 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 逐项核对 | pass | dependencies/LOCAL_MODIFICATIONS.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-004-R007 | 每项绕过的应用/撤销、验证及性能/合同边界 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 逐项核对 | pass | dependencies/LOCAL_MODIFICATIONS.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-004-R008 | 可加载冻结配置入口与完整传递依赖 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R009 | CLI/环境覆盖、加载顺序、cwd，区分未设置/空值/显式值 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R010 | 最终有效配置/默认值快照或缺项与历史恢复依据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R011 | 实际加载归档配置的身份与运行证据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R012 | 基线/继承修复/本地增量逐项问题与源码/产物组成 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R013 | issue/PR及合入状态快照，归档引用未冒充修复完成 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R014 | patch可应用、tracked恢复、完整恢复、构建和运行范围分别记录 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R015 | 跨仓库资产归属及源码commit/元数据commit/二进制身份分别固定 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R016 | 每个外部资产用途、必需/可选、内容/排除项、格式与大小/hash | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 逐项核对 | pass | ASSETS.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-004-R017 | 每个外部资产固定获取方式、权限、责任人及核对日期 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 Release 与下载 | pass | ASSETS.md, validation/material_check.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-004-R018 | planned/prepared/uploaded/download_verified与实际动作相符 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对状态 | pass | ASSETS.md, validation/material_check.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-004-R019 | runtime必需依赖与历史可选trace分开，工具链按操作提供 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R020 | Release/README/PR/索引的内容、身份、能力和限制一致 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R021 | 跨仓库result/ELF/diss/runtime/config身份双向核对，无未来commit伪填 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R022 | 组合附件的适用基包、必需关系、顺序、替换文件与最终身份 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-004-R023 | 当前组合与历史索引范围、受影响版本的命令勘误可定位 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/*/manifest.json；依赖身份已固定；API 来源缺 GAP-API-1。 |
| WORK-ARCHIVE-005-R001 | 输入/生成器的获取、seed、shape、格式、大小与身份 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对数据身份 | partial | dependencies/dataset/ | lvhao / 未复核 | 输入/生成器/seed/大小/哈希已记录；参考 ELF 未发布 |
| WORK-ARCHIVE-005-R002 | golden独立性、参考算法、比较范围及容差依据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 golden 来源 | partial | results/r1.../README.md, validation/negative_control/ | lvhao / 未复核 | smoke checker 用独立 oracle 扫描；dataset verify 的 golden 来自参考 ELF |
| WORK-ARCHIVE-005-R003 | 执行器/ELF、实际输入/读回、误差与退出状态记录 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 check 日志 | partial | results/r1.../README.md, validation/negative_control/ | lvhao / 未复核 | check 运行日志与退出状态有；无输出读回记录 |
| WORK-ARCHIVE-005-R004 | 新结果排除陈旧输出与缺失/错误大小输出 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/dataset；输入哈希/check 引用已记录；checker 身份与负控缺。 |
| WORK-ARCHIVE-005-R005 | 数值覆盖仅限实际shape/非零/布局/边界/特殊值范围 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | dependencies/dataset；输入哈希/check 引用已记录；checker 身份与负控缺。 |
| WORK-ARCHIVE-005-R006 | 新/修改检查器拒绝错误结果的实际负控 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | A/B 负控 | pass | validation/negative_control/ | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R001 | 原始Total Cycles与kernel/pure边界、核/线程聚合方法 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对周期与边界 | pass | results/*/README.md, ASSETS.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R002 | 初始化、准备、同步、排空、转换、读回计入范围 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对口径 | pass | results/*/README.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R003 | 工作量、MAC/FLOP、padding、峰值和指标公式/单位 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对公式/分母 | pass | results/*/README.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R004 | 运行命令、cwd、环境、seed、执行身份及重复次数/随机取样 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对运行字段 | pass | results/*/README.md, config/ | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R005 | 每次gfsim性能执行的日志、full及nocounters与采集入口 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 5 个 P 运行 | pass | ASSETS.md, outputs/.../r3_extra/ | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R006 | 压缩trace的恢复命令、原始字节数与hash | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对解压命令/hash | pass | ASSETS.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R007 | PipeView原始数据/开关/解析与时序关系，未采原因和补采方式 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对原始数据/开关 | pass | ASSETS.md (A-PIPEV) | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R008 | trace完整/截断/窗口范围及执行完成、退出证据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对完成证据 | pass | results/*/README.md, validation/negative_control/r1check_ok.log | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R009 | 语义标注原字段/flows、双JSON同步、原名恢复与源/目标身份 | 不适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对是否提供标注 | not_applicable | results/*/README.md | lvhao / 未复核 | 未提供/引用语义标注，按条件项不适用 |
| WORK-ARCHIVE-006-R010 | 宿主OS/架构、限制/超时、磁盘需求与代表性case资源观测 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对资源记录 | pass | results/r1.../README.md, A-PIPEV | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R011 | RSS/耗时/字节的测量性质、未知项和并发资源预算 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对测量性质 | pass | results/r1.../README.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-006-R012 | 取消/超时/OOM等失败身份、原因、退出信号与证据保留/清理 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对失败运行 | partial | results/r3.../README.md | lvhao / 未复核 | r3 gfsim 断言失败身份/退出码已记录；无超时/OOM 运行 |
| WORK-ARCHIVE-006-R013 | 离线trace处理与新仿真分开记录 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对离线处理 | partial | build/run_swimlane.py | lvhao / 未复核 | trace 处理仅限 gzip/nocounters 生成；无离线重放分析记录 |
| WORK-ARCHIVE-007-R001 | 优化过程的基线、采用方案、关键失败/对照与选择理由 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-007-R002 | 比较双方口径、变化项及可支持的因果证据强度 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-007-R003 | 主线模型性能与分阶段改动收益；缺同口径对照不授予主线收益 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-007-R004 | bound/理论下界及诊断观察与实测/整芯片验收分开 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-007-R005 | 理论推导、单位、假设、扫描代码和原始数据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-007-R006 | 关键唯一证据保留、重复/无效中间文件清理前依赖核对 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | pass | 见 note / results、dependencies | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-008-R001 | work/archive/numerical/performance/review各状态独立且符合证据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R002 | 本地/共享重放、模型/算子重建和数值状态独立 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R003 | 必须字段有值/显式状态，未知不默认成功 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R004 | 缺项前搜索盘点并区分已共享/本地未交付/待确认/确实缺失/不适用 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R005 | 每个gap的原因、影响、负责人和下一步/恢复动作 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-008-R006 | 旧规范/issue/历史验收快照保留，新事实不回写旧记录 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；状态独立性与缺项登记见 README/COMPLIANCE。 |
| WORK-ARCHIVE-009-R001 | 任务/归档/共享复核与公共依赖维护责任明确 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | README；reviewer 未指派；其余协作项已记录。 |
| WORK-ARCHIVE-009-R002 | 冻结后不原地覆盖，执行/执行语义/文档修订按规则分配身份 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 revision | pass | README.md (revision), git log | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-009-R003 | supersedes/撤回原因、当前索引与历史版本保留 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 revision 记录 | pass | README.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-009-R004 | 无并行写同一冻结目标，归档后统一更新索引 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | README；reviewer 未指派；其余协作项已记录。 |
| WORK-ARCHIVE-009-R005 | 首页剩余工作、阻塞与关闭条件 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 人工核对归档材料 | partial | 见 note / results、dependencies | lvhao / 未复核 | README；reviewer 未指派；其余协作项已记录。 |
| WORK-ARCHIVE-010-R001 | 最终staged、payload及实际下载解压内容的个人路径检查 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 下载+解压+扫描 | pass | validation/material_check.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R002 | 二进制/压缩内容实际检查，扫描工具/文件/身份/退出码留存 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对记录 | pass | validation/material_check.md, validation/PATH_SCAN.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R003 | 路径命中逐项修复/误报依据，不排除目录或删改许可证换通过 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 逐项复核 | pass | validation/PATH_SCAN.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R004 | prefix-map覆盖各语言/生成源码/子项目实际命令及新产物复核 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R005 | 脱敏/派生副本的原件、变换和最终身份及可执行/审阅范围 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对变换记录 | pass | release_r3/sanitization.json, validation/PATH_SCAN.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R006 | 异目录/无关cwd的实际包内执行，排除原目录/未交付链接依赖 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R007 | 逐操作/case的执行与证据类型、runner及完整执行身份 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R008 | 独立复核者、日期、revision、命令、观察、证据和实际层级 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R009 | material/run/build层级所需实际动作，metadata检查不越级 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R010 | 每项适用性、方法/预期、实际状态/退出码和可定位证据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 逐项生成并核对 | pass | validation/check_records.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R011 | 每项检查责任/复核及未完成项，未做不预填PASS | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对字段 | pass | validation/check_records.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R012 | 明确提供/不提供哪些操作，依赖/权限和未完成能力分别陈述 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R013 | 历史证据原记录、新旧身份/入口差异、复用依据与未覆盖变化 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R014 | 新/修改的可用文档示例按原命令实测并定位证据，修订后评估影响 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R015 | 最终结构化文件语法、ID/引用/大小/hash/状态及索引一致性 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 运行校验 | pass | FILE_MANIFEST.json, sha256sum -c | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R016 | 变换后派生hash与引用清单重算，工具范围/结果/局限留存 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 重算并核对 | pass | validation/diss_normalization.md, release_r3/sanitization.json | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R017 | 固定规范/登记表身份，稳定ID维护及新旧/retired关系 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R018 | 全要求适用性与operation/result/case/子项覆盖映射 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对行数与统计 | pass | requirements_coverage.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R019 | 未映射/缺实例/未查/失败/自定义/失效引用与独立复核分别统计 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对统计 | pass | requirements_coverage.md, COMPLIANCE.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-010-R020 | 缺依赖/错身份/错误输出等适用正反向场景实际检查 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 执行负控 | partial | validation/negative_control/ | lvhao / 未复核 | 缺 ELF/MODEL/CLANG 与错误输出已测；错误身份/错配置组合未测 |
| WORK-ARCHIVE-010-R021 | 空case/未知case/零执行汇总的拒绝检查 | 不适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对入口 | not_applicable | validation/negative_control/README.md | lvhao / 未复核 | 入口是固定 6144 query kernel，无可传空 case 列表 |
| WORK-ARCHIVE-010-R022 | 污染环境下的清理/固定/拒绝实测及继承范围 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对污染场景 | partial | results/r3.../README.md | lvhao / 未复核 | 主线模型拒绝该 ELF 已观察并记录；未做受控污染环境实测 |
| WORK-ARCHIVE-010-R023 | 链接逃逸、输出嵌入payload和冻结输入写入的拒绝检查 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R024 | 陈旧输出、缺trace/完成证据时的失败处理；mock范围明确 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R025 | 缺overlay/不兼容基包/错误组合的拒绝检查 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R026 | 关闭前源码/提交/staged/unstaged/untracked资产保护和校验 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R027 | 关闭前运行/编译/协作者/实时配置依赖检查 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R028 | 固定产物保留、恢复命令核对及可恢复关闭 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-010-R029 | 任务/环境登记一致及新增/关闭/仍保留入口报告 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 未执行 | not_checked | 见 note / results、dependencies | lvhao / 未复核 | COMPLIANCE.md；验收层级未做独立复核（非归档人）。 |
| WORK-ARCHIVE-011-R001 | 算法数学定义、精度及目标/实现差异 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对算法定义 | pass | results/r2.../algorithm.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-011-R002 | 数值、形状、存储、并行、流水、ISA/硬件假设逐项登记 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 逐项核对 | pass | results/r2.../algorithm.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-011-R003 | 每项前提的来源、影响、检查/证明依据 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 逐项核对 | pass | results/r2.../algorithm.md, dependencies/LOCAL_MODIFICATIONS.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-011-R004 | 违反前提时的拒绝/回退/未支持行为及实际验证范围 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 | pass | results/r2.../algorithm.md | lvhao / 未复核 | 无 |
| WORK-ARCHIVE-011-R005 | 假设/支持范围变化新revision，不默认泛化支持 | 适用 | revision 3 归档文件与外部资产（见 ASSETS） | 核对 | pass | README.md (revision) | lvhao / 未复核 | 无 |
