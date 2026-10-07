# PTO element for 全量迁移与验证

目标是把活跃的稀疏、histogram、gather、scatter 和不规则访问 workload 写成
`#pragma pto element for` 标记的真实元素循环。源码以逻辑 element array 索引
Tile，中文注释解释有效元素、条件访存和输出顺序；每个应用 kernel 保留完整算法。
正式 TileOp API 负责存储与 view，编译器负责表达式分析和中间 Tile 分配。

源文件清单在 [pto_element_migration.json](pto_element_migration.json)。清单逐项
记录未完成状态；迁移和模型执行完成后才填写精确提交、ELF 和独立 golden 证据。
多线程 wrapper 有单独的执行覆盖要求，不能由单线程结果推断完成。

## 当前缺口

- 标准 CFG/SSA 编译器已闭合 P1a U32/M32/32 算术与 P1b 零 inactive gather。
  LLVM34ade53 的 rebuilt Clang 与 APIb223de6、benchd4fabe0、model34a3d799
  均为 clean head。15 个 focused lit 通过；同一 expression ELF 的五段及 gather
  ELF 的四段独立 golden 在 gfrun/gfsim 均通过。gather 保留263输出、257全空
  poison分区和guards。精确 heads、ELF/content ID、29/30 hash 的 frozen artifact
  记录在 JSON 的 foundation profiles；这些记录不关闭原应用清单。
- P1b 只接受 U32 索引、i32有效数与证明过的 CFG 零值合流。宽界限、非零else、
  不安全cast、错误分支方向、clobber和额外load等边界明确诊断。P2 正推进
  tail/equality predicate 与 relaxed atomic；O0 marked region 的 typed spill、
  一般PHI/select、cast/math、scatter/dynamic probe 仍有缺口。
- 正式 ElementTile 已在既有 API headers 支持 U32/S32/F32 的32/128 typed
  storage/view/transport，clean API native10/10与hostmakecheck通过；三dtype
  transport另有同ELF双模型七区段golden。编译器尚不接受 marked S32/F32 region，
  不能用 API 类型存在来宣称该dtype的元素循环已可执行。默认Tile存储保持一致，
  追加profile参数改变C++ mangled type identity，依赖接口需要重新编译。
- gather原应用仍需rank-2 FP32行索引/广播与原四配置；不能以U32一维foundation
  代替。concat原配置含S32与FP16/512元素，完整输入容量与运行时shape映射均须
  保留，不能为了第一阶段把512改成32或缩小原数据集。JSON已明确登记配置。
- QLI 中的私有 histogram 汇编、稀疏 MLA 的主存 spill/标量索引阶段和哈希查找的
  动态探测需要各自的完整迁移，不能用新建的小型 histogram 代替原 workload。

## 必须取得的证据

1. 前端实际解析 canonical pragma；错误拼写、错误后继语句及不支持语义应诊断，
   不得忽略标记或无声退化成普通标量循环。
2. 对表达式、临时变量、predicate、typed byte index 做 IR 与物理指令检查。
   算术输入/输出类型、mask、逻辑 shape 和布局保持一致。
3. 每个 kernel 和适用配置由指定 compiler 与正式安装的 API 重新编译；同一 ELF
   分别在 gfrun、gfsim 执行，输入、输出、副作用和 guard 对照独立 golden。
4. gfsim 的 A3 计数为零、队列守恒正常，不允许通过放大资源/超时、伪进度、
   finisher override 或修改 golden 隐藏错误。
5. 分类型记录 pointwise 基础算子、gather/scatter、原子/非原子、稀疏和多线程
   覆盖。几个 opcode 或一个 shape 的结果不支持“所有基础 TileOp”完成声明。

原始 opcode conformance 用例保留其直接测试目的，并补对应的 element-loop
覆盖。历史 two-level、retired mnemonic 和缺失头文件的旧 driver 单独登记，
不能把目录中存在源码当作可执行覆盖，也不能隐式恢复已退休 ISA 语义。

## 关联跟踪

- [NPUbench #202](https://github.com/PTO-ISA/SuperNPUBench/pull/202)
- [LLVM #117](https://github.com/LinxISA/llvm-project/pull/117)
- [TileOp API #249](https://github.com/LinxISA/Linx-TileOP-API/pull/249)
- [模型 #902](https://github.com/LinxISA/SuperScalarModel/pull/902)
- [规格 #369](https://github.com/PTO-ISA/pto-spec/pull/369)
- [集成 issue #370](https://github.com/PTO-ISA/pto-spec/issues/370)

整个目标尚未完成。后续每个变更都必须让这份全量清单中的真实应用路径更接近
最终表达与执行要求，不能把任务缩减为当前九个 foundation ELF（八个 suite case 和独立主 Top-K） 的兼容性改名。
