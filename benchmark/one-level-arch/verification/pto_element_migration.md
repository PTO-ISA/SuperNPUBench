# PTO element for 全量迁移与验证

目标是把活跃的稀疏、histogram、gather、scatter 和不规则访问 workload 写成
`#pragma pto element for` 标记的真实元素循环。源码以逻辑 element array 索引
Tile，中文注释解释有效元素、条件访存和输出顺序；每个应用 kernel 保留完整算法。
正式 TileOp API 负责存储与 view，编译器负责表达式分析和中间 Tile 分配。

源文件清单在 [pto_element_migration.json](pto_element_migration.json)。清单逐项
记录未完成状态；迁移和模型执行完成后才填写精确提交、ELF 和独立 golden 证据。
多线程 wrapper 有单独的执行覆盖要求，不能由单线程结果推断完成。

## 当前缺口

- `element for` 已进入正常 C++ CFG/SSA，由必需的 LLVM region pass 降低；
  Astra/xhigh 的 linx-simt 复用方案已落实第一阶段：U32/M32/32 的直线算术、
  SSA 临时值和证明过的 carrier transport。旧 AST body matcher 已删除。
  当前源代码的 expression kernel 已通过独立 review、11 个 lit 测试和同一 ELF
  在 gfrun/gfsim 的五区段 golden；提交后 rebuilt Clang 的版本为 e46b264，clean-head provenance 也已通过。
  精确四 repo heads、ELF/content ID 和 frozen artifact 路径见 JSON 中
  `foundation_expression_profile.evidence`；该记录不关闭原应用清单。
- 标准 CFG 路径尚未支持条件 gather、一般 predicate/PHI、atomic、cast 和其他
  dtype；这些边界明确诊断。历史九个 foundation ELF 是旧路径检查点，不能
  推断新路径已跑通整个 suite。O0 也因 typed spill/reload 尚未闭合而明确拒绝。
- 既有 masked gather/scatter intrinsic 的布局合同不足以直接接 M32 element
  carrier；条件 mask 的 basis type 也必须与 widened byte index 合法匹配。
  这些需要在编译器/模型边界闭合，不能靠删除 legality 检查绕过。
- `ElementTile` 首版仅 U32；其他 dtype 的物理 carrier 并不自动成为 typed
  element array。更多位宽和浮点必须先实现正确视图和独立测试。
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
