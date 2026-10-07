# PTO element for 全量迁移与验证

目标是把活跃的稀疏、histogram、gather、scatter 和不规则访问 workload 写成
`#pragma pto element for` 标记的真实元素循环。源码以逻辑 element array 索引
Tile，中文注释解释有效元素、条件访存和输出顺序；每个应用 kernel 保留完整算法。
正式 TileOp API 负责存储与 view，编译器负责表达式分析和中间 Tile 分配。

源文件清单在 [pto_element_migration.json](pto_element_migration.json)。清单逐项
记录未完成状态；迁移和模型执行完成后才填写精确提交、ELF 和独立 golden 证据。
多线程 wrapper 有单独的执行覆盖要求，不能由单线程结果推断完成。

## 当前缺口

- 用户要求标准编译器路径：`element for` 进入正常 C++ CFG/SSA，复用 Linx SIMT
  的循环/控制流分析，再用现有 Tile 后端降低。当前 AST 模式匹配仍是 foundation，
  不能把某一种 `if` 或表达式写法当作通用编译能力。Astra/xhigh 已完成实现方案审查；
  通用 region pass、依赖分析和 typed view 扩展仍待实现。

- 当前前端支持 U32 histogram atomic-if、一种 FP32 add/sub 分支，以及纯 U32
  的十种二元表达式和一元负号/补码。纯表达式先无副作用地验证完整 AST，再生成
  32-element Tile SSA；局部临时变量复用现有寄存器分配。其他 dtype、一般条件、
  casts/math 和普通指针索引仍有缺口，不能称为通用 element-wise。
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
