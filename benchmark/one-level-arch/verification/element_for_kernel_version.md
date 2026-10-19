# Element-for kernel 交付版本

核心示例是 [`element_atomic_topk.hpp`](../kernels/basic_op/topk/element_atomic_topk.hpp)。
它处理 U32 承载的 16-bit key，保持 Tile 加载、元素计算、Tile 写回的源码顺序。
4 个 `#pragma pto element for` 区域分别承担两组融合的工作区清零和两轮 radix histogram。
元素循环内的移位、按位与、条件和原子加使用普通 C++，没有手写 barrier。

```cpp
TLOAD(keys, input + begin, valid);
auto parts = TPARTVIEW<32>(keys, valid);
// 取出当前 logical part；布局由公共 API 保留。
auto view = parts.part(part);
TMOV(values, view);
const auto &key = TPARTELEMENT(values);
auto &old_value = TPARTELEMENT(old);
#pragma pto element for
for (unsigned element = 0; element < 32; ++element) {
  if (element < valid_elements)
    old_value[element] = __atomic_fetch_add(
        &high_hist[key[element] >> 8], 1u, __ATOMIC_RELAXED);
  else
    old_value[element] = 0;
}
TSTORE(workspace.old_high + begin, old, parts, part);
```

代码片段展示一个 part 的计算；完整初始化、分区和两轮选择见头文件。
cutoff 扫描和紧凑输出包含跨元素依赖，仍按普通顺序 C++ 执行。
输入无需 padding；原子 old-value 工作区按 128 元素补齐，写回按逻辑输入顺序。
重复 key 按 TopK 配额保留，输出集合无序。

QLI 的 [`qli_element.hpp`](../kernels/solution/qli/qli_element.hpp) 及
simple/dynamic kernel 还覆盖反复的 Tile reduction → element update → Tile reduction，
包括 outlined Local Tile helper、运行期边界和相同 workspace 重复调用。

## Masked Tile 编程契约

`#pragma pto element for` 是编译约束：已支持的元素数据路径必须生成带执行
mask 的原生 Tile 运算；不支持的表达式、别名或控制流应报错，不能回退到 scalar
循环。开发版编译器要求 `-mllvm -linxv5-enable-generic-element=true`；缺少或关闭
该选项、`-O0`、以及残留未完成的 marked region 均拒绝编译。

外层分块、PE 调度、统一地址计算和 packed predicate 的 GPR 运算仍可使用
scalar 指令。验收禁止的是逐元素数据计算、加载、写回或原子操作退化为 scalar。
TopK 的 cutoff 扫描和紧凑输出有跨元素依赖，仍是明确保留的顺序阶段。

MoE 的单 PE `group_token_old` 和四 PE 动态 `group_token_vec_mt_dyn` histogram
采用一个 32 元素 fused marked loop：直接读取 `input[element]`，检查有效长度
和 expert 范围，再执行原子计数。编译器生成 masked gather 和 masked atomic。输入不要求新增 padding，清零同样使用 masked scatter。
PE 行分工、跨 PE barrier、后续 prefix sum 和稳定排序保持原顺序。
动态 histogram 与 driver 在同一编译单元内，普通 Tile 和 fused element loop
交替执行。编译器从本地 helper 的全部直接调用方证明缓冲区分离；不在源码中
添加 `restrict`，未知或重叠的调用仍拒绝降低。

`run_solution_gfrun_regressions.py` 对这两个迁移用例先检查具体 helper 的源码、
优化 IR 和 ELF 反汇编，再运行数值测试。检查拒绝 scalar 数据 load/store、
scalar atomic、原子 helper 和 vector extract/insert；缺失执行 mask 也必须失败。
`test_check_moe_element_histograms.py` 保留对应的故障注入测试。
同一 runner 显式选择 `--only group_token_histogram_element` 可运行 11 个边界长度、
重复调用、非法 expert id 和输出 guard 的独立测试；该补充用例不改变原 21 项分母。
MoE combine 的算术已有 Tile 运算；prefix/rank 和 MegaMoE 的变长内循环仍需独立
依赖分析与编译支持，不把普通外层循环机械改成 element loop。

## 优先使用 fused 源码

当相邻元素阶段没有跨元素依赖、需要保留的中间 Tile 快照或归约边界时，
优先放在同一个 element loop 内。MoE histogram 的推荐形式为：

```cpp
#pragma pto element for
for (unsigned element = 0; element < 32; ++element) {
  if (element < valid) {
    const unsigned expert = input[element];
    if (expert < histogramSize)
      __atomic_fetch_add(histogram + expert, 1u, __ATOMIC_RELAXED);
  }
}
```

本次合并两个 MoE read/count 循环对，以及 TopK 两个相同零值 store 循环对。
TopK 仍写回四个原始缓冲区；减少的是 region 边界，不是写回数量。
编译器仅在对齐足以排除部分字节重叠时接受可交换的零值写入。

QLI `make_key`、gather/view-copy、transpose、concat 的部分拆分保留整块输入
快照与既有 alias 行为；element→Tile reduction→element 保留归约顺序。
清零与 histogram 累加之间也保留阶段边界，防止后续 lane 清零覆盖已有计数。
这些边界不应为了减少 pragma 数量而合并。

QLI 的多 PE driver 由各 PE 分担输入预处理，再由 PE0 执行 score 与 TopK。
`run_qli_pe0` 用 `PTO_PE_DOMAIN(8)` 声明这段调用链的 Tile 参与范围；8 是
架构原始 mask `1000`，对应 PE0。编译器将该范围传递到 helper、Tile producer、
consumer 和生成的复制指令。该声明不会给普通 C++ 标量代码添加 PE 条件，
因此调用仍位于 `tid != 0` 返回之后。

PE mask 与 element loop 的 execution mask 分别检查：前者限制参与的 PE，
后者保留尾部和 `if` 条件选择的元素。QLI runner 检查 PE0 调用链上每个
`B.IOT` 的 mask，并检查每个 element atomic 的 `B.IOR` execution mask。
`--profile-set original` 保留原五项用例；`original-four-pe` 验证原四项静态
配置的四 PE 路径，`upstream-four-pe` 验证新增的两项 Sq32/Skv8192 配置。

## 构建与运行

使用彼此匹配的开发版 LLVM、TileOP API 和 gfrun。完整运行库提供 musl sysroot；
`COMPILER_DIR` 可以指向独立编译器 build 的 bin 目录。每个运行目录必须是新目录。

```bash
export COMPILER_DIR=/path/to/frozen-compiler/bin
export COMPILER_SOURCE_ROOT=/path/to/compiler-source
export LINX_RUNTIME_ROOT=/path/to/linx_blockisa_llvm_musl
export SYSROOT="$LINX_RUNTIME_ROOT/sysroot"
export RESOURCE_DIR=/path/to/private-clang-resource
export API_INCLUDE=/path/to/Linx-TileOP-API/include
export SSM=/path/to/SuperScalarModel
export MODEL_SOURCE_ROOT="$SSM"
export GFRUN=/path/to/frozen/gfrun
export MODELS=gfrun

ARTIFACT_DIR=/path/to/new/topk bash run_element_atomic_topk.sh
ARTIFACT_DIR=/path/to/new/suite bash run_tile_element_suite.sh
```

上述命令在本文件目录执行。primary runner 在运行目录创建私有 resource overlay，
suite runner 将选定 API 安装到 `RESOURCE_DIR`，因此后者应是本次验证专用目录。
二者都保留原有 120 秒 gfrun 超时，验证 ELF 指令、独立 golden 和运行前后的输入哈希。
失败或缺失输出不能以 finisher 文本代替正确性证明。

[solution 回归入口](run_solution_gfrun_regressions.py) 另外覆盖 LLVM #129 的 21 个
原始配置，使用隔离对象目录、逐例状态，以及已有的数值 oracle。
QLI 和原始 low-precision matmul 的运行入口见
[`pto_element_migration.md`](pto_element_migration.md)。

## 验收范围

当前验证批次的精确版本与结果写入 `pto_element_migration.json` 的最新 checkpoint。
最终验收需要包含 default suite 64 项、原 777 元素 TopK、四个真实四 PE wrapper、
17 个 TopK 边界配置，以及 QLI、predicate 和原始 matmul 的相应数值用例。
本轮只验收 gfrun。gfsim/TimingSim 另行验证；CUBE_M16/A16 完整 element 迁移及
剩余 QSMLA 配置仍保留在原始 21 应用清单中，不由 TopK 示例替代。

## 2026-10-19 固定版本验收

LLVM `03019e0f794d6f3cf57407508e3d7aedc705fc81`、API `03ef51d`、
gfrun `0a237ccfe`、benchmark `aa97d6d` 在干净源码上重新构建并完成验收：

- element suite **64/64**，含四个真实四 PE wrapper 和 TopK 边界配置；
- 核心 777 元素 TopK **9/9 内存段**；solution 回归 **21/21**；
- QLI key48、histogram72、NORM13、PopN18、静态 TopK7、原始 driver5、动态 TopK4；
- 原始低精度 matmul **7/7**，predicate 原始/浮点边界各 7 项，D6 和 issue-130；
- MoE 补充边界 driver **1/1**，11 个长度各重复调用两次。

独立最终审计通过，核验 **2,843 项哈希**，没有缺失或失败项。
精确版本、结果和 artifact 路径见 `pto_element_migration.json` 的
`masked_element_gfrun_20261019`；本次文档提交发生在验收之后，不改写已记录版本。

以上关闭本轮 gfrun 配置验收，不代表 LLVM 全部测试可合入：28 个旧 pragma
fixture 仍需迁移到 masked-only 契约，issue-60 的合法 guard-correlated 程序仍有
路径证明缺口。公共 pointer TLOAD 的部分 parent 搬运可能使用 marked region
之外的 scalar tail copy；本轮 MoE helper 已改用 bounded masked MGATHER。
完整原始应用库存仍为 17/21 个已验证 profile，四条 pending 路径不由
solution 的 21/21 数值回归替代。gfsim 继续留到后续 PR。

## Fused 版本验收

最新融合版本使用 LLVM `a3b798ae3cf22f452b0fe58a61191764a5464623`、
benchmark `f4a20b82ec7fc5658a03cc45a7c5f8c1a7379292`，API `03ef51d` 与
gfrun `0a237ccfe` 保持不变。该 benchmark 提交包含 TopK 检查器对共享地址偏移
的适配；四条 masked store 及其各自执行 mask 的检查仍保留。

最终结果：element suite **64/64**、solution **21/21**、核心 TopK **9/9 内存段**、
原始 matmul **7/7**、QLI 全组、predicate 原始/边界、D6、issue-130 和 MoE
重复边界测试全部通过。独立审计核验 **2,844 项哈希**，错误和待完成项均为零。
精确版本与范围见 JSON 的 `fused_element_gfrun_20261019`。

同一 v16 编译器下，动态 MoE histogram helper 从拆分版的 5 条 TCI、3 条 TCMPS、
1 条 TSEL，减少为 3 条 TCI、2 条 TCMPS、0 条 TSEL。两版都保留 2 条 TLEA、
1 条 masked MGATHER 和 1 条 masked MGATHER.ADD。这是静态指令比较，未作周期加速结论。

动态 MoE 已移除独立 TU workaround，可在完整 driver 中交替执行 Tile 与 fused
元素循环。前述 LLVM legacy pragma/issue-60、原应用库存和 gfsim 范围限制仍分别跟踪。
