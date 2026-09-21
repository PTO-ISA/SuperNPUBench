# topk radix-select tile-op 化：不支持项与绕行清单

算子：`benchmark/one-level-arch/kernels/multi_thread/topk/topk.hpp`
（branch `topk-radix-select`，cand 追加 tile 化版本）。
依据：`topk_scatter_atomic_add_tileop.md`（本目录设计稿）、gfrun
（`model/emulator/`）、工具链头文件（`tileop-api/jcore/template_asm.hpp`、
`common/pto_tile.hpp`）。suffix cumsum 见
`kernels/multi_thread/histogram_cumsum_m32/histogram_cumsum_m32.md`。

已经跑在 tile op 上的部分（对照用，后文不再列）：

- bin/key 计算（TLOAD/TCVT/TSHRS/TMULS/TADDS/TXOR/TANDS/TSTORE）
- 候选值回读（手写 MGATHER，PR #313 形式）
- 结果写出（手写 MSCATTER_MASK）
- 256-bin suffix cumsum（独立算子 `histogram_cumsum_m32`：1 条 grouped ND2M32
  TLOAD + B.SUBVIEW pair-tree 组内后缀 + TSHUF 跨组 scan）
- find_threshold（TCMPS + TSELS + TEXPANDS + TCOLSUM + cell TSTORE）
- scratch 清零（TEXPANDS + 分组 TSTORE）
- 候选追加紧凑化（TCMPS EQ/LT + TSELS + TMUL 得 0/1 mask → GM 上
  Hillis-Steele 5 轮前缀和 → MSCATTER_MASK 按 exclusive scan 位移写
  cand；不依赖任何原子操作）

---

## A. gfrun（功能模型）缺失 —— 阻断性

### A1. GM 原子操作 MGATHER_ADD / MSCATTER_ADD 未实现

- 证据：`model/emulator/engine/TMAEngine.cpp` 的 TLSU dispatch 只有
  TMOV/GMOV/TLOAD/TSTORE/TPREFETCH/MGATHER/MSCATTER/MGATHER_MASK/
  MSCATTER_MASK/MGATHER_CAS/VGATHER/VSCATTER，无 *_ADD（TLSU
  function 12/21）。
- 被阻断的环节：
  - 直方图累加 `hist[bin]++`（stage1_scan / round_scan 的 collect=false 路径）；
  - 槽位分配 `hist[bin+1]++` 的先取后增（collect=true 路径的输出位置）。
- 现状绕行：PE 私有 GM 直方图 + 标量核 RMW；单 PE 程序序天然串行化重复
  bin，语义与原子 RMW 一致。这也是 kernel 里剩余标量 lane 循环的根因。
- 解开后的收益：stage1_scan / round_scan 的 lane 循环可整体换成
  MSCATTER_ADD（直方图）+ MGATHER_ADD（槽位分配），msk/idx/off 的标量
  物化随之消失。
- 注：候选追加曾是本项的第三个被阻断环节，现已用前缀和 +
  MSCATTER_MASK  tile 化（见文首清单），紧凑排列的位移本来就要自己算，
  整条链不需要原子。

### A2. TCMPS 的 GPR predicate 载体未实现

- 证据：`TEPLEngine.cpp ExecuteTCMPS` 断言 dst 必须是
  `TileStorageKind::PREDICATE`（PredicateCell 载体）；没有任何
  predicate → GPR 的回读指令。
- 被阻断的环节：伪代码 3.4 的 threshold 查找形式（TCMPS 输出 64-bit
  predicate GPR → 标量 AND → CTZ 取首个 true bit）。
- 现状绕行：tile 上计数 —— TCMPS<GE> 得 PredicateCell，TSELS 物化成
  0/1，TCOLSUM 求和，`thr = count - 1`（hist 单调不增时与原谓词严格
  等价，hist[0] >= rem 保证 count >= 1）。

### A3. tile 寄存器 dtype 标签 + TSTORE 校验，无寄存器内位重解释

- 证据：`AccumulateBlockInfo.cpp ValidateLocalTlsu` 按"最后一次写者"
  的 dtype 校验 TSTORE。
- 被阻断的环节：FP16 位型 → U16 的零成本 reinterpret（sortable-key
  变换的第一步）。
- 现状绕行：GM 往返（TSTORE fp16 → TLOAD u16），每个块声明的 dtype 与
  标签一致。

### A4. 比较/选择块的维度必须等于 tile 静态元数据

- 证据：`AccumulateBlockInfo.cpp ValidateCompareSelectTepl` 要求
  TCMP/TCMPS/TSELS 的块维度（lb0/lb1/lb2）与 src tile 元数据的
  validRow/validCol/col 逐一相等；tile 元数据来自 C++ Tile 类型的静态
  形状，因此 lb1 里放运行时 vc（尾部块屏蔽）在 tail chunk 上必然
  断言。
- 被阻断的环节：用运行时 validRow 让执行器自动屏蔽尾部的写法。
- 现状绕行：所有比较/选择块用静态 32 lane 维度，尾部用显式 keep 谓词
  （`TCMPS<LT>(lane, vc)`）与 eq 谓词各自 TSELS 物化成 0/1 后 TMUL
  合并；尾部 lane 的垃圾 bin 不会进入前缀和与 scatter。

---

## B. 工具链 wrapper 缺失 / 偏差

### B1. VecTileM32 行数上限 32，且 \<32,8\> TLOAD 转置

- 证据：`pto_tile.hpp` `static_assert(CubeM32 Rows <= 32)`；\<32,8\>
  形态的 ND2M32 TLOAD 使 GM 序与 CELL payload 序互为转置。
- 影响：256 行的 M32 分组 tile（cumsum 的 1KB 分组 tile、find_threshold
  的 hist tile）无法用 wrapper tile 表达。
- 现状绕行：裸 `linx_tile_carrier<1024>` + 手写块（照抄反汇编的 wrapper
  形式，ValidRow=256）。S32 \<256,1\> M32 物理字节连续是一切成立的基础。

### B2. MGATHER / MSCATTER_MASK wrapper 仍发旧版 B.IOR

- 证据：wrapper 发非零 stride 的旧版 B.IOR；PR #313（issue #301，未合入）
  约定 IndexTile = 字节位移、B.IOR 仅 BaseGPR。
- 现状绕行：两条指令在 kernel 里手写 asm（`mgather_u32_m32`、
  `mscatter_mask_i32_m32`）。PR 合入后可回到 wrapper。

### B3. TCMPS / TSELS wrapper 不发 CUBE layout selector

- 影响：M32 tile 上走 wrapper 时块里没有 CUBE_M32 DATR。
- 现状：单 cell（\<32,1\>）M32 tile 已验证可直接走 wrapper（U32 单
  cell payload 各布局字节一致，执行器不需要 DATR 寻址）；只有裸 1KB
  分组 tile 仍需手写块显式携带（TCMPS 用 `B.DATR Zero, GE`，TSELS 按
  wrapper 形式无 DATR）。

### B4. TCOLSUM wrapper 的形状断言与分组 tile 不兼容

- 证据：wrapper `static_assert` 要求 dst ValidRow=1、列数/布局与 src
  一致；src 是裸 1KB 分组 tile 时无从谈起。
- 现状绕行：手写 TCOLSUM 块（src = 分组 tile，dst = I32Tile）。

### B5. tile 数组动态下标会降级到栈

- 现象：tile 数组一旦被动态下标访问即 demote 到栈，每次访问变成 1KB
  S64 NORM spill/reload。
- 现状绕行：所有 tile 数组循环 `#pragma clang loop unroll(full)`。

### B6. TCI 只收 RowMajor，且产出的 tile 不带 \<32,1\> tileInfo

- 证据：wrapper `static_assert` 要求 unboxed RowMajor 且 ValidRow==1；
  手写块绕过后，gfrun 执行器又硬编码 validRow=1 只写 validCol 个元素，
  产物的 tileInfo 与 \<32,1\> M32 tile 的静态元数据不符，下游比较块被
  A4 的校验拒收。
- 现状绕行：lane 序列 0..31 在 GM scratch 里标量初始化一次，用cell
  TLOAD 取回（TLOAD 产物 tileInfo 必然一致）；`base + lane` 用一条
  TADDS 得到。

### B7. tile 对象跨非内联函数边界被降级到栈

- 现象：tile 对象经引用参数（或 struct 字段）跨非内联函数边界传递时被
  demote 到栈，CUBE 标签 tile 的 spill 本身非法
  （`RecordRawTileTransport` 断言）。
- 现状绕行：所有经手 tile 的辅助函数 `__attribute__((always_inline))`，
  需要跨阶段的值一律经 GM scratch 物化。

### B8. 控制流依赖的 tile 活区间触发非法 TMOV

- 现象：tile 的活区间若依赖控制流（条件块内定义、循环回边携带），
  regalloc 会在 join/回边插 TLSU TMOV，gfrun 以 Local TMOV legality
  断言拒绝。
- 现状绕行：tile 代码保持直线型，循环全部展开；不在跨 chunk 迭代的
  回边上持有 tile。

---

## C. ISA / 伪代码层缺口（规范本身没有，或伪代码形式有坑）

### C1. REDUCE_FIRST_TRUE 不是 ISA 指令（伪代码已用 CTZ 消除）

- 伪代码 3.4 已把 first-true 改写为当前 ISA 可表达的
  `TCMPS(GPR) + AND + CTZ`，不再使用 `REDUCE_FIRST_TRUE`；CTZ 本身是
  真实存在的标量 ALU 指令（`isa/asl/scalar/alu/CTZ.asl`）。所以规范侧
  无缺口，真正的阻断在 A2：gfrun 没有 TCMPS 的 GPR predicate 载体，
  CTZ 没有可消费的 predicate GPR。现状用计数法（见 A2 绕行）。

### C2. B.SUBVIEW 逻辑 range 操作数无 M32 实现

- 伪代码 cumsum 的 `T112[:,q:q+1]`：B.SUBVIEW 的 parent 必须是 Matrix
  tile；M32 的 range 沿列推进，抽不出"第 q 个 cell"的列向视图。
- 现状绕行：经 GM 物化 —— 1 次分组 TSTORE + 8 次 cell TLOAD 取出 8 个
  cell。机理（ND2M32 源序、为什么必须拆 8 次 TLOAD、stride/绑定都试过）
  见 `../m32_nd2m32_layout_usage.md`。

### C3. 伪代码 3.4 的 \<2,32,2\> 成对 TLOAD 存在转置

- dense TLOAD 的 GM 序是 row*2+col，而 M32 payload 序是 col*32+row，
  两者不一致；即使 A2（GPR 载体）解开，H 的 GM 布局也需要相应调整或
  换装载形式，不能照抄伪代码的 TLOAD。

### C4. fence.d 排序在功能模型中未建模

- 伪代码要求 tile TSTORE → 标量 lw 之间插 fence.d（relaxed load 对
  tile 写的可见性）。gfrun 功能模型按程序序执行，当前代码不需要；
  gfsim 定时模型侧未验证。

---

## D. 已本地修复、未上游

### D1. gfrun ExecuteMSCATTER_MASK 的 CUBE 操作数寻址

- `model/emulator/engine/TMAEngine.cpp`：CUBE 操作数改走
  `CubeCellPayloadIndexForTile`（旧版 dense 读只匹配 RowMajor）。
  **改动在 model 工作树，未提交**；`model/bin/gfrun` 已用
  `python3 build.py build --target gfrun -j8` 重建。
- 没有它：CUBE_M32 的 U8 mask tile 读出错误（U8 cell 元素 (row,0) 在
  字节 row*4），res_check 退回全 -1。

---

## 剩余标量环节 → 阻断项对照

| 标量环节 | 位置 | 阻断项 | 备注 |
|---|---|---|---|
| 直方图累加 `hist[b]++` | stage1_scan / round_scan | A1 | 解开即 tile 化 |
| 槽位分配 `hist[b+1]++` / off·idx·msk 物化 | 同上 collect 路径 | A1 | 解开即 tile 化 |
| 候选追加 `cand[p] = in_idx` | 同上 | 无（已 tile 化） | 前缀和 + MSCATTER_MASK，见文首清单 |
| `any` 检查（msk 的 OR 归约） | scatter_masked | 依附 A1 | lane 循环消失后可用 TCOLSUM |
| `rem -= hist[thr+1]` 的 lw | run() | 无 | 伪代码 3.4 明确保持标量 lw |
| starts/ends clamp、prefix、round 控制 | run() | 无 | 本质标量控制流 |
