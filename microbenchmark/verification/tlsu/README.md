# TLSU 功能验证用例

**这不是 benchmark。** 本目录下的用例只做 TLSU（`BSTART.TLSU`）搬运语义的正确性
验证：产生激励、把结果区 dump 出来、用**不依赖被测模型**的判据判对错，并在不同
模型/工具链之间做差分。它们不测 cycle，不计入 microbench 的用例统计，也不被
`microbenchmark/compile_all.sh` 编译。

指令级的 TLSU **性能**微基准在 `microbenchmark/memory/`（25 个用例），与本目录无关。

## 用例分组

| 组 | 用例 | 考察点 |
| --- | --- | --- |
| S | `s1_copy_i32_32x32` / `s2_strided_i32_8x128` / `s2b_stride_elem_i32_8x128` / `s3_two_dst_i32` | 单条搬运的形状、stride（按元素表达）、多目的地 |
| C | `c1_store_load_i32` / `c2_load_store_i32` / `c3_load_store_load_i32` | 短序列下的 RAW / WAR 可见性 |
| R | `r1_random_seq_i32` | 先把 16 个区各装一份唯一图样，再跑 200 块随机 TLOAD/TSTORE —— 全程只在区与区之间搬运，填满 LIQ/STQ/SCB |
| — | `copy_i32_8x128` | 最初的端到端 copy 骨架 |

公共骨架：`tlsu_bench.hpp`（自识别图样、区域划分）、`tlsu_finish.h`（结果缓冲符号
与收尾 store）。

自识别编码贯穿全部用例与工具。S/C 组用的是基础形式：

```
元素值 = (tag << 28) | ((row + 1) << 16) | (col + 1)
```

行列都从 1 起，所以整个 32 位为 0 恒表示"从未被写"。tag：`A` 主图样 / `B` 覆写 /
`C` 预置 / `D` witness 预填 / `F` 越界填充（出现在结果里即为过读）。

R 组的 workspace 初始图样多带一个区号（`TlsuWsTag`）：

```
元素值 = (0x1 << 28) | (rid << 24) | ((row + 1) << 16) | (col + 1)
```

区号占的是编码里空闲的 `[27:24]` —— 有效行 `r+1 ≤ 8`，即便 pad 行也只到 32，占到
`[21:16]` 为止，不会与区号相碰。hex dump 按 nibble 直读：`0x15010001` = 图样 W、
区 5、第 1 行、第 1 列。越界填充仍是 `TLSU_TAG_PAD`，所以"过读"与"错读了别的区"
是两种互不混淆的签名。

## 构建

```bash
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
cd microbenchmark/verification/tlsu
make TESTCASE=s1_copy_i32_32x32
```

产物：`output/microbenchmark/verification/elf/verification/<case>.elf`。

本目录嵌在 `verification/` 下一层，`Makefile.common` 由路径推导出的 `CATEGORY` 是
`verification`，`-I$(MICROBENCH_ROOT)/$(CATEGORY)` 只到上一级，所以本地 `Makefile`
补了一条 `INCLUDE += -I$(CURDIR)`，否则 `src/*.cpp` 找不到 `tlsu_bench.hpp`。

## 运行与判定

ELF 跑在 SuperScalarModel 的 `gfrun`/`gfsim` 上。结果缓冲是 `cross_model_result`，
尺寸由绝对符号 `cross_model_result_size` 给出（`run_diff.py` 取的是符号的 `st_value`
而非内容）。跨模型比对由 SuperScalarModel 的 `scripts/run_tlsu_compare.sh` 驱动，
dump 出的就是这块区域的架构内存。

R 组另有一套**脱离模型**的判据，见下面的 `--check`。

---

## R 组生成器：`gen_random_case.py`

生成的序列分两段：

1. **初始化**（32 块 = 16 TLOAD + 16 TSTORE）把 16 份**互不相同**的图样各装进一个区
2. **随机体**（`--blocks` 块）只在区与区之间搬运，**不再读只读图样**

第 2 点是这版的关键。早先随机体有一半概率去读只读图样，那一半 TLOAD 读的是永远不
会被写的地址，天然不会与任何 store 冲突。现在每次 TLOAD 读的都是可能刚被写过的区，
RAW/WAR/WAW 的密度显著上来了。

代价是"错读了别的区"必须靠内容区分，所以初始化时每个区装的图样都不一样（区号编进
元素值的 `[27:24]`）——16 个区两两不同，错读任何一个区都会在 dump 里现形。

随机的只有**序列**——load/store 怎么交错、每一步碰哪个 tile 寄存器和哪块内存。形状、
dtype、行距、对齐一律固定，失败时唯一的变量就是定序与交错，不必再去排除形状或寻址
的嫌疑。

### 命令行参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--seed` | `20260818` | 随机种子；换种子即换一条序列 |
| `--blocks` | `200` | **随机体**的块数；初始化的 32 块不计在内 |
| `-o, --output` | 必填 | 输出的 .cpp 路径；文件名同时用于文件头里的"重新生成"命令 |

### 仓库里这份用例实际用的参数

```bash
python3 gen_random_case.py --seed 20260818 --blocks 200 -o src/r1_random_seq_i32.cpp
# r1_random_seq_i32: seed=20260818 init=32 blocks=200 (103 TLOAD + 97 TSTORE) -> src/r1_random_seq_i32.cpp
```

即 `src/r1_random_seq_i32.cpp` = 种子 `20260818` + 随机体 200 块（**103 TLOAD +
97 TSTORE**），加上固定的 32 块初始化，序列共 **232 块**。同样的参数复跑必然得到逐
字节相同的文件（`random.Random(seed)` 固定序列），这几个数字也原样写在生成文件的
文件头里。

### 脚本内写死、命令行不暴露的量

改这些要直接改 `gen_random_case.py` 顶部的常量：

| 常量 | 值 | 说明 |
| --- | --- | --- |
| `ROWS` × `COLS` | 8 × 128 | tile 形状 |
| `ELEM_BYTES` | 4 | int32 |
| `TILE_BYTES` | 4096 | `8×128×4`，是 256 的整数倍，所以按 tile 紧排的区域基址天然满足 256 字节对齐 |
| `NUM_REGIONS` | 16 | 工作区切成 16 块 → `RESULT_SIZE = 16 × 4096 = 65536` 字节。同时是初始图样的份数，上限 16（区号只有 4 bit） |
| `NUM_TILES` | 8 | 源码里可用的 tile 寄存器变量 `t0..t7` |

### 序列的随机策略

- 初始化段是确定的：区 `r` 装图样 `W{r}`，tile 变量轮着用。走完这 32 块 `t0..t7`
  全部已定义、16 个区全部有内容，所以随机体第一步就可以是 store。
- 随机体里 TSTORE 与 TLOAD 各 50% 概率，tile 与区都在全集里均匀取 —— 两侧都只碰
  工作区。
- 序列里因此密集地长出 RAW（store 后读同址）、WAR、WAW 和长距离 in-flight 交错。
  C 组每个用例只有两三个 tile op，队列压力浅；这里一次 200 块，LIQ/STQ/SCB 会被真正
  填满。

**已知的代价：图样多样性会随序列长度塌缩。** 随机体只在工作区内部搬运，每次 TSTORE
都把某个区的内容复制到另一个区，存活的源图样种类只减不增。实测这条序列：

| 走到第几块 | 32（初始化完） | 75 | 100 | 150 | 232（末尾） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 个区里的不同图样数 | 16 | 9 | 6 | 5 | 3 |

判据只看**终态**，所以序列越靠后发生的"搬错源区"越容易被同内容掩盖。这是"只在工作区
内搬运"这个设定的固有代价，不是 bug。想要靠前段的高分辨力，就把 `--blocks` 调小多跑
几条种子，而不是把单条序列拉长。

改前后的密度对比（同为 200 块随机体）：

| | 读只读图样的 TLOAD | 近距 RAW（≤16 块内读回刚写的区） | 区域→区域搬运 |
| --- | ---: | ---: | ---: |
| 旧（混读只读图样） | 46 | 28（占 TLOAD 31%） | 49 |
| 新（只读工作区） | 16（仅初始化） | 42（占 TLOAD 35%） | 86 |

### 提交约定

生成的 `.cpp` **要提交进仓库**：种子固定、谁都能重现，而且构建不依赖 Python。改序列
就换种子重新生成，把新文件一并提交；文件头带 `不要手改` 标记，别手工编辑它。

---

## 可视化与独立判据：`viz_random_case.py`

同一套"重放序列 → 推导每个区最终装什么"的逻辑，两种出口：`-o` 出可视化，`--check`
出判定。两者读的都只是用例 `.cpp` 源码，不碰任何模型。

### 生成可视化

```bash
python3 viz_random_case.py src/r1_random_seq_i32.cpp -o r1_flow.html
# src/r1_random_seq_i32.cpp: 232 步 / 42 个近距 RAW（≤16 块）/ 86 次区域→区域搬运 -> r1_flow.html
```

参数：位置参数是生成的用例 `.cpp`，`-o/--output` 是输出 html。`-o` 与 `--check` 至少
给一个。

统计行的三个数：总步数（含 32 块初始化）、**近距 RAW** 次数、区域→区域搬运次数。

近距 RAW 的口径值得说明：初始化之后每个区都有内容，"读回写过的区"恒真、不再有区分
度，所以这里统计的是**距离** —— 上一次写这个区之后隔了不超过 `RAW_WINDOW`（默认 16）
块就读回来。这只是个看图用的启发式，不参与判据。

用浏览器打开 html，逐步看数据从哪搬到哪、GM 各区当前应装什么：

| 控件 | 作用 |
| --- | --- |
| `⏮ ◀ ▶ ⏭` | 首步 / 上一步 / 下一步 / 末步 |
| `▶ 播放` | 自动播放，`速度` 按钮在 1× / 2× / 4× 之间循环 |
| `下一次 GM 变化` | 跳到下一条改写 GM 的 TSTORE |
| `下一个近距 RAW` | 跳到下一处"刚写完就读回" |
| 键盘 | `←` `→` 单步，空格播放/暂停 |

页面按主题自适应明暗，颜色区分 TLOAD / TSTORE 与数据的搬运深度；未被写过的区域显示
为全零底色。生成的 `*.html` 已 gitignore —— 随时可重新生成，不必提交。

### 独立判据校验 dump

```bash
python3 viz_random_case.py src/r1_random_seq_i32.cpp --check dump.bin
# PASS  dump.bin 与独立推导的期望值逐字节一致（65536 字节）
# FAIL  首处不一致 @字节 12344（R3 第 0 行 第 14 列）期望 d001000f 实际 d001ff0f
```

`dump.bin` 必须正好是 `cross_model_result` 这 **65536 字节**（16 区 × 4096），长度不符
会直接报错退出。退出码：一致 `0`，不一致或长度不符 `1`，可以直接接进 CI。

**为什么要有它**：端到端比对一直以 gfrun 为黄金参考，而 gfrun 与 gfsim **共用 `isa/`
解码器** —— 共用层错了两边一起错，逐字节比对照样报 IDENTICAL（2026-08-14 的 ADDTPC
回归就是这么把整套用例伪装成全绿的）。`--check` 的期望值只依赖用例源码与自识别编码，
因此能独立判死。

推导之所以成立：每次搬运都是整块 8×128 的 1:1 拷贝，所以一个区最终装什么，完全由
"最后写入的数据源自哪份图样"决定 —— 只要跟踪图样 id 的流向，就能把 65536 字节全部
算出来。16 份初始图样两两不同，所以搬错了源区一定会在期望值里现形，不会被同内容掩盖。

区数不写死在脚本里：`--check` 从用例源码的 `#define RESULT_SIZE (N * 4096)` 读出 `N`，
改了生成器的 `NUM_REGIONS` 这边自动对上。
