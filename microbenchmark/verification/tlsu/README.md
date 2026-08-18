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
| R | `r1_random_seq_i32` | 200 块随机 TLOAD/TSTORE 序列，16 份两两不同的只读图样，填满 LIQ/STQ/SCB |
| — | `copy_i32_8x128` | 最初的端到端 copy 骨架 |

公共骨架：`tlsu_bench.hpp`（自识别图样、区域划分）、`tlsu_finish.h`（结果缓冲符号
与收尾 store）。

自识别编码贯穿全部用例与工具。S/C 组用的是基础形式：

```
元素值 = (tag << 28) | ((row + 1) << 16) | (col + 1)
```

行列都从 1 起，所以整个 32 位为 0 恒表示"从未被写"。tag：`A` 主图样 / `B` 覆写 /
`C` 预置 / `D` witness 预填 / `F` 越界填充（出现在结果里即为过读）。

R 组要 16 份两两不同的只读图样，4 bit 的 tag 不够用，于是多带一个图样编号
（`TlsuIdTag`）：

```
元素值 = (0x1 << 28) | (pid << 24) | ((row + 1) << 16) | (col + 1)
```

编号占的是编码里空闲的 `[27:24]` —— 有效行 `r+1 ≤ 8`，即便 pad 行也只到 32，占到
`[21:16]` 为止，不会与编号相碰。hex dump 按 nibble 直读：`0x15010001` = 编号图样 5、
第 1 行、第 1 列。越界填充仍是 `TLSU_TAG_PAD`，所以"过读"与"搬错了源"是两种互不
混淆的签名。

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

TLOAD 一半概率从只读图样装载，一半概率把写过的区读回来 —— 后者制造 RAW，是这组
用例的重点；前者则不断把新图样注入工作区。

只读图样有 **16 份且两两不同**，这一点有两层作用：

1. **可分辨**：图样只有 4 份时，"搬错了源"有 1/4 概率被同内容掩盖，判据看不见。
2. **不塌缩**：TSTORE 只复制不创造，区与区之间搬得越久、存活的图样种类越少（就是
   遗传漂变）。src 装载持续把新编号注回工作区，序列再长也不会收敛成少数几种内容。

第 2 点是长序列能不能用的关键。实测同为 200 块（只统计已写过的区）：

| 走到第几块 | 50 | 100 | 150 | 200 |
| --- | ---: | ---: | ---: | ---: |
| 4 份源：不同图样数 / 错读被掩盖的概率 | 2 种 / 70% | 4 种 / 22% | 3 种 / 46% | 4 种 / 28% |
| 16 份源：不同图样数 / 错读被掩盖的概率 | 4 种 / 31% | 6 种 / 14% | 6 种 / 18% | 6 种 / **13%** |

随机的只有**序列**——多少个块、load/store 怎么交错、每一步碰哪个 tile 寄存器和哪块
内存。形状、dtype、行距、对齐一律固定，失败时唯一的变量就是定序与交错，不必再去
排除形状或寻址的嫌疑。

### 命令行参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--seed` | `20260818` | 随机种子；换种子即换一条序列 |
| `--blocks` | `200` | tile op 总块数（TLOAD + TSTORE 之和） |
| `-o, --output` | 必填 | 输出的 .cpp 路径；文件名同时用于文件头里的"重新生成"命令 |

### 仓库里这份用例实际用的参数

```bash
python3 gen_random_case.py --seed 20260818 --blocks 200 -o src/r1_random_seq_i32.cpp
# r1_random_seq_i32: seed=20260818 blocks=200 (99 TLOAD + 101 TSTORE) srcs=16 -> src/r1_random_seq_i32.cpp
```

即 `src/r1_random_seq_i32.cpp` = 种子 `20260818` + 200 块，展开成 **99 个 TLOAD +
101 个 TSTORE**。同样的参数复跑必然得到逐字节相同的文件（`random.Random(seed)` 固定
序列），这几个数字也原样写在生成文件的文件头里。

### 脚本内写死、命令行不暴露的量

改这些要直接改 `gen_random_case.py` 顶部的常量：

| 常量 | 值 | 说明 |
| --- | --- | --- |
| `ROWS` × `COLS` | 8 × 128 | tile 形状 |
| `ELEM_BYTES` | 4 | int32 |
| `TILE_BYTES` | 4096 | `8×128×4`，是 256 的整数倍，所以按 tile 紧排的区域基址天然满足 256 字节对齐 |
| `NUM_REGIONS` | 16 | 结果缓冲切成 16 块 → `RESULT_SIZE = 16 × 4096 = 65536` 字节 |
| `NUM_TILES` | 8 | 源码里可用的 tile 寄存器变量 `t0..t7` |
| `NUM_SRCS` | 16 | 只读图样份数 `gSrc0..gSrc15`，两两不同；上限 16（编号只有 `[27:24]` 四位） |

### 序列的随机策略

- 头几步没有可搬的数据，只能先 TLOAD；之后 TSTORE 与 TLOAD 各 50% 概率。
- TLOAD 时若已有区域被写过，则 50% 概率**读回写过的区域**，否则从只读图样装载。
  前者制造 RAW，是 R 组的重点；后者把新图样注入工作区，维持可分辨性。
- 序列里因此天然长出 RAW（store 后读同址）、WAR、WAW 和长距离 in-flight 交错。C 组
  每个用例只有两三个 tile op，队列压力浅；这里一次 200 块，LIQ/STQ/SCB 会被真正填满。

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
# src/r1_random_seq_i32.cpp: 200 步 / 27 个近距 RAW（≤16 块）/ 45 次区域→区域搬运 -> r1_flow.html
```

参数：位置参数是生成的用例 `.cpp`，`-o/--output` 是输出 html。`-o` 与 `--check` 至少
给一个。

统计行的三个数：总步数、**近距 RAW** 次数、区域→区域搬运次数（数据上一跳来自另一个
区域，而非只读图样）。

近距 RAW 的口径：光是"读回写过的区"只说明有依赖、不说明压力，所以这里统计的是
**距离** —— 上一次写这个区之后隔了不超过 `RAW_WINDOW`（默认 16）块就读回来。这只是
个看图用的启发式，不参与判据。

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
"最后写入的数据源自哪份图样"决定 —— 只要跟踪图样编号的流向，就能把 65536 字节全部
算出来。16 份只读图样两两不同，所以搬错了源一定会在期望值里现形，不会被同内容掩盖。

区数不写死在脚本里：`--check` 从用例源码的 `#define RESULT_SIZE (N * 4096)` 读出 `N`，
改了生成器的 `NUM_REGIONS` 这边自动对上。
