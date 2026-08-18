#!/usr/bin/env python3
"""R 组用例生成器：随机长度、随机交错的 TLOAD/TSTORE 序列。

序列分两段：

  1. 初始化   把 16 份**互不相同**的图样各装进一个区，工作区从此自带身份
  2. 随机体   只在区与区之间搬运，不再读只读图样

第 2 点是这版的关键。早先随机体有一半概率去读只读图样，那一半 TLOAD 读的是
永远不会被写的地址，天然不会与任何 store 冲突。现在每次 TLOAD 读的都是可能刚
被写过的区，RAW/WAR/WAW 的密度显著上来了。

代价是"错读了别的区"必须靠内容区分，所以初始化时每个区装的图样都不一样（区号
编进元素值的 [27:24]，见 tlsu_bench.hpp 的 TlsuWsTag）——16 个区两两不同，错读
任何一个区都会在 dump 里现形，不会被同内容掩盖。

随机的只有**序列**——load/store 怎么交错、每一步碰哪个 tile 寄存器和哪块内存。
形状、dtype、行距一律固定（8x128 int32，稠密行距），地址一律 256 字节对齐。这样
失败时唯一的变量就是定序与交错，不必再去排除形状或寻址。

判据有两条：gfrun 三方比对（由 SuperScalarModel 的 run_tlsu_compare.sh 驱动），
以及 viz_random_case.py --check 的独立推导（不碰任何模型）。

生成的 .cpp 要提交进仓库：种子固定，谁都能重现，而且构建不依赖 Python。改序列
就换种子重新生成，把新文件一并提交。

用法:
  python3 gen_random_case.py --seed 20260818 --blocks 200 \\
      -o src/r1_random_seq_i32.cpp
"""
import argparse
import random

# 形状固定。8x128 int32 = 4096 B，正好是 256 的整数倍，所以按 tile 紧排的
# 区域天然满足 256 字节对齐的基址要求。
ROWS = 8
COLS = 128
ELEM_BYTES = 4
TILE_BYTES = ROWS * COLS * ELEM_BYTES      # 4096

NUM_REGIONS = 16       # 工作区切成几块；区号编进图样，上限 16（编码 [27:24]）
NUM_TILES = 8          # 可用的 tile 寄存器变量

HEADER = '''// R 组 —— 随机 TLOAD/TSTORE 序列。**本文件由 gen_random_case.py 生成，不要手改。**
//
//   种子   : {seed}
//   初始化 : {init_blocks} 块（{regions} 个 TLOAD + {regions} 个 TSTORE），每个区装一份唯一图样
//   随机体 : {blocks} 块（{loads} 个 TLOAD + {stores} 个 TSTORE），只在区与区之间搬运
//   形状   : {rows}x{cols} int32，稠密行距，基址 256 字节对齐
//   重新生成: python3 gen_random_case.py --seed {seed} --blocks {blocks} \\
//                 -o src/{name}.cpp
//
// 随机的只有序列本身：load/store 的交错、每一步用哪个 tile 寄存器、读写哪一块
// 内存。形状、dtype、行距全部固定 —— 失败时唯一的变量就是定序与交错，不必再
// 排除形状或寻址的嫌疑。
//
// 随机体不读只读图样：初始化之后 {regions} 个区全部有内容，之后每一次 TLOAD 读的
// 都是可能刚被写过的区。序列里因此密集地长出 RAW（store 后读同址）、WAR、WAW
// 和长距离 in-flight 交错 —— 这正是 S/C 两组用固定小序列够不到的地方：C 组每个
// 用例只有两三个 tile op，队列压力浅；这里一次 {blocks} 个块，LIQ/STQ/SCB 会被真正
// 填满。
//
// 每个区的初始图样互不相同（区号在元素值的 [27:24]），所以"错读了别的区"不会被
// 同内容掩盖 —— 任何一次搬错源都在 dump 里可见。元素编码见 tlsu_bench.hpp：
//   0xT_R_rr_cccc = 图样 tag、区号、行号、列号，hex dump 按 nibble 直读。
#include "tlsu_bench.hpp"

#define RESULT_SIZE ({regions} * {tile_bytes})
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = {rows};
constexpr int N = {cols};

'''


def gen(seed, blocks, name):
    rng = random.Random(seed)

    # 初始化：区 r 装图样 W{r}。tile 变量轮着用，用完这一段 t0..t7 都已定义，
    # 随机体因此第一步就可以是 store。
    init = [(r % NUM_TILES, r) for r in range(NUM_REGIONS)]

    # 随机体：load 与 store 各半，两侧都只碰工作区。
    ops = []
    for _ in range(blocks):
        if rng.random() < 0.5:
            ops.append(("store", rng.randrange(NUM_TILES), rng.randrange(NUM_REGIONS)))
        else:
            ops.append(("load", rng.randrange(NUM_TILES), rng.randrange(NUM_REGIONS)))

    loads = sum(1 for o in ops if o[0] == "load")
    stores = blocks - loads

    out = [HEADER.format(seed=seed, blocks=blocks, loads=loads, stores=stores,
                         rows=ROWS, cols=COLS, regions=NUM_REGIONS,
                         init_blocks=2 * NUM_REGIONS,
                         tile_bytes=TILE_BYTES, name=name)]

    out.append("// 每个区一份唯一的初始图样：区号编进元素值，错读别的区在 dump 里可见。\n")
    for r in range(NUM_REGIONS):
        out.append("alignas(256) constinit auto gWs{0} = "
                   "MakeTlsuWsPattern<int32_t, M, N>({0});\n".format(r))

    out.append("""
int main()
{
    using gm_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using iter_t = global_iterator<gm_t, tile_t>;

""")

    out.append("    // 区域基址按 tile 紧排；TILE_BYTES 是 256 的整数倍，\n"
               "    // 所以每个区域的基址天然 256 字节对齐。\n")
    for r in range(NUM_REGIONS):
        out.append("    int32_t *R{} = (int32_t *)(cross_model_result + {} * {});\n"
                   .format(r, r, TILE_BYTES))
    out.append("\n")

    for r in range(NUM_REGIONS):
        out.append("    iter_t gw{0}(gWs{0}.v);  auto w{0} = gw{0}(0, 0);\n".format(r))
    for r in range(NUM_REGIONS):
        out.append("    iter_t gr{0}(R{0});  auto r{0} = gr{0}(0, 0);\n".format(r))
    out.append("\n")
    out.append("    tile_t " + ", ".join("t{}".format(i) for i in range(NUM_TILES)) + ";\n\n")

    out.append("    BENCHSTART;\n")
    out.append("    // ── 初始化：每个区装入自己的那份图样 ──\n")
    for ti, r in init:
        out.append("    TLOAD(t{}, w{});\n".format(ti, r))
        out.append("    TSTORE(r{}, t{});\n".format(r, ti))
    out.append("\n    // ── 随机体：只在区与区之间搬运 ──\n")
    for kind, ti, rj in ops:
        if kind == "store":
            out.append("    TSTORE(r{}, t{});\n".format(rj, ti))
        else:
            out.append("    TLOAD(t{}, r{});\n".format(ti, rj))
    out.append("    BENCHEND;\n\n")

    out.append("    TlsuDrain();\n    tlsu_finish(1);\n    return 0;\n}\n")
    return "".join(out), loads, stores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--blocks", type=int, default=200,
                    help="随机体的块数；初始化的 2*NUM_REGIONS 块不计在内")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    name = args.output.split("/")[-1].rsplit(".", 1)[0]
    text, loads, stores = gen(args.seed, args.blocks, name)
    with open(args.output, "w") as f:
        f.write(text)
    print("{}: seed={} init={} blocks={} ({} TLOAD + {} TSTORE) -> {}"
          .format(name, args.seed, 2 * NUM_REGIONS, args.blocks,
                  loads, stores, args.output))


if __name__ == "__main__":
    main()
