#!/usr/bin/env python3
"""R 组用例生成器：随机长度、随机交错的 TLOAD/TSTORE 序列。

随机的只有**序列**——多少个块、load/store 怎么交错、每一步碰哪个 tile 寄存器
和哪块内存。形状、dtype、行距一律固定（8x128 int32，稠密行距），地址一律
256 字节对齐。这样失败时唯一的变量就是定序与交错，不必再去排除形状或寻址。

判据是 gfrun（三方比对由 SuperScalarModel 的 run_tlsu_compare.sh 驱动），所以
这里不计算期望值。但每个元素仍带自识别编码 0xT_rrr_cccc，dump 里能直接读出
"这一格来自哪份图样的第几行第几列"，缩例时不用反推地址。

生成的 .cpp 要提交进仓库：种子固定，谁都能重现，而且构建不依赖 Python。改序列
就换种子重新生成，把新文件一并提交。

用法:
  python3 gen_random_case.py --seed 20260814 --blocks 200 \\
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

NUM_REGIONS = 16       # 结果缓冲切成几块
NUM_TILES = 8          # 可用的 tile 寄存器变量
SRC_TAGS = ["TLSU_TAG_A", "TLSU_TAG_B", "TLSU_TAG_C", "TLSU_TAG_D"]

HEADER = '''// R 组 —— 随机 TLOAD/TSTORE 序列。**本文件由 gen_random_case.py 生成，不要手改。**
//
//   种子   : {seed}
//   块数   : {blocks}（{loads} 个 TLOAD + {stores} 个 TSTORE）
//   形状   : {rows}x{cols} int32，稠密行距，基址 256 字节对齐
//   重新生成: python3 gen_random_case.py --seed {seed} --blocks {blocks} \\
//                 -o src/{name}.cpp
//
// 随机的只有序列本身：块数、load/store 的交错、每一步用哪个 tile 寄存器、
// 读写哪一块内存。形状、dtype、行距全部固定 —— 失败时唯一的变量就是定序与
// 交错，不必再排除形状或寻址的嫌疑。
//
// 序列里天然会长出 RAW（store 后读同址）、WAR、WAW 和长距离 in-flight 交错，
// 这正是 S/C 两组用固定小序列够不到的地方：C 组每个用例只有两三个 tile op，
// 队列压力浅；这里一次 {blocks} 个块，LIQ/STQ/SCB 会被真正填满。
//
// 判据是 gfrun 的架构内存 dump（三方逐字节比对）。元素带自识别编码
// 0xT_rrr_cccc，dump 里能直接读出来源图样与行列号。
#include "tlsu_bench.hpp"

#define RESULT_SIZE ({regions} * {tile_bytes})
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = {rows};
constexpr int N = {cols};

'''


def gen(seed, blocks, name):
    rng = random.Random(seed)

    # 先规划序列，再落成源码 —— 这样统计量（load/store 各多少）能写进文件头。
    ops = []
    defined = set()          # 已经装载过、可以用作 TSTORE 数据源的 tile
    written = set()          # 已经被写过的区域，可以再被读回来

    for _ in range(blocks):
        # 头几步没有可搬的数据，只能先 load。之后 load/store 各半。
        want_store = bool(defined) and rng.random() < 0.5
        if want_store:
            ti = rng.choice(sorted(defined))
            rj = rng.randrange(NUM_REGIONS)
            ops.append(("store", ti, rj))
            written.add(rj)
        else:
            ti = rng.randrange(NUM_TILES)
            # 一半概率从只读图样装载，一半概率把写过的区域读回来 —— 后者才
            # 制造 RAW，是这组用例的重点。
            if written and rng.random() < 0.5:
                ops.append(("load_region", ti, rng.choice(sorted(written))))
            else:
                ops.append(("load_src", ti, rng.randrange(len(SRC_TAGS))))
            defined.add(ti)

    loads = sum(1 for o in ops if o[0] != "store")
    stores = blocks - loads

    out = [HEADER.format(seed=seed, blocks=blocks, loads=loads, stores=stores,
                         rows=ROWS, cols=COLS, regions=NUM_REGIONS,
                         tile_bytes=TILE_BYTES, name=name)]

    for i, tag in enumerate(SRC_TAGS):
        out.append("alignas(256) constinit auto gSrc{} = "
                   "MakeTlsuPattern<int32_t, M, N>({});\n".format(i, tag))

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

    for i in range(len(SRC_TAGS)):
        out.append("    iter_t gs{0}(gSrc{0}.v);  auto s{0} = gs{0}(0, 0);\n".format(i))
    for r in range(NUM_REGIONS):
        out.append("    iter_t gr{0}(R{0});  auto r{0} = gr{0}(0, 0);\n".format(r))
    out.append("\n")
    out.append("    tile_t " + ", ".join("t{}".format(i) for i in range(NUM_TILES)) + ";\n\n")

    out.append("    BENCHSTART;\n")
    for kind, ti, x in ops:
        if kind == "store":
            out.append("    TSTORE(r{}, t{});\n".format(x, ti))
        elif kind == "load_region":
            out.append("    TLOAD(t{}, r{});\n".format(ti, x))
        else:
            out.append("    TLOAD(t{}, s{});\n".format(ti, x))
    out.append("    BENCHEND;\n\n")

    out.append("    TlsuDrain();\n    tlsu_finish(1);\n    return 0;\n}\n")
    return "".join(out), loads, stores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--blocks", type=int, default=200)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    name = args.output.split("/")[-1].rsplit(".", 1)[0]
    text, loads, stores = gen(args.seed, args.blocks, name)
    with open(args.output, "w") as f:
        f.write(text)
    print("{}: seed={} blocks={} ({} TLOAD + {} TSTORE) -> {}"
          .format(name, args.seed, args.blocks, loads, stores, args.output))


if __name__ == "__main__":
    main()
