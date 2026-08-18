// C 组 —— c1_store_load：TSTORE X → TLOAD X，验 RAW。
//
// 序列（X = 结果区第 0 子区，全等于 8x128 int32 一块）：
//
//   TLOAD  tPre  ← gPre        图样 C
//   TSTORE X     ← tPre        X := PRE      预置，好让"没读到新值"和
//                                            "什么都没发生"区分得开
//   TLOAD  tNew  ← gNew        图样 B
//   TSTORE X     ← tNew        X := NEW      ← 被测的 store
//   TLOAD  tRead ← X                         ← 被测的 load，必须看到 NEW
//   TSTORE W0    ← tRead
//
// W0 在序列开始前先被 TSTORE 预填成图样 D。这不是装饰 —— 没有它，"witness
// 那条 store 压根没落地" 和 "落地了但搬的是零" 在 dump 里都表现为全零，两个
// 完全不同的故障分不开。
//
// 期望（由 gfrun 顺序执行给出，这里只说明该怎么读 dump）：
//   X  全是 0xB…  W0 全是 0xB…
// 失败签名：
//   W0 = 0xC…   load 越过了 store，读到预置值 —— RAW 违例
//   W0 = 0xD…   最后那条 TSTORE 没落地（预填还在）
//   W0 = 0      load 读到了从未被写的内存，且这些零确实被写了出去
//   X  = 0xC…   后一条 store 根本没落地，或两条 store 次序反了（WAW）
#include "tlsu_bench.hpp"

#define RESULT_SIZE (2 * TLSU_PHYS_BYTES)
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;

alignas(256) constinit auto gPre = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_C);
alignas(256) constinit auto gNew = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_B);
alignas(256) constinit auto gFill = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_D);

int main()
{
    int32_t *X = TLSU_REGION(0);
    int32_t *W0 = TLSU_REGION(1);

    using gm_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using iter_t = global_iterator<gm_t, tile_t>;

    iter_t gP(gPre.v), gN(gNew.v), gF(gFill.v), gX(X), gW(W0);
    auto gP0 = gP(0, 0), gN0 = gN(0, 0), gF0 = gF(0, 0);
    auto gX0 = gX(0, 0), gW0 = gW(0, 0);
    tile_t tFill, tPre, tNew, tRead;

    BENCHSTART;
    TLOAD(tFill, gF0);
    TSTORE(gW0, tFill);

    TLOAD(tPre, gP0);
    TSTORE(gX0, tPre);

    TLOAD(tNew, gN0);
    TSTORE(gX0, tNew);

    TLOAD(tRead, gX0);
    TSTORE(gW0, tRead);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
