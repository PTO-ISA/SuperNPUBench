// S 组 —— s3_two_dst：两条独立的 copy，写两个互不相干的目的区。
//
//   TLOAD  tA ← gSrcA (图样 A)    TSTORE R0 ← tA
//   TLOAD  tB ← gSrcB (图样 B)    TSTORE R1 ← tB
//
// 完全没有冲突可言：两条链读不同的源、写不同的目的，中间没有任何重叠。这是
// C 组失败后为了归因而退化出来的对照 —— C 组里 R0 写进去了、R1 全零，但 C 组
// 同时含 RAW/WAR 语义，无法区分"定序错了"和"第二个目的区根本写不进去"。
// 本用例把定序全部拿掉，只留"两个目的区"这一个变量。
//
// R1 若仍为零，问题与冲突无关，是多目的区的 tile store 本身丢了写。
#include "tlsu_bench.hpp"

#define RESULT_SIZE (2 * TLSU_REGION_STRIDE)
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;

alignas(256) constinit auto gSrcA = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_A);
alignas(256) constinit auto gSrcB = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_B);

int main()
{
    int32_t *R0 = TLSU_REGION(0);
    int32_t *R1 = TLSU_REGION(1);

    using gm_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using iter_t = global_iterator<gm_t, tile_t>;

    iter_t gA(gSrcA.v), gB(gSrcB.v), g0(R0), g1(R1);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gR0 = g0(0, 0), gR1 = g1(0, 0);
    tile_t tA, tB;

    BENCHSTART;
    TLOAD(tA, gA0);
    TSTORE(gR0, tA);

    TLOAD(tB, gB0);
    TSTORE(gR1, tB);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
