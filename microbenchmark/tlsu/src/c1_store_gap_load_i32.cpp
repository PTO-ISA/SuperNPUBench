// C 组归因用例 —— c1 的唯一变体：在 TSTORE X 和 TLOAD X 之间插一段长延时。
//
// c1 失败后有两种解释，架构内存这个判据本身分不开：
//
//   (甲) 定序错了 —— store 的数据其实会到，只是还没到 load 就放行了，
//        hazard 检查该拦没拦。
//   (乙) 可见性错了 —— tile store 写下去的数据根本不进入 tile load 能读到
//        的那份状态，等多久都没用。
//
// 两者预测的 c1 dump 完全一样。插入延时就能把它们分开：延时后读对了是 (甲)，
// 仍然读零是 (乙)。(乙) 比 (甲) 严重得多 —— 那不是定序问题，是数据通路断了。
//
// 除这一段延时外，与 c1_store_load_i32 逐字相同。
//
// 延时长度由 -DTLSU_GAP=<n> 给出（独立于收尾的 TLSU_DRAIN，否则把 GAP 调到 0
// 会连收尾排空一起去掉，两个变量搅在一起）。扫这个值就能量出"多久之后 store
// 才对后续 tile load 可见"这个窗口有多宽。
#ifndef TLSU_GAP
#define TLSU_GAP 20000
#endif

#include "tlsu_bench.hpp"

static inline void TlsuGap()
{
    static volatile int gapSpin;
    for (int k = 0; k < TLSU_GAP; ++k) {
        gapSpin = gapSpin + 1;
    }
}

#define RESULT_SIZE (2 * TLSU_REGION_STRIDE)
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
    BENCHEND;

    TlsuGap();   // ← 与 c1 的唯一差别

    BENCHSTART;
    TLOAD(tRead, gX0);
    TSTORE(gW0, tRead);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
