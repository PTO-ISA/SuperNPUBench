// C 组 —— c2_load_store：TLOAD X → TSTORE X，验 WAR。
//
// 序列：
//
//   TLOAD  tPre  ← gPre        图样 C
//   TSTORE X     ← tPre        X := PRE      预置
//   TLOAD  tRead ← X                         ← 被测的 load，必须看到 PRE
//   TLOAD  tNew  ← gNew        图样 B
//   TSTORE X     ← tNew        X := NEW      ← 被测的 store，不得污染 tRead
//   TSTORE W0    ← tRead
//
// 期望：X 全是 0xB…，W0 全是 0xC…。两边都非零，谁跑到谁前面一目了然。
// 失败签名：
//   W0 = 0xB…   store 越过了 load —— WAR 违例
//   X  = 0xC…   后写的 NEW 丢了
#include "tlsu_bench.hpp"

#define RESULT_SIZE (2 * TLSU_PHYS_BYTES)
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;

alignas(256) constinit auto gPre = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_C);
alignas(256) constinit auto gNew = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_B);

int main()
{
    int32_t *X = TLSU_REGION(0);
    int32_t *W0 = TLSU_REGION(1);

    using gm_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using iter_t = global_iterator<gm_t, tile_t>;

    iter_t gP(gPre.v), gN(gNew.v), gX(X), gW(W0);
    auto gP0 = gP(0, 0), gN0 = gN(0, 0), gX0 = gX(0, 0), gW0 = gW(0, 0);
    tile_t tPre, tNew, tRead;

    BENCHSTART;
    TLOAD(tPre, gP0);
    TSTORE(gX0, tPre);

    TLOAD(tRead, gX0);

    TLOAD(tNew, gN0);
    TSTORE(gX0, tNew);

    TSTORE(gW0, tRead);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
