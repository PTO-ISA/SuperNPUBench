// C 组 —— c3_load_store_load：TLOAD X → TSTORE X → TLOAD X。
//
// c1 和 c2 的合成：同一块 X 上一条 store 夹在两条 load 中间，前一条必须读旧、
// 后一条必须读新。单条 witness 只能证伪一个方向，两条一起才能排除"整条链
// 被整体推迟/整体提前"这类看起来两边都对的退化解。
//
//   TLOAD  tPre ← gPre         图样 C
//   TSTORE X    ← tPre         X := PRE
//   TLOAD  tR0  ← X                        ← 必须看到 PRE
//   TLOAD  tNew ← gNew         图样 B
//   TSTORE X    ← tNew         X := NEW
//   TLOAD  tR1  ← X                        ← 必须看到 NEW
//   TSTORE W0   ← tR0
//   TSTORE W1   ← tR1
//
// 期望：X = 0xB…，W0 = 0xC…，W1 = 0xB…
// 失败签名：
//   W0 = 0xB…             WAR 违例
//   W1 = 0xC… 或 0        RAW 违例
//   W0 == W1              两条 load 塌成了一条（去重/转发做过头）
#include "tlsu_bench.hpp"

#define RESULT_SIZE (3 * TLSU_PHYS_BYTES)
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;

alignas(256) constinit auto gPre = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_C);
alignas(256) constinit auto gNew = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_B);

int main()
{
    int32_t *X = TLSU_REGION(0);
    int32_t *W0 = TLSU_REGION(1);
    int32_t *W1 = TLSU_REGION(2);

    using gm_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using iter_t = global_iterator<gm_t, tile_t>;

    iter_t gP(gPre.v), gN(gNew.v), gX(X), gWa(W0), gWb(W1);
    auto gP0 = gP(0, 0), gN0 = gN(0, 0), gX0 = gX(0, 0);
    auto gW0 = gWa(0, 0), gW1 = gWb(0, 0);
    tile_t tPre, tNew, tR0, tR1;

    BENCHSTART;
    TLOAD(tPre, gP0);
    TSTORE(gX0, tPre);

    TLOAD(tR0, gX0);

    TLOAD(tNew, gN0);
    TSTORE(gX0, tNew);

    TLOAD(tR1, gX0);

    TSTORE(gW0, tR0);
    TSTORE(gW1, tR1);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
