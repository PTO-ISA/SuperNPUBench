// S 组 —— 形状扫描：32x32 int32 的 copy（TLOAD + TSTORE）。
//
// 与 P0 的 8x128 同为 4096 B，但行宽只有 128 B —— 不足一条 L2 cacheline
// (256 B)、正好一个 beat (128 B)。P0 那档每行跨 2 条 cacheline / 4 个 beat，
// 这一档反过来是多行挤在同一条 cacheline 里。两档合起来把 AGU 的行/线切分
// 逻辑两个方向都压到。
#include "tlsu_bench.hpp"

#define RESULT_SIZE TLSU_PHYS_BYTES
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 32;
constexpr int N = 32;

alignas(256) constinit auto gSrc = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_A);

int main()
{
    int32_t *dst = TLSU_REGION(0);

    using gm_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using iter_t = global_iterator<gm_t, tile_t>;

    iter_t gA(gSrc.v), gC(dst);
    auto gA0 = gA(0, 0), gC0 = gC(0, 0);
    tile_t tA;

    BENCHSTART;
    TLOAD(tA, gA0);
    TSTORE(gC0, tA);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
