#include "guard_common.hpp"
// TileOP-API doc guard: TCONCAT (SFU layout) — concatenate tiles.
// Source: docs/tileop-usage/engines.md 仅分类 layout-and-rearrangement;无签名。
// gfrun earlier hinted "TCONCAT requires 2 source tiles". 按二元 (dst,s0,s1) 猜,
// dst 列数 = 两源之和 (类比 TMRGSORT 契约)。
int main() {
    constexpr int M = 16, N = 8;
    float a[M * N], b[M * N], c[M * (2 * N)];
    gfill_seq(a, M * N); gfill_seq(b, M * N, 100.0f); gzero(c, M * 2 * N);

    iter_t<float, M, N> gA((float *)a), gB((float *)b);
    iter_t<float, M, 2 * N> gC(c);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    vtile_t<float, M, N> tA, tB;
    vtile_t<float, M, 2 * N> tC;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    BENCHSTART;
    TCONCAT(tC, tA, tB);
    BENCHEND;
    TSTORE(gC0, tC);
    return 0;
}
