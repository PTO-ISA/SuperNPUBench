#include "guard_common.hpp"
// TileOP-API doc guard: TTRANS (SFU layout) — transpose.
// Source: docs/tileop-usage/engines.md 仅一行分类 layout-and-rearrangement;
//   layout.md 提及 TTRANS 是 SFU op 但 NOT 给签名。
// NOTE(doc-gap): 无签名/无示例。按转置语义猜 (dst, src);dst 形状取 N x M。
int main() {
    constexpr int M = 16, N = 16, NE = M * N;   // 方阵，避免 N x M 形状纠纷
    float a[NE], c[NE];
    gfill_seq(a, NE); gzero(c, NE);
    BENCHSTART;
    g_unary<float, M, N>(c, a, [](auto& d, auto& s){ TTRANS(d, s); });
    BENCHEND;
    return 0;
}
