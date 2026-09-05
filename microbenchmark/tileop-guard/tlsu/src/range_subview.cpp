#include "guard_common.hpp"
// TileOP-API guard: range::Subview — source-side range carrier over TSTORE.
// ROOT CAUSE (recovered from model Block.cpp:1052 HandleBSubview): the Subview
// parent MUST be a CUBE-layout tile — the model asserts IsCubeLayout(parent) and
// derives the sub-region via CubeCellDescribeSubview. This Vec/RowMajor parent
// compiles (the Subview header does NOT restrict the parent) but is rejected at
// runtime: "illegal TSTORE operand or descriptor contract". docs/range-modifiers.md
// shows a Vec-tile example and does NOT state the cube-parent requirement — a doc
// gap, same "persistent cube/Matrix source" family as TEXTRACT/TIMG2COL.
// A correct cube demo (CubeTileM32 + TLOAD_CUBE + Subview + store) additionally
// hits further undocumented cube contracts — TSTORE_CUBE requires GM/CUBE dtype
// match AND its const& parameter is incompatible with Subview::data() (non-const)
// — so a full cube-subview demo needs deeper cube support / owner clarification.
// Left as a Vec witness that pinpoints the cube-parent requirement.
int main() {
    constexpr int M = 4, N = 8;
    float ha[M * N], hc[M * N];
    gfill_seq(ha, M * N);
    gzero(hc, M * N);

    using Src = vtile_t<float, M, N>;             // Vec/RowMajor — NOT cube; model rejects
    Src s;
    iter_t<float, M, N> gA(ha);
    auto gA0 = gA(0, 0);
    global_tensor<float, RowMajor<M, N>> gm(hc);

    TLOAD(s, gA0);
    BENCHSTART;
    auto sv = range::Subview<Src, 1, /*Off*/ 0, /*RegSrc*/ 0>(s, 0);
    TSTORE(gm, sv);   // gfrun: needs IsCubeLayout(parent) -> descriptor contract
    BENCHEND;
    return 0;
}
