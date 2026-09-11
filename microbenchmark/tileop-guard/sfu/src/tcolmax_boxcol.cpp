#include "guard_common.hpp"
#include "guard_io.h"
// 场景(新 API 问题): TSTORE 的源 tile = colReduce 的直接输出, 该 tile valid col < physical col
// (列 boxed, 如 physical 64 / valid 40), 直接 TSTORE 它, 中间不插任何 elementwise op
// (不经 TCVT/TABS/TSEL/COPY)。涉及列归约族 TCOLMAX/SUM/MIN/PROD/COL{ARG}MAX/MIN。
//
// 约束: TCOLMAX API static_assert(template_asm.hpp:13769) 要求 dst.Cols==src.Cols 且
// dst.ValidCol==src.ValidCol, 故源与目的同为 physical 64 / valid 40。
// 现象: 编译通过; gfrun 直接 TSTORE 该 boxed-col colReduce 输出时崩:
//   ASSERTION: IsLegalLocalTileDescriptor(srcs[1]) "Local TSTORE requires one legal source Tile descriptor"
// 对照(已验证): 非 boxed(64/64) 直接 TSTORE PASS; boxed 输出插 TABS 也崩(TABS 亦拒该 boxed-col 源)。
// run-fail witness(崩在 TSTORE, 无可校输出)。
constexpr int M = 16, PN = 64, VN = 40;   // physical col 64, valid col 40 (boxed)
static float a[M * PN], c[1 * PN];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", a, sizeof(a));
#else
    for (int i = 0; i < M * PN; ++i) a[i] = (float)(i % 97) * 0.5f;
#endif
    using SrcTile = Tile<Location::Vec, float, M, PN, BLayout::RowMajor, M, VN>;  // 物理 M×64 valid M×40
    using OutTile = Tile<Location::Vec, float, 1, PN, BLayout::RowMajor, 1, VN>;  // 物理 1×64 valid 1×40 (boxed)
    iter_t<float, M, PN> gA(a);
    global_iterator<gm_t<float, 1, PN>, OutTile> gC(c);
    auto a0 = gA(0, 0);
    auto c0 = gC(0, 0);
    SrcTile tA;
    OutTile tC;
    TLOAD(tA, a0);
    TCOLMAX(tC, tA);          // colReduce 直接输出 (boxed col: valid 40 < physical 64)
    TSTORE(c0, tC);           // 直接 TSTORE, 无中间 elementwise op → gfrun 崩
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", c, sizeof(c));
#endif
    return 0;
}
