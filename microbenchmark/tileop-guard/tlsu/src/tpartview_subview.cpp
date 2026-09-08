#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TPARTVIEW（推荐高层 B.SUBVIEW 入口）+ region 一元 op。
// range-modifiers-developer-guide.md：推荐通过 TPARTVIEW / TileArray / TASSEMBLY 描述
// parent Tile 的分区与组装；range::subview/assemble 仅作底层兼容/测试接口。TPARTVIEW 把
// 一个 CUBE/Matrix parent 划分为 R×C 个同布局 fragment，槽 SubTileView 被 region 一元 op
// （TABS，pto_region_unary）消费时发出 `BSTART.TEPL … B.SUBVIEW`（接受 IsCubeLayout fragment）。
//
// spec 侧：B.SUBVIEW 只接受 Matrix+CUBE parent（subview-descriptor.asl），故此处用 CUBE parent。
//
// 状态 = run-only（能编译 + 跑通，但精度不 golden）：API 自述 region producer 的 CUBE 路径
// **尚未实现/验证**（range-modifiers.md line 118-123："region producer inline-asm path is
// intentionally limited to RowMajor+NoneBox ... do not use ... row-wise region producers with
// Cube fragments until the Cube binder/CELL ordering path is implemented and validated"）。
// 实测 out == abs(in).T（CUBE cell ordering 未实现的表现），是 API 未完成路径而非规范语义，
// 按 golden 纪律不焊进 oracle。此 demo 看护"TPARTVIEW+region op 的 cube B.SUBVIEW 能编译+
// 跑通不被 ASL 拒"，并见证 cube cell-ordering 尚未落地。CUBE binder 实现后可升级为精度 golden。
constexpr int M = 32, N = 32, NE = M * N;
static float ha[NE], hc[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
#else
    gfill_seq(ha, NE);
#endif
    CubeTileM32<float, M, N> parent;                 // Matrix(Left) + CUBE layout
    global_tensor<float, RowMajor<M, N>> gA(ha);
    global_tensor<float, RowMajor<M, N>> gm(hc);
    TLOAD_CUBE(parent, gA);
    auto tpv = TPARTVIEW<CubeTileM32<float, M, N>, 1, 1>(parent);
    auto slot = tpv[0][0];                            // SubTileView = 合法 B.SUBVIEW 源
    CubeAccumulatorM32<float, M, N> out;
    BENCHSTART;
    TABS(out, slot);                                 // region unary → emits B.SUBVIEW
    BENCHEND;
    TSTORE_CUBE(gm, out);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
#endif
    return 0;
}
