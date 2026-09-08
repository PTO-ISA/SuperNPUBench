#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: range::subview — source-side range carrier over TSTORE
// （低层 subview 路径看护；CUBE 路径由 tpartview_subview.cpp 另行看护）。
//
// 写法本身合规：range-modifiers.md 状态清单列 "Implemented: Local source Subview
// over TSTORE"，developer-guide 的 store_subview 示例即 `TSTORE(gm, range::subview(t))`。
//
// 但这是 **fail-closed witness**（API↔spec 缺口，非模型 bug、非写法错）：
//  - API：`range::subview`+TSTORE 路径带 requires(!IsCubeLayout)，只对 **RowMajor**
//    parent 发 B.SUBVIEW（本 demo 即此形式，Vec/RowMajor parent）。
//  - 模型：`subview-descriptor.asl::BundleCubeSubviewDescriptorOf` 要求 parent
//    location==Matrix && CUBE layout，非 CUBE 一律 Fault_TileLegality（ASL-correct）。
//  → RowMajor subview 被模型正确拒（gfrun: illegal TSTORE operand or descriptor
//    contract）。API 的 CUBE subview binder 尚未实现（range-modifiers.md line 118-123
//    自述 "region producer ... intentionally limited to RowMajor+NoneBox ... until the
//    Cube binder/CELL ordering path is implemented and validated"）。
//  归属 = Linx-TileOP-API 未实现 cube subview；run-fail 作缺口 witness，无 golden。
constexpr int M = 4, N = 8, NE = M * N;
static float ha[NE], hc[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", ha, sizeof(ha));
#else
    gfill_seq(ha, NE);
#endif
    using Src = vtile_t<float, M, N>;
    Src s;
    iter_t<float, M, N> gA(ha);
    auto gA0 = gA(0, 0);
    global_tensor<float, RowMajor<M, N>> gm(hc);
    TLOAD(s, gA0);
    BENCHSTART;
    auto sv = range::subview(s);       // lowercase helper
    TSTORE(gm, sv);
    BENCHEND;
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", hc, sizeof(hc));
#endif
    return 0;
}
