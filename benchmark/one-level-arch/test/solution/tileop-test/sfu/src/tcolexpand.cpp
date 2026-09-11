#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TCOLEXPAND — copy-expand from a per-column broadcast source.
// v0.58 contract (recovered from header + emulator + ValidateReduceAndExpandTepl):
//   TCOLEXPAND(dst, src): 广播源为 1 x N 行（validRow=1, validCol=N）,dst[i,j]=src[0,j] 全行广播。
//   **2026-09-08 对齐文档（model 49547742）**: 源改用真 1 x N（`Tile<Vec,float,1,N,RowMajor>`）与
//   TCOLEXPAND.md 示例一致。spec TCOLEXPAND.asl 只要求源逻辑 ValidRow==1/ValidCol==dst、物理由 layout 派生,
//   真 1 x N 合法(实测编译+跑通+全 M 行正确广播)。旧注释"模型只填 row0、退化 expand"已过时(模型已修)。
constexpr int M = 16, N = 16;
using SrcTile = vtile_t<float, 1, N>;                 // GENUINE 1 x N broadcast row
static float src[N];
static float out[M * N];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", src, sizeof(src));
#endif
    for (int i = 0; i < N; ++i) if (src[i] == 0.0f) src[i] = (float)(i + 1) * 0.25f;
    iter_t<float, 1, N> gS(src);
    iter_t<float, M, N> gO(out);
    auto s0 = gS(0, 0);
    auto o0 = gO(0, 0);
    SrcTile tS;
    vtile_t<float, M, N> tD;
    TLOAD(tS, s0);
    BENCHSTART;
    TCOLEXPAND(tD, tS);
    BENCHEND;
    TSTORE(o0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
