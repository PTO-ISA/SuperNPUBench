#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TTRI — triangular fill. v0.58 header signature is 1-arg
// (dst only): the op self-generates a triangular pattern into dst (no input
// tile). docs/tileop-usage 早期无签名（现 per-op 文档已补）; 1-arg form recovered from the
// header. Semantics (which triangle / fill value) are not pinned by docs, so
// 现已注册 golden（check_ttri，PASS）.
constexpr int M = 16, N = 16, NE = M * N;
static float out[NE];
int main() {
    iter_t<float, M, N> gO(out);
    auto gO0 = gO(0, 0);
    vtile_t<float, M, N> tD;
    BENCHSTART;
    TTRI(tD);
    BENCHEND;
    TSTORE(gO0, tD);
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
