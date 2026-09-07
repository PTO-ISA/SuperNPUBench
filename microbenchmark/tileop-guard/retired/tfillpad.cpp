#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: TFILLPAD — copy the VALID source rectangle into dst and fill
// the physical padding with zero. Authoritative semantics: TFILLPAD.md ("复制有效
// 源区域, 并将绑定的标量写入 padding") + its worked example (InputTile valid 9x9 in
// a 16x16 physical, OutputTile PadValue::Zero); cpu_sim TFillPad.hpp static_assert
// PadVal==Zero confirms zero-pad only. Signature + tile-shape recipe are taken
// verbatim from the doc example. Precision: res_check; golden = zero-padded copy
// of the VR x VC valid rectangle (dst[i,j]=src[i,j] for i<VR & j<VC, else 0).
constexpr int M = 16, N = 16, VR = 9, VC = 9, NE = M * N;
static float gin[NE], gout[NE];
int main() {
    guard_read_bin(CHK_DIR "/in_a.bin", gin, sizeof(gin));
    using GM = global_tensor<float, RowMajor<M, N>>;
    using InputTile  = Tile<Location::Vec, float, M, N, BLayout::RowMajor, VR, VC>;
    using OutputTile = Tile<Location::Vec, float, M, N, BLayout::RowMajor,
                            M, N, SLayout::NoneBox, 512, PadValue::Zero>;
    GM gsrc(gin), gdst(gout);
    InputTile src;
    OutputTile dst;
    TLOAD(src, gsrc);
    BENCHSTART;
    TFILLPAD(dst, src);
    BENCHEND;
    TSTORE(gdst, dst);
    guard_dump_bin(CHK_DIR "/out.bin", gout, sizeof(gout));
    return 0;
}
