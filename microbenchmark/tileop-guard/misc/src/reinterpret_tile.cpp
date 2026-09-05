#include "guard_common.hpp"
#include "guard_io.h"
// TileOP-API guard: reinterpret_tile<NewDType>(src) — zero-instruction bitcast view.
// reinterpret_tile returns a ReinterpretedTileView (a distinct view type; is_tile
// is true). Tileop templates bind BOTH operands to the SAME tile_shape, so a view
// cannot be mixed with a plain Tile in one call — that is the reported
// "no matching function for call to 'TABS'". Correct usage: reinterpret BOTH
// operands so they share the view type. TANDS accepts the views fine. The store
// pattern follows the model's own cross_model test
// (bf16_backing_tands_u16_then_tmuls_bf16): after the integer-view op the backing
// carries the INT32 runtime dtype tag, so a subsequent NATIVE-type op is used to
// re-tag the backing back to FP32 before TSTORE (TSTORE keys the block dtype off
// the tile's declared type and rejects view operands). Here:
//   bitcast fp32->int32 view; TANDS &0x7fffffff (clear sign -> |x| bits);
//   TMULS(tOut, tOut, 1.0f) re-tags FP32; TSTORE fp32.  golden = abs.
// (The TCMP variant of this pattern is a model gap — see reinterpret_tcmp.cpp.)
constexpr int M = 16, N = 16, NE = M * N;
static float in[NE], out[NE];
int main() {
#ifdef RES_CHECK
    guard_read_bin(CHK_DIR "/in_a.bin", in, sizeof(in));
#else
    for (int i = 0; i < NE; ++i) in[i] = (i % 2 ? -1.f : 1.f) * (float)i * 0.5f;
#endif
    iter_t<float, M, N> gI(in), gO(out);
    auto i0 = gI(0, 0), o0 = gO(0, 0);
    vtile_t<float, M, N> tIn, tOut;
    TLOAD(tIn, i0);
    BENCHSTART;
    auto vIn = reinterpret_tile<int32_t>(tIn);     // fp32 bits viewed as int32
    auto vOut = reinterpret_tile<int32_t>(tOut);   // same view type as vIn
    TANDS(vOut, vIn, (int32_t)0x7fffffff);         // clear sign bit -> |x| (backing tagged INT32)
    TMULS(tOut, tOut, 1.0f);                       // native fp32 op re-tags backing FP32
    BENCHEND;
    TSTORE(o0, tOut);                              // fp32 store now matches -> |x|
#ifdef RES_CHECK
    guard_dump_bin(CHK_DIR "/out.bin", out, sizeof(out));
#endif
    return 0;
}
