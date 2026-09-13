#include <common/pto_tileop.hpp>
#include <cstdint>

// Emit-capability probe for the fp16/bf16/fp32 input support (plan step 3).
// Confirms the toolchain compiles+links+disassembles the NEW tile ops the
// dispatched helpers will rely on, and that none trip an alignment/round-mode
// assertion. This is an EMIT check only (runtime blocked by toolchain<->emulator
// skew); correctness is verified separately by op-review against AscendC.
//
// Probed ops (per plan「探针」):
//   1. TCVT half -> bf16      (OCP-half maxExp regularize: cast then AND exp bits)
//   2. TCVT half -> fp32      (cuBLAS-half amax + half data-path pre-mul cast)
//   3. TCVT uint32 -> uint16  (OCP-fp32 narrow after uint32 reduce + >>16)
//   4. uint32 TROWMAX/TCOLMAX (OCP-fp32 exponent reduce in the 32b domain)
//   5. fp32 TABS + fp32 TROWMAX/TCOLMAX (cuBLAS-fp32 amax, full-precision reduce)
//   6. TSHRS uint32           (OCP-fp32 >>16 exp realign; already used by cublas core)
//
// NOTE round-mode: the high-level TCVT wrapper uses the default RMode (LINX_RNONE);
// AscendC's half->bf16 uses TRUNC (RTZ). The default is fine for the EMIT check;
// the round-mode match is a code-review item on the real helper, not a probe gate.
using namespace pto;

static __half   xh[8 * 32] __attribute__((aligned(4096))) = {};
static float    xf_in[8 * 32] __attribute__((aligned(4096))) = {};
static uint32_t xu32[8 * 32] __attribute__((aligned(4096))) = {};

static __bf16   y_h2b[8 * 32] __attribute__((aligned(4096))) = {};
static float    y_h2f[8 * 32] __attribute__((aligned(4096))) = {};
static uint16_t y_u32_row[8 * 1] __attribute__((aligned(4096))) = {};
static uint16_t y_u32_col[1 * 32] __attribute__((aligned(4096))) = {};
static float    y_f32_row[8 * 1] __attribute__((aligned(4096))) = {};
static float    y_f32_col[1 * 32] __attribute__((aligned(4096))) = {};

int main() {
    // ---- shapes -------------------------------------------------------------
    using tile_h    = Tile<Location::Vec, __half,    8, 32, BLayout::RowMajor>;
    using tile_b    = Tile<Location::Vec, __bf16,    8, 32, BLayout::RowMajor>;
    using tile_f    = Tile<Location::Vec, float,     8, 32, BLayout::RowMajor>;
    using tile_u32  = Tile<Location::Vec, uint32_t,  8, 32, BLayout::RowMajor>;
    using tile_u16r = Tile<Location::Vec, uint16_t,  8, 32, BLayout::RowMajor, 8, 1>;
    using tile_u16c = Tile<Location::Vec, uint16_t,  8, 32, BLayout::RowMajor, 1, 32>;
    using tile_u32r = Tile<Location::Vec, uint32_t,  8, 32, BLayout::RowMajor, 8, 1>;
    using tile_u32c = Tile<Location::Vec, uint32_t,  8, 32, BLayout::RowMajor, 1, 32>;
    using tile_fr   = Tile<Location::Vec, float,     8, 32, BLayout::RowMajor, 8, 1>;
    using tile_fc   = Tile<Location::Vec, float,     8, 32, BLayout::RowMajor, 1, 32>;

    using gm_h   = global_tensor<__half,    RowMajor<8, 32>>;
    using gm_f   = global_tensor<float,     RowMajor<8, 32>>;
    using gm_u32 = global_tensor<uint32_t,  RowMajor<8, 32>>;
    using gm_b   = global_tensor<__bf16,    RowMajor<8, 32>>;
    using gm_u16r = global_tensor<uint16_t, RowMajor<8, 1>>;
    using gm_u16c = global_tensor<uint16_t, RowMajor<1, 32>>;
    using gm_fr   = global_tensor<float,    RowMajor<8, 1>>;
    using gm_fc   = global_tensor<float,    RowMajor<1, 32>>;

    global_iterator<gm_h,   tile_h>   hi(xh);
    global_iterator<gm_f,   tile_f>   fi(xf_in);
    global_iterator<gm_u32, tile_u32> ui(xu32);

    auto ghi = hi(0, 0);
    auto gfi = fi(0, 0);
    auto gui = ui(0, 0);
    tile_h xhq; TLOAD(xhq, ghi);
    tile_f xfq; TLOAD(xfq, gfi);
    tile_u32 xuq; TLOAD(xuq, gui);

    // 1. half -> bf16 -----------------------------------------------------------
    tile_b hb; TCVT(hb, xhq);
    { global_iterator<gm_b, tile_b> o(y_h2b); auto g = o(0, 0); TSTORE(g, hb); }

    // 2. half -> fp32 -----------------------------------------------------------
    tile_f hf; TCVT(hf, xhq);
    { global_iterator<gm_f, tile_f> o(y_h2f); auto g = o(0, 0); TSTORE(g, hf); }

    // 4. uint32 reduce (row+col) + 6. TSHRS + 3. uint32 -> uint16 narrow --------
    tile_u32 exp32; TANDS(exp32, xuq, static_cast<uint32_t>(0x7F800000));
    TSHRS(exp32, exp32, static_cast<uint32_t>(16));
    tile_u32r mrow; TROWMAX(mrow, exp32);
    tile_u16r nrow; TCVT(nrow, mrow);
    { global_iterator<gm_u16r, tile_u16r> o(y_u32_row); auto g = o(0, 0); TSTORE(g, nrow); }
    tile_u32c mcol; TCOLMAX(mcol, exp32);
    tile_u16c ncol; TCVT(ncol, mcol);
    { global_iterator<gm_u16c, tile_u16c> o(y_u32_col); auto g = o(0, 0); TSTORE(g, ncol); }

    // 5. fp32 TABS + fp32 reduce (row+col) --------------------------------------
    tile_f absf; TABS(absf, xfq);
    tile_fr frow; TROWMAX(frow, absf);
    { global_iterator<gm_fr, tile_fr> o(y_f32_row); auto g = o(0, 0); TSTORE(g, frow); }
    tile_fc fcol; TCOLMAX(fcol, absf);
    { global_iterator<gm_fc, tile_fc> o(y_f32_col); auto g = o(0, 0); TSTORE(g, fcol); }

    return 0;
}
