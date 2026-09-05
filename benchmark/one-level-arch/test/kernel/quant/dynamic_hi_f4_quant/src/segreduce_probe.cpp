#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// Segmented-max + grouped-broadcast-mul emit probe for dynamic_hi_f4_quant.
// ============================================================================
// GOAL: verify the two core hi_f4 tile-op idioms (DESIGN §3.1 问题1/2, RECORD
// 问题1/2) can be EMITTED on the current one-level toolchain:
//
//   (A) segmented max 64->16->8->1 via zero-cost TRESHAPE + whole-row TROWMAX,
//       每 4 / 每 2 相邻一组 (maxPer4 / maxPer2 / max).
//   (B) grouped broadcast-mul 16->64 via TRESHAPE + TROWEXPANDMUL (per-row [R,1]).
//
// ============================================================================
// RESULT (empirically confirmed 2026-08-14, see RECORD.md 问题10): this probe
// DOES NOT COMPILE. The DESIGN §3.1 "zero-cost TRESHAPE 化整为零 [.,4]" workaround
// for adjacent segmented max is NOT realizable on the one-level toolchain. TWO
// independent walls:
//
//   (1) 32B COLUMN-ALIGNMENT (pto_tile.hpp:649 static_assert). A RowMajor NoneBox
//       tile requires `Cols * sizeof(DType) % 32 == 0`; a boxed (SFractal!=NoneBox)
//       tile requires `Cols % InnerCols == 0` with InnerCols==16 for bf16
//       (pto_tile.hpp:562). Either way a bf16 tile's Cols MUST be a multiple of 16.
//       The compact reshape targets [.,4] / [.,2] / [.,1] (m16c, m8c, m8, vmax)
//       ALL fail this static_assert at TYPE-INSTANTIATION time — independent of
//       TRESHAPE. Adjacent 4/2-grouping fundamentally needs a width-4/2 tile,
//       which the layout rules forbid.
//   (2) TRESHAPE HEADER GAP. The top-level `TRESHAPE` wrapper lives in
//       common/tileop_api.hpp:52 but tileop_api_impl.hpp only pulls the aarch64/
//       + cpu_sim/ TReshape.hpp, NOT jcore/ — so under -D__linx (jcore path via
//       common/pto_tileop.hpp) `TRESHAPE` is an undeclared identifier. Same
//       header-gap class as TCAST / TINTERLEAVE (RECORD 问题5). Even calling
//       TRESHAPE_Impl directly cannot rescue wall (1).
//
// CONCLUSION: the segmented-max / grouped-broadcast reshape route is BLOCKED.
// A viable one-level implementation must keep every intermediate tile's Cols a
// multiple of 16 (bf16) / 8 (fp32/uint32) — i.e. NO width-4/2/1 tiles. The
// grouping must instead come from strided HBM loads (split each block into
// wide ≥16-col sub-tiles + elementwise TMAX tree) OR from adopting the emulator's
// strided (transpose) grouping semantics (§2.4). DECISION DEFERRED to user.
//
// The code below is kept as the FAILING witness (mirrors the e6m2_probe Stage-1
// honest-path pattern). Build `make ... TYPE=SEGREDUCE_PROBE` to reproduce the
// two static_assert failures at m8c / m8 / vmax and the TRESHAPE undeclared error.
// ============================================================================
using namespace pto;

static constexpr int M  = 8;   // rows == blocks (one 64-elem block per row)
static constexpr int BS = 64;  // hi_f4 fixed block size

static __bf16 x[M * BS]  __attribute__((aligned(4096))) = {};
static __bf16 y[M * BS]  __attribute__((aligned(4096))) = {}; // normalized data out
static __bf16 vm[M * 8]  __attribute__((aligned(4096))) = {}; // block-max out (valid col 1)

int main() {
    // ---- tile shapes (all RowMajor, compact unless noted) --------------------
    using tile_v    = Tile<Location::Vec, __bf16, M,      BS, BLayout::RowMajor>;         // [8,64]
    using tile_r16  = Tile<Location::Vec, __bf16, M * 16, 4,  BLayout::RowMajor>;         // [128,4]
    using tile_m16c = Tile<Location::Vec, __bf16, M * 16, 1,  BLayout::RowMajor>;         // [128,1]
    using tile_m16  = Tile<Location::Vec, __bf16, M,      16, BLayout::RowMajor>;         // [8,16]
    using tile_r8   = Tile<Location::Vec, __bf16, M * 8,  2,  BLayout::RowMajor>;         // [64,2]
    using tile_m8c  = Tile<Location::Vec, __bf16, M * 8,  1,  BLayout::RowMajor>;         // [64,1]
    using tile_m8   = Tile<Location::Vec, __bf16, M,      8,  BLayout::RowMajor>;         // [8,8]
    using tile_vmax = Tile<Location::Vec, __bf16, M,      8,  BLayout::RowMajor, M, 1>;   // [8,8] valid col 1

    using gm_v = global_tensor<__bf16, RowMajor<M, BS>>;
    using gm_m = global_tensor<__bf16, RowMajor<M, 8>>;

    global_iterator<gm_v, tile_v>    xi(x);
    global_iterator<gm_v, tile_v>    yi(y);
    global_iterator<gm_m, tile_vmax> vmi(vm);

    tile_v xv;
    TLOAD(xv, xi(0, 0));

    // (A) segmented max 64 -> 16 -> 8 -> 1 -------------------------------------
    tile_v vabs;
    TABS(vabs, xv);

    tile_r16  r16;  TRESHAPE(r16, vabs);       // [8,64] -> [128,4], data() alias
    tile_m16c m16c; TROWMAX(m16c, r16);        // per-4-adjacent max -> [128,1]
    tile_m16  m16;  TRESHAPE(m16, m16c);       // -> [8,16] (16 L3-maxes per row)

    tile_r8   r8;   TRESHAPE(r8, m16);         // [8,16] -> [64,2]
    tile_m8c  m8c;  TROWMAX(m8c, r8);          // per-2-adjacent max -> [64,1]
    tile_m8   m8;   TRESHAPE(m8, m8c);         // -> [8,8] (8 L2-maxes per row)

    tile_vmax vmax; TROWMAX(vmax, m8);         // whole-row max -> [8,8] valid col 1
    TSTORE(vmi(0, 0), vmax);

    // (B) grouped broadcast-mul 16 -> 64 --------------------------------------
    // reuse m16c[128,1] as the per-group factor; multiply each group of 4.
    tile_r16 prod;
    TROWEXPANDMUL(prod, r16, m16c);            // [128,4] * [128,1] broadcast -> [128,4]
    tile_v yv; TRESHAPE(yv, prod);             // [128,4] -> [8,64]
    TSTORE(yi(0, 0), yv);
    return 0;
}
