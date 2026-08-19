#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4_bigbs.hpp"

namespace supernpu::tile_isa::mxquant {

// Non-tail-axis, OCP scale (scaleAlg=0), FP4 output (E2M1 default, E1M2 valid).
// Quantize axis is rows (TCOLMAX); fp4 packs 2/byte along the contiguous Post
// axis, so the output tile is [BlockSize, TileN/2] and gm_y is
// RowMajor<Axis, Post/2>. emax derived from OutT.
//
// fp4 output tile [BlockSize, TileN/2] is plain RowMajor NoneBox; the 32B column
// alignment (pto_tile.hpp:649, RECORD problem 3) requires (TileN/2)*8 % 256 == 0
// -> TileN % 64 == 0, i.e. one tile spans >=2 MX blocks along Post. The packed
// axis (Post) is orthogonal to the reduce axis (rows), so this widening does not
// touch the per-column TCOLMAX reduce. Default TileN=64 = 2 blocks.
//
// scale: E8M0 1 byte/block, compact planar [scaleRows, Post] with
// scaleRows = evenAlign(numKb) (reduce-axis collapsed by BlockSize + even-aligned)
// — same as dynamic_mx_quant_nontail_cublas_fp8. NOTE: this is NOT the final
// AscendC interleaved layout [ceil(numKb/2), Post, 2]; the parity zip is a
// documented gap (PTO Tile-ISA has no interleave/zip intrinsic, only TCONCAT +
// reduce-internal butterfly shuffle). See DESIGN §5.3 / README.
// Supported BlockSize range (plain single-load path): BlockSize ∈ {32, 64}.
// The whole [BlockSize, TileN] block is loaded in ONE tile, so the contiguous
// axis TileN carries BOTH the fp4 32B alignment LOWER bound (TileN % 64 == 0,
// i.e. TileN ≥ 64) and the TileSize UPPER bound (16b input tile:
// BlockSize*TileN*2 ≤ 8192 → TileN ≤ 4096/BlockSize). A legal TileN exists iff
// 64 ≤ 4096/BlockSize → BlockSize ≤ 64. BlockSize ≥ 96 (next multiple of 32:
// 96*64=6144 > 4096) has NO legal TileN here → use
// dynamic_mx_quant_nontail_ocp_fp4_bigbs (方案A, splits the reduce axis).
//
// This is the PLAIN single-load implementation, kept behind an internal name.
// The public entry `dynamic_mx_quant_nontail_ocp_fp4` (below) DERIVES TileN at
// compile time from Post + the InT budget and AUTO-ROUTES to this plain path when
// a legal TileN exists, or to the 方案A split-reduce `_bigbs` kernel when it does
// not (large BlockSize). TileN stays an explicit param here so the dispatcher can
// feed the derived value.
template <int Axis, int Post, int BlockSize = 32, int TileN = 64, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16>
static void nontail_ocp_fp4_plain(InT *x, OutT *y, uint8_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % TileN == 0, "Post must be multiple of TileN");
    static_assert(TileN % 64 == 0,
                  "fp4 output tile is plain RowMajor NoneBox: (TileN/2)*8 % 256 == 0 "
                  "requires TileN a multiple of 64 (>=2 MX blocks along Post)");
    // BlockSize range: single-load path is capped at BlockSize ≤ 64. With
    // TileN ≥ 64, the 16b input tile budget BlockSize*TileN ≤ 4096 forces
    // BlockSize ≤ 64. BlockSize ≥ 96 -> no legal TileN; use the _bigbs kernel.
    // NOTE: unlike nontail_cublas_fp8, this BS ≤ 64 bound is FORMAL AND
    // PERMANENT — OCP extracts the exponent in the bf16/uint16 (16b) domain
    // (reinterpret_u16_to_bf16), never a fp32 32b roundtrip, so the binding tile
    // is already the 16b input. The 4096 budget does NOT relax when the compiler
    // gains a register-level reinterpret (问题4); large BS still needs _bigbs.
    static_assert(BlockSize * TileN <= 4096,
                  "plain non-tail OCP-FP4 supports BlockSize ∈ {32,64} only (16b input "
                  "tile BlockSize*TileN <= 4096, and TileN >= 64 forces BlockSize <= 64). "
                  "For BlockSize >= 96 use dynamic_mx_quant_nontail_ocp_fp4_bigbs "
                  "(方案A, split reduce axis).");

    constexpr int numKb = Axis / BlockSize;
    constexpr int numN  = Post / TileN;
    // reduce-axis block count, even-aligned (padding block-row left zero).
    constexpr int scaleRows = ((numKb + 1) / 2) * 2;

    using namespace pto;

    using tile_x  = Tile<Location::Vec, InT,    BlockSize, TileN,     BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  BlockSize, TileN,     BLayout::RowMajor>;
    using tile_o  = Tile<Location::Vec, OutT,   BlockSize, TileN / 2, BLayout::RowMajor>;
    // Full uint16 input view (bit-reinterpret of bf16) for the boxed OCP reduce.
    using tile_xu = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor>;
    // Boxed (valid row=1) per-block scale/recip: one scalar per Post column.
    using tile_sred      = Tile<Location::Vec, uint16_t,   BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_bf1 = Tile<Location::Vec, __bf16,   BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_f1  = Tile<Location::Vec, float,    BlockSize, TileN, BLayout::RowMajor, 1, TileN>;

    using gm_x  = global_tensor<InT,      RowMajor<Axis, Post>>;
    using gm_xu = global_tensor<uint16_t, RowMajor<Axis, Post>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<Axis, Post / 2>>;
    // scale: E8M0, compact planar [scaleRows, Post], one byte per block.
    using gm_s  = global_tensor<__fp8_e8m0, RowMajor<scaleRows, Post>>;

    global_iterator<gm_x,  tile_x>  x_iter(x);
    global_iterator<gm_xu, tile_xu> xu_iter(reinterpret_cast<uint16_t *>(x));
    global_iterator<gm_y,  tile_o>  y_iter(reinterpret_cast<uint8_t *>(y));

    for (int kb = 0; kb < numKb; ++kb) {
        for (int n = 0; n < numN; ++n) {
            auto gx  = x_iter(kb, n);
            auto gy  = y_iter(kb, n);
            // Compact scale: fold block-row index (kb) into the base pointer since
            // the iterator's i-stride is the PHYSICAL tile height, not 1. Each
            // block-row writes TileN bytes at scale + kb*Post.
            global_iterator<gm_s, tile_se8m0> s_iter(reinterpret_cast<__fp8_e8m0 *>(scale) + kb * Post);
            auto gs = s_iter(0, n);

            tile_se8m0 scale_e8m0;
            tile_sred recip;
            // Regularize InT -> uint16 bf16-exponent bits for the shared OCP reduce.
            tile_xu x_u16;
            if constexpr (std::is_same_v<InT, __bf16>) {
                auto gxu = xu_iter(kb, n);
                TLOAD(x_u16, gxu);
            } else {
                tile_x xin;
                TLOAD(xin, gx);
                if constexpr (std::is_same_v<InT, __half>) {
                    half_to_bf16bits<4, BlockSize, TileN>(xin, x_u16);
                } else {
                    f32_to_bf16expbits<4, BlockSize, TileN>(xin, x_u16);
                }
            }
            compute_ocp_scale_not_tail_boxed<OutT, BlockSize, TileN>(x_u16, scale_e8m0, recip);
            // scale_e8m0 boxed valid row=1: E8M0 byte produced directly by
            // Cast<bf16->e8m0>, store 1 byte/block (no narrowing TCVT).
            // MISSING INTERLEAVE: stored as COMPACT planar [scaleRows, Post]
            // (block-rows in order). AscendC's mxScale is PARITY-INTERLEAVED
            // [ceil(numKb/2), Post, 2] -- even/odd block-rows zipped via
            // Reg::Interleave. Blocked on TINTERLEAVE/TDEINTERLEAVE not being
            // exposed in the -D__linx header (RECORD 问题5); insert the even/odd
            // zip here once available.
            TSTORE(gs, scale_e8m0);

            tile_recip_bf1 inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
            reinterpret_u16_to_bf16<2, BlockSize, TileN, 1, TileN>(recip, inv_bf16);
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            tile_x xq;
            TLOAD(xq, gx);
            tile_o oq;
            if constexpr (std::is_same_v<InT, float>) {
                TCOLEXPANDMUL(xq, xq, inv_scale_f); // fp32 domain mul (no pre-cast)
                TCVT(oq, xq); // fp32 -> packed fp4_e2m1x2 (Post halved)
            } else {
                tile_f xf;
                TCVT(xf, xq); // bf16/half -> fp32
                TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
                TCVT(oq, xf); // fp32 -> packed fp4_e2m1x2 (Post halved)
            }
            TSTORE(gy, oq);
        }
    }
}

// Public entry: TileN is NOT a caller knob. It is DERIVED at compile time from
// Post + the InT budget (pick_tilen). If a legal TileN >= the fp4 64B lower bound
// exists, route to the plain single-load path; otherwise (large BlockSize leaves
// no legal TileN) auto-route to the 方案A split-reduce `_bigbs` kernel with a
// budget-derived R_sub. `if constexpr` guarantees the untaken branch is not
// instantiated. InT drives BOTH the budget AND the compute domain: scale-reduce and
// data paths are InT-dispatched (bf16/half/fp32) via `if constexpr`.
template <int Axis, int Post, int BlockSize = 32, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16>
void dynamic_mx_quant_nontail_ocp_fp4(InT *x, OutT *y, uint8_t *scale) {
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    constexpr int TileN = pick_tilen<BlockSize, Post, OutT, InT, /*IsCublas=*/false>();
    if constexpr (TileN >= nontail_align_lower<OutT>()) {
        nontail_ocp_fp4_plain<Axis, Post, BlockSize, TileN, OutT, InT>(x, y, scale);
    } else {
        constexpr int BigTileN = nontail_align_lower<OutT>();
        constexpr int Rsub = max_rsub<BlockSize, BigTileN, InT, /*IsCublas=*/false>();
        dynamic_mx_quant_nontail_ocp_fp4_bigbs<Axis, Post, BlockSize, BigTileN, Rsub, OutT, InT>(x, y, scale);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
