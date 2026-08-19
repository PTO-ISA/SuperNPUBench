#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_BIGBS_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_BIGBS_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Non-tail-axis, OCP scale (scaleAlg=0), FP4 output — LARGE-BlockSize branch
// (方案A: split reduce axis + register accumulation). See RECORD 问题2 /
// DESIGN §7.5.
//
// Why a separate kernel: the plain non-tail kernel loads the whole
// [BlockSize, TileN] block in one shot, so the contiguous axis TileN carries
// BOTH the fp4 32B alignment LOWER bound (TileN % 64 == 0) and the TileSize
// UPPER bound (Rows*Cols*sizeof <= 8192 -> TileN <= 4096/BlockSize on the 16b
// input tile). Valid TileN exists iff lower <= upper, which fails once
// BlockSize >= 128 (upper < 64). This branch breaks the Rows*Cols product by
// tiling the REDUCE axis (rows, length BlockSize) into R_sub-row sub-chunks:
// TileSize now binds R_sub*TileN <= 4096 with R_sub a free knob, so TileN can
// always meet the 64 alignment regardless of BlockSize.
//
// Pass 1 (reduce): for each R_sub sub-chunk, TANDS(exp) + TCOLMAX -> partial
// [1, TileN], accumulate into a register-resident running-max via elementwise
// TMAX (max is associative, so cross-sub-chunk accumulation is exact). No shared
// memory: the accumulator is loop-carried within one PE-thread. After the last
// sub-chunk, finalize once (ocp_scale_from_maxexp_not_tail_boxed) -> scale_byte
// + recip, store one E8M0 byte per block.
//
// Pass 2 (data): reinterpret recip -> bf16 -> fp32 inv_scale (per-column
// scalar, valid row=1); for each R_sub sub-chunk re-load x, TCVT bf16->fp32,
// TCOLEXPANDMUL broadcasts the per-column inv_scale to all R_sub rows, TCVT
// fp32->fp4, TSTORE. The same per-column scale applies to every sub-chunk row.
//
// scale layout: E8M0 1 byte/block, compact planar [scaleRows, Post] — same as
// dynamic_mx_quant_nontail_ocp_fp4. NOT the AscendC parity-interleaved layout
// (interleave intrinsic is a header gap; see RECORD 问题5 / README).

// OCP not-tail scale from an ALREADY-reduced per-column max exponent (boxed
// valid row=1). Local to this kernel (方案A-only): pass1 reduces the BlockSize
// rows ACROSS sub-chunks (running TMAX into a [R_sub, TileN] valid=1 tile), then
// this finalizes once. Uses the new bf16-multiply + Cast<e8m0> scale (mirrors
// compute_ocp_scale_not_tail_boxed's mul+cast core); kept here (not in
// common.hpp) since only the split-reduce path needs a maxexp-first finalize.
// R is the sub-chunk height R_sub, NOT BlockSize.
template <typename OutT, int R, int TileN, int ValidN = TileN>
static void ocp_scale_from_maxexp_not_tail_boxed_bigbs(
    pto::Tile<pto::Location::Vec, uint16_t,   R, TileN, pto::BLayout::RowMajor, 1, ValidN> &max_exp,
    pto::Tile<pto::Location::Vec, __fp8_e8m0, R, TileN, pto::BLayout::RowMajor, 1, ValidN> &scale_e8m0,
    pto::Tile<pto::Location::Vec, uint16_t,   R, TileN, pto::BLayout::RowMajor, 1, ValidN> &recip_out) {
    using namespace pto;
    ocp_scale_mulcast_from_maxexp<OutT, 3, R, TileN, 1, ValidN>(max_exp, scale_e8m0, recip_out);
}

// Supported BlockSize range (方案A, split reduce axis):
//   BlockSize ∈ { multiples of R_sub, R_sub..∞ }  (R_sub | BlockSize)
// Unlike the plain `dynamic_mx_quant_nontail_ocp_fp4` (single-load, capped at
// BlockSize ≤ 64 because fp4 alignment lower bound 64 > TileSize upper 4096/BS
// once BS ≥ 128), this branch decouples them: TileSize now binds R_sub*TileN
// (a free knob) instead of BlockSize*TileN, so ANY BlockSize is legal as long
// as R_sub | BlockSize and R_sub*TileN ≤ 4096. Intended for the BS ≥ 128 range
// the plain kernel cannot cover; small BS also works but the plain kernel is
// cheaper there (no split/re-read). Default R_sub=32, TileN=64.
template <int Axis, int Post, int BlockSize, int TileN = 64, int R_sub = 32,
          typename OutT = __fp4_e2m1x2, typename InT = __bf16>
void dynamic_mx_quant_nontail_ocp_fp4_bigbs(InT *x, OutT *y, uint8_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % TileN == 0, "Post must be multiple of TileN");
    static_assert(TileN % 64 == 0,
                  "fp4 output tile is plain RowMajor NoneBox: (TileN/2)*8 % 256 == 0 "
                  "requires TileN a multiple of 64 (>=2 MX blocks along Post)");
    // BlockSize range: any multiple of R_sub (方案A decouples reduce axis).
    static_assert(BlockSize % R_sub == 0,
                  "BlockSize must be a multiple of R_sub (方案A splits the reduce axis "
                  "into R_sub-row sub-chunks); supported BlockSize = R_sub, 2*R_sub, ...");
    // 16b sub-chunk input tile: R_sub*TileN elems, <=4096 (Rows*Cols*2 <=8192).
    static_assert(R_sub * TileN <= 4096,
                  "sub-chunk R_sub*TileN must be <= 4096 (16b TileSize budget)");

    constexpr int numKb  = Axis / BlockSize;
    constexpr int numSub = BlockSize / R_sub;
    constexpr int numN   = Post / TileN;
    // reduce-axis block count, even-aligned (padding block-row left zero).
    constexpr int scaleRows = ((numKb + 1) / 2) * 2;

    using namespace pto;

    // Sub-chunk data/input tiles: physical [R_sub, TileN].
    using tile_x  = Tile<Location::Vec, InT,      R_sub, TileN,     BLayout::RowMajor>;
    using tile_xu = Tile<Location::Vec, uint16_t, R_sub, TileN,     BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,    R_sub, TileN,     BLayout::RowMajor>;
    using tile_o  = Tile<Location::Vec, OutT,     R_sub, TileN / 2, BLayout::RowMajor>;
    // Boxed (valid row=1) accumulator / scale / recip: one scalar per Post column.
    using tile_box       = Tile<Location::Vec, uint16_t,   R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_bf1 = Tile<Location::Vec, __bf16,   R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_f1  = Tile<Location::Vec, float,    R_sub, TileN, BLayout::RowMajor, 1, TileN>;

    using gm_x  = global_tensor<InT,      RowMajor<Axis, Post>>;
    using gm_xu = global_tensor<uint16_t, RowMajor<Axis, Post>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<Axis, Post / 2>>;
    // scale: uint8 E8M0, compact planar [scaleRows, Post], one byte per block.
    using gm_s  = global_tensor<__fp8_e8m0, RowMajor<scaleRows, Post>>;

    global_iterator<gm_x,  tile_x>  x_iter(x);
    global_iterator<gm_xu, tile_xu> xu_iter(reinterpret_cast<uint16_t *>(x));
    global_iterator<gm_y,  tile_o>  y_iter(reinterpret_cast<uint8_t *>(y));

    for (int kb = 0; kb < numKb; ++kb) {
        for (int n = 0; n < numN; ++n) {
            // ---- Pass 1: reduce BlockSize rows across R_sub sub-chunks ----
            tile_box max_exp_acc;
            TEXPANDS(max_exp_acc, static_cast<uint16_t>(0)); // running max seed
            for (int s = 0; s < numSub; ++s) {
                // absolute sub-chunk row-block index; R_sub == tile height ==
                // iterator i-stride, so no base-pointer fold needed for x.
                const int rb = kb * numSub + s;
                // Regularize InT input -> uint16 bf16-exponent bits (shared reduce).
                tile_xu x_u16;
                if constexpr (std::is_same_v<InT, __bf16>) {
                    auto gxu = xu_iter(rb, n);
                    TLOAD(x_u16, gxu);
                } else {
                    auto gx = x_iter(rb, n);
                    tile_x xin;
                    TLOAD(xin, gx);
                    if constexpr (std::is_same_v<InT, __half>) {
                        half_to_bf16bits<4, R_sub, TileN>(xin, x_u16);
                    } else {
                        f32_to_bf16expbits<4, R_sub, TileN>(xin, x_u16);
                    }
                }
                tile_xu exp_bits;
                TANDS(exp_bits, x_u16, BF16_EXP_MASK);
                tile_box partial;
                TCOLMAX(partial, exp_bits);              // rows -> valid row=1
                TMAX(max_exp_acc, max_exp_acc, partial); // cross-sub-chunk accum
            }

            tile_se8m0 scale_e8m0;
            tile_box recip;
            ocp_scale_from_maxexp_not_tail_boxed_bigbs<OutT, R_sub, TileN>(
                max_exp_acc, scale_e8m0, recip);

            // Compact scale store: fold block-row index (kb) into the base
            // pointer (iterator i-stride is physical tile height, not 1). Each
            // block-row writes TileN bytes at scale + kb*Post. The new OCP path
            // produces the E8M0 byte directly (Cast<bf16->e8m0>), so it stores
            // with no narrowing TCVT.
            global_iterator<gm_s, tile_se8m0> s_iter(
                reinterpret_cast<__fp8_e8m0 *>(scale) + kb * Post);
            auto gs = s_iter(0, n);
            // MISSING INTERLEAVE: stored as COMPACT planar [scaleRows, Post]
            // (block-rows in order). AscendC's mxScale is PARITY-INTERLEAVED
            // [ceil(numKb/2), Post, 2] -- even/odd block-rows zipped via
            // Reg::Interleave. Blocked on TINTERLEAVE/TDEINTERLEAVE not being
            // exposed in the -D__linx header (RECORD 问题5); insert the even/odd
            // zip here once available.
            TSTORE(gs, scale_e8m0);

            // ---- Pass 2: apply per-column inv_scale, cast to fp4 ----
            tile_recip_bf1 inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题4
            reinterpret_u16_to_bf16<3, R_sub, TileN, 1, TileN>(recip, inv_bf16);
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            for (int s = 0; s < numSub; ++s) {
                const int rb = kb * numSub + s;
                auto gx = x_iter(rb, n);
                auto gy = y_iter(rb, n);
                tile_x xq;
                TLOAD(xq, gx);
                tile_o oq;
                if constexpr (std::is_same_v<InT, float>) {
                    TCOLEXPANDMUL(xq, xq, inv_scale_f); // fp32 domain mul (no pre-cast)
                    TCVT(oq, xq); // fp32 -> packed fp4_e2m1x2 (Post halved)
                } else {
                    tile_f xf;
                    TCVT(xf, xq); // bf16/half -> fp32
                    TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast
                    TCVT(oq, xf); // fp32 -> packed fp4_e2m1x2 (Post halved)
                }
                TSTORE(gy, oq);
            }
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
