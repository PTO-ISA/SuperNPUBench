#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_BIGBS_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_BIGBS_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Non-tail-axis, cuBLAS scale (scaleAlg=1), FP8 output — LARGE-BlockSize branch
// (方案A: split reduce axis + register accumulation). Mirrors
// dynamic_mx_quant_nontail_ocp_fp4_bigbs; see RECORD 问题2 / 问题4 / DESIGN §7.5.
//
// Why a separate kernel: the plain non-tail cuBLAS kernel loads the whole
// [BlockSize, TileN] block in one shot, so the contiguous axis TileN carries
// BOTH the fp8 32B alignment LOWER bound (TileN % 32 == 0) and the TileSize
// UPPER bound. cuBLAS extracts the exponent in the fp32 domain via a scratch-HBM
// reinterpret roundtrip (问题4), so the binding tile is 32b: current upper bound
// R*C <= 2048, formal (post-bitcast) 4096. A valid TileN exists iff lower <=
// upper, which fails at large BlockSize. This branch breaks the Rows*Cols
// product by tiling the REDUCE axis (rows, length BlockSize) into R_sub-row
// sub-chunks: TileSize now binds R_sub*TileN with R_sub a free knob, so TileN
// can always meet the 32 alignment regardless of BlockSize.
//
// Pass 1 (reduce): for each R_sub sub-chunk, TABS + TCOLMAX -> partial per-column
// bf16 amax [1, TileN], accumulate into a register-resident running amax via
// elementwise TMAX (max is associative, so cross-sub-chunk accumulation is
// exact). After the last sub-chunk, TCVT the reduced bf16 amax to fp32 and
// finalize once via compute_cublas_core (guarded exponent extract, identical to
// the plain kernel) -> scale_byte + recip. No shared memory: the accumulator is
// loop-carried within one PE-thread.
//
// Pass 2 (data): reinterpret recip -> bf16 -> fp32 inv_scale (per-column scalar,
// valid row=1); for each R_sub sub-chunk re-load x, TCVT bf16->fp32,
// TCOLEXPANDMUL broadcasts the per-column inv_scale to all R_sub rows, TCVT
// fp32->fp8, TSTORE. The same per-column scale applies to every sub-chunk row.
//
// scale layout: E8M0 1 byte/block, compact planar [scaleRows, Post] — same as
// dynamic_mx_quant_nontail_cublas_fp8. NOT the AscendC parity-interleaved layout
// (interleave intrinsic is a header gap; see RECORD 问题5 / README).
//
// NOTE vs the plain kernel: this branch requires Post % TileN == 0 (no N_tail
// column handling) to keep the structure identical to nontail_ocp_fp4_bigbs; the
// plain kernel still covers the Post % TileN != 0 tail-column case at small BS.

// Supported BlockSize range (方案A, split reduce axis): any multiple of R_sub
// (R_sub | BlockSize). Unlike the plain dynamic_mx_quant_nontail_cublas_fp8
// (single-load, capped at BS<=64 current / BS<=128 formal because the fp8
// alignment lower bound 32 collides with the TileSize upper on the single TileN
// axis), this branch decouples them: TileSize binds R_sub*TileN (a free knob)
// instead of BlockSize*TileN, so ANY BlockSize is legal as long as R_sub |
// BlockSize and R_sub*TileN <= budget. The R_sub*TileN budget has the same two
// values (问题4): the assert below encodes the FORMAL bound 4096; the CURRENT
// fp32 32b roundtrip caps the toolchain at R_sub*TileN <= 2048, so R_sub=32/
// TileN=32 (1024) builds today while R_sub=64/TileN=64 waits on 问题4. Intended
// for the BS>=96 range the plain kernel cannot cover; small BS also works but the
// plain kernel is cheaper there (no split/re-read). Default R_sub=32, TileN=32.
template <int Axis, int Post, int BlockSize, int TileN = 32, int R_sub = 32,
          typename OutT = __fp8_e4m3, typename InT = __bf16,
          uint32_t MaxLowBoundBits = 0x2b8cbcccu>
void dynamic_mx_quant_nontail_cublas_fp8_bigbs(InT *x, OutT *y, uint8_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % TileN == 0, "Post must be multiple of TileN");
    static_assert(TileN % 32 == 0,
                  "fp8 output tile is plain RowMajor NoneBox: TileN*8 % 256 == 0 "
                  "requires TileN a multiple of 32 (one MX block is 32B-aligned)");
    // BlockSize range: any multiple of R_sub (方案A decouples the reduce axis).
    static_assert(BlockSize % R_sub == 0,
                  "BlockSize must be a multiple of R_sub (方案A splits the reduce axis "
                  "into R_sub-row sub-chunks); supported BlockSize = R_sub, 2*R_sub, ...");
    // FORMAL (post-bitcast) sub-chunk budget: 16b bf16 input tile R_sub*TileN
    // <= 4096. The CURRENT fp32 32b reinterpret roundtrip (问题4, compute_cublas_
    // core -> reinterpret_f32_to_u32) tightens the toolchain to R_sub*TileN <=
    // 2048; that tighter effective limit is documented, not asserted.
    static_assert(R_sub * TileN <= 4096,
                  "sub-chunk R_sub*TileN must be <= 4096 (formal 16b TileSize budget)");

    constexpr int numKb  = Axis / BlockSize;
    constexpr int numSub = BlockSize / R_sub;
    constexpr int numN   = Post / TileN;
    // reduce-axis block count, even-aligned (padding block-row left zero).
    constexpr int scaleRows = ((numKb + 1) / 2) * 2;

    using namespace pto;

    // Sub-chunk data/input tiles: physical [R_sub, TileN].
    using tile_x  = Tile<Location::Vec, InT,      R_sub, TileN, BLayout::RowMajor>;
    using tile_xu = Tile<Location::Vec, uint16_t, R_sub, TileN, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,    R_sub, TileN, BLayout::RowMajor>;
    using tile_o  = Tile<Location::Vec, OutT,     R_sub, TileN, BLayout::RowMajor>;
    // Boxed (valid row=1) accumulator / scale / recip: one scalar per Post column.
    using tile_box       = Tile<Location::Vec, uint16_t, R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_amax_bf1  = Tile<Location::Vec, __bf16,   R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_amax_in   = Tile<Location::Vec, InT,      R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_f32_1     = Tile<Location::Vec, float,    R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_sstore    = Tile<Location::Vec, uint8_t,  R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_bf1 = Tile<Location::Vec, __bf16,   R_sub, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_f1  = Tile<Location::Vec, float,    R_sub, TileN, BLayout::RowMajor, 1, TileN>;

    using gm_x  = global_tensor<InT,      RowMajor<Axis, Post>>;
    using gm_xu = global_tensor<uint16_t, RowMajor<Axis, Post>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<Axis, Post>>;
    // scale: uint8 E8M0, compact planar [scaleRows, Post], one byte per block.
    using gm_s  = global_tensor<uint8_t,  RowMajor<scaleRows, Post>>;

    global_iterator<gm_x,  tile_x>  x_iter(x);
    global_iterator<gm_xu, tile_xu> xu_iter(reinterpret_cast<uint16_t *>(x));
    global_iterator<gm_y,  tile_o>  y_iter(reinterpret_cast<uint8_t *>(y));

    for (int kb = 0; kb < numKb; ++kb) {
        for (int n = 0; n < numN; ++n) {
            // ---- Pass 1: reduce BlockSize rows across R_sub sub-chunks ----
            // Reduce amax in the uint16 abs-BIT domain. This is OP-FAITHFUL to
            // AscendC's non-tail ComputeScaleCuBlas bf16 branch, which reduces the
            // SAME way: And(x, BF16_ABS_MASK) + uint16 Reg::Max accumulate into
            // maxU16 (dynamic_mx_quant_not_tail_axis_optimize_high_perf_large_tail.h
            // :426-440) -- NOT a bf16 value-domain reduce. For non-negative bf16 the
            // unsigned bit order is monotonic in magnitude, so max-of-abs-bits ==
            // max-of-abs-value for finite inputs (inf/NaN are caught by the core's
            // `finite` mask). Choosing this domain also sidesteps a bf16 TEXPANDS
            // seed: the running-max needs an initial value, and TEXPANDS of a bf16
            // immediate crashes the LinxV5 backend ("illegal type in inline asm",
            // getCopyToParts) -- but a uint16 TEXPANDS(0) seed is legal. (NOTE: bf16
            // elementwise TMAX itself does NOT crash -- verified by probe -- so a
            // bf16 value-domain reduce with a peeled-first-sub-chunk TCOLMAX seed
            // would also compile; we pick the uint16 abs-bit domain because it
            // MATCHES AscendC exactly, not because bf16 TMAX is unavailable.) The
            // plain nontail_cublas_fp8 uses a bf16 value-domain reduce (TABS + bf16
            // TCOLMAX, single-pass so no cross-chunk seed) which is numerically
            // equivalent for finite non-negative values.
            tile_f32_1 max_f;
            if constexpr (std::is_same_v<InT, __bf16>) {
                // bf16: uint16 abs-BIT domain reduce (OP-FAITHFUL to AscendC bf16
                // branch; uint16 TEXPANDS(0) seed legal where a bf16 seed crashes).
                tile_box max_abs_acc;
                TEXPANDS(max_abs_acc, static_cast<uint16_t>(0)); // running max seed
                for (int s = 0; s < numSub; ++s) {
                    // absolute sub-chunk row-block index; R_sub == tile height ==
                    // iterator i-stride, so no base-pointer fold needed for x.
                    const int rb = kb * numSub + s;
                    auto gxu = xu_iter(rb, n);
                    tile_xu x_u16;
                    TLOAD(x_u16, gxu);
                    tile_xu abs_bits;
                    TANDS(abs_bits, x_u16, BF16_ABS_MASK);   // drop sign bit
                    tile_box partial;
                    TCOLMAX(partial, abs_bits);              // rows -> valid row=1
                    TMAX(max_abs_acc, max_abs_acc, partial); // cross-sub-chunk accum
                }
                // reinterpret accumulated abs bits -> bf16 -> fp32 amax for the core.
                tile_amax_bf1 amax_bf16;
                // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题4
                reinterpret_u16_to_bf16<2, R_sub, TileN, 1, TileN>(max_abs_acc, amax_bf16);
                TCVT(max_f, amax_bf16);                  // cast reduced amax to fp32
            } else if constexpr (std::is_same_v<InT, float>) {
                // fp32: reduce amax in the fp32 VALUE domain (full precision, no
                // reinterpret). Peel sub-chunk 0 to seed the running max (avoids any
                // seed immediate); accumulate the rest via elementwise TMAX. The
                // reduced amax is already fp32 -> flows straight into the core.
                {
                    auto gx0 = x_iter(kb * numSub + 0, n);
                    tile_x xq0; TLOAD(xq0, gx0);
                    tile_x abs0; TABS(abs0, xq0);
                    TCOLMAX(max_f, abs0);                // seed = sub-chunk 0 col-max
                }
                for (int s = 1; s < numSub; ++s) {
                    const int rb = kb * numSub + s;
                    auto gx = x_iter(rb, n);
                    tile_x xq; TLOAD(xq, gx);
                    tile_x abs_x; TABS(abs_x, xq);
                    tile_f32_1 partial; TCOLMAX(partial, abs_x);
                    TMAX(max_f, max_f, partial);
                }
            } else {
                // half: reduce amax in the half VALUE domain (keeps full mantissa; no
                // half->bf16 pre-cast). Peeled seed + elementwise TMAX, then TCVT the
                // reduced half amax to fp32. Mirrors generic compute_cublas_scale.
                tile_amax_in acc;
                {
                    auto gx0 = x_iter(kb * numSub + 0, n);
                    tile_x xq0; TLOAD(xq0, gx0);
                    tile_x abs0; TABS(abs0, xq0);
                    TCOLMAX(acc, abs0);                  // seed = sub-chunk 0 col-max
                }
                for (int s = 1; s < numSub; ++s) {
                    const int rb = kb * numSub + s;
                    auto gx = x_iter(rb, n);
                    tile_x xq; TLOAD(xq, gx);
                    tile_x abs_x; TABS(abs_x, xq);
                    tile_amax_in partial; TCOLMAX(partial, abs_x);
                    TMAX(acc, acc, partial);
                }
                TCVT(max_f, acc);                        // half amax -> fp32
            }

            tile_box scale_byte;
            tile_box recip;
            // guarded exponent extract, identical to the plain kernel's
            // compute_cublas_scale_not_tail (which is TCOLMAX + this same core).
            compute_cublas_core<OutT, R_sub, TileN, MaxLowBoundBits, 1, TileN>(
                max_f, scale_byte, recip);

            // Compact scale store: fold block-row index (kb) into the base
            // pointer (iterator i-stride is physical tile height, not 1). Each
            // block-row writes TileN bytes at scale + kb*Post.
            global_iterator<gm_s, tile_sstore> s_iter(scale + kb * Post);
            auto gs = s_iter(0, n);
            tile_sstore scale_u8;
            TCVT(scale_u8, scale_byte);
            // MISSING INTERLEAVE: stored as COMPACT planar [scaleRows, Post]
            // (block-rows in order). AscendC's mxScale is PARITY-INTERLEAVED
            // [ceil(numKb/2), Post, 2] -- even/odd block-rows zipped via
            // Reg::Interleave (..._not_tail_axis_optimize_high_perf_large_tail.h:511).
            // Blocked on TINTERLEAVE/TDEINTERLEAVE not being exposed in the -D__linx
            // header (RECORD 问题5); insert the even/odd zip here once available.
            TSTORE(gs, scale_u8);

            // ---- Pass 2: apply per-column inv_scale, cast to fp8 ----
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
                    TCOLEXPANDMUL(xq, xq, inv_scale_f); // per-column scalar broadcast
                    TCVT(oq, xq); // fp32 -> fp8
                } else {
                    tile_f xf;
                    TCVT(xf, xq); // bf16/half -> fp32
                    TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast
                    TCVT(oq, xf); // fp32 -> fp8
                }
                TSTORE(gy, oq);
            }
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
