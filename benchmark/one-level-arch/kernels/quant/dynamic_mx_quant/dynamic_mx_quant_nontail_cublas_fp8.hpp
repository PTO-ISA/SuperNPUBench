#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"
#include "quant/dynamic_mx_quant/dynamic_mx_quant_nontail_cublas_fp8_bigbs.hpp"

namespace supernpu::tile_isa::mxquant {

// Non-tail-axis, cuBLAS scale (scaleAlg=1), FP8 output (E4M3 default, E5M2).
// cuBLAS consumes the bf16 VALUE view (abs -> TCOLMAX -> fp32 amax, guarded
// exponent extract). Two-pass structure keeps peak live tiles low.
//
// Supported BlockSize range (plain single-load path). The whole
// [BlockSize, TileN] block is loaded in one tile, so the contiguous axis TileN
// carries BOTH the fp8 32B alignment LOWER bound (TileN % 32 == 0, i.e.
// TileN >= 32) and the TileSize UPPER bound. A legal TileN exists iff
// lower <= upper. The upper bound has TWO values because cuBLAS extracts the
// exponent in the fp32 domain via a scratch-HBM reinterpret roundtrip
// (compute_cublas_core -> reinterpret_f32_to_u32, RECORD 问题4):
//   - FORMAL (post-bitcast) model — the assert below encodes THIS: once the
//     compiler exposes a register-level reinterpret, the 32b roundtrip
//     disappears and the binding tile falls back to the 16b bf16 input, so the
//     budget is BlockSize*TileN <= 4096 -> BlockSize <= 128 (BS=128 -> TileN=32).
//   - CURRENT (workaround) model — the fp32 32b roundtrip tile binds a tighter
//     IsValidActiveSize budget BlockSize*TileN <= 2048 -> effective BlockSize <= 64
//     (BS=32 -> TileN in {32,64}; BS=64 -> only 32). For 64 < BlockSize <= 128 the
//     assert passes but the build still stops at the toolchain IsValidActiveSize
//     check until 问题4 is fixed. This tighter current limit is the documented gap.
// For BlockSize >= 160 (formal) there is NO legal TileN even after the fix; that
// large-BS range needs the 方案A split-reduce kernel (see
// dynamic_mx_quant_nontail_ocp_fp4_bigbs for the analogous structure).
//
// This is the PLAIN single-load implementation, kept behind an internal name.
// The public entry `dynamic_mx_quant_nontail_cublas_fp8` (below) DERIVES TileN at
// compile time from Post + the InT budget and AUTO-ROUTES to this plain path when
// a legal TileN exists, or to the 方案A split-reduce `_bigbs` kernel when it does
// not (large BlockSize). TileN stays an explicit param here so the dispatcher can
// feed the derived value.
template <int Axis, int Post, int BlockSize = 32, int TileN = 32, typename OutT = __fp8_e4m3,
          typename InT = __bf16, uint32_t MaxLowBoundBits = 0x2b8cbcccu>
static void nontail_cublas_fp8_plain(InT *x, OutT *y, uint8_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    // Axis must be whole blocks (a block is exactly BlockSize along the quant
    // axis); Post need NOT be a multiple of TileN: full column tiles + N_tail.
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    // FORMAL (post-bitcast) TileSize bound: 16b bf16 input tile BlockSize*TileN
    // <= 4096, and TileN >= 32 forces BlockSize <= 128. The CURRENT fp32 32b
    // reinterpret roundtrip (问题4 workaround) further caps the toolchain at
    // BlockSize <= 64; that tighter effective limit is documented above, not
    // asserted, so the code already targets the fixed-compiler model.
    static_assert(BlockSize * TileN <= 4096,
                  "plain non-tail cuBLAS-FP8 supports BlockSize <= 128 (formal: 16b "
                  "input tile BlockSize*TileN <= 4096, TileN >= 32). For BlockSize >= 160 "
                  "use a 方案A split-reduce kernel (see nontail_ocp_fp4_bigbs).");

    constexpr int numKb  = Axis / BlockSize;
    constexpr int numN   = Post / TileN;   // full column tiles
    constexpr int N_tail = Post % TileN;   // trailing partial column tile
    // AscendC scale even-pads the quant-axis block count (交织/interleaving):
    // scaleRows = ceil_even(numKb). The trailing padding block-row is left zero.
    constexpr int scaleRows = ((numKb + 1) / 2) * 2;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, InT,      BlockSize, TileN, BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,    BlockSize, TileN, BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, OutT,     BlockSize, TileN, BLayout::RowMajor>;
    // Compact scale store (transposed): the cuBLAS core emits scale_byte/recip
    // boxed valid row=1 (one per-column scalar per block-row), so we narrow to
    // uint8 and store one byte per block — no intermediate TCOLMAX.
    using tile_sred   = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_sstore = Tile<Location::Vec, uint8_t,  BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    // Per-column-scalar (valid row=1) reciprocal, reinterpreted + cast to fp32 and
    // fused into the data pass via TCOLEXPANDMUL (col broadcast mul).
    using tile_recip_bf1 = Tile<Location::Vec, __bf16, BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_recip_f1  = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    // Inlined scale-compute intermediates (boxed valid row=1): InT-domain reduced
    // amax and the uint32 bit-math working set for the expanded compute_cublas_core.
    using tile_in1   = Tile<Location::Vec, InT,      BlockSize, TileN, BLayout::RowMajor, 1, TileN>;
    using tile_u32_1 = Tile<Location::Vec, uint32_t, BlockSize, TileN, BLayout::RowMajor, 1, TileN>;

    using gm_x = global_tensor<InT,      RowMajor<Axis, Post>>;
    using gm_y = global_tensor<uint8_t,  RowMajor<Axis, Post>>;
    // AscendC scale layout: uint8 E8M0, one byte per block. Transposed (quant axis
    // is rows): compact [scaleRows, Post] with scaleRows = evenAlign(numKb); the
    // trailing padding block-row is left zero.
    using gm_s = global_tensor<uint8_t,  RowMajor<scaleRows, Post>>;

    global_iterator<gm_x, tile_x> x_iter(x);
    global_iterator<gm_y, tile_o> y_iter(reinterpret_cast<uint8_t *>(y));

    for (int kb = 0; kb < numKb; ++kb) {
        for (int n = 0; n < numN; ++n) {
            auto gx = x_iter(kb, n);
            auto gy = y_iter(kb, n);
            // Compact scale: base pointer folds the block-row index (kb) since the
            // iterator's i-stride is the PHYSICAL tile height, not 1. Iterate the
            // Post columns via s_iter(0, n); each block-row writes TileN bytes.
            global_iterator<gm_s, tile_sstore> s_iter(scale + kb * Post);
            auto gs = s_iter(0, n);

            tile_sred scale_byte;
            tile_sred recip;
            tile_x xq_s;
            TLOAD(xq_s, gx);
            // ================================================================
            // 内联展开：等价于 common::compute_cublas_scale_not_tail<OutT,InT,
            // BlockSize,TileN,MaxLowBoundBits> + common::compute_cublas_core（含其
            // 末尾保留的 IDEAL CmpMode 版）。就地展开以规避 RECORD 问题8（tile 作真实
            // 函数入参 → S64 栈往返 → gfrun 拒）。两处规避已换正式方案：
            //   · reinterpret_f32_to_u32（scratch-HBM，问题4）→ reinterpret_tile<>（零指令视图）
            //   · GT/LT/NE 的 min/max+默认-EQ 模拟（问题3）→ 带 CmpMode 的原生 TCMPS
            // scale 交织（问题5）仍无正式方案，保持 planar（见下方 MISSING INTERLEAVE）。
            // -- compute_cublas_scale_not_tail：InT 域 TABS+TCOLMAX，仅把归约量转 fp32 --
            tile_x abs_x;
            TABS(abs_x, xq_s);
            tile_recip_f1 max_f;
            if constexpr (std::is_same_v<InT, float>) {
                TCOLMAX(max_f, abs_x);      // fp32：直接归约到 fp32（免前置 cast）
            } else {
                tile_in1 max_r;
                TCOLMAX(max_r, abs_x);      // reduce rows -> valid row=1（InT 域）
                TCVT(max_f, max_r);         // bf16/half -> fp32（仅归约后的 per-col 标量）
            }
            // -- compute_cublas_core（IDEAL CmpMode 版，对照 AscendC ComputeScaleCublas）--
            // finite/nonzero 掩码须在原地 clamp 前从 raw 视图算完（视图与 max_f 同寄存器）。
            auto raw = reinterpret_tile<uint32_t>(max_f);        // 问题4 正式方案：零指令
            tile_u32_1 finite;
            TCMPS<CmpMode::LT>(finite, raw, FP32_EXP_MASK);      // raw < 0x7f800000
            tile_u32_1 nonzero;
            TCMPS<CmpMode::NE>(nonzero, raw, static_cast<uint32_t>(0));
            TMAXS(max_f, max_f, __builtin_bit_cast(float, MaxLowBoundBits)); // 原地 clamp
            TMULS(max_f, max_f, inv_dst_max<OutT>());
            // clamp 后再开视图（零指令），再用 u32->u32 恒等 TCVT 把位型物化到真实
            // uint32 tile：后续 TSHRS/TANDS/TAND/TOR/TSEL 都是单模板参（dst/src 必须同类型），
            // 视图类型 ≠ 真实 tile，故须先物化一次；相比 scratch-HBM 往返，这里只一条寄存器级 TCVT。
            auto s32v = reinterpret_tile<uint32_t>(max_f);
            tile_u32_1 s32;
            TCVT(s32, s32v);
            tile_u32_1 exp32;
            TSHRS(exp32, s32, FP32_SHR_NUM);
            tile_u32_1 man32;
            TANDS(man32, s32, FP32_MANTISSA_MASK);
            // p0 = (exp>0) && (exp<254) && (man>0)
            tile_u32_1 p0a; TCMPS<CmpMode::GT>(p0a, exp32, static_cast<uint32_t>(0));
            tile_u32_1 p0b; TCMPS<CmpMode::LT>(p0b, exp32, FP32_NUMBER_254);
            tile_u32_1 p0c; TCMPS<CmpMode::GT>(p0c, man32, static_cast<uint32_t>(0));
            tile_u32_1 pa;
            TAND(pa, p0a, p0b);
            TAND(pa, pa, p0c);
            // p1 = (exp==0) && (man>0x400000)
            tile_u32_1 p1a; TCMPS<CmpMode::EQ>(p1a, exp32, static_cast<uint32_t>(0));
            tile_u32_1 p1b; TCMPS<CmpMode::GT>(p1b, man32, FP32_NUMBER_HALF);
            tile_u32_1 pb;
            TAND(pb, p1a, p1b);
            tile_u32_1 roundup;
            TOR(roundup, pa, pb);
            // extractExp = roundup? exp+1 : exp ; finite? .. : 0xff ; nonzero? .. : 0
            tile_u32_1 exp_p1;
            TADDS(exp_p1, exp32, static_cast<uint32_t>(1));
            tile_u32_1 sel;
            TADDS(sel, exp32, static_cast<uint32_t>(0));
            TSEL(sel, roundup, exp_p1);
            tile_u32_1 nanb;
            TEXPANDS(nanb, FP32_FP8_NAN);
            TSEL(nanb, finite, sel);        // finite? sel : 0xff
            tile_u32_1 extract;
            TEXPANDS(extract, static_cast<uint32_t>(0));
            TSEL(extract, nonzero, nanb);   // nonzero? .. : 0
            TCVT(scale_byte, extract);      // narrow low16
            // recip = 0x7f00 - (extractExp<<7) ; finite? .. : 0x7f81 ; nonzero? .. : 0
            tile_u32_1 sh;
            TSHLS(sh, extract, static_cast<uint32_t>(BF16_SHR_NUM));
            tile_u32_1 bias;
            TEXPANDS(bias, FP32_EXP_BIAS_CUBLAS);
            tile_u32_1 half;
            TSUB(half, bias, sh);
            tile_u32_1 rnan;
            TEXPANDS(rnan, FP32_NAN_PACK);
            TSEL(rnan, finite, half);       // finite? half : 0x7f81
            tile_u32_1 rsel;
            TEXPANDS(rsel, static_cast<uint32_t>(0));
            TSEL(rsel, nonzero, rnan);      // nonzero? .. : 0
            TCVT(recip, rsel);
            // ================================================================
            // scale_byte already boxed valid row=1; narrow to uint8, store 1 byte/block.
            tile_sstore scale_u8;
            TCVT(scale_u8, scale_byte);
            // MISSING INTERLEAVE: this stores scale as a COMPACT planar [scaleRows,
            // Post] layout (block-rows in order). AscendC's mxScale is PARITY-
            // INTERLEAVED [ceil(numKb/2), Post, 2] -- even/odd block-rows zipped via
            // Reg::Interleave (..._not_tail_axis_optimize_high_perf_large_tail.h:511).
            // The zip needs TINTERLEAVE/TDEINTERLEAVE, which LinxISA 0.57 defines but
            // the -D__linx header does not expose (RECORD 问题5). Once exposed, insert
            // a TINTERLEAVE of even/odd block-rows right here before the store.
            TSTORE(gs, scale_u8); // store scale early; scale_byte now dead

            // 问题4 正式方案：reinterpret_tile 零指令把 recip(uint16) 视为 bf16，替代
            // scratch-HBM 的 reinterpret_u16_to_bf16。recip 为具名 uint16 lvalue，满足视图约束。
            auto inv_bf16 = reinterpret_tile<__bf16>(recip);
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            tile_x xq;
            TLOAD(xq, gx);
            tile_o oq;
            if constexpr (std::is_same_v<InT, float>) {
                TCOLEXPANDMUL(xq, xq, inv_scale_f);
                TCVT(oq, xq);
            } else {
                tile_f xf;
                TCVT(xf, xq); // bf16/half -> fp32
                TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
                TCVT(oq, xf);
            }
            TSTORE(gy, oq);
        }
    }

    // Tail column tile: N_tail (< TileN) leftover Post columns. Keep the PHYSICAL
    // tile shape at BlockSize x TileN (so logicalTileBytes stays >= 512B; a
    // TileN'=N_tail recursion would create sub-512B tiles and fail
    // IsValidActiveSize) but box every tile to ValidCol = N_tail so only the live
    // columns are touched. The column tile is addressed at index numN (iterator
    // j-stride uses PHYSICAL TileN).
    if constexpr (N_tail > 0) {
        using tile_x_r      = Tile<Location::Vec, InT,      BlockSize, TileN, BLayout::RowMajor, BlockSize, N_tail>;
        using tile_f_r      = Tile<Location::Vec, float,    BlockSize, TileN, BLayout::RowMajor, BlockSize, N_tail>;
        using tile_o_r      = Tile<Location::Vec, OutT,     BlockSize, TileN, BLayout::RowMajor, BlockSize, N_tail>;
        using tile_sred_r   = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;
        using tile_sstore_r = Tile<Location::Vec, uint8_t,  BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;
        using tile_recip_bf1_r = Tile<Location::Vec, __bf16, BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;
        using tile_recip_f1_r  = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;
        // Inlined scale-compute intermediates (boxed valid row=1, valid col=N_tail).
        using tile_in1_r   = Tile<Location::Vec, InT,      BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;
        using tile_u32_1_r = Tile<Location::Vec, uint32_t, BlockSize, TileN, BLayout::RowMajor, 1, N_tail>;

        global_iterator<gm_x, tile_x_r> x_iter_r(x);
        global_iterator<gm_y, tile_o_r> y_iter_r(reinterpret_cast<uint8_t *>(y));

        for (int kb = 0; kb < numKb; ++kb) {
            auto gx = x_iter_r(kb, numN);
            auto gy = y_iter_r(kb, numN);
            global_iterator<gm_s, tile_sstore_r> s_iter_r(scale + kb * Post);
            auto gs = s_iter_r(0, numN);

            tile_sred_r scale_byte;
            tile_sred_r recip;
            tile_x_r xq_s;
            TLOAD(xq_s, gx);
            // 内联展开：等价于 common::compute_cublas_scale_not_tail<...,N_tail> +
            // compute_cublas_core（IDEAL CmpMode 版），就地展开规避问题8；两处规避
            // 换正式方案（问题4 reinterpret_tile / 问题3 CmpMode）。详见 full loop 注释。
            tile_x_r abs_x;
            TABS(abs_x, xq_s);
            tile_recip_f1_r max_f;
            if constexpr (std::is_same_v<InT, float>) {
                TCOLMAX(max_f, abs_x);      // fp32：直接归约到 fp32（免前置 cast）
            } else {
                tile_in1_r max_r;
                TCOLMAX(max_r, abs_x);      // reduce rows -> valid row=1（InT 域）
                TCVT(max_f, max_r);         // bf16/half -> fp32（仅归约后的 per-col 标量）
            }
            auto raw = reinterpret_tile<uint32_t>(max_f);        // 问题4 正式方案
            tile_u32_1_r finite;
            TCMPS<CmpMode::LT>(finite, raw, FP32_EXP_MASK);      // raw < 0x7f800000
            tile_u32_1_r nonzero;
            TCMPS<CmpMode::NE>(nonzero, raw, static_cast<uint32_t>(0));
            TMAXS(max_f, max_f, __builtin_bit_cast(float, MaxLowBoundBits)); // 原地 clamp
            TMULS(max_f, max_f, inv_dst_max<OutT>());
            // clamp 后再开视图（零指令）+ u32->u32 恒等 TCVT 物化到真实 uint32 tile（见 full loop 注释）。
            auto s32v = reinterpret_tile<uint32_t>(max_f);
            tile_u32_1_r s32;
            TCVT(s32, s32v);
            tile_u32_1_r exp32;
            TSHRS(exp32, s32, FP32_SHR_NUM);
            tile_u32_1_r man32;
            TANDS(man32, s32, FP32_MANTISSA_MASK);
            // p0 = (exp>0) && (exp<254) && (man>0)
            tile_u32_1_r p0a; TCMPS<CmpMode::GT>(p0a, exp32, static_cast<uint32_t>(0));
            tile_u32_1_r p0b; TCMPS<CmpMode::LT>(p0b, exp32, FP32_NUMBER_254);
            tile_u32_1_r p0c; TCMPS<CmpMode::GT>(p0c, man32, static_cast<uint32_t>(0));
            tile_u32_1_r pa;
            TAND(pa, p0a, p0b);
            TAND(pa, pa, p0c);
            // p1 = (exp==0) && (man>0x400000)
            tile_u32_1_r p1a; TCMPS<CmpMode::EQ>(p1a, exp32, static_cast<uint32_t>(0));
            tile_u32_1_r p1b; TCMPS<CmpMode::GT>(p1b, man32, FP32_NUMBER_HALF);
            tile_u32_1_r pb;
            TAND(pb, p1a, p1b);
            tile_u32_1_r roundup;
            TOR(roundup, pa, pb);
            tile_u32_1_r exp_p1;
            TADDS(exp_p1, exp32, static_cast<uint32_t>(1));
            tile_u32_1_r sel;
            TADDS(sel, exp32, static_cast<uint32_t>(0));
            TSEL(sel, roundup, exp_p1);
            tile_u32_1_r nanb;
            TEXPANDS(nanb, FP32_FP8_NAN);
            TSEL(nanb, finite, sel);        // finite? sel : 0xff
            tile_u32_1_r extract;
            TEXPANDS(extract, static_cast<uint32_t>(0));
            TSEL(extract, nonzero, nanb);   // nonzero? .. : 0
            TCVT(scale_byte, extract);      // narrow low16
            tile_u32_1_r sh;
            TSHLS(sh, extract, static_cast<uint32_t>(BF16_SHR_NUM));
            tile_u32_1_r bias;
            TEXPANDS(bias, FP32_EXP_BIAS_CUBLAS);
            tile_u32_1_r half;
            TSUB(half, bias, sh);
            tile_u32_1_r rnan;
            TEXPANDS(rnan, FP32_NAN_PACK);
            TSEL(rnan, finite, half);       // finite? half : 0x7f81
            tile_u32_1_r rsel;
            TEXPANDS(rsel, static_cast<uint32_t>(0));
            TSEL(rsel, nonzero, rnan);      // nonzero? .. : 0
            TCVT(recip, rsel);
            tile_sstore_r scale_u8;
            TCVT(scale_u8, scale_byte);
            TSTORE(gs, scale_u8);

            // 问题4 正式方案：reinterpret_tile 零指令把 recip(uint16) 视为 bf16。
            auto inv_bf16 = reinterpret_tile<__bf16>(recip);
            tile_recip_f1_r inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            tile_x_r xq;
            TLOAD(xq, gx);
            tile_o_r oq;
            if constexpr (std::is_same_v<InT, float>) {
                TCOLEXPANDMUL(xq, xq, inv_scale_f);
                TCVT(oq, xq);
            } else {
                tile_f_r xf;
                TCVT(xf, xq); // bf16/half -> fp32
                TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
                TCVT(oq, xf);
            }
            TSTORE(gy, oq);
        }
    }
}

// Public entry: TileN is NOT a caller knob. It is DERIVED at compile time from
// Post + the InT budget (pick_tilen). If a legal TileN >= the fp8 32B lower bound
// exists, route to the plain single-load path; otherwise (large BlockSize leaves
// no legal TileN) auto-route to the 方案A split-reduce `_bigbs` kernel with a
// budget-derived R_sub. `if constexpr` guarantees the untaken branch is not
// instantiated, so plain's TileN=0 / bigbs's illegal R_sub never fire an assert.
// InT drives BOTH the budget (a wider input dtype shrinks it) AND the compute domain:
// scale-reduce and data paths are InT-dispatched (bf16/half/fp32) via `if constexpr`.
template <int Axis, int Post, int BlockSize = 32, typename OutT = __fp8_e4m3,
          typename InT = __bf16, uint32_t MaxLowBoundBits = 0x2b8cbcccu>
void dynamic_mx_quant_nontail_cublas_fp8(InT *x, OutT *y, uint8_t *scale) {
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    constexpr int TileN = pick_tilen<BlockSize, Post, OutT, InT, /*IsCublas=*/true>();
    if constexpr (TileN >= nontail_align_lower<OutT>()) {
        nontail_cublas_fp8_plain<Axis, Post, BlockSize, TileN, OutT, InT, MaxLowBoundBits>(x, y, scale);
    } else {
        constexpr int BigTileN = nontail_align_lower<OutT>();
        constexpr int Rsub = max_rsub<BlockSize, BigTileN, InT, /*IsCublas=*/true>();
        dynamic_mx_quant_nontail_cublas_fp8_bigbs<Axis, Post, BlockSize, BigTileN, Rsub, OutT,
                                                  InT, MaxLowBoundBits>(x, y, scale);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
