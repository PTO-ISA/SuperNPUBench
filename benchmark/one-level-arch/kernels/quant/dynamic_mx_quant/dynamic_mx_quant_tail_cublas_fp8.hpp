#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_CUBLAS_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_CUBLAS_FP8_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// Tail-axis, cuBLAS scale (scaleAlg=1), FP8 output (E4M3 default, E5M2 valid).
// cuBLAS consumes the fp32 VALUE view (amax via TABS+TROWMAX, guarded exponent
// extract). Two-pass structure keeps peak live tiles low.
//
// BlockSize range: UNBOUNDED. Unlike the non-tail kernels, the tail contiguous
// axis is BlockSize (the quant axis, always a multiple of 32 -> naturally 32B
// aligned), and the free axis is M rows tiled by the tunable TileM. There is no
// alignment-vs-TileSize conflict on a single axis: the binding tile budget
// (TileM*BlockSize) is met by shrinking TileM, so ANY BlockSize is legal. The
// budget itself has two values from the fp32 reinterpret roundtrip (问题4):
// CURRENT (fp32 32b roundtrip) TileM*BlockSize <= 2048; FORMAL (post-bitcast,
// 16b) TileM*BlockSize <= 4096. Either way BlockSize is free — only TileM shrinks
// — so no static_assert on BlockSize is needed here.
//
// TileM is NOT a caller knob: it is DERIVED at compile time from M + the InT
// binding-tile budget (max_tilem<M, BlockSize, InT, /*IsCublas=*/true>()), clamped
// to [tilem_min(>=512B tile), budget/BlockSize] and to M. InT drives BOTH the budget
// (a wider input dtype shrinks TileM) AND the compute domain: scale-reduce and data
// paths are InT-dispatched (bf16/half/fp32) via `if constexpr` (static_assert below).
template <int M, int K, int BlockSize = 32, typename OutT = __fp8_e4m3,
          typename InT = __bf16, uint32_t MaxLowBoundBits = 0x2b8cbcccu>
void dynamic_mx_quant_tail_cublas_fp8(InT *x, OutT *y, uint8_t *scale) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(K % BlockSize == 0, "K must be multiple of BlockSize");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    constexpr int TileM  = max_tilem<M, BlockSize, InT, /*IsCublas=*/true>();
    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;
    constexpr int numKb  = K / BlockSize;
    // AscendC scale layout: uint8 E8M0, one byte per block, compact [M, scaleCols]
    // with the block count even-aligned (scaleColNum_ = CeilDiv(numKb,2)*2). The
    // trailing padding column is left zero. Mirrors dynamic_mx_quant_tail_axis_fp8.h:168.
    constexpr int scaleCols = ((numKb + 1) / 2) * 2;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, InT,      TileM, BlockSize, BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,    TileM, BlockSize, BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, OutT,     TileM, BlockSize, BLayout::RowMajor>;
    using tile_scale = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;
    // Compact scale store: the cuBLAS core now emits scale_byte/recip already
    // boxed valid col=1 (one per-row scalar per block), so we narrow straight to
    // uint8 and store one byte per block — no intermediate TROWMAX.
    using tile_sred   = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    using tile_sstore = Tile<Location::Vec, uint8_t,  TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    // Per-row-scalar (valid col=1) reciprocal, reinterpreted + cast to fp32 and
    // fused into the data pass via TROWEXPANDMUL (row broadcast mul).
    using tile_recip_f1  = Tile<Location::Vec, float,  TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    // Inlined scale-compute intermediates (boxed valid col=1): InT-domain reduced
    // amax and the uint32 bit-math working set for the expanded compute_cublas_core.
    using tile_in1   = Tile<Location::Vec, InT,      TileM, BlockSize, BLayout::RowMajor, TileM, 1>;
    using tile_u32_1 = Tile<Location::Vec, uint32_t, TileM, BlockSize, BLayout::RowMajor, TileM, 1>;

    using gm_x = global_tensor<InT,      RowMajor<M, K>>;
    using gm_y = global_tensor<uint8_t,  RowMajor<M, K>>;
    using gm_s = global_tensor<uint8_t,  RowMajor<M, scaleCols>>;

    global_iterator<gm_x, tile_x> x_iter(x);
    global_iterator<gm_y, tile_o> y_iter(reinterpret_cast<uint8_t *>(y));

    for (int m = 0; m < full_m; ++m) {
        for (int kb = 0; kb < numKb; ++kb) {
            auto gx = x_iter(m, kb);
            auto gy = y_iter(m, kb);
            // Compact scale: base pointer folds the block index (kb) since the
            // iterator's j-stride is the PHYSICAL tile width, not 1. Iterate rows
            // only via s_iter(m, 0); each row writes one byte at column kb.
            global_iterator<gm_s, tile_sstore> s_iter(scale + kb);
            auto gs = s_iter(m, 0);

            // ComputeScale pass: reduce the amax in bf16, cast only the reduced
            // per-row scalar to fp32 (mirrors AscendC; no whole-tile CVT).
            tile_sred scale_byte;
            tile_sred recip;
            tile_x xq_s;
            TLOAD(xq_s, gx);
            // ================================================================
            // 内联展开：等价于 common::compute_cublas_scale_tail<OutT,InT,TileM,
            // BlockSize,MaxLowBoundBits> + common::compute_cublas_core（含其末尾保留
            // 的 IDEAL CmpMode 版）。就地展开以规避 RECORD 问题8（tile 作真实函数入参
            // → S64 栈往返 → gfrun 拒）。两处规避已换正式方案：
            //   · reinterpret_f32_to_u32（scratch-HBM，问题4）→ reinterpret_tile<>（零指令视图）
            //   · GT/LT/NE 的 min/max+默认-EQ 模拟（问题3）→ 带 CmpMode 的原生 TCMPS
            // scale 无需交织（尾轴块行行内已连续，compact 平铺即等价，问题5）。
            // -- compute_cublas_scale_tail：InT 域 TABS+TROWMAX，仅把归约量转 fp32 --
            tile_x abs_x;
            TABS(abs_x, xq_s);
            tile_recip_f1 max_f;
            if constexpr (std::is_same_v<InT, float>) {
                TROWMAX(max_f, abs_x);      // fp32：直接归约到 fp32（免前置 cast）
            } else {
                tile_in1 max_r;
                TROWMAX(max_r, abs_x);      // reduce cols -> valid col=1（InT 域）
                TCVT(max_f, max_r);         // bf16/half -> fp32（仅归约后的 per-row 标量）
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
            // clamp 后再开视图（零指令），再用 u32->u32 恒等 TCVT 把位型物化到真实 uint32
            // tile：后续 TSHRS/TANDS/TAND/TOR/TSEL 都是单模板参（dst/src 必须同类型），视图
            // 类型 ≠ 真实 tile，故须先物化一次；相比 scratch-HBM 往返，这里只一条寄存器级 TCVT。
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
            // scale_byte already boxed valid col=1; narrow to uint8, store 1 byte/block.
            tile_sstore scale_u8;
            TCVT(scale_u8, scale_byte);
            TSTORE(gs, scale_u8); // store scale early; scale_byte now dead

            // 问题4 正式方案：reinterpret_tile 零指令把 recip(uint16) 视为 bf16，替代
            // scratch-HBM 的 reinterpret_u16_to_bf16。recip 为具名 uint16 lvalue，满足视图约束。
            auto inv_bf16 = reinterpret_tile<__bf16>(recip);
            tile_recip_f1 inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            // ComputeData pass: reload the value view now.
            tile_x xq;
            TLOAD(xq, gx);
            tile_o oq;
            if constexpr (std::is_same_v<InT, float>) {
                // fp32 input already in the compute domain: mul in fp32 directly
                // (mirrors AscendC ComputeData fp32 branch, no pre-cast).
                TROWEXPANDMUL(xq, xq, inv_scale_f);
                TCVT(oq, xq);
            } else {
                tile_f xf;
                TCVT(xf, xq); // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, inv_scale_f); // per-row scalar broadcast-mul
                TCVT(oq, xf);
            }
            TSTORE(gy, oq);
        }
    }

    // Tail block: M_tail (< TileM) leftover rows. Keep the PHYSICAL tile shape at
    // TileM x BlockSize (so logicalTileBytes stays >= 512B; a TileM'=M_tail
    // recursion would create sub-512B tiles and fail IsValidActiveSize) but box
    // every tile to ValidRow = M_tail so only the live rows are touched. The row
    // block is addressed at index full_m (iterator i-stride uses PHYSICAL Rows).
    if constexpr (M_tail > 0) {
        using tile_x_r      = Tile<Location::Vec, InT,      TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_f_r      = Tile<Location::Vec, float,    TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_o_r      = Tile<Location::Vec, OutT,     TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_scale_r  = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, M_tail, BlockSize>;
        using tile_sred_r   = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_sstore_r = Tile<Location::Vec, uint8_t,  TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_recip_f1_r  = Tile<Location::Vec, float,  TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_in1_r   = Tile<Location::Vec, InT,      TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;
        using tile_u32_1_r = Tile<Location::Vec, uint32_t, TileM, BlockSize, BLayout::RowMajor, M_tail, 1>;

        global_iterator<gm_x, tile_x_r> x_iter_r(x);
        global_iterator<gm_y, tile_o_r> y_iter_r(reinterpret_cast<uint8_t *>(y));

        for (int kb = 0; kb < numKb; ++kb) {
            auto gx = x_iter_r(full_m, kb);
            auto gy = y_iter_r(full_m, kb);
            global_iterator<gm_s, tile_sstore_r> s_iter_r(scale + kb);
            auto gs = s_iter_r(full_m, 0);

            tile_sred_r scale_byte;
            tile_sred_r recip;
            tile_x_r xq_s;
            TLOAD(xq_s, gx);
            // 内联展开（同 full loop，boxed ValidM=M_tail、valid col=1）：等价
            // common::compute_cublas_scale_tail<...,M_tail> + compute_cublas_core（IDEAL 版）。
            // 规避问题8；reinterpret_tile 替 scratch-HBM(问题4)；原生 CmpMode 替模拟(问题3)。
            tile_x_r abs_x;
            TABS(abs_x, xq_s);
            tile_recip_f1_r max_f;
            if constexpr (std::is_same_v<InT, float>) {
                TROWMAX(max_f, abs_x);
            } else {
                tile_in1_r max_r;
                TROWMAX(max_r, abs_x);
                TCVT(max_f, max_r);
            }
            auto raw = reinterpret_tile<uint32_t>(max_f);
            tile_u32_1_r finite;
            TCMPS<CmpMode::LT>(finite, raw, FP32_EXP_MASK);
            tile_u32_1_r nonzero;
            TCMPS<CmpMode::NE>(nonzero, raw, static_cast<uint32_t>(0));
            TMAXS(max_f, max_f, __builtin_bit_cast(float, MaxLowBoundBits));
            TMULS(max_f, max_f, inv_dst_max<OutT>());
            auto s32v = reinterpret_tile<uint32_t>(max_f);
            tile_u32_1_r s32;
            TCVT(s32, s32v);
            tile_u32_1_r exp32;
            TSHRS(exp32, s32, FP32_SHR_NUM);
            tile_u32_1_r man32;
            TANDS(man32, s32, FP32_MANTISSA_MASK);
            tile_u32_1_r p0a; TCMPS<CmpMode::GT>(p0a, exp32, static_cast<uint32_t>(0));
            tile_u32_1_r p0b; TCMPS<CmpMode::LT>(p0b, exp32, FP32_NUMBER_254);
            tile_u32_1_r p0c; TCMPS<CmpMode::GT>(p0c, man32, static_cast<uint32_t>(0));
            tile_u32_1_r pa;
            TAND(pa, p0a, p0b);
            TAND(pa, pa, p0c);
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
            TSEL(nanb, finite, sel);
            tile_u32_1_r extract;
            TEXPANDS(extract, static_cast<uint32_t>(0));
            TSEL(extract, nonzero, nanb);
            TCVT(scale_byte, extract);
            tile_u32_1_r sh;
            TSHLS(sh, extract, static_cast<uint32_t>(BF16_SHR_NUM));
            tile_u32_1_r bias;
            TEXPANDS(bias, FP32_EXP_BIAS_CUBLAS);
            tile_u32_1_r half;
            TSUB(half, bias, sh);
            tile_u32_1_r rnan;
            TEXPANDS(rnan, FP32_NAN_PACK);
            TSEL(rnan, finite, half);
            tile_u32_1_r rsel;
            TEXPANDS(rsel, static_cast<uint32_t>(0));
            TSEL(rsel, nonzero, rnan);
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
                TROWEXPANDMUL(xq, xq, inv_scale_f);
                TCVT(oq, xq);
            } else {
                tile_f_r xf;
                TCVT(xf, xq); // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, inv_scale_f); // per-row scalar broadcast-mul
                TCVT(oq, xf);
            }
            TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
