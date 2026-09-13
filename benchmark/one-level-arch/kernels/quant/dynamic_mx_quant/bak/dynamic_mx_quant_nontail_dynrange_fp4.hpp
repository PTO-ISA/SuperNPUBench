#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_DYNRANGE_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_DYNRANGE_FP4_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// =====================================================================
// 未调试 (UNDEBUGGED) — 待改：未套用 nontail_ocp_fp4 的 TileN=64 plain tile 方案，
//   DynRange scale 核心也未逐 op 对齐 AscendC。编译+链接通过不构成验证。
//   测试入口已从 test/.../src 移除。
//   （注：fp4 发射本身已验证可用，非 toolchain-blocked；见 RECORD 问题2。）
//   状态定义/调试标准见本目录 README.md「状态总览」。
// =====================================================================
// Non-tail-axis, DynamicDtypeRange scale (scaleAlg=2), FP4_E2M1 output only.
// DynRange is legal solely for FP4_E2M1 in AscendC. Quantize axis is rows
// (TCOLMAX); fp4 packs 2/byte along contiguous Post -> tile [BlockSize, TileN/2],
// gm_y RowMajor<Axis, Post/2>.
template <int Axis, int Post, int BlockSize = 32, int TileN = 32, typename OutT = __fp4_e2m1x2>
void dynamic_mx_quant_nontail_dynrange_fp4(__bf16 *x, OutT *y, uint16_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % TileN == 0, "Post must be multiple of TileN");
    static_assert(TileN % 2 == 0, "fp4 packs 2/byte along Post; TileN must be even");
    static_assert(std::is_same_v<OutT, __fp4_e2m1x2>,
                  "DynamicDtypeRange is legal only for FP4_E2M1 output");

    constexpr int numKb = Axis / BlockSize;
    constexpr int numN  = Post / TileN;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, __bf16,   BlockSize, TileN,     BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,    BlockSize, TileN,     BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, OutT,     BlockSize, TileN / 2, BLayout::RowMajor>;
    using tile_scale = Tile<Location::Vec, uint16_t, BlockSize, TileN,     BLayout::RowMajor>;

    using gm_x  = global_tensor<__bf16,   RowMajor<Axis, Post>>;
    using gm_xu = global_tensor<uint16_t, RowMajor<Axis, Post>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<Axis, Post / 2>>;
    using gm_s  = global_tensor<uint16_t, RowMajor<numKb * BlockSize, Post>>;

    global_iterator<gm_x,  tile_x>     x_iter(x);
    global_iterator<gm_xu, tile_scale> xu_iter(reinterpret_cast<uint16_t *>(x));
    global_iterator<gm_y,  tile_o>     y_iter(reinterpret_cast<uint8_t *>(y));
    global_iterator<gm_s,  tile_scale> s_iter(scale);

    for (int kb = 0; kb < numKb; ++kb) {
        for (int n = 0; n < numN; ++n) {
            auto gx  = x_iter(kb, n);
            auto gxu = xu_iter(kb, n);
            auto gy  = y_iter(kb, n);
            auto gs  = s_iter(kb, n);

            tile_scale scale_byte;
            tile_scale recip;
            tile_scale x_u16;
            TLOAD(x_u16, gxu);
            compute_dynamic_range_scale_not_tail<OutT, BlockSize, TileN>(x_u16, scale_byte, recip);
            TSTORE(gs, scale_byte);

            tile_x inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
            reinterpret_u16_to_bf16<2, BlockSize, TileN>(recip, inv_bf16);
            tile_f inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            tile_x xq;
            TLOAD(xq, gx);
            tile_f xf;
            TCVT(xf, xq);
            TCOLEXPANDMUL(xf, xf, inv_scale_f);
            tile_o oq;
            TCVT(oq, xf); // fp32 -> packed fp4_e2m1x2 (Post halved)
            TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
