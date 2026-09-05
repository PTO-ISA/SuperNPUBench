#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP8_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP8_HPP

#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// =====================================================================
// 未调试 (UNDEBUGGED) — 仍用广播版 compute_ocp_scale_not_tail（uint16 广播 scale，
//   归约轴未 ÷BlockSize），OCP scale 核心未逐 op 对齐 AscendC，正确性存疑。
//   编译+链接通过不构成验证。测试入口已从 test/.../src 移除。
//   状态定义/调试标准见本目录 README.md「状态总览」。
// =====================================================================
// Non-tail-axis, OCP scale (scaleAlg=0), FP8 output (E4M3 default, E5M2 valid).
// Quantize axis is rows (Axis), reduced along BlockSize rows via TCOLMAX; the
// per-column scale broadcasts down each block via TCOLEXPANDMUL. Two-pass
// structure keeps peak live tiles low (LinxV5RegisterInfo.cpp:403).
template <int Axis, int Post, int BlockSize = 32, int TileN = 32, typename OutT = __fp8_e4m3>
void dynamic_mx_quant_nontail_ocp_fp8(__bf16 *x, OutT *y, uint16_t *scale) {
    static_assert(Axis > 0 && Post > 0, "dims must be positive");
    static_assert(Axis % BlockSize == 0, "Axis must be multiple of BlockSize");
    static_assert(Post % TileN == 0, "Post must be multiple of TileN");

    constexpr int numKb = Axis / BlockSize;
    constexpr int numN  = Post / TileN;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, __bf16,   BlockSize, TileN, BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,    BlockSize, TileN, BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, OutT,     BlockSize, TileN, BLayout::RowMajor>;
    using tile_scale = Tile<Location::Vec, uint16_t, BlockSize, TileN, BLayout::RowMajor>;

    using gm_x  = global_tensor<__bf16,   RowMajor<Axis, Post>>;
    using gm_xu = global_tensor<uint16_t, RowMajor<Axis, Post>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<Axis, Post>>;
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
            compute_ocp_scale_not_tail<OutT, BlockSize, TileN>(x_u16, scale_byte, recip);
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
            TCVT(oq, xf);
            TSTORE(gy, oq);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
