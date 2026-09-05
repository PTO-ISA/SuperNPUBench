#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_DYNRANGE_FP4_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_DYNRANGE_FP4_HPP

#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// =====================================================================
// 未调试 (UNDEBUGGED) — 待改：未套用 tail_ocp_fp4 已落地的「每对 block 子归约 +
//   scratch-HBM concat 到 64 宽再单次打包 fp4」方案，DynRange scale 核心也未逐 op
//   对齐 AscendC。编译+链接通过不构成验证。测试入口已从 test/.../src 移除。
//   （注：fp4 发射本身已验证可用，非 toolchain-blocked；见 RECORD 问题2。）
//   状态定义/调试标准见本目录 README.md「状态总览」。
// =====================================================================
// Tail-axis, DynamicDtypeRange scale (scaleAlg=2), FP4_E2M1 output only.
// DynRange is legal solely for FP4_E2M1 in AscendC (add-value 0x003F rounding,
// emax=0x0100). Output packed 2 fp4/byte -> tile Cols=BlockSize/2, gm_y K/2.
template <int M, int K, int TileM = 8, int BlockSize = 32, typename OutT = __fp4_e2m1x2>
void dynamic_mx_quant_tail_dynrange_fp4(__bf16 *x, OutT *y, uint16_t *scale) {
    static_assert(M > 0 && K > 0, "dim must be positive");
    static_assert(K % BlockSize == 0, "K must be multiple of BlockSize");
    static_assert(BlockSize % 2 == 0, "fp4 packs 2/byte; BlockSize must be even");
    static_assert(std::is_same_v<OutT, __fp4_e2m1x2>,
                  "DynamicDtypeRange is legal only for FP4_E2M1 output");

    constexpr int full_m = M / TileM;
    constexpr int M_tail = M % TileM;
    constexpr int numKb  = K / BlockSize;

    using namespace pto;

    using tile_x     = Tile<Location::Vec, __bf16,   TileM, BlockSize,     BLayout::RowMajor>;
    using tile_f     = Tile<Location::Vec, float,    TileM, BlockSize,     BLayout::RowMajor>;
    using tile_o     = Tile<Location::Vec, OutT,     TileM, BlockSize / 2, BLayout::RowMajor>;
    using tile_scale = Tile<Location::Vec, uint16_t, TileM, BlockSize,     BLayout::RowMajor>;

    using gm_xu = global_tensor<uint16_t, RowMajor<M, K>>;
    using gm_x  = global_tensor<__bf16,   RowMajor<M, K>>;
    using gm_y  = global_tensor<uint8_t,  RowMajor<M, K / 2>>;
    using gm_s  = global_tensor<uint16_t, RowMajor<M, numKb * BlockSize>>;

    global_iterator<gm_x,  tile_x>     x_iter(x);
    global_iterator<gm_xu, tile_scale> xu_iter(reinterpret_cast<uint16_t *>(x));
    global_iterator<gm_y,  tile_o>     y_iter(reinterpret_cast<uint8_t *>(y));
    global_iterator<gm_s,  tile_scale> s_iter(scale);

    for (int m = 0; m < full_m; ++m) {
        for (int kb = 0; kb < numKb; ++kb) {
            auto gx  = x_iter(m, kb);
            auto gxu = xu_iter(m, kb);
            auto gy  = y_iter(m, kb);
            auto gs  = s_iter(m, kb);

            tile_scale scale_byte;
            tile_scale recip;
            tile_scale x_u16;
            TLOAD(x_u16, gxu);
            compute_dynamic_range_scale_tail<OutT, TileM, BlockSize>(x_u16, scale_byte, recip);
            TSTORE(gs, scale_byte);

            tile_x inv_bf16;
            // WORKAROUND: 寄存器级 reinterpret 未支持，经 HBM 字节别名规避，详见 RECORD.md 问题5
            reinterpret_u16_to_bf16<2, TileM, BlockSize>(recip, inv_bf16);
            tile_f inv_scale_f;
            TCVT(inv_scale_f, inv_bf16);

            tile_x xq;
            TLOAD(xq, gx);
            tile_f xf;
            TCVT(xf, xq);
            TMUL(xf, xf, inv_scale_f);
            tile_o oq;
            TCVT(oq, xf); // fp32 -> packed fp4_e2m1x2 (Cols halved)
            TSTORE(gy, oq);
        }
    }

    if constexpr (M_tail > 0) {
        dynamic_mx_quant_tail_dynrange_fp4<M_tail, K, TileM, BlockSize, OutT>(
            x + full_m * K,
            reinterpret_cast<OutT *>(reinterpret_cast<uint8_t *>(y) + full_m * (K / 2)),
            scale + full_m * numKb * BlockSize);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
