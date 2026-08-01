#ifndef SUPERNPU_DYNAMIC_MX_QUANT_COMMON_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_COMMON_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

namespace supernpu::tile_isa::mxquant {

enum class ScaleAlg { OCP, CUBLAS, DYNAMIC_RANGE };

constexpr uint16_t BF16_EXP_MASK = 0x7F80;
constexpr uint16_t BF16_ABS_MASK = 0x7FFF;
constexpr uint16_t BF16_EXP_BIAS = 0x7F00;
constexpr uint16_t BF16_SHR_NUM = 7;
constexpr uint16_t BF16_NAN_PATTERN = 0x7F81;
constexpr uint16_t BF16_SPECIAL_EXP = 0x0040;

constexpr uint16_t FP8_E4M3_EMAX = 0x0400;
constexpr float FP8_E4M3_DST_MAX = 448.0f;
constexpr float FP8_E4M3_INV_DST_MAX = 1.0f / 448.0f;
constexpr uint16_t FP8_NAN_BYTE = 0x00FF;

constexpr uint32_t FP32_EXP_MASK = 0x7F800000;
constexpr uint32_t FP32_MANTISSA_MASK = 0x007FFFFF;
constexpr uint32_t FP32_EXP_BIAS = 0x3F800000;
constexpr uint32_t FP32_SHR_NUM = 23;

constexpr float CLAMP_MIN = 1e-12f;
constexpr int kScaleStride = 8;

template <int TileM, int BlockSize>
void compute_ocp_scale_byte(
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &x_u16,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &scale_byte
) {
    using namespace pto;
    using tile_u16 = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;

    tile_u16 exp_bits;
    TANDS(exp_bits, x_u16, BF16_EXP_MASK);

    tile_u16 max_exp;
    TROWMAX(max_exp, exp_bits);

    tile_u16 eq_nan;
    TCMPS(eq_nan, max_exp, BF16_EXP_MASK);

    TMAXS(max_exp, max_exp, FP8_E4M3_EMAX);

    tile_u16 emax_tile;
    TEXPANDS(emax_tile, FP8_E4M3_EMAX);

    tile_u16 shared_exp;
    TSUB(shared_exp, max_exp, emax_tile);

    TSHRS(scale_byte, shared_exp, static_cast<uint16_t>(BF16_SHR_NUM));

    tile_u16 nan_byte;
    TEXPANDS(nan_byte, FP8_NAN_BYTE);
    TSEL(scale_byte, eq_nan, nan_byte);
}

template <int TileM, int BlockSize>
void compute_cublas_scale_byte(
    Tile<Location::Vec, float, TileM, BlockSize, BLayout::RowMajor> &x_f32,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &scale_byte
) {
    using namespace pto;
    using tile_f32 = Tile<Location::Vec, float, TileM, BlockSize, BLayout::RowMajor>;
    using tile_u16 = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;
    using tile_u32 = Tile<Location::Vec, uint32_t, TileM, BlockSize, BLayout::RowMajor>;

    tile_f32 abs_x;
    TABS(abs_x, x_f32);

    tile_f32 max_abs;
    TROWMAX(max_abs, abs_x);

    tile_f32 s_fp32;
    TMULS(s_fp32, max_abs, FP8_E4M3_INV_DST_MAX);

    tile_u32 s_bits;
    TCAST(s_bits, s_fp32);

    tile_u32 exp_bits;
    TSHRS(exp_bits, s_bits, static_cast<uint32_t>(FP32_SHR_NUM));

    tile_u32 man_bits;
    TANDS(man_bits, s_bits, FP32_MANTISSA_MASK);

    tile_u32 zero_u32;
    TEXPANDS(zero_u32, static_cast<uint32_t>(0));
    tile_u32 man_nz;
    TCMP(man_nz, man_bits, zero_u32);

    tile_u32 exp_plus1;
    TADDS(exp_plus1, exp_bits, static_cast<uint32_t>(1));
    TSEL(exp_bits, man_nz, exp_plus1);

    tile_u16 scale_u16;
    TCAST(scale_u16, exp_bits);
    TSEL(scale_byte, scale_u16, scale_byte);
}

template <int TileM, int BlockSize>
void compute_dynamic_range_scale_byte(
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &x_u16,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &scale_byte
) {
    using namespace pto;
    using tile_u16 = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;

    constexpr uint16_t ADD_VALUE = 0x003F;

    tile_u16 abs_x;
    TANDS(abs_x, x_u16, BF16_ABS_MASK);

    tile_u16 max_abs;
    TROWMAX(max_abs, abs_x);

    tile_u16 max_abs_rounded;
    TADDS(max_abs_rounded, max_abs, ADD_VALUE);

    tile_u16 exp_bits;
    TANDS(exp_bits, max_abs_rounded, BF16_EXP_MASK);

    tile_u16 eq_nan;
    TCMPS(eq_nan, exp_bits, BF16_EXP_MASK);

    TMAXS(exp_bits, exp_bits, FP8_E4M3_EMAX);

    tile_u16 emax_tile;
    TEXPANDS(emax_tile, FP8_E4M3_EMAX);

    tile_u16 shared_exp;
    TSUB(shared_exp, exp_bits, emax_tile);

    TSHRS(scale_byte, shared_exp, static_cast<uint16_t>(BF16_SHR_NUM));

    tile_u16 nan_byte;
    TEXPANDS(nan_byte, FP8_NAN_BYTE);
    TSEL(scale_byte, eq_nan, nan_byte);
}

template <ScaleAlg Alg, int TileM, int BlockSize>
void compute_scale_byte(
    Tile<Location::Vec, __bf16, TileM, BlockSize, BLayout::RowMajor> &x_bf16,
    Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor> &scale_byte
) {
    using tile_u16 = Tile<Location::Vec, uint16_t, TileM, BlockSize, BLayout::RowMajor>;
    tile_u16 x_u16;
    TCAST(x_u16, x_bf16);

    if constexpr (Alg == ScaleAlg::OCP) {
        compute_ocp_scale_byte<TileM, BlockSize>(x_u16, scale_byte);
    } else if constexpr (Alg == ScaleAlg::CUBLAS) {
        using tile_f32 = Tile<Location::Vec, float, TileM, BlockSize, BLayout::RowMajor>;
        tile_f32 x_f32;
        TCVT(x_f32, x_bf16);
        compute_cublas_scale_byte<TileM, BlockSize>(x_f32, scale_byte);
    } else {
        compute_dynamic_range_scale_byte<TileM, BlockSize>(x_u16, scale_byte);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
