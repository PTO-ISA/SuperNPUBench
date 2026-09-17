// ============================================================================
// GELU 算子 — PTO 一层编程模型
//
// 原始 gelu.hpp 策略:
//   TCOPYIN (half) -> __vec__ gelu_simd (多项式拟合) -> TCOPYOUT (half)
//   __vec__ 块逐元素: fp16→fp32, clamp, Horner 多项式, exp, 除法, fp32→fp16
//
// PTO 一层策略:
//   TLOAD (half) -> TCVT(fp16→fp32) -> tile 指令链计算 GELU -> TCVT(fp32→fp16) -> TSTORE (half)
//   全部用 Tile 级内联函数, 无 __vec__ 块
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │                    当前编译器不支持 / 不完整的指令汇总                   │
// ├──────────┬──────────────────┬──────────────────────────────────────────┤
// │ Pto ISA  │ 当前编译器状态   │ 说明                                     │
// │ 指令     │                  │                                          │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TLOAD    │ API 有(名不同)， │ PTO ISA 名 TLOAD；当前编译器名 TCOPYIN；  │
// │          │ 二层实现         │ jcore/TCopyIn.hpp 用 __vec__ 实现        │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TCVT     │ API 有(签名不同)，│ PTO ISA: TCVT(dst,src,tmp,mode,satMode) │
// │          │ 二层实现         │ 当前编译器: TCVT(dst,src) 无 tmp/mode；  │
// │          │                  │ jcore/TCvt.hpp 用 __vec__ 实现           │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TMAXS    │ API 有，二层实现 │ jcore/TMaxs.hpp 用 __vec__ 实现          │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TMINS    │ API 有，二层实现 │ jcore/TMins.hpp 用 __vec__ 实现          │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TMUL     │ API 有，二层实现 │ jcore/TMul.hpp 用 __vec__ 实现           │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TMULS    │ API 有，二层实现 │ jcore/TMuls.hpp 用 __vec__ 实现          │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TADDS    │ API 有，二层实现 │ jcore/TAdds.hpp 用 __vec__ 实现          │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TEXP     │ API 有，二层实现 │ jcore/TExp.hpp 用 __vec__ 实现           │
// │          │                  │ (template_asm.h 有 TEXP_TEPL 内联汇编,    │
// │          │                  │  但不在 pto_tileop.hpp API 中)            │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TRECIP   │ API 有，二层实现 │ jcore/TRecip.hpp 用 __vec__ 实现         │
// │          │                  │ (template_asm.h 有 TRECIP_TEPL 内联汇编,  │
// │          │                  │  但不在 pto_tileop.hpp API 中)            │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TSTORE   │ API 有(名不同)， │ PTO ISA 名 TSTORE；当前编译器名 TCOPYOUT；│
// │          │ 二层实现         │ jcore/TCopyOut.hpp 用 __vec__ 实现       │
// └──────────┴──────────────────┴──────────────────────────────────────────┘
//
// PTO ISA 文档签名 (Declared in include/pto/pto_instr.hpp):
//
//   TLOAD:
//     template <typename TileData, typename GlobalData, typename... WaitEvents>
//     PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
//
//   TCVT:
//     template <typename TileDataD, typename TileDataS, typename TmpTileData,
//               typename... WaitEvents>
//     PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp,
//                               RoundMode mode, SaturationMode satMode,
//                               WaitEvents &... events);
//
//   TMAXS:
//     template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
//     PTO_INST RecordEvent TMAXS(TileDataDst &dst, TileDataSrc &src,
//                                typename TileDataSrc::DType scalar,
//                                WaitEvents &... events);
//
//   TMINS:
//     template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
//     PTO_INST RecordEvent TMINS(TileDataDst &dst, TileDataSrc &src,
//                                typename TileDataSrc::DType scalar,
//                                WaitEvents &... events);
//
//   TMUL:
//     template <typename TileDataDst, typename TileDataSrc0,
//               typename TileDataSrc1, typename... WaitEvents>
//     PTO_INST RecordEvent TMUL(TileDataDst &dst, TileDataSrc0 &src0,
//                               TileDataSrc1 &src1, WaitEvents &... events);
//
//   TMULS:
//     template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
//     PTO_INST RecordEvent TMULS(TileDataDst &dst, TileDataSrc &src0,
//                                typename TileDataSrc::DType scalar,
//                                WaitEvents &... events);
//
//   TADDS:
//     template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
//     PTO_INST RecordEvent TADDS(TileDataDst &dst, TileDataSrc &src0,
//                                typename TileDataSrc::DType scalar,
//                                WaitEvents &... events);
//
//   TEXP:
//     template <auto PrecisionType = ExpAlgorithm::DEFAULT,
//               typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
//     PTO_INST RecordEvent TEXP(TileDataDst &dst, TileDataSrc &src,
//                               WaitEvents &... events);
//
//   TRECIP:
//     template <auto PrecisionType = RecipAlgorithm::DEFAULT,
//               typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
//     PTO_INST RecordEvent TRECIP(TileDataDst &dst, TileDataSrc &src,
//                                 WaitEvents &... events);
//
//   TSTORE:
//     template <typename TileData, typename GlobalData,
//               AtomicType atomicType = AtomicType::AtomicNone,
//               typename... WaitEvents>
//     PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src,
//                                 WaitEvents &... events);
// ============================================================================

#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>
// #include <pto/pto_instr.hpp>            // [!] PTO ISA C++ Intrinsic — 当前编译器未提供

#include <cstdint>
#include <cstdio>

// ============================================================================
// GELU 多项式拟合系数 (与原始 gelu.hpp 一致)
// GELU(x) = x / (1 + exp(t * P(t²)))
// P(t²) = A5*t²⁵ + A4*t²⁴ + A3*t²³ + A2*t²² + A1*t² + A0 + AM1/t²
// (Horner: p = ((((A5*t2 + A4)*t2 + A3)*t2 + A2)*t2 + A1)*t2 + A0)*t2 + AM1)
// ============================================================================
namespace gelu_coeffs {
    constexpr float A5  = -3.5123395303315874e-09f;
    constexpr float A4  =  2.6452661927578447e-07f;
    constexpr float A3  = -7.9294877650681883e-06f;
    constexpr float A2  =  1.1061238183174282e-04f;
    constexpr float A1  =  6.5189960878342390e-05f;
    constexpr float A0  = -7.2666168212890625e-02f;
    constexpr float AM1 = -1.5957698822021484e+00f;
    constexpr float CLAMP_MAX = 5.75f;
}

// ----------------------------------------------------------------------------
// gelu_impl: 用 PTO ISA Tile 指令计算 GELU (一层编程, 无 __vec__ 块)
//
// 输入:  inTile  — fp16 tile, shape (1, tM)
// 输出:  outTile — fp16 tile, shape (1, tM)
// 中间:  全部在 fp32 tile 上计算
//
// 算法:
//   x  = (float)in
//   t  = clamp(x, -5.75, 5.75)
//   t2 = t * t
//   p  = Horner(t2, [A5,A4,A3,A2,A1,A0,AM1])
//   e  = exp(t * p)
//   y  = x * (1 / (1 + e))           // 用 TRECIP + TMUL 代替除法
//   out = (half)y
// ----------------------------------------------------------------------------
template<typename tile_shapeData, typename tile_shapeFP32>
void gelu_impl(
    tile_shapeData  &inTile,
    tile_shapeData  &outTile,
    tile_shapeFP32  &tmpCvt          // TCVT 需要的临时 tile
) {
    using fp_t = typename tile_shapeFP32::DType;   // float

    tile_shapeFP32 xTile;        // x = (float)input
    tile_shapeFP32 tTile;        // t = clamp(x)
    tile_shapeFP32 t2Tile;       // t²
    tile_shapeFP32 pTile;        // 多项式值
    tile_shapeFP32 scratchTile;  // 复用: tp -> exp -> denom -> recip -> y

    // ---- Step 1: fp16 -> fp32 ----
    // [当前编译器] TCVT(dst, src) 无 tmp/mode/satMode 参数; jcore 为 __vec__
    TCVT(xTile, inTile);

    // ---- Step 2: clamp x to [-5.75, 5.75] ----
    // [当前编译器] TMAXS/TMINS API 有, jcore 为 __vec__
    TMAXS(tTile, xTile, (fp_t)(-gelu_coeffs::CLAMP_MAX));   // t = max(x, -5.75)
    TMINS(tTile, tTile, (fp_t)gelu_coeffs::CLAMP_MAX);       // t = min(t, 5.75)

    // ---- Step 3: t² = t * t ----
    // [当前编译器] TMUL API 有, jcore 为 __vec__
    TMUL(t2Tile, tTile, tTile);

    // ---- Step 4: 多项式 Horner 法 ----
    // p = A5*t2 + A4
    // [当前编译器] TMULS/TADDS API 有, jcore 为 __vec__
    TMULS(pTile, t2Tile, gelu_coeffs::A5);
    TADDS(pTile, pTile, gelu_coeffs::A4);

    // p = p*t2 + A3
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::A3);

    // p = p*t2 + A2
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::A2);

    // p = p*t2 + A1
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::A1);

    // p = p*t2 + A0
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::A0);

    // p = p*t2 + AM1
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::AM1);

    // ---- Step 5: exp_val = exp(t * p) ----
    // scratch = t * p
    TMUL(scratchTile, tTile, pTile);
    // exp_val = exp(scratch)
    // [当前编译器] TEXP API 有, jcore 为 __vec__
    // (template_asm.h 有 TEXP_TEPL 内联汇编, 但不在 pto_tileop.hpp 中)
    TEXP(scratchTile, scratchTile);          // scratch = exp(t*p)

    // ---- Step 6: y = x / (1 + exp_val) ----
    // denom = 1 + exp_val
    TADDS(scratchTile, scratchTile, (fp_t)1.0f);   // scratch = 1 + exp
    // recip = 1 / denom
    // [当前编译器] TRECIP API 有, jcore 为 __vec__
    // (template_asm.h 有 TRECIP_TEPL 内联汇编, 但不在 pto_tileop.hpp 中)
    TRECIP(scratchTile, scratchTile);               // scratch = 1 / (1+exp)
    // y = x * recip
    TMUL(scratchTile, xTile, scratchTile);           // scratch = x * recip = y

    // ---- Step 7: fp32 -> fp16 ----
    TCVT(outTile, scratchTile);
}


// ----------------------------------------------------------------------------
// gelu: 主入口, 接口与原 gelu.hpp 一致
// ----------------------------------------------------------------------------
template<typename dtype, int gM, int tM>
void gelu(
    dtype *in_ptr,
    dtype *out_ptr,
    bool approximate = false
    ) {
    using gm_shape       = global_tensor<dtype, RowMajor<1, gM>>;
    using tile_shapeData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>;
    using tile_shapeFP32 = Tile<Location::Vec, float, 1, tM, BLayout::RowMajor>;
    using tile_shapeData_rmd = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor, 1, gM % tM>;
    using tile_shapeFP32_rmd = Tile<Location::Vec, float, 1, tM, BLayout::RowMajor, 1, gM % tM>;

    const int Mb    = gM / tM;
    const int rmd_M = gM % tM;

    using itIn  = global_iterator<gm_shape, tile_shapeData>;
    using itOut = global_iterator<gm_shape, tile_shapeData>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

    tile_shapeData inTile, outTile;
    tile_shapeFP32 tmpCvt;                          // TCVT 临时 tile
    tile_shapeData_rmd inTile_rmd, outTile_rmd;
    tile_shapeFP32_rmd tmpCvt_rmd;

    for (int i = 0; i < Mb; ++i) {
        auto gI = gIIter(0, i);
        auto gO = gOIter(0, i);

        // TLOAD: GM -> UB
        // [当前编译器] 名为 TCOPYIN, jcore 为 __vec__
        TLOAD(inTile, gI);

        gelu_impl<tile_shapeData, tile_shapeFP32>(inTile, outTile, tmpCvt);

        // TSTORE: UB -> GM
        // [当前编译器] 名为 TCOPYOUT, jcore 为 __vec__
        TSTORE(gO, outTile);
    }
    if constexpr (rmd_M) {
        auto gI = gIIter(0, Mb);
        auto gO = gOIter(0, Mb);

        TLOAD(inTile_rmd, gI);
        gelu_impl<tile_shapeData_rmd, tile_shapeFP32_rmd>(inTile_rmd, outTile_rmd, tmpCvt_rmd);
        TSTORE(gO, outTile_rmd);
    }
}
