// ============================================================================
// GELU 算子 — PTO 一层编程模型 (debug 变体: tile 不跨函数传递)
//
// 与 gelu_pto.hpp 的区别:
//   - 删除 gelu_impl 函数模板, 把 tile 计算体 (TCVT→clamp→t²→Horner→exp→
//     recip→TCVT) 直接内联进 gelu_debug() 的 full-tile 主循环 与 rmd 余数分支
//     (两处直接重复, 各自用本地 tile_shapeFP32 / tile_shapeFP32_rmd 类型)。
//   - 所有 PTO tile 指令与 tile 对象在同一函数作用域内, 不跨函数传递 tile ——
//     对当前 __vec__ 后端与未来真实 tile-register 内联汇编后端都更稳。
//   - 删除死参数 tmpCvt / tmpCvt_rmd (原 gelu_impl 体内从不引用)。
//   - 入口改名 gelu_debug; 算法/系数逐行与 gelu_pto.hpp 一致。
//
// PTO 一层策略 (与 gelu_pto.hpp 一致):
//   TLOAD (half) -> TCVT(fp16→fp32) -> tile 指令链计算 GELU -> TCVT(fp32→fp16)
//   -> TSTORE (half); 全部用 Tile 级内联函数, 无 __vec__ 块。
//
// (PTO ISA 指令/编译器状态表与 TCVT/TMAXS/TMINS/TMUL/TMULS/TADDS/TEXP/TRECIP
//  的签名说明见 gelu_pto.hpp 顶部文档注释, 此处不重复。)
// ============================================================================

#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>
// #include <pto/pto_instr.hpp>            // [!] PTO ISA C++ Intrinsic — 当前编译器未提供

#include <cstdint>
#include <cstdio>

// ============================================================================
// GELU 多项式拟合系数 (与 gelu.hpp 一致)
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
// gelu_debug: 主入口 (tile 不跨函数传递)
//
// 算法 (与 gelu_pto.hpp::gelu_impl 一致, 此处内联):
//   x  = (float)in
//   t  = clamp(x, -5.75, 5.75)
//   t2 = t * t
//   p  = Horner(t2, [A5,A4,A3,A2,A1,A0,AM1])
//   e  = exp(t * p)
//   y  = x * (1 / (1 + e))           // 用 TRECIP + TMUL 代替除法
//   out = (half)y
//
// full-tile 主循环 与 rmd 余数分支 各内联一份计算体; 所有 tile 在本函数作用域
// 内声明与使用, 不经函数参数传递。
// ----------------------------------------------------------------------------
template<typename dtype, int gM, int tM>
void gelu_debug(
    dtype *in_ptr,
    dtype *out_ptr,
    bool /*approximate*/ = false     // PTO 版本只做多项式拟合, 与 gelu_pto.hpp 一致保留接口
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
    tile_shapeData_rmd inTile_rmd, outTile_rmd;

    for (int i = 0; i < Mb; ++i) {
        auto gI = gIIter(0, i);
        auto gO = gOIter(0, i);

        // TLOAD: GM -> UB
        // [当前编译器] 名为 TCOPYIN, jcore 为 __vec__
        TLOAD(inTile, gI);

        // ---- GELU compute (inlined, tiles stay in gelu_debug scope) ----
        //   输入:  inTile  — fp16 tile, shape (1, tM)
        //   输出:  outTile — fp16 tile, shape (1, tM)
        //   中间:  全部在 fp32 tile 上计算
        {
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
            // [当前编译器] TMULS/TADDS API 有, jcore 为 __vec__
            TMULS(pTile, t2Tile, gelu_coeffs::A5);
            TADDS(pTile, pTile, gelu_coeffs::A4);

            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::A3);

            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::A2);

            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::A1);

            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::A0);

            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::AM1);

            // ---- Step 5: exp_val = exp(t * p) ----
            TMUL(scratchTile, tTile, pTile);
            // [当前编译器] TEXP API 有, jcore 为 __vec__
            // (template_asm.h 有 TEXP_TEPL 内联汇编, 但不在 pto_tileop.hpp 中)
            TEXP(scratchTile, scratchTile);          // scratch = exp(t*p)

            // ---- Step 6: y = x / (1 + exp_val) ----
            TADDS(scratchTile, scratchTile, (fp_t)1.0f);   // scratch = 1 + exp
            // [当前编译器] TRECIP API 有, jcore 为 __vec__
            // (template_asm.h 有 TRECIP_TEPL 内联汇编, 但不在 pto_tileop.hpp 中)
            TRECIP(scratchTile, scratchTile);               // scratch = 1 / (1+exp)
            TMUL(scratchTile, xTile, scratchTile);           // scratch = x * recip = y

            // ---- Step 7: fp32 -> fp16 ----
            TCVT(outTile, scratchTile);
        }

        // TSTORE: UB -> GM
        // [当前编译器] 名为 TCOPYOUT, jcore 为 __vec__
        TSTORE(gO, outTile);
    }

    if constexpr (rmd_M) {
        auto gI = gIIter(0, Mb);
        auto gO = gOIter(0, Mb);

        TLOAD(inTile_rmd, gI);

        // ---- GELU compute (inlined, rmd tiles) ----
        //   与上面 full-tile 计算体逐行一致, 仅 tile 类型换为 *_rmd
        {
            using fp_t = typename tile_shapeFP32_rmd::DType;   // float

            tile_shapeFP32_rmd xTile;
            tile_shapeFP32_rmd tTile;
            tile_shapeFP32_rmd t2Tile;
            tile_shapeFP32_rmd pTile;
            tile_shapeFP32_rmd scratchTile;

            TCVT(xTile, inTile_rmd);

            TMAXS(tTile, xTile, (fp_t)(-gelu_coeffs::CLAMP_MAX));
            TMINS(tTile, tTile, (fp_t)gelu_coeffs::CLAMP_MAX);

            TMUL(t2Tile, tTile, tTile);

            TMULS(pTile, t2Tile, gelu_coeffs::A5);
            TADDS(pTile, pTile, gelu_coeffs::A4);
            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::A3);
            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::A2);
            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::A1);
            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::A0);
            TMUL(pTile, pTile, t2Tile);
            TADDS(pTile, pTile, gelu_coeffs::AM1);

            TMUL(scratchTile, tTile, pTile);
            TEXP(scratchTile, scratchTile);
            TADDS(scratchTile, scratchTile, (fp_t)1.0f);
            TRECIP(scratchTile, scratchTile);
            TMUL(scratchTile, xTile, scratchTile);

            TCVT(outTile_rmd, scratchTile);
        }

        TSTORE(gO, outTile_rmd);
    }
}
