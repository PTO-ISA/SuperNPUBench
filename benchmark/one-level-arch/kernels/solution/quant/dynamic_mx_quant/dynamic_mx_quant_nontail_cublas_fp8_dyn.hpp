#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_DYN_HPP

#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// NON-TAIL-CUBLAS-FP8 —— 运行期动态 shape 版（DYNAMIC SHAPE，最终形态 + 临时入口校验）
//
// 与静态 dynamic_mx_quant_nontail_cublas_fp8 逐 op 等价（cuBLAS scale：InT 域 TABS+
// TCOLMAX 沿行归约 per-column amax，转 fp32，原地 clamp + 位运算抽指数 + 嵌套 TSEL 守卫，
// recip 位补，TCOLEXPANDMUL 逐列广播乘），唯一区别：**Axis(量化轴 M)、Post(自由轴 N)
// 编译期不可知，运行期由 tiling 指针传入**。
//
// ---------------------------------------------------------------------------
// ★ 动态列 + 任意 N ★
//   column tile 声明 physical 列 = 编译期 TileN、valid 列 = 运行期 -1（DYNAMIC），每个列 tile
//   用运行期 validN = min(TileN, Post - n*TileN) 构造。最右列 tile 自然 validN<TileN → **支持
//   任意 N**（实证 Post=176 4-PE gfrun R2=0 + output/scale byte-exact）。量化轴 M 整块 → 行恒满,
//   行维保持编译期静态 BlockSize。注：Axis % BlockSize == 0 是调用方前提（运行期值无法
//   static_assert，同 tail_ocp_fp8_dyn 的 K%BlockSize==0）。
//
// ---------------------------------------------------------------------------
// 动态化范式：
//   · physical tile [BlockSize, TileN] 编译期锁定；ValidRow=BlockSize 静态、ValidCol=-1 动态。
//     TileN 是编译期模板参（Post 运行期未知，无法从 Post 派生；默认 32 = fp8 32B 列对齐下界）。
//   · 运行期下放：Axis/Post/numKb/numN/scaleRows、每 tile validN、kb 分派、global stride。
//   · global_tensor 用 RowMajor<-1,-1>，per-(kb,n) 基址偏移 + 运行期 (Axis,Post) 定行 stride
//     + TLOAD/TSTORE（弃 global_iterator，它依赖编译期 RowStride）。
//   · PE 分派沿量化轴块行 kb 连续切分（静态版本就是运行期公式，此处零改动）。
//
// cuBLAS 核心内联展开（规避问题8 tile 作函数参 S64 往返）+ 合规守卫掩码
// （PTO-REQ-TEPL-COMPARISON-001：TCMPS 产 packed predicate，复合条件用嵌套 TSEL，含
// CmpMode::LT/NE/GT/EQ）——逐行对齐静态版（见 dynamic_mx_quant_nontail_cublas_fp8.hpp 注释）。
//
// SPMD：kPeNum=1（默认，单 PE 全算）/ kPeNum=4（按 kb 切 4 段，须 gfrun -s
//   softcore.multiThreadNum=4）。tiling[0]=Axis（量化轴 M），tiling[1]=Post（自由轴 N）。
// ===========================================================================
template <int BlockSize = 32, int TileN = 128, typename OutT = __fp8_e4m3,
          typename InT = __bf16, uint32_t MaxLowBoundBits = 0x2b8cbcccu, int kPeNum = 1>
void dynamic_mx_quant_nontail_cublas_fp8_dyn(InT *x, OutT *y, uint8_t *scale,
                                             const int64_t *tiling) {
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore.h kCorePeCount)");
    // TileN 编译期约束（physical 列 = tilesize，必须编译期）：32B 列对齐下界 + 256KB 上限。
    static_assert(TileN % nontail_align_lower<OutT>() == 0,
                  "cuBLAS-FP8 output tile is plain RowMajor: TileN must be a multiple "
                  "of the fp8 32B column-alignment lower bound (nontail_align_lower<OutT>)");
    static_assert(BlockSize * TileN <= 65536,
                  "non-tail cuBLAS-FP8 single-load tile exceeds the 256KB TilesizeCode "
                  "ceiling (BlockSize*TileN*4 <= 256KB for the 32b intermediates).");

    using namespace pto;

    // ---- 运行期 shape 与派生量 ----
    const int64_t Axis      = tiling[0];          // 量化轴 M（TCOLMAX 沿其归约）
    const int64_t Post      = tiling[1];          // 自由轴 N（列，per-column 独立量化组）
    // 任意 N：最右列 tile 用 validN<TileN 列 boxed（见头注 WHY），无 Post%TileN 约束。
    const int64_t numKb     = Axis / BlockSize;   // 块行数（= 输出行块 / scale 行）
    const int64_t numN      = (Post + TileN - 1) / TileN; // ceil：缺口修复后自动含 partial 尾列
    const int64_t scaleRows = ((numKb + 1) / 2) * 2; // even-align 补行（尾块 padding 行留零）

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return;   // 冗余 PE 不发指令

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    // 动态 tile：physical [BlockSize, TileN]，ValidRow=BlockSize 静态（行恒满），ValidCol=-1
    //   动态（ctor 传 validN）。colReduce 输出行向量 physical row=1（对齐模型 Block.cpp:2349）。
    using tile_x     = Tile<Location::Vec, InT,      BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_f     = Tile<Location::Vec, float,    BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_o     = Tile<Location::Vec, OutT,     BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_sred   = Tile<Location::Vec, uint16_t, 1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_sstore = Tile<Location::Vec, uint8_t,  1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_recip_f1 = Tile<Location::Vec, float,  1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_in1    = Tile<Location::Vec, InT,      1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_u32_1  = Tile<Location::Vec, uint32_t, 1, TileN, BLayout::RowMajor, 1, -1>;

    // 动态 global_tensor：RowMajor<-1,-1>。x/y 行 stride=Post；scale planar 行 stride=Post。
    using gm_x = global_tensor<InT,     RowMajor<-1, -1>>;
    using gm_y = global_tensor<uint8_t, RowMajor<-1, -1>>;
    using gm_s = global_tensor<uint8_t, RowMajor<-1, -1>>;

    // 单个 [BlockSize, TileN] tile 块的完整计算（scale pass + data pass）。kb = 块行索引，
    //   n = 列 tile 索引。validN = 该列 tile 的有效列宽（满 tile: TileN；缺口修复后尾列<TileN）。
    //   逐 op 对齐静态 nontail_cublas_fp8_plain 内层循环；所有 column tile 用 (vN) 构造。
    auto process_tile = [&](int64_t kb, int64_t n) {
        const int64_t validN = (Post - n * TileN < TileN) ? (Post - n * TileN) : TileN;
        const size_t vN = static_cast<size_t>(validN);
        gm_x gx(x + kb * BlockSize * Post + n * TileN,
                static_cast<int>(Axis), static_cast<int>(Post));
        gm_y gy(y_u8 + kb * BlockSize * Post + n * TileN,
                static_cast<int>(Axis), static_cast<int>(Post));
        gm_s gs(scale + kb * Post + n * TileN,
                static_cast<int>(scaleRows), static_cast<int>(Post));

        tile_sred scale_byte(vN);
        tile_sred recip(vN);
        tile_x xq_s(vN);
        TLOAD(xq_s, gx);
        // -- compute_cublas_scale_not_tail：InT 域 TABS+TCOLMAX，仅把归约量转 fp32 --
        tile_x abs_x(vN);
        TABS(abs_x, xq_s);
        tile_recip_f1 max_f(vN);
        if constexpr (std::is_same_v<InT, float>) {
            TCOLMAX(max_f, abs_x);      // fp32：直接归约到 fp32（免前置 cast）
        } else {
            tile_in1 max_r(vN);
            TCOLMAX(max_r, abs_x);      // reduce rows -> valid row=1（InT 域）
            TCVT(max_f, max_r);         // bf16/half -> fp32（仅归约后的 per-col 标量）
        }
        // -- compute_cublas_core（IDEAL CmpMode 版）--
        auto raw = reinterpret_tile<uint32_t>(max_f);        // 问题4 正式方案：零指令
        tile_u32_1 finite(vN);
        TCMPS<CmpMode::LT>(finite, raw, FP32_EXP_MASK);      // raw < 0x7f800000
        tile_u32_1 nonzero(vN);
        TCMPS<CmpMode::NE>(nonzero, raw, static_cast<uint32_t>(0));
        TMAXS(max_f, max_f, __builtin_bit_cast(float, MaxLowBoundBits)); // 原地 clamp
        TMULS(max_f, max_f, inv_dst_max<OutT>());
        auto s32v = reinterpret_tile<uint32_t>(max_f);
        tile_u32_1 s32(vN);
        TCVT(s32, s32v);
        tile_u32_1 exp32(vN);
        TSHRS(exp32, s32, FP32_SHR_NUM);
        tile_u32_1 man32(vN);
        TANDS(man32, s32, FP32_MANTISSA_MASK);
        // extractExp = ((exp>0 && exp<254 && man>0) || (exp==0 && man>0x400000)) ? exp+1 : exp
        // 合规写法：复合条件用嵌套 TSEL，每个 TSEL 只吃单个直接 compare predicate。
        tile_u32_1 exp_p1(vN);
        TADDS(exp_p1, exp32, static_cast<uint32_t>(1));
        tile_u32_1 sel(vN);
        TADDS(sel, exp32, static_cast<uint32_t>(0));   // 默认 extractExp = exp
        tile_u32_1 c1(vN); TCMPS<CmpMode::GT>(c1, exp32, static_cast<uint32_t>(0));
        tile_u32_1 c2(vN); TCMPS<CmpMode::LT>(c2, exp32, FP32_NUMBER_254);
        tile_u32_1 c3(vN); TCMPS<CmpMode::GT>(c3, man32, static_cast<uint32_t>(0));
        tile_u32_1 n3(vN); TADDS(n3, sel, static_cast<uint32_t>(0)); TSEL(n3, c3, exp_p1); // c3? e+1 : e
        tile_u32_1 n2(vN); TADDS(n2, sel, static_cast<uint32_t>(0)); TSEL(n2, c2, n3);     // c2? n3 : e
        TSEL(sel, c1, n2);                                                                 // c1? n2 : e
        tile_u32_1 c4(vN); TCMPS<CmpMode::EQ>(c4, exp32, static_cast<uint32_t>(0));
        tile_u32_1 c5(vN); TCMPS<CmpMode::GT>(c5, man32, FP32_NUMBER_HALF);
        tile_u32_1 u5(vN); TADDS(u5, sel, static_cast<uint32_t>(0)); TSEL(u5, c5, exp_p1); // c5? e+1 : sel
        TSEL(sel, c4, u5);                                                                 // c4? u5 : sel
        // finite? .. : 0xff ; nonzero? .. : 0
        tile_u32_1 nanb(vN);
        TEXPANDS(nanb, FP32_FP8_NAN);
        TSEL(nanb, finite, sel);        // finite? sel : 0xff
        tile_u32_1 extract(vN);
        TEXPANDS(extract, static_cast<uint32_t>(0));
        TSEL(extract, nonzero, nanb);   // nonzero? .. : 0
        TCVT(scale_byte, extract);      // narrow low16
        // recip = 0x7f00 - (extractExp<<7) ; finite? .. : 0x7f81 ; nonzero? .. : 0
        tile_u32_1 sh(vN);
        TSHLS(sh, extract, static_cast<uint32_t>(BF16_SHR_NUM));
        tile_u32_1 bias(vN);
        TEXPANDS(bias, FP32_EXP_BIAS_CUBLAS);
        tile_u32_1 half_u(vN);
        TSUB(half_u, bias, sh);
        tile_u32_1 rnan(vN);
        TEXPANDS(rnan, FP32_NAN_PACK);
        TSEL(rnan, finite, half_u);     // finite? half : 0x7f81
        tile_u32_1 rsel(vN);
        TEXPANDS(rsel, static_cast<uint32_t>(0));
        TSEL(rsel, nonzero, rnan);      // nonzero? .. : 0
        TCVT(recip, rsel);
        tile_sstore scale_u8(vN);
        TCVT(scale_u8, scale_byte);
        TSTORE(gs, scale_u8); // store scale early; scale_byte now dead

        // 问题4 正式方案：reinterpret_tile 零指令把 recip(uint16) 视为 bf16。
        auto inv_bf16 = reinterpret_tile<__bf16>(recip);
        tile_recip_f1 inv_scale_f(vN);
        TCVT(inv_scale_f, inv_bf16);

        tile_x xq(vN);
        TLOAD(xq, gx);
        tile_o oq(vN);
        if constexpr (std::is_same_v<InT, float>) {
            TCOLEXPANDMUL(xq, xq, inv_scale_f);
            TCVT(oq, xq);
        } else {
            tile_f xf(vN);
            TCVT(xf, xq); // bf16/half -> fp32
            TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
            TCVT(oq, xf);
        }
        TSTORE(gy, oq);
    };

    // ---- SPMD：按块行 kb 连续切分（运行期公式；前 kb_rem 个 PE 各多 1 块行）----
    const int64_t kb_base  = numKb / kPeNum;
    const int64_t kb_rem   = numKb % kPeNum;
    const int64_t itid     = static_cast<int64_t>(tid);
    const int64_t SubKb    = kb_base + (itid < kb_rem ? 1 : 0);
    const int64_t kb_begin = (itid < kb_rem)
                                 ? itid * (kb_base + 1)
                                 : kb_rem * (kb_base + 1) + (itid - kb_rem) * kb_base;
    const int64_t kb_end   = kb_begin + SubKb;

    for (int64_t kb = kb_begin; kb < kb_end; ++kb) {
        for (int64_t n = 0; n < numN; ++n) {
            process_tile(kb, n);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
