#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_SPLITN_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_CUBLAS_FP8_SPLITN_DYN_HPP

#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// NON-TAIL-CUBLAS-FP8 —— 运行期动态 shape 版，**按自由轴 N 切分 PE**（SPLIT-N）
//
// 计算逻辑与 dynamic_mx_quant_nontail_cublas_fp8_dyn **逐 op 完全一致**（同一个 process_tile:
// InT 域 TABS+TCOLMAX per-column amax，转 fp32，原地 clamp + 位运算抽指数 + 嵌套 TSEL 守卫，
// recip 位补，TCOLEXPANDMUL 逐列广播乘，最终形态 ValidCol=-1 + 运行期 validN + 入口校验）。
// **唯一区别在 SPMD 切分维度**：
//   · 本 kernel（SPLIT-N）：沿**自由轴列 tile 索引 n** 切分 —— 每个 PE 拿一段列 tile
//     [n_begin, n_end)，内部遍历**全部块行 kb**。是尾轴切 M（自由轴）的干净转置。
//   · kb 版（默认 nontail_..._dyn）：沿**量化轴块行 kb** 切分 —— 每个 PE 拿一段块行，内部遍历
//     全部列 tile。
//
// 切分粒度 = **TileN（一个列 tile）**，不是裸列 —— 切分线永远落在 tile 边界上，故不产生跨 PE
//   被切开的 tile / 内部 partial 列。每个 PE 写**不重叠的列带**（各自的 n 段 × 全部行），
//   scale/output 互不依赖，**无 barrier**（与 kb 版同）。
//
// 负载均衡取舍：SPLIT-N 在 **numN >= kPeNum**（列 tile 够多）时均衡；kb 版在
//   **numKb >= kPeNum**（块行够多）时均衡。扁矩阵（numKb 小、Post 大）宜用 SPLIT-N；
//   高瘦矩阵（numKb 大、Post 小）宜用 kb 版。
//
// ★ 动态列 + 任意 N，同 kb 版（见 dynamic_mx_quant_nontail_cublas_fp8_dyn.hpp 头详注）：
//   column tile ValidCol=-1 + 运行期 validN=min(TileN,Post-n*TileN)，最右列 tile 列 boxed →
//   支持任意 N。Axis%BlockSize==0 是调用方前提（行恒满，行维编译期静态）。
//
// SPMD：kPeNum=1（默认，单 PE 全算）/ kPeNum=4（按 n 切，须 gfrun -s
//   softcore.multiThreadNum=4）。tiling[0]=Axis（量化轴 M），tiling[1]=Post（自由轴 N）。
// ===========================================================================
template <int BlockSize = 32, int TileN = 128, typename OutT = __fp8_e4m3,
          typename InT = __bf16, uint32_t MaxLowBoundBits = 0x2b8cbcccu, int kPeNum = 1>
void dynamic_mx_quant_nontail_cublas_fp8_splitN_dyn(InT *x, OutT *y, uint8_t *scale,
                                                    const int64_t *tiling) {
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore.h kCorePeCount)");
    static_assert(TileN % nontail_align_lower<OutT>() == 0,
                  "cuBLAS-FP8 output tile is plain RowMajor: TileN must be a multiple "
                  "of the fp8 32B column-alignment lower bound (nontail_align_lower<OutT>)");
    static_assert(BlockSize * TileN <= 65536,
                  "non-tail cuBLAS-FP8 single-load tile exceeds the 256KB TilesizeCode "
                  "ceiling (BlockSize*TileN*4 <= 256KB for the 32b intermediates).");

    using namespace pto;

    const int64_t Axis      = tiling[0];          // 量化轴 M（TCOLMAX 沿其归约）
    const int64_t Post      = tiling[1];          // 自由轴 N（列，per-column 独立量化组）
    // 任意 N：最右列 tile 用 validN<TileN 列 boxed（见头注 WHY），无 Post%TileN 约束。
    const int64_t numKb     = Axis / BlockSize;   // 块行数
    const int64_t numN      = (Post + TileN - 1) / TileN; // ceil：缺口修复后自动含 partial 尾列
    const int64_t scaleRows = ((numKb + 1) / 2) * 2; // even-align 补行

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return;   // 冗余 PE 不发指令

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    using tile_x     = Tile<Location::Vec, InT,      BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_f     = Tile<Location::Vec, float,    BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_o     = Tile<Location::Vec, OutT,     BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_sred   = Tile<Location::Vec, uint16_t, 1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_sstore = Tile<Location::Vec, uint8_t,  1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_recip_f1 = Tile<Location::Vec, float,  1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_in1    = Tile<Location::Vec, InT,      1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_u32_1  = Tile<Location::Vec, uint32_t, 1, TileN, BLayout::RowMajor, 1, -1>;

    using gm_x = global_tensor<InT,     RowMajor<-1, -1>>;
    using gm_y = global_tensor<uint8_t, RowMajor<-1, -1>>;
    using gm_s = global_tensor<uint8_t, RowMajor<-1, -1>>;

    // 单个 [BlockSize, TileN] tile 块的完整计算（与 kb 版 process_tile 逐 op 一致）。
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
        tile_x abs_x(vN);
        TABS(abs_x, xq_s);
        tile_recip_f1 max_f(vN);
        if constexpr (std::is_same_v<InT, float>) {
            TCOLMAX(max_f, abs_x);      // fp32：直接归约到 fp32
        } else {
            tile_in1 max_r(vN);
            TCOLMAX(max_r, abs_x);      // reduce rows -> valid row=1（InT 域）
            TCVT(max_f, max_r);         // bf16/half -> fp32
        }
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
        tile_u32_1 exp_p1(vN);
        TADDS(exp_p1, exp32, static_cast<uint32_t>(1));
        tile_u32_1 sel(vN);
        TADDS(sel, exp32, static_cast<uint32_t>(0));   // 默认 extractExp = exp
        tile_u32_1 c1(vN); TCMPS<CmpMode::GT>(c1, exp32, static_cast<uint32_t>(0));
        tile_u32_1 c2(vN); TCMPS<CmpMode::LT>(c2, exp32, FP32_NUMBER_254);
        tile_u32_1 c3(vN); TCMPS<CmpMode::GT>(c3, man32, static_cast<uint32_t>(0));
        tile_u32_1 n3(vN); TADDS(n3, sel, static_cast<uint32_t>(0)); TSEL(n3, c3, exp_p1);
        tile_u32_1 n2(vN); TADDS(n2, sel, static_cast<uint32_t>(0)); TSEL(n2, c2, n3);
        TSEL(sel, c1, n2);
        tile_u32_1 c4(vN); TCMPS<CmpMode::EQ>(c4, exp32, static_cast<uint32_t>(0));
        tile_u32_1 c5(vN); TCMPS<CmpMode::GT>(c5, man32, FP32_NUMBER_HALF);
        tile_u32_1 u5(vN); TADDS(u5, sel, static_cast<uint32_t>(0)); TSEL(u5, c5, exp_p1);
        TSEL(sel, c4, u5);
        tile_u32_1 nanb(vN);
        TEXPANDS(nanb, FP32_FP8_NAN);
        TSEL(nanb, finite, sel);        // finite? sel : 0xff
        tile_u32_1 extract(vN);
        TEXPANDS(extract, static_cast<uint32_t>(0));
        TSEL(extract, nonzero, nanb);   // nonzero? .. : 0
        TCVT(scale_byte, extract);      // narrow low16
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
        TSTORE(gs, scale_u8);

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

    // ---- SPMD：按**列 tile 索引 n** 连续切分（TILE 粒度；前 n_rem 个 PE 各多 1 列 tile）----
    const int64_t n_base  = numN / kPeNum;
    const int64_t n_rem   = numN % kPeNum;
    const int64_t itid    = static_cast<int64_t>(tid);
    const int64_t SubN    = n_base + (itid < n_rem ? 1 : 0);
    const int64_t n_begin = (itid < n_rem)
                                ? itid * (n_base + 1)
                                : n_rem * (n_base + 1) + (itid - n_rem) * n_base;
    const int64_t n_end   = n_begin + SubN;

    for (int64_t n = n_begin; n < n_end; ++n) {
        for (int64_t kb = 0; kb < numKb; ++kb) {
            process_tile(kb, n);
        }
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
