#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_SPLITN_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_SPLITN_DYN_HPP

#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// NON-TAIL-OCP-FP4 —— 运行期动态 shape 版，**按自由轴 N 切分 PE**（SPLIT-N）
//
// 计算逻辑与 dynamic_mx_quant_nontail_ocp_fp4_dyn **逐 op 完全一致**（同一 process_tile:
// InT 分派值域 TCOLMAX per-column amax，floor 指数，直转 e8m0；倒数位补 + inf/zero/special
// 三守卫；fp4 打包输出；最终形态 ValidCol=-1 + 运行期 validN + 入口校验）。
// **唯一区别在 SPMD 切分维度**：
//   · 本 kernel（SPLIT-N）：沿**自由轴列 tile 索引 n** 切分 —— 每个 PE 拿一段列 tile
//     [n_begin, n_end)，内部遍历**全部块行 kb**。是尾轴切 M（自由轴）的干净转置。
//   · kb 版（默认 nontail_..._dyn）：沿**量化轴块行 kb** 切分。
//
// 切分粒度 = **TileN（一个列 tile）**，切分线永远落在 tile 边界，不产生跨 PE 被切开的 tile /
//   内部 partial 列。每个 PE 写不重叠的列带（n 段 × 全部行），**无 barrier**。
// 负载均衡：SPLIT-N 宜 numN>=kPeNum（列 tile 够多），kb 版宜 numKb>=kPeNum。
//
// ★ 动态列 + 任意 N，同 kb 版（见 dynamic_mx_quant_nontail_ocp_fp4_dyn.hpp 头详注）：
//   column tile ValidCol=-1 + 运行期 validN=min(TileN,Post-n*TileN)，最右列 tile 列 boxed →
//   支持任意 N（Post 偶数，fp4 打包固有）。Axis%BlockSize==0 是调用方前提（行恒满，行维编译期静态）。
//
// SPMD：kPeNum=1 / kPeNum=4（按 n 切，须 gfrun -s softcore.multiThreadNum=4）。
//   tiling[0]=Axis（量化轴 M），tiling[1]=Post（自由轴 N）。
// ===========================================================================
template <int BlockSize = 32, int TileN = 128, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16, int kPeNum = 1>
void dynamic_mx_quant_nontail_ocp_fp4_splitN_dyn(InT *x, OutT *y, uint8_t *scale,
                                                 const int64_t *tiling) {
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore kCorePeCount)");
    static_assert(TileN % 64 == 0,
                  "fp4 output tile is plain RowMajor NoneBox: (TileN/2)*8 % 256 == 0 "
                  "requires TileN a multiple of 64 (>=2 MX blocks along Post)");
    static_assert(BlockSize * TileN <= 65536,
                  "non-tail OCP-FP4 single-load tile exceeds the 256KB TilesizeCode "
                  "ceiling (BlockSize*TileN*4 <= 256KB).");

    using namespace pto;

    constexpr uint16_t RECIP_EMAX = recip_emax_bits<OutT>();

    const int64_t Axis      = tiling[0];            // 量化轴 M（TCOLMAX 沿其归约）
    const int64_t Post      = tiling[1];            // 自由轴 N（列，per-column 量化组）
    // 任意 N（Post 偶数）：最右列 tile 用 validN<TileN 列 boxed（见头注 WHY），无 Post%TileN 约束。
    const int64_t numKb     = Axis / BlockSize;     // 块行数
    const int64_t numN      = (Post + TileN - 1) / TileN; // ceil：缺口修复后自动含 partial 尾列
    const int64_t scaleRows = ((numKb + 1) / 2) * 2; // even-align 补行

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return;   // 冗余 PE 不发指令

    uint8_t    *y_u8     = reinterpret_cast<uint8_t *>(y);
    __fp8_e8m0 *scale_e8 = reinterpret_cast<__fp8_e8m0 *>(scale);

    using tile_x   = Tile<Location::Vec, InT,    BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_f   = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_o   = Tile<Location::Vec, OutT,   BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_maxh      = Tile<Location::Vec, __half,     1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_maxf      = Tile<Location::Vec, float,      1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, 1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_recip_bf1 = Tile<Location::Vec, __bf16,     1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_recip_f1  = Tile<Location::Vec, float,      1, TileN, BLayout::RowMajor, 1, -1>;

    using gm_x = global_tensor<InT,        RowMajor<-1, -1>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<-1, -1>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<-1, -1>>;

    // 单个 [BlockSize, TileN] tile 块的完整计算（与 kb 版 process_tile 逐 op 一致）。
    auto process_tile = [&](int64_t kb, int64_t n) {
        const int64_t validN = (Post - n * TileN < TileN) ? (Post - n * TileN) : TileN;
        const size_t vN = static_cast<size_t>(validN);
        gm_x gx(x + kb * BlockSize * Post + n * TileN,
                static_cast<int>(Axis), static_cast<int>(Post));

        // --- 值域归约（InT 分派，TABS 白名单 FP16/FP32 → bf16 先转 fp32）---
        tile_x xin(vN);
        TLOAD(xin, gx);
        tile_recip_bf1 max_bf(vN);
        if constexpr (std::is_same_v<InT, __half>) {
            tile_x abs_h(vN); TABS(abs_h, xin);
            tile_maxh max_h(vN); TCOLMAX(max_h, abs_h);   // reduce 行 -> valid row=1
            TCVT(max_bf, max_h);                            // half -> bf16
        } else if constexpr (std::is_same_v<InT, float>) {
            tile_x abs_f(vN); TABS(abs_f, xin);
            tile_maxf max_f(vN); TCOLMAX(max_f, abs_f);
            TCVT(max_bf, max_f);                            // fp32 -> bf16
        } else { // bf16
            tile_f xf32(vN); TCVT(xf32, xin);               // bf16 -> fp32
            tile_f abs_f(vN); TABS(abs_f, xf32);
            tile_maxf max_f(vN); TCOLMAX(max_f, abs_f);
            TCVT(max_bf, max_f);                            // fp32 -> bf16
        }
        // --- 清尾数留 2^E_max，乘 2^-emax，直转 e8m0 ---
        auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
        TANDS(max_u16, max_u16, BF16_EXP_MASK);
        tile_recip_bf1 shared_bf(vN);
        TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX));
        tile_se8m0 scale_e8m0(vN);
        TCVT(scale_e8m0, shared_bf);                    // bf16 -> e8m0

        // --- finalize_recip_u16 内联（问题8）：同载体 uint16 视图，三守卫 ---
        auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
        tile_recip_bf1 recip_bf(vN), eqinf_bf(vN), eqzero_bf(vN), eqspc_bf(vN), k_bf(vN);
        auto recip_u16  = reinterpret_tile<uint16_t>(recip_bf);
        auto eq_inf     = reinterpret_tile<uint16_t>(eqinf_bf);
        auto eq_zero    = reinterpret_tile<uint16_t>(eqzero_bf);
        auto eq_special = reinterpret_tile<uint16_t>(eqspc_bf);
        auto k_u16      = reinterpret_tile<uint16_t>(k_bf);
        TCMPS(eq_inf,     max_u16,    BF16_EXP_MASK);              // NOT finite
        TCMPS(eq_zero,    max_u16,    static_cast<uint16_t>(0));   // 全零块
        TCMPS(eq_special, shared_u16, BF16_EXP_BIAS);             // shared==0x7f00
        TEXPANDS(k_u16, BF16_EXP_BIAS);
        TSUB(recip_u16, k_u16, shared_u16);                       // 0x7f00 - shared
        TEXPANDS(k_u16, BF16_NAN_PATTERN);   TSEL(recip_u16, eq_inf, k_u16);        // inf -> 0x7f81
        TEXPANDS(k_u16, static_cast<uint16_t>(0)); TSEL(recip_u16, eq_zero, k_u16); // 全零 -> 0
        TEXPANDS(k_u16, BF16_SPECIAL_EXP);   TSEL(recip_u16, eq_special, k_u16);    // special -> 0x0040

        gm_s gs(scale_e8 + kb * Post + n * TileN,
                static_cast<int>(scaleRows), static_cast<int>(Post));
        TSTORE(gs, scale_e8m0);

        tile_recip_f1 inv_scale_f(vN);
        TCVT(inv_scale_f, recip_bf);                    // 问题4 消除：recip_bf 直接转 fp32

        // --- data pass ---
        tile_x xq(vN);
        TLOAD(xq, gx);
        tile_o oq(vN);
        if constexpr (std::is_same_v<InT, float>) {
            TCOLEXPANDMUL(xq, xq, inv_scale_f); // fp32 domain mul
            TCVT(oq, xq);                        // fp32 -> packed fp4_e2m1x2
        } else {
            tile_f xf(vN);
            TCVT(xf, xq); // bf16/half -> fp32
            TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
            TCVT(oq, xf);                        // fp32 -> packed fp4_e2m1x2
        }
        gm_y gy(y_u8 + kb * BlockSize * (Post / 2) + n * (TileN / 2),
                static_cast<int>(Axis), static_cast<int>(Post / 2));
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
