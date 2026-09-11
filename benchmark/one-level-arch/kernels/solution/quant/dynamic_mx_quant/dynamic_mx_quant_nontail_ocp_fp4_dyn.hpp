#ifndef SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_NONTAIL_OCP_FP4_DYN_HPP

#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// NON-TAIL-OCP-FP4 —— 运行期动态 shape 版（DYNAMIC SHAPE，最终形态 + 临时入口校验）
//
// 与静态 dynamic_mx_quant_nontail_ocp_fp4 逐 op 等价（OCP scale：InT 分派值域 TCOLMAX
// 沿行归约 per-column amax，floor 指数，直转 e8m0；倒数位补主路径 + inf/zero/special 三守卫；
// fp4(e2m1) 打包输出），唯一区别：**Axis(量化轴 M)、Post(自由轴 N) 编译期不可知，运行期由
// tiling 指针传入**。
//
// ---------------------------------------------------------------------------
// ★ 动态列 + 任意 N ★
//   column tile 声明 physical 列 = 编译期 TileN、valid 列 = 运行期 -1（DYNAMIC），每个列 tile
//   用运行期 validN = min(TileN, Post - n*TileN) 构造。最右列 tile 自然 validN<TileN → **支持
//   任意 N**（Post 需偶数，fp4 打包 2/字节的固有要求；实证 Post=160 4-PE gfrun R2=0 + output/
//   scale pass）。量化轴 M 整块 → 行恒满，行维保持编译期静态。注：Axis%BlockSize==0 是调用方
//   前提（运行期值无法 static_assert）。
//
// ---------------------------------------------------------------------------
// 动态化范式：physical tile [BlockSize, TileN] 编译期锁定，ValidRow=BlockSize 静态、
//   ValidCol=-1 动态；TileN 编译期模板参（默认 64 —— fp4 打包输出 [BlockSize, TileN/2] 字节需
//   32B 列对齐 → (TileN/2)*8 % 256 == 0 → TileN%64==0，一个 tile 跨 ≥2 个 MX 块）。运行期下放:
//   Axis/Post/numKb/numN/scaleRows、每 tile validN、kb 分派、global stride（RowMajor<-1,-1>：
//   x 行 stride=Post，y 打包字节域行 stride=Post/2，scale planar 行 stride=Post；基址偏移 +
//   TLOAD/TSTORE，弃 global_iterator）。PE 分派沿量化轴块行 kb 连续切分。
//
// fp4 专属（逐 op 对齐静态版）：InT 分派 reduce（bf16 经 fp32 域 TABS）；e8m0 直转；
//   finalize_recip_u16 内联（同载体 uint16 视图，三守卫 TCMPS+TSEL）；fp4 element-列形输出
//   tile [BlockSize, TileN]（TCVT 打包两 4bit/字节，落 Post/2 字节域）；emax 由 OutT 派生。
//
// SPMD：kPeNum=1 / kPeNum=4（按 kb 切，须 gfrun -s softcore.multiThreadNum=4）。
//   tiling[0]=Axis（量化轴 M），tiling[1]=Post（自由轴 N）。
// ===========================================================================
template <int BlockSize = 32, int TileN = 128, typename OutT = __fp4_e2m1x2,
          typename InT = __bf16, int kPeNum = 1>
void dynamic_mx_quant_nontail_ocp_fp4_dyn(InT *x, OutT *y, uint8_t *scale,
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

    // emax 由 OutT 派生（逐值等价静态版）。
    constexpr uint16_t RECIP_EMAX = recip_emax_bits<OutT>();

    // ---- 运行期 shape 与派生量 ----
    const int64_t Axis      = tiling[0];            // 量化轴 M（TCOLMAX 沿其归约）
    const int64_t Post      = tiling[1];            // 自由轴 N（列，per-column 量化组）
    // 任意 N（Post 偶数）：最右列 tile 用 validN<TileN 列 boxed（见头注 WHY），无 Post%TileN 约束。
    const int64_t numKb     = Axis / BlockSize;     // 块行数
    const int64_t numN      = (Post + TileN - 1) / TileN; // ceil：缺口修复后自动含 partial 尾列
    const int64_t scaleRows = ((numKb + 1) / 2) * 2; // even-align 补行（padding 行留零）

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return;   // 冗余 PE 不发指令

    uint8_t    *y_u8     = reinterpret_cast<uint8_t *>(y);
    __fp8_e8m0 *scale_e8 = reinterpret_cast<__fp8_e8m0 *>(scale);

    // 动态 tile：physical [BlockSize, TileN]，ValidRow=BlockSize 静态、ValidCol=-1 动态。
    //   fp4 输出 element-列形（physical Cols=TileN → TCVT 走 fp4 打包 specialization）。
    //   colReduce 输出行向量 physical row=1（对齐模型 Block.cpp:2349）。
    using tile_x   = Tile<Location::Vec, InT,    BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_f   = Tile<Location::Vec, float,  BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_o   = Tile<Location::Vec, OutT,   BlockSize, TileN, BLayout::RowMajor, BlockSize, -1>;
    using tile_maxh      = Tile<Location::Vec, __half,     1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_maxf      = Tile<Location::Vec, float,      1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_se8m0     = Tile<Location::Vec, __fp8_e8m0, 1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_recip_bf1 = Tile<Location::Vec, __bf16,     1, TileN, BLayout::RowMajor, 1, -1>;
    using tile_recip_f1  = Tile<Location::Vec, float,      1, TileN, BLayout::RowMajor, 1, -1>;

    // 动态 global_tensor：RowMajor<-1,-1>。
    using gm_x = global_tensor<InT,        RowMajor<-1, -1>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<-1, -1>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<-1, -1>>;

    // 单个 [BlockSize, TileN] tile 块的完整计算。validN = 该列 tile 有效列宽（满 tile: TileN；
    //   缺口修复后尾列<TileN）。逐 op 对齐静态 nontail_ocp_fp4_plain；所有 column tile 用 (vN)。
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
            tile_f xf32(vN); TCVT(xf32, xin);               // bf16 -> fp32（规避 TABS 拒 bf16）
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
        TCVT(scale_e8m0, shared_bf);                    // bf16 -> e8m0（inf/nan->0xff 硬件）

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

        // scale store：planar [scaleRows, Post] one byte/block（PTO-ISA Shared B-scale [G,N]，
        //   无 parity interleave，见静态版注释）。
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
            TCOLEXPANDMUL(xq, xq, inv_scale_f); // fp32 domain mul (no pre-cast)
            TCVT(oq, xq);                        // fp32 -> packed fp4_e2m1x2 (Post halved)
        } else {
            tile_f xf(vN);
            TCVT(xf, xq); // bf16/half -> fp32
            TCOLEXPANDMUL(xf, xf, inv_scale_f); // per-column scalar broadcast-mul
            TCVT(oq, xf);                        // fp32 -> packed fp4_e2m1x2 (Post halved)
        }
        // fp4 打包字节域 gm_y：块行 kb 偏 kb*BlockSize 行 × (Post/2) 字节行距，列块 n 偏
        //   n*(TileN/2) 字节；element-列形 tile store 两 4bit/字节。
        gm_y gy(y_u8 + kb * BlockSize * (Post / 2) + n * (TileN / 2),
                static_cast<int>(Axis), static_cast<int>(Post / 2));
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
