#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_CUBLAS_FP8_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_CUBLAS_FP8_DYN_HPP

#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// TAIL-CUBLAS-FP8 —— 运行期动态 shape 版（DYNAMIC SHAPE）
//
// 与静态 dynamic_mx_quant_tail_cublas_fp8 逐 op 等价（cuBLAS scale：InT 域 TABS+
// TROWMAX 归约 amax，per-row 标量转 fp32，原地 clamp + 位运算抽指数 + 嵌套 TSEL 守卫，
// recip 位补，TROWEXPANDMUL 融合），唯一区别：**M、K 编译期不可知，运行期由 tiling
// 指针传入**。BlockSize/OutT/InT/MaxLowBoundBits 是属性 → 保留模板参。动态化范式与
// tail_ocp_fp8_dyn 完全一致：
//   · physical tile 行高 TileM 仅由 BlockSize + InT 预算决定（编译期，M 无关；M 运行期
//     未知，故取预算上界，validRows boxing 兜住 M<TileM——max_tilem 注释保证 physical
//     TileM>M 安全）。
//   · Valid 有效尺寸全部下放运行期：数据/reduce tile 声明 ValidRow=-1（DYNAMIC），ctor
//     传运行期 vr。列 valid 保持编译期静态（BlockSize 或 1）—— B.DIM 立即数要求列维
//     编译期可知。
//   · global_tensor 用 RowMajor<-1,-1>，ctor 传运行期 shape → 行 stride 运行期；用基址
//     偏移 + TLOAD/TSTORE（弃 global_iterator）。
//   · full-tile 与尾块共用同一 Valid=-1 类型，process_tile 退化为普通运行期 lambda，PE
//     分派退化为运行期公式。
//
// cuBLAS 核心内联展开（规避问题8 tile 作函数参 S64 往返）+ 合规守卫掩码
// （PTO-REQ-TEPL-COMPARISON-001：TCMPS 产 packed predicate，复合条件用嵌套 TSEL，含
// CmpMode::LT/NE/GT/EQ）——逐行对齐静态版（见 dynamic_mx_quant_tail_cublas_fp8.hpp 注释）。
//
// SPMD：kPeNum=1（默认，单 PE 全算）/ kPeNum=4（按 tid 切 4 段，须 gfrun -s
//   softcore.multiThreadNum=4）。tiling[0]=M（行/自由轴），tiling[1]=K（尾轴/量化轴，
//   K%BlockSize==0）。
// ===========================================================================
template <int BlockSize = 32, typename OutT = __fp8_e4m3, typename InT = __bf16,
          uint32_t MaxLowBoundBits = 0x2b8cbcccu, int kPeNum = 1>
void dynamic_mx_quant_tail_cublas_fp8_dyn(InT *x, OutT *y, uint8_t *scale,
                                          const int64_t *tiling) {
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore.h kCorePeCount)");

    using namespace pto;

    // physical TileM —— 仅由 BlockSize + InT 预算决定（编译期）。M 运行期未知：用一个
    //   足够大的 sentinel M 逼 max_tilem 落到预算上界（min(M,tilem_max) → tilem_max）；
    //   validRows boxing 处理真实 M<TileM 的尾块（max_tilem 注释：physical TileM>M 安全）。
    constexpr int TileM = max_tilem<(1 << 20), BlockSize, InT, /*IsCublas=*/true>(); // BS=32/fp16 -> 32

    // ---- 运行期 shape 与派生量 ----
    const int64_t M         = tiling[0];
    const int64_t K         = tiling[1];
    const int64_t numKb     = K / BlockSize;
    const int64_t scaleCols = ((numKb + 1) / 2) * 2; // even-align 补列

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return; // 冗余 PE 不发指令

    uint8_t *y_u8 = reinterpret_cast<uint8_t *>(y);

    // 动态 global_tensor：RowMajor<-1,-1>。x/y 行 stride=K；scale compact 行 stride=scaleCols。
    using gm_x = global_tensor<InT,     RowMajor<-1, -1>>;
    using gm_y = global_tensor<uint8_t, RowMajor<-1, -1>>;
    using gm_s = global_tensor<uint8_t, RowMajor<-1, -1>>;

    // 动态 Valid tile：physical [TileM, 列]，ValidRow=-1（ctor 传 vr），列静态。全宽
    //   tile Cols=BlockSize；列向量中间 tile Cols=1（匹配 model rowReduce 无条件 col=1，
    //   令 reduce→下游 physical 列全等，见静态版注释）。
    using tile_x        = Tile<Location::Vec, InT,      TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using tile_f        = Tile<Location::Vec, float,    TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using tile_o        = Tile<Location::Vec, OutT,     TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using tile_sred     = Tile<Location::Vec, uint16_t, TileM, 1, BLayout::RowMajor, -1, 1>;
    using tile_sstore   = Tile<Location::Vec, uint8_t,  TileM, 1, BLayout::RowMajor, -1, 1>;
    using tile_recip_f1 = Tile<Location::Vec, float,    TileM, 1, BLayout::RowMajor, -1, 1>;
    using tile_in1      = Tile<Location::Vec, InT,      TileM, 1, BLayout::RowMajor, -1, 1>;
    using tile_u32_1    = Tile<Location::Vec, uint32_t, TileM, 1, BLayout::RowMajor, -1, 1>;

    // 单个 tile-行块的完整计算（scale pass + data pass）。row0 = 全局起始行；validRows =
    //   活跃行数（full-tile: TileM；尾块: seg_tail<TileM）。全部 tile 用同一 Valid=-1 类型。
    auto process_tile = [&](int64_t row0, int64_t validRows) {
        const size_t vr = static_cast<size_t>(validRows);
        for (int64_t kb = 0; kb < numKb; ++kb) {
            gm_x gx(x + row0 * K + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(K));
            gm_y gy(y_u8 + row0 * K + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(K));
            gm_s gs(scale + row0 * scaleCols + kb,
                    static_cast<int>(M), static_cast<int>(scaleCols));

            // ComputeScale pass：InT 域归约 amax，仅把归约后的 per-row 标量转 fp32。
            tile_sred scale_byte(vr);
            tile_sred recip(vr);
            tile_x    xq_s(vr);
            TLOAD(xq_s, gx);
            // -- compute_cublas_scale_tail：InT 域 TABS+TROWMAX --
            tile_x abs_x(vr);
            TABS(abs_x, xq_s);
            tile_recip_f1 max_f(vr);
            if constexpr (std::is_same_v<InT, float>) {
                TROWMAX(max_f, abs_x);      // fp32：直接归约到 fp32（免前置 cast）
            } else {
                tile_in1 max_r(vr);
                TROWMAX(max_r, abs_x);      // reduce cols -> valid col=1（InT 域）
                TCVT(max_f, max_r);         // bf16/half -> fp32（仅归约后的 per-row 标量）
            }
            // -- compute_cublas_core（IDEAL CmpMode 版）--
            // finite/nonzero 掩码须在原地 clamp 前从 raw 视图算完（视图与 max_f 同寄存器）。
            auto raw = reinterpret_tile<uint32_t>(max_f);        // 问题4 正式方案：零指令
            tile_u32_1 finite(vr);
            TCMPS<CmpMode::LT>(finite, raw, FP32_EXP_MASK);      // raw < 0x7f800000
            tile_u32_1 nonzero(vr);
            TCMPS<CmpMode::NE>(nonzero, raw, static_cast<uint32_t>(0));
            TMAXS(max_f, max_f, __builtin_bit_cast(float, MaxLowBoundBits)); // 原地 clamp
            TMULS(max_f, max_f, inv_dst_max<OutT>());
            // clamp 后再开视图（零指令）+ u32->u32 恒等 TCVT 物化到真实 uint32 tile。
            auto s32v = reinterpret_tile<uint32_t>(max_f);
            tile_u32_1 s32(vr);
            TCVT(s32, s32v);
            tile_u32_1 exp32(vr);
            TSHRS(exp32, s32, FP32_SHR_NUM);
            tile_u32_1 man32(vr);
            TANDS(man32, s32, FP32_MANTISSA_MASK);
            // extractExp = ((exp>0 && exp<254 && man>0) || (exp==0 && man>0x400000)) ? exp+1 : exp
            // 合规写法：复合条件用嵌套 TSEL，每个 TSEL 只吃单个直接 compare predicate。
            tile_u32_1 exp_p1(vr);
            TADDS(exp_p1, exp32, static_cast<uint32_t>(1));
            tile_u32_1 sel(vr);
            TADDS(sel, exp32, static_cast<uint32_t>(0));   // 默认 extractExp = exp
            tile_u32_1 c1(vr); TCMPS<CmpMode::GT>(c1, exp32, static_cast<uint32_t>(0));
            tile_u32_1 c2(vr); TCMPS<CmpMode::LT>(c2, exp32, FP32_NUMBER_254);
            tile_u32_1 c3(vr); TCMPS<CmpMode::GT>(c3, man32, static_cast<uint32_t>(0));
            tile_u32_1 n3(vr); TADDS(n3, sel, static_cast<uint32_t>(0)); TSEL(n3, c3, exp_p1); // c3? e+1 : e
            tile_u32_1 n2(vr); TADDS(n2, sel, static_cast<uint32_t>(0)); TSEL(n2, c2, n3);     // c2? n3 : e
            TSEL(sel, c1, n2);                                                                 // c1? n2 : e
            tile_u32_1 c4(vr); TCMPS<CmpMode::EQ>(c4, exp32, static_cast<uint32_t>(0));
            tile_u32_1 c5(vr); TCMPS<CmpMode::GT>(c5, man32, FP32_NUMBER_HALF);
            tile_u32_1 u5(vr); TADDS(u5, sel, static_cast<uint32_t>(0)); TSEL(u5, c5, exp_p1); // c5? e+1 : sel
            TSEL(sel, c4, u5);                                                                 // c4? u5 : sel
            // finite? .. : 0xff ; nonzero? .. : 0
            tile_u32_1 nanb(vr);
            TEXPANDS(nanb, FP32_FP8_NAN);
            TSEL(nanb, finite, sel);        // finite? sel : 0xff
            tile_u32_1 extract(vr);
            TEXPANDS(extract, static_cast<uint32_t>(0));
            TSEL(extract, nonzero, nanb);   // nonzero? .. : 0
            TCVT(scale_byte, extract);      // narrow low16
            // recip = 0x7f00 - (extractExp<<7) ; finite? .. : 0x7f81 ; nonzero? .. : 0
            tile_u32_1 sh(vr);
            TSHLS(sh, extract, static_cast<uint32_t>(BF16_SHR_NUM));
            tile_u32_1 bias(vr);
            TEXPANDS(bias, FP32_EXP_BIAS_CUBLAS);
            tile_u32_1 half_u(vr);
            TSUB(half_u, bias, sh);
            tile_u32_1 rnan(vr);
            TEXPANDS(rnan, FP32_NAN_PACK);
            TSEL(rnan, finite, half_u);     // finite? half : 0x7f81
            tile_u32_1 rsel(vr);
            TEXPANDS(rsel, static_cast<uint32_t>(0));
            TSEL(rsel, nonzero, rnan);      // nonzero? .. : 0
            TCVT(recip, rsel);
            // scale_byte already boxed valid col=1; narrow to uint8, store 1 byte/block.
            tile_sstore scale_u8(vr);
            TCVT(scale_u8, scale_byte);
            TSTORE(gs, scale_u8); // store scale early; scale_byte now dead

            // 问题4 正式方案：reinterpret_tile 零指令把 recip(uint16) 视为 bf16。
            auto inv_bf16 = reinterpret_tile<__bf16>(recip);
            tile_recip_f1 inv_scale_f(vr);
            TCVT(inv_scale_f, inv_bf16);

            // ComputeData pass: reload the value view now.
            tile_x xq(vr);
            TLOAD(xq, gx);
            tile_o oq(vr);
            if constexpr (std::is_same_v<InT, float>) {
                TROWEXPANDMUL(xq, xq, inv_scale_f);
                TCVT(oq, xq);
            } else {
                tile_f xf(vr);
                TCVT(xf, xq); // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, inv_scale_f); // per-row scalar broadcast-mul
                TCVT(oq, xf);
            }
            TSTORE(gy, oq);
        }
    };

    // ---- L1 行切分（运行期公式）：前 row_rem 个 PE 各多 1 行，起点连续 ----
    const int64_t row_base = M / kPeNum;
    const int64_t row_rem  = M % kPeNum;
    const int64_t itid     = static_cast<int64_t>(tid);
    const int64_t SubM     = row_base + (itid < row_rem ? 1 : 0);
    const int64_t row_begin = (itid < row_rem)
                                  ? itid * (row_base + 1)
                                  : row_rem * (row_base + 1) + (itid - row_rem) * row_base;
    if (SubM == 0) return; // M<kPeNum 时的空 PE

    // ---- L2 段内 tiling（运行期）：seg_full 个 full-tile + 可选 seg_tail 尾块 ----
    const int64_t seg_full = SubM / TileM;
    const int64_t seg_tail = SubM % TileM;
    for (int64_t lm = 0; lm < seg_full; ++lm) {
        process_tile(row_begin + lm * TileM, TileM);
    }
    if (seg_tail > 0) {
        process_tile(row_begin + seg_full * TileM, seg_tail);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
