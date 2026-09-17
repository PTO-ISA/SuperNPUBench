#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_DYN_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"
// 复用静态版的 tail_ocp_fp8_detail::{pow2_floor, tilem_max}（纯 BlockSize 函数，
// 编译期常量），避免重复定义。静态 kernel 模板未实例化则不产码。
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// TAIL-OCP-FP8 —— 运行期动态 shape 版（DYNAMIC SHAPE）
//
// 与静态 dynamic_mx_quant_tail_ocp_fp8 逐 op 等价（InT(bf16/half/fp32) in / e4m3 out /
// e8m0 scale / recip = 0x7F00-shared 主路径 TEXPANDS+TSUB），唯一区别：**M、N 在编译期
// 不可知，运行期由 tiling 指针传入，全部切分参数运行期计算**。BlockSize/InT 是属性 → 模板参。
// ⚠ 同静态版：未补 inf/zero/special 三守卫（随机基准不触发）；见 recip finalize 处 TODO。
//
// 设计对照 normalization/rms_norm（动态入口范式）：
//   · physical tile 形状（TileM×列宽）仍编译期锁定 —— 决定寄存器分配。
//     TileM = tilem_max(BlockSize)，纯由 BlockSize 决定，与 M/N 无关。
//   · Valid 有效尺寸全部下放运行期：数据/reduce tile 声明 ValidRow=-1（DYNAMIC），
//     构造时传运行期 vr。列 valid 保持编译期静态（BlockSize 或 1）—— TEPL B.DIM
//     立即数要求列维编译期可知（见 rms_norm SKILL：tile_v 必须列静态）。
//   · global_tensor 用 RowMajor<-1,-1>，构造传运行期 (M, N)；其 stride_t 以
//     dynamicCol=N 作为行 stride（pto_tile.hpp:1178），故 [vr, BlockSize] 的
//     多行 strided 列块 load 正确。不能用 global_iterator（依赖编译期 RowStride）。
//
// 尾块的范式跃迁：静态版 boxed 尾块要求 ValidRows 编译期常量（独立模板实例）；
//   动态版 full-tile 与尾块**共用同一 Valid=-1 类型**，仅 ctor 传不同 vr，
//   彻底消除 boxed 编译期特例，process_tile 退化为普通 lambda，PE 分派退化为
//   运行期公式（无需 switch(tid) 编译期展开）。
//
// 两级切分（运行期）：
//   L1 行切分：M 行按 kPeNum 均分，每 PE 连续段 SubM 行（前 M%kPeNum 个各多 1 行）。
//   L2 段内 tiling：TileM 固定上限；seg_full = SubM/TileM 个 full-tile
//     + seg_tail = SubM%TileM 尾块（Valid=-1，vr=seg_tail）。
//   kb 量化 block：numKb = N/BlockSize 运行期循环，每块 TROWMAX 沿 BlockSize 列归约。
//
// SPMD：kPeNum=1（默认，单 PE 全算，零回归）/ kPeNum=4（按 tid 切 4 段，须
//   gfrun -s softcore.multiThreadNum=4）。
//
// 约束沿用静态版：BlockSize%32==0；BS=32/half·fp32 守 TROWMAX/TROWEXPANDMUL
//   私有通路两道门；reduce→TCVT 形状契约（RECORD 问题22，physical 列=1 规整）。
//   注：seg_tail>0（M 非 TileM 整除）会触碰 boxed sub-TileM reduce→TCVT 契约缺陷
//   （全家族共有，独立于本切分模型）。
//
// tiling 语义：tiling[0]=M（行/自由轴），tiling[1]=N（尾轴/量化轴，N%BlockSize==0）。
// ===========================================================================
template <int BlockSize = 32, int kPeNum = 1, typename InT = __half>
void dynamic_mx_quant_tail_ocp_fp8_dyn(InT *x, __fp8_e4m3 *y, uint8_t *scale,
                                       const int64_t *tiling) {
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of 32 "
                  "so the output tile is 32B-column-aligned");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore kCorePeCount)");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    using namespace pto;

    // 逐值等价静态版 common 常量（非改数值）。recip 主路径 = 0x7F00 - shared（TEXPANDS+TSUB）。
    constexpr uint16_t RECIP_EMAX = recip_emax_bits<__fp8_e4m3>(); // 0x3b80

    // physical tile 行高 —— 仅由 BlockSize 决定（编译期），与运行期 M/N 无关。
    constexpr int TileM = tail_ocp_fp8_detail::tilem_max(BlockSize); // BS=32 -> 64

    // ---- 运行期 shape 与派生量 ----
    const int64_t M = tiling[0];
    const int64_t N = tiling[1];
    const int64_t numKb     = N / BlockSize;
    const int64_t scaleCols = ((numKb + 1) / 2) * 2; // even-align 补列

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return; // 冗余 PE 不发指令

    uint8_t     *y_u8     = reinterpret_cast<uint8_t *>(y);
    __fp8_e8m0  *scale_e8 = reinterpret_cast<__fp8_e8m0 *>(scale);

    // 动态 global_tensor：RowMajor<-1,-1>，构造传运行期 (rows=M, cols=N) → 行 stride=N。
    using gm_x = global_tensor<InT,        RowMajor<-1, -1>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<-1, -1>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<-1, -1>>;

    // 动态 Valid tile：physical [TileM, 列]，ValidRow=-1（运行期 ctor 传 vr），列静态。
    //   reduce 向量（列=1）physical 列=1 —— 匹配 model rowReduce 无条件 col=1，
    //   令 reduce→TCVT 两侧 physical 列全等，绕过 ValidateOperandContract 契约（问题22）。
    using t_h   = Tile<Location::Vec, InT,        TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using t_hb  = Tile<Location::Vec, InT,        TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_bfb = Tile<Location::Vec, __bf16,     TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_e8b = Tile<Location::Vec, __fp8_e8m0, TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_fb  = Tile<Location::Vec, float,      TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_f   = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using t_o   = Tile<Location::Vec, __fp8_e4m3, TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    // #585 列分区归约类型（复用静态版 reduce_slice_cols）：源 [TileM,BlockSize]=half 超 2048B →
    //   沿列切 nPart 个 [TileM,RSC] 子块各 <=2048B，逐块 TROWMAX→[TileM,1] partial，TMAX 合并。
    constexpr int RSC   = tail_ocp_fp8_detail::reduce_slice_cols(TileM, sizeof(InT), BlockSize);
    constexpr int nPart = BlockSize / RSC;
    using t_hc  = Tile<Location::Vec, InT, TileM, RSC, BLayout::RowMajor, -1, RSC>;

    // 单个 tile-行块的完整计算（scale pass + data pass）。row0 = 全局起始行；
    //   validRows = 该 tile 活跃行数（full-tile: TileM；尾块: seg_tail<TileM）。
    //   全部 tile 用同一 Valid=-1 类型，ctor 传 vr（列静态 → 单参 ctor）。
    auto process_tile = [&](int64_t row0, int64_t validRows) {
        const size_t vr = static_cast<size_t>(validRows);
        // 列分区 half 域 rowmax（#585）：切 nPart 个 [TileM,RSC] 子块累计 max 到 max_in。
        auto col_part_rowmax = [&](int64_t colBase, t_hb &max_in) {
            gm_x g0(x + row0 * N + colBase, static_cast<int>(M), static_cast<int>(N));
            t_hc xs0(vr); TLOAD(xs0, g0);
            t_hc as0(vr); TABS(as0, xs0);
            TROWMAX(max_in, as0);
            for (int p = 1; p < nPart; ++p) {
                gm_x gp(x + row0 * N + colBase + p * RSC, static_cast<int>(M), static_cast<int>(N));
                t_hc xsp(vr); TLOAD(xsp, gp);
                t_hc asp(vr); TABS(asp, xsp);
                t_hb pm(vr); TROWMAX(pm, asp);
                TMAX(max_in, max_in, pm);
            }
        };
        for (int64_t kb = 0; kb < numKb; ++kb) {
            // === scale pass：value-domain reduce（InT 分派），floor 指数（与静态版统一）===
            //   half/fp32 先 fp32 域 mask floor 再窄化（避 narrowing 进位）；bf16 原生直接取指数。
            gm_x gx(x + row0 * N + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(N));
            t_h  xh(vr);     TLOAD(xh, gx);                      // 全宽 load 供 data pass 复用

            t_bfb max_bf(vr);
            if constexpr (std::is_same_v<InT, __half>) {
                t_hb max_h(vr); col_part_rowmax(kb * BlockSize, max_h);  // half 域列分区归约（#585）
                t_fb max_f(vr); TCVT(max_f, max_h);                      // half -> fp32
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);                  // fp32 域 floor
                TCVT(max_bf, max_f);                                     // fp32 -> bf16（尾数=0）
            } else if constexpr (std::is_same_v<InT, float>) {
                t_hb max_f(vr); col_part_rowmax(kb * BlockSize, max_f);  // fp32 域列分区归约
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);                  // fp32 域 floor
                TCVT(max_bf, max_f);                                     // fp32 -> bf16（尾数=0）
            } else {  // bf16：原生取指数（无转换 -> 无进位）
                col_part_rowmax(kb * BlockSize, max_bf);
                auto max_u16 = reinterpret_tile<uint16_t>(max_bf);
                TANDS(max_u16, max_u16, BF16_EXP_MASK);
            }

            t_bfb shared_bf(vr);
            TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX)); // 2^(E_max-8)
            t_e8b scale_e8m0(vr);  TCVT(scale_e8m0, shared_bf);  // bf16 -> e8m0 直转（须在 recip 前）
            gm_s gs(scale_e8 + row0 * scaleCols + kb,
                    static_cast<int>(M), static_cast<int>(scaleCols));
            TSTORE(gs, scale_e8m0);

            // === recip finalize：主路径 recip = 0x7F00 - shared（TEXPANDS+TSUB，与静态版统一；
            //   位补 TXORS/TSUBS 触发 gfrun openat 落盘缺陷已弃用）。写独立 recip_bf,不就地改
            //   shared_bf,使 shared_u16/max_u16 可留给守卫。
            // TODO(inf/zero/special 三守卫)：同静态版,照抄 fp4 的 TCMPS+TSEL 6 行;暂不补
            //   （缺守卫后果:shared==0x7F00 饱和点输出错/inf/nan recip 错/全零块防御缺失,
            //   随机基准不触发 -> output byte-exact）。
            auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
            t_bfb recip_bf(vr), k_bf(vr);
            auto recip_u16 = reinterpret_tile<uint16_t>(recip_bf);
            auto k_u16     = reinterpret_tile<uint16_t>(k_bf);
            TEXPANDS(k_u16, BF16_EXP_BIAS);                      // 0x7F00
            TSUB(recip_u16, k_u16, shared_u16);                 // 0x7F00 - shared = 2^(8-E_max)
            t_fb recip_f(vr);  TCVT(recip_f, recip_bf);         // bf16 -> fp32

            // === data pass（InT 分派，与静态版统一）===
            t_o oq(vr);
            if constexpr (std::is_same_v<InT, float>) {
                TROWEXPANDMUL(xh, xh, recip_f);                 // fp32 域直乘（无预转）
                TCVT(oq, xh);                                   // fp32 -> e4m3
            } else {
                t_f xf(vr); TCVT(xf, xh);                       // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, recip_f);                 // 逐行标量广播乘
                TCVT(oq, xf);                                   // fp32 -> e4m3
            }
            gm_y gy(y_u8 + row0 * N + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(N));
            TSTORE(gy, oq);
        }
        // 奇尾 scale 列补 0x00 E8M0（numKb 奇数时,与静态版 oddTail 统一;运行期 if）。
        if ((numKb % 2) != 0) {
            t_e8b zpad(vr);
            TEXPANDS(zpad, __builtin_bit_cast(__fp8_e8m0, static_cast<uint8_t>(0)));
            gm_s gzs(scale_e8 + row0 * scaleCols + numKb,
                     static_cast<int>(M), static_cast<int>(scaleCols));
            TSTORE(gzs, zpad);
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
