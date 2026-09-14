#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_DYN_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"
// 复用静态版的 tail_ocp_fp8_detail::{pow2_floor, tilem_max}（纯 BlockSize 函数，
// 编译期常量），避免重复定义。静态 kernel 模板未实例化则不产码。
#include "quant/dynamic_mx_quant/dynamic_mx_quant_tail_ocp_fp8.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// TAIL-OCP-FP8 —— 运行期动态 shape 版（DYNAMIC SHAPE）
//
// 与静态 dynamic_mx_quant_tail_ocp_fp8 逐 op 等价（half in / e4m3 out /
// e8m0 scale / 位补求倒数），唯一区别：**M、N 在编译期不可知，运行期由 tiling
// 指针传入，全部切分参数运行期计算**。BlockSize 是属性 → 保留模板参（编译期）。
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
template <int BlockSize = 32, int kPeNum = 1>
void dynamic_mx_quant_tail_ocp_fp8_dyn(__half *x, __fp8_e4m3 *y, uint8_t *scale,
                                       const int64_t *tiling) {
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of 32 "
                  "so the output tile is 32B-column-aligned");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore kCorePeCount)");

    using namespace pto;

    // 逐值等价静态版 common 常量（非改数值）。
    constexpr uint16_t RECIP_EMAX     = recip_emax_bits<__fp8_e4m3>(); // 0x3b80
    constexpr uint16_t RECIP_XOR_NOT  = 0xFFFF; // 按位取反 (int16: -1)
    constexpr uint16_t RECIP_COMPL_SUB = 0x80FF; // 0xFFFF - 0x7F00 (int16 补码减法)

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
    using gm_x = global_tensor<__half,     RowMajor<-1, -1>>;
    using gm_y = global_tensor<uint8_t,    RowMajor<-1, -1>>;
    using gm_s = global_tensor<__fp8_e8m0, RowMajor<-1, -1>>;

    // 动态 Valid tile：physical [TileM, 列]，ValidRow=-1（运行期 ctor 传 vr），列静态。
    //   reduce 向量（列=1）physical 列=1 —— 匹配 model rowReduce 无条件 col=1，
    //   令 reduce→TCVT 两侧 physical 列全等，绕过 ValidateOperandContract 契约（问题22）。
    using t_h   = Tile<Location::Vec, __half,     TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using t_hb  = Tile<Location::Vec, __half,     TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_bfb = Tile<Location::Vec, __bf16,     TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_e8b = Tile<Location::Vec, __fp8_e8m0, TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_fb  = Tile<Location::Vec, float,      TileM, 1,         BLayout::RowMajor, -1, 1>;
    using t_f   = Tile<Location::Vec, float,      TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;
    using t_o   = Tile<Location::Vec, __fp8_e4m3, TileM, BlockSize, BLayout::RowMajor, -1, BlockSize>;

    // 单个 tile-行块的完整计算（scale pass + data pass）。row0 = 全局起始行；
    //   validRows = 该 tile 活跃行数（full-tile: TileM；尾块: seg_tail<TileM）。
    //   全部 tile 用同一 Valid=-1 类型，ctor 传 vr（列静态 → 单参 ctor）。
    auto process_tile = [&](int64_t row0, int64_t validRows) {
        const size_t vr = static_cast<size_t>(validRows);
        for (int64_t kb = 0; kb < numKb; ++kb) {
            // === scale pass ===
            gm_x gx(x + row0 * N + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(N));
            t_h  xh(vr);     TLOAD(xh, gx);
            t_h  abs_h(vr);  TABS(abs_h, xh);
            t_hb max_h(vr);  TROWMAX(max_h, abs_h);
            t_fb max_f(vr);  TCVT(max_f, max_h);                 // half -> fp32（精确）
            auto max_u32 = reinterpret_tile<uint32_t>(max_f);
            TANDS(max_u32, max_u32, FP32_EXP_MASK);              // floor 到 2^E_max
            t_bfb max_bf(vr);  TCVT(max_bf, max_f);              // fp32 -> bf16（尾数=0）
            t_bfb shared_bf(vr);
            TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX)); // 2^(E_max-8)
            t_e8b scale_e8m0(vr);  TCVT(scale_e8m0, shared_bf);  // bf16 -> e8m0 直转
            gm_s gs(scale_e8 + row0 * scaleCols + kb,
                    static_cast<int>(M), static_cast<int>(scaleCols));
            TSTORE(gs, scale_e8m0);

            // === data pass (NEWCALC: 位补求倒数, 复用 xh) ===
            auto sh_u16 = reinterpret_tile<int16_t>(shared_bf);
            TXORS(sh_u16, sh_u16, RECIP_XOR_NOT);                // 0xFFFF - bits
            TSUBS(sh_u16, sh_u16, RECIP_COMPL_SUB);              // -> 0x7F00 - bits
            t_fb recip_f(vr);  TCVT(recip_f, shared_bf);         // bf16 -> fp32
            t_f  xf(vr);       TCVT(xf, xh);                     // half -> fp32（复用 xh）
            TROWEXPANDMUL(xf, xf, recip_f);                      // x * (1/scale)
            t_o  oq(vr);       TCVT(oq, xf);                     // fp32 -> e4m3
            gm_y gy(y_u8 + row0 * N + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(N));
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
