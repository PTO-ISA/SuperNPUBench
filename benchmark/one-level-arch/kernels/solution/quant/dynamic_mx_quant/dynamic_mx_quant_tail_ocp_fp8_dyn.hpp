#ifndef SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_DYN_HPP
#define SUPERNPU_DYNAMIC_MX_QUANT_TAIL_OCP_FP8_DYN_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "solution/quant/dynamic_mx_quant/dynamic_mx_quant_common.hpp"

namespace supernpu::tile_isa::mxquant {

// ===========================================================================
// TAIL-OCP-FP8 —— 运行期动态 shape 版 · **M32（CUBE_M32 cell）布局，TileM=32**
//
// 【本版 = M32 实验版（RowMajor V1 已备份为 *_V1-09181200.hpp）】
//
// 【2026-10-09】原「已知阻塞」已解除：TileOP-API#187 为 reduction-prefix 发射路径补齐
//   ValidRow<0 → 寄存器 B.DIM 分支（TREDUCEPREFIXVIEW 的 "i" 立即数约束不再排斥运行期
//   ValidRow），本 _dyn 现可编译 + gfrun 4-PE 逐字节 PASS（见 Makefile/run_precision_check 注释）。
//
// 与 RowMajor V1 逐 op 等价（InT in / e4m3 out / e8m0 scale / recip = 0x7F00-shared 主路径
// + inf/zero/special 三守卫），唯一改动：**tile 全部改 VecTileM32（CUBE_M32），TileM=32**。
// 参考 kernels/basic_op/mxquant/mxquant.hpp。
//
// M32 相对 RowMajor 的净变化（仅布局，数值不变）：
//   · 归约源 [32,BlockSize] bf16 = 2048B 恰 <= #585 → **输入 load 一次**（去掉 RowMajor 的 4 片重载）；
//   · 广播源 [32,1] M32 = 128B <= 128B → 天然满足 gfsim #605（去掉 TileM=128 的 #605 触发）；
//   · TROWMAX 的 dest 须 = 源列宽整块（PTO #311）→ reduce/shared/recip 载体声明 physical Col=BlockSize、
//     valid Col=1（不能是 [32,1]，否则 #311 crash）。
//
// 三守卫（inf/zero/special）已补齐（2026-10-09，镜像静态版 7864c78）：CUBE_M32 上的
//   TCMPS/TSEL 已由 SuperScalarModel #785（按 ASL 实现 CUBE PredicateCell）修复，静态 M32
//   版先行验证；本版为同一守卫序列在运行期 ValidRow tile 上的移植。
//
// 切分（不改）：L1 M 行按 kPeNum 均分（每 PE 连续 SubM 行）；L2 TileM=32、seg_full+seg_tail
//   （M32 cell 32 行，SubM<32 时 boxed valid<32）；内层 kb 循环 numKb=N/BlockSize。
// tiling：tiling[0]=M，tiling[1]=N（N%BlockSize==0）。
// ===========================================================================
template <int BlockSize = 32, int kPeNum = 1, typename InT = __half>
void dynamic_mx_quant_tail_ocp_fp8_dyn(InT *x, __fp8_e4m3 *y, uint8_t *scale,
                                       const int64_t *tiling) {
    static_assert(BlockSize % 32 == 0,
                  "fp8 block = BlockSize bytes; BlockSize must be a multiple of 32");
    static_assert(kPeNum == 1 || kPeNum == 4,
                  "kPeNum must be 1 (single PE) or 4 (SoftCore kCorePeCount)");
    static_assert(std::is_same_v<InT, __bf16> || std::is_same_v<InT, __half> ||
                      std::is_same_v<InT, float>,
                  "InT must be one of {__bf16, __half, float}");

    using namespace pto;

    constexpr uint16_t RECIP_EMAX = recip_emax_bits<__fp8_e4m3>(); // 0x3b80
    constexpr int TileM = 32;  // M32 cell 行高（固定）

    const int64_t M = tiling[0];
    const int64_t N = tiling[1];
    const int64_t numKb     = N / BlockSize;
    const int64_t scaleCols = ((numKb + 1) / 2) * 2;

    const uint32_t tid = get_thread_idx();
    if (static_cast<int>(tid) >= kPeNum) return;

    __fp8_e8m0  *scale_e8 = reinterpret_cast<__fp8_e8m0 *>(scale);

    // M32 TSTORE_CUBE 要求 GM 与 CUBE tile dtype 一致（RowMajor 曾用 uint8 视图，M32 不允许）。
    using gm_x = global_tensor<InT,         RowMajor<-1, -1>>;
    using gm_y = global_tensor<__fp8_e4m3,  RowMajor<-1, -1>>;
    using gm_s = global_tensor<__fp8_e8m0,  RowMajor<-1, -1>>;

    // ---- M32 tile 类型（VecTileM32）：#311 宽 reduce carrier + TREDUCEPREFIXVIEW 消费范式（照 fa_lowp）----
    //   #311 让 CUBE 行归约结果逻辑 [M,1] 但要「宽 physical carrier（Col=源列宽）」；宽 carrier 无法直接
    //   被 TCVT/#291 elementwise 消费（它们要 physical==valid 派生）。正解：TROWMAX 出宽 carrier →
    //   TREDUCEPREFIXVIEW 取「第一个 128B CELL」的零拷贝窄视图 → 用 mixed TADD(zero+view) materialize 成
    //   普通窄 tile（TMULS/TCVT/TSUB 不能直接吃 view，见 fa_lowp 注释）→ 之后标量链全在窄 tile 上。
    using t_blk = VecTileM32<InT,        32, BlockSize, -1, BlockSize>; // 输入/abs（宽，full valid）
    using t_rw  = VecTileM32<InT,        32, BlockSize, -1, 1>;         // reduce 宽 carrier（#311）
    using t_rin = VecTileM32<InT,        32, 1,         -1, 1>;         // 窄 InT（view SubTile + materialize 目标）
    using t_row = VecTileM32<__bf16,     32, 1,         -1, 1>;         // 窄 bf16（shared/recip）
    using t_frw = VecTileM32<float,      32, 1,         -1, 1>;         // 窄 fp32（floor/recip_f）
    using t_e8b = VecTileM32<__fp8_e8m0, 32, 1,         -1, 1>;         // scale（窄）
    using t_f   = VecTileM32<float,      32, BlockSize, -1, BlockSize>; // data-pass fp32
    using t_o   = VecTileM32<__fp8_e4m3, 32, BlockSize, -1, BlockSize>; // 输出

    auto process_tile = [&](int64_t row0, int64_t validRows) {
        const size_t vr = static_cast<size_t>(validRows);
        for (int64_t kb = 0; kb < numKb; ++kb) {
            // === scale pass：M32 全宽 load 一次，行归约（无 #585 切片）===
            gm_x gx(x + row0 * N + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(N));
            t_blk xin(vr); TLOAD(xin, gx);            // [32,BlockSize] M32，一次 load
            t_blk absx(vr); TABS(absx, xin);          // |x|（M32 #291 layout-preserving）

            // #311 归约 + TREDUCEPREFIXVIEW 消费（照 fa_lowp）：宽 carrier 归约 → 零拷贝窄视图 →
            //   TADD(zero+view) materialize 成普通窄 InT tile。
            t_rw  max_w(vr);  TROWMAX(max_w, absx);                    // 宽 carrier 归约（#311 ✓）
            auto  max_view = TREDUCEPREFIXVIEW<t_rin>(max_w);         // 零拷贝窄视图（第一个 128B CELL）
            t_rin zero_in(vr); TEXPANDS(zero_in, static_cast<InT>(0.0f));
            t_rin max_in(vr);  TADD(max_in, zero_in, max_view);       // materialize → 普通窄 InT

            // floor 指数 → shared + 收集 inf/zero 守卫 predicate（读 floor 后 max 位型；与静态版统一）。
            t_row shared_bf(vr), eqinf_bf(vr), eqzero_bf(vr);
            auto eq_inf  = reinterpret_tile<uint16_t>(eqinf_bf);
            auto eq_zero = reinterpret_tile<uint16_t>(eqzero_bf);
            if constexpr (std::is_same_v<InT, __bf16>) {
                auto max_u16 = reinterpret_tile<uint16_t>(max_in);
                TANDS(max_u16, max_u16, BF16_EXP_MASK);               // bf16 floor
                TCMPS(eq_inf,  max_u16, BF16_EXP_MASK);               // NOT finite (max_exp==0x7F80)
                TCMPS(eq_zero, max_u16, static_cast<uint16_t>(0));    // all-zero block
                TMULS(shared_bf, max_in, __builtin_bit_cast(__bf16, RECIP_EMAX)); // 2^(E_max-8)
            } else {
                t_frw max_f(vr); TCVT(max_f, max_in);                 // half/fp32 -> 窄 fp32
                auto max_u32 = reinterpret_tile<uint32_t>(max_f);
                TANDS(max_u32, max_u32, FP32_EXP_MASK);               // fp32 域 floor
                t_row max_bf(vr); TCVT(max_bf, max_f);                // 窄 fp32 -> 窄 bf16（尾数=0）
                auto maxbf_u16 = reinterpret_tile<uint16_t>(max_bf);
                TCMPS(eq_inf,  maxbf_u16, BF16_EXP_MASK);
                TCMPS(eq_zero, maxbf_u16, static_cast<uint16_t>(0));
                TMULS(shared_bf, max_bf, __builtin_bit_cast(__bf16, RECIP_EMAX));
            }
            t_e8b scale_e8m0(vr);  TCVT(scale_e8m0, shared_bf);  // bf16 -> e8m0
            gm_s gs(scale_e8 + row0 * scaleCols + kb,
                    static_cast<int>(M), static_cast<int>(scaleCols));
            TSTORE(gs, scale_e8m0);

            // === recip finalize：主路径 recip = 0x7F00 - shared（TEXPANDS+TSUB）+ inf/zero/special 三守卫 ===
            //   与静态版（7864c78）/V1/tail_ocp_fp4 统一：inf/nan->0x7F81 / 全零->0 /
            //   special(shared==0x7F00)->0x0040。CUBE_M32 compare-select 几何已由 #785 修复。
            auto shared_u16 = reinterpret_tile<uint16_t>(shared_bf);
            t_row recip_bf(vr), eqspc_bf(vr), k_bf(vr);
            auto recip_u16  = reinterpret_tile<uint16_t>(recip_bf);
            auto eq_special = reinterpret_tile<uint16_t>(eqspc_bf);
            auto k_u16      = reinterpret_tile<uint16_t>(k_bf);
            TCMPS(eq_special, shared_u16, BF16_EXP_BIAS);        // shared==0x7F00
            TEXPANDS(k_u16, BF16_EXP_BIAS);                      // 0x7F00
            TSUB(recip_u16, k_u16, shared_u16);                  // 0x7F00 - shared = 2^(8-E_max)
            TEXPANDS(k_u16, BF16_NAN_PATTERN);
            TSEL(recip_u16, eq_inf, k_u16);                      // inf/nan -> 0x7F81
            TEXPANDS(k_u16, static_cast<uint16_t>(0));
            TSEL(recip_u16, eq_zero, k_u16);                     // all-zero -> 0
            TEXPANDS(k_u16, BF16_SPECIAL_EXP);
            TSEL(recip_u16, eq_special, k_u16);                  // special -> 0x0040
            t_frw recip_f(vr);  TCVT(recip_f, recip_bf);         // 窄 bf16 -> 窄 fp32

            // === data pass ===
            t_o oq(vr);
            if constexpr (std::is_same_v<InT, float>) {
                TROWEXPANDMUL(xin, xin, recip_f);               // fp32 域直乘
                TCVT(oq, xin);                                  // fp32 -> e4m3
            } else {
                t_f xf(vr); TCVT(xf, xin);                      // bf16/half -> fp32
                TROWEXPANDMUL(xf, xf, recip_f);                 // 逐行标量广播乘
                TCVT(oq, xf);                                   // fp32 -> e4m3
            }
            gm_y gy(y + row0 * N + kb * BlockSize,
                    static_cast<int>(M), static_cast<int>(N));
            TSTORE(gy, oq);
        }
        // 奇尾 scale 列补 0x00 E8M0（numKb 奇数）。
        if ((numKb % 2) != 0) {
            t_e8b zpad(vr);
            TEXPANDS(zpad, __builtin_bit_cast(__fp8_e8m0, static_cast<uint8_t>(0)));
            gm_s gzs(scale_e8 + row0 * scaleCols + numKb,
                     static_cast<int>(M), static_cast<int>(scaleCols));
            TSTORE(gzs, zpad);
        }
    };

    // ---- L1 行切分 ----
    const int64_t row_base = M / kPeNum;
    const int64_t row_rem  = M % kPeNum;
    const int64_t itid     = static_cast<int64_t>(tid);
    const int64_t SubM     = row_base + (itid < row_rem ? 1 : 0);
    const int64_t row_begin = (itid < row_rem)
                                  ? itid * (row_base + 1)
                                  : row_rem * (row_base + 1) + (itid - row_rem) * row_base;
    if (SubM == 0) return;

    // ---- L2 段内 tiling（M32 cell 32 行）----
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
