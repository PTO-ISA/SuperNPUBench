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
// 切分（V3 2D）：L1 把 M 按 TileM=32 切成 rowTiles 个行 tile，行方向铺 min(rowTiles,4) 个
//   PE（tile 粒度均分、连续区间）；行方向用不满 4 PE 时把剩余 PE 切到 N 轴（peCols=4/peRows：
//   rowTiles=1→1×4、=2→2×2、=3→3×1〔PE3 闲〕、>=4→4×1）。kb（列块）区间在 colLane 间均分，
//   PE 只扫自己的 [kbBegin, kbBegin+myKb)——kb 块沿 N 两两独立，y/scale 以全局 kb 寻址，
//   区间天然不重叠；oddTail scale 补齐列由持有最末 kb 的 PE 写（ownsLastKb）。L2 每 tile
//   全宽 32 行（最后一个全局 tile M%32≠0 时 boxed）。
//   动机（2026-10-10 性能分析）：瓶颈 = per-kb 串行链周期 × 每 PE 链数（PipeView 实测链 ~350cyc、
//   Vector BLOCK Retire 均值 ~1.2K；STQ 99.61% 在等生产者数据）。旧均分下 M<128 每 PE 扫全 N
//   且 tile 半幅（M=64：4PE×16行×512链）；2D 切分让链数/PE 随 peCols 减少（M=64 → 256 链满幅）。
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

    auto process_tile = [&](int64_t row0, int64_t validRows, int64_t kbBegin,
                            int64_t kbCount, bool ownsLastKb) {
        const size_t vr = static_cast<size_t>(validRows);
        for (int64_t kb = kbBegin; kb < kbBegin + kbCount; ++kb) {
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
        // 奇尾 scale 列补 0x00 E8M0（numKb 奇数）：2D 切分下同一行 tile 的 kb 区间分属多个
        //   PE，补齐列只由持有最末 kb 区间者写（ownsLastKb），保证每行恰好写一次。
        //   构造链（两个约束叠加）：① TEXPANDS dtype 白名单无 E8M0 → 先产 bf16 再 TCVT；
        //   ② e8m0 无法表示 0（模型 ConvertFloatToE8M0：零输入→0xFF invalid），而 ADR-0101
        //   契约 pad = 2^-127（byte 0x00）→ 用 bf16 subnormal 0x0040（=2^-127，frac 64 ×
        //   2^-133）经 TCVT 精确落 byte 0x00。V1/V2 存档用 TEXPANDS-e8m0 非法元组（奇
        //   numKb 触发 emulator 断言；历史用例 numKb 全偶从未暴露）。
        if (ownsLastKb && (numKb % 2) != 0) {
            t_row zpad_bf(vr);
            TEXPANDS(zpad_bf, __builtin_bit_cast(__bf16, static_cast<uint16_t>(0x0040)));  // 2^-127
            t_e8b zpad(vr);
            TCVT(zpad, zpad_bf);             // bf16 2^-127 -> e8m0 0x00
            gm_s gzs(scale_e8 + row0 * scaleCols + numKb,
                     static_cast<int>(M), static_cast<int>(scaleCols));
            TSTORE(gzs, zpad);
        }
    };

    // ---- L1 2D 切分：行 tile 粒度 × N 轴补切 ----
    //   rowTiles 个 32 行 tile 优先铺满行方向；行方向用不满 kPeNum 个 PE 时，剩余 PE 切到
    //   N 轴（peCols=kPeNum/peRows，仅整除时>1：1→1×4、2→2×2、3→3×1、>=4→4×1）。
    //   注：rowTiles>=4 且不被 4 整除时 tile 粒度有 ±1 tile 负载差（如 M=160 → 2:1:1:1），
    //   换取全部满幅链；此类形状旧均分更均衡，属已知取舍（网络形状 M 通常为 128 的倍数）。
    const int64_t itid = static_cast<int64_t>(tid);
    const int64_t rowTiles = (M + TileM - 1) / TileM;
    const int64_t peRows   = rowTiles < kPeNum ? rowTiles : kPeNum;
    const int64_t peCols   = kPeNum / peRows;             // 整除：4/3→1（PE3 闲）
    const int64_t rowLane  = itid / peCols;
    const int64_t colLane  = itid % peCols;
    if (rowLane >= peRows) return;                        // 空 PE（如 3×1 时的 tid3）

    // 行方向：rowTiles 均分到 peRows 个 lane（前 rem 个各 +1 tile，连续 tile 区间）。
    const int64_t tilesBase = rowTiles / peRows;
    const int64_t tilesRem  = rowTiles % peRows;
    const int64_t myTiles   = tilesBase + (rowLane < tilesRem ? 1 : 0);
    const int64_t myTileBegin =
        rowLane * tilesBase + (rowLane < tilesRem ? rowLane : tilesRem);
    if (myTiles <= 0) return;

    // 列方向：numKb 均分到 peCols 个 lane（前 rem 个各 +1 kb，连续 kb 区间）。
    const int64_t kbBase = numKb / peCols;
    const int64_t kbRem  = numKb % peCols;
    const int64_t myKb   = kbBase + (colLane < kbRem ? 1 : 0);
    if (myKb <= 0) return;                                // 行有份但列区间空（numKb<peCols）
    const int64_t kbBegin = colLane * kbBase + (colLane < kbRem ? colLane : kbRem);
    const bool ownsLastKb = (kbBegin + myKb == numKb);    // oddTail 补齐列归属

    // ---- L2 段内 tiling：每 tile 全宽 32 行；最后一个全局 tile M%32≠0 时 boxed ----
    for (int64_t t = 0; t < myTiles; ++t) {
        const int64_t tileIdx = myTileBegin + t;
        const int64_t row0 = tileIdx * TileM;
        const int64_t validRows =
            (tileIdx == rowTiles - 1) ? (M - (rowTiles - 1) * TileM) : TileM;
        process_tile(row0, validRows, kbBegin, myKb, ownsLastKb);
    }
}

} // namespace supernpu::tile_isa::mxquant

#endif
