#ifndef FA_BWD_DIRECT_SHARED
#define FA_BWD_DIRECT_SHARED 0
#endif
#pragma once

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace pto;

#ifndef FA_BWD_Q_COMPACT_AUX
#define FA_BWD_Q_COMPACT_AUX 1
#endif

// FlashAttention backward, TileLang-shaped (single loop over Q, no pass split).
//
//   outer loop over KV tiles (the grid dimension); inner loop over Q blocks.
//   One Q iteration does exactly what TileLang's flash_bwd body does:
//
//     S^T [Tk, Tn] = K_j . Q_b^T          (group_M = Tk, KV rows split over PEs)
//     P^T          = exp(scale*S^T - lse_b)
//     dP^T [Tk,Tn] = V_j . dO_b^T
//     dS^T         = scale*P^T .* (dP^T - delta_b)
//     dV  [Tk,vD] += P^T . dO_b           (accumulate over Q, register)
//     dK  [Tk,qD] += dS^T . Q_b           (accumulate over Q, register)
//     dQ  [Tn,qD] += (dS^T)^T . K_j       (FP32 GM atomic add over KV)
//
// Q/K/V/dO and the rounded P/dS matmul operands use MatDType; every accumulator,
// lse, delta and the elementwise path stay fp32. The default FP8 E4M3 D=256
// configuration uses four 128x256 operand tiles plus one 128x128 dS tile:
// 144KiB of logical shared operands. K Left/Right views share one allocation.
//
// ISA differences from TileLang that shape the code (not algorithm choices):
//   * FA_BWD_DIRECT_SHARED=1 uses PTO #399 CUBE-to-Shared-ND TMOV;
//     =0 preserves dS MatDType GM staging (TSTORE_CUBE then Shared TLOAD).
//   * Shared TLOAD requires tile columns == buffer pitch, so lse/delta/stage are
//     stored block-contiguous ([*kTk, kTn]) instead of wide [*, Sq].
// dQ must be zero-initialized by the caller. MSCATTER_ADD reduces each KV
// contribution into [Sq,qD], using a reusable U32 byte-displacement tile.
// FA_BWD_Q_COMPACT_AUX=1 loads lse/delta as [1,kTn] from compact [Sq]
// GM vectors and subtracts with TCOLEXPANDSUB, without an expanded temporary.
// A [1,kTn] M32 source still allocates 32 physical rows: this reduces GM
// payload, but does not promise a smaller CellReg allocation.
template <typename MatDType, typename DType, int Sq, int Skv, int qD, int vD,
          int kTk, int kTn, int scaleD = qD>
void flash_attention_bwd_q_impl(DType *dqPtr, DType *dkPtr, DType *dvPtr,
                                MatDType *qPtr, MatDType *kPtr,
                                MatDType *vPtr, MatDType *doPtr,
                                DType *lseBPtr, DType *deltaBPtr,
                                MatDType *dsStagePtr,
                                float dqOutScale, float dkOutScale, float dvOutScale) {
    constexpr int kPeNum = 4;
    constexpr int kGroupM = kTk <= 128 ? kTk : 128;
    constexpr int kPeM = kGroupM <= 64 ? 16 : 32;
    static_assert(kGroupM == 128 && kPeM == 32, "long-context square config");
    static_assert(kTn == kGroupM, "Q block must match the group_M width");

    const uint32_t tid = get_thread_idx();

    constexpr int kQBlocks = Sq / kTn;
    constexpr int kKVBlocks = Skv / kTk;
    static_assert(Sq % kTn == 0 && Skv % kTk == 0, "shapes must divide");

    using GmQ = global_tensor<MatDType, RowMajor<Sq, qD>>;
    using GmK = global_tensor<MatDType, RowMajor<Skv, qD>>;
    using GmV = global_tensor<MatDType, RowMajor<Skv, vD>>;
    using GmDo = global_tensor<MatDType, RowMajor<Sq, vD>>;
#if FA_BWD_Q_COMPACT_AUX
    using GmAux = global_tensor<DType, RowMajor<kQBlocks, kTn>>;
#else
    using GmAux = global_tensor<DType, RowMajor<kQBlocks * kTk, kTn>>;
#endif
    using GmStage = global_tensor<MatDType, RowMajor<kKVBlocks * kQBlocks * kTk,
                                                     kTn>>;
    using GmDk = global_tensor<DType, RowMajor<Skv, qD>>;
    using GmDv = global_tensor<DType, RowMajor<Skv, vD>>;

    using TileKjL = SharedMatrixLeft<MatDType, kTk, qD>;
    using TileVj = SharedMatrixLeft<MatDType, kTk, vD>;
    using TileQBlk = SharedMatrixRight<MatDType, kTn, qD>;
    using TileDoBlk = SharedMatrixRight<MatDType, kTn, vD>;
    using TileDst = SharedMatrixLeft<MatDType, kTk, kTn>;

    using TileKjLSh = SharedTile<TileKjL>;
    using TileVjSh = SharedTile<TileVj>;
    using TileQBlkSh = SharedTile<TileQBlk>;
    using TileDoBlkSh = SharedTile<TileDoBlk>;
    using TileDstSh = SharedTile<TileDst>;

    using TileST = CubeAccumulatorM32<DType, kPeM, kTn>;
#if FA_BWD_Q_COMPACT_AUX
    using TileAux = CubeAccumulatorM32<DType, 1, kTn>;
#else
    using TileAux = TileST;
#endif
    using TilePTL = CubeTileM32<MatDType, kPeM, kTn>;
    using TileDSTL = CubeTileM32<MatDType, kPeM, kTn>;
    using TileDv = CubeAccumulatorM32<DType, kPeM, vD>;
    using TileDk = CubeAccumulatorM32<DType, kPeM, qD>;
    using TileDq = CubeAccumulatorM32<DType, kPeM, qD>;

    using ItQBlk = global_iterator<GmQ, TileQBlk>;
    using ItDoBlk = global_iterator<GmDo, TileDoBlk>;
    using ItKjL = global_iterator<GmK, TileKjL>;
    using ItVj = global_iterator<GmV, TileVj>;
    using ItLse = global_iterator<GmAux, TileAux>;
    using ItDelta = global_iterator<GmAux, TileAux>;
    using ItStageW = global_iterator<GmStage, TileDSTL>;
    using ItStageR = global_iterator<GmStage, TileDst>;
    using ItDk = global_iterator<GmDk, TileDk>;
    using ItDv = global_iterator<GmDv, TileDv>;

    ItQBlk gQBlk(qPtr);
    ItDoBlk gDoBlk(doPtr);
    ItKjL gKjL(kPtr);
    ItVj gVj(vPtr);
    ItLse gLse(lseBPtr);
    ItDelta gDelta(deltaBPtr);
    ItStageW gStageW(dsStagePtr);
    ItStageR gStageR(dsStagePtr);
    ItDk gDk(dkPtr);
    ItDv gDv(dvPtr);

    // Single-KV ordinary-store control: no atomic index tile or index load.
    const float scale = 1.0f / sqrt(static_cast<float>(scaleD));

    constexpr auto sOpt = fixp::keep_acc();
    constexpr auto lrOpt = fixp::keep_acc().transpose_b();
    constexpr auto dqOpt = fixp::keep_acc().transpose_a().transpose_b();

#pragma clang loop unroll(full)
    for (int kvi = 0; kvi < kKVBlocks; ++kvi) {
        TileKjLSh tKjL;
        {
            auto g = gKjL(kvi, 0);
            TLOAD<TileKjL, 1>(tKjL, g);
        }
        auto tKjR = reinterpret_shared_tile<Location::Right>(tKjL);
        TileVjSh tVj;
        {
            auto g = gVj(kvi, 0);
            TLOAD<TileVj, 1>(tVj, g);
        }

        TileDv tDv;
        TileDk tDk;

        // Two Q blocks in flight. Source/order semantics match the baseline;
        // preload i+2 after the final Q/dO consumer of i, and cover P latency
        // with QK/dP from its independent neighbour. No extra GM staging.
        static_assert(kQBlocks == FA_BWD_PRELOAD_QBLOCKS);
        static_assert(kQBlocks >= 2 && kQBlocks <= 32 && kQBlocks % 2 == 0);
#if FA_BWD_PRELOAD_QBLOCKS > 0
        TileQBlkSh tQ0; TileDoBlkSh tDo0;
        TileST tST0, tPT0, tDPT0;
        TileAux tLse0, tDelta0;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 1
        TileQBlkSh tQ1; TileDoBlkSh tDo1;
        TileST tST1, tPT1, tDPT1;
        TileAux tLse1, tDelta1;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 2
        TileQBlkSh tQ2; TileDoBlkSh tDo2;
        TileST tST2, tPT2, tDPT2;
        TileAux tLse2, tDelta2;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 3
        TileQBlkSh tQ3; TileDoBlkSh tDo3;
        TileST tST3, tPT3, tDPT3;
        TileAux tLse3, tDelta3;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 4
        TileQBlkSh tQ4; TileDoBlkSh tDo4;
        TileST tST4, tPT4, tDPT4;
        TileAux tLse4, tDelta4;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 5
        TileQBlkSh tQ5; TileDoBlkSh tDo5;
        TileST tST5, tPT5, tDPT5;
        TileAux tLse5, tDelta5;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 6
        TileQBlkSh tQ6; TileDoBlkSh tDo6;
        TileST tST6, tPT6, tDPT6;
        TileAux tLse6, tDelta6;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 7
        TileQBlkSh tQ7; TileDoBlkSh tDo7;
        TileST tST7, tPT7, tDPT7;
        TileAux tLse7, tDelta7;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 8
        TileQBlkSh tQ8; TileDoBlkSh tDo8;
        TileST tST8, tPT8, tDPT8;
        TileAux tLse8, tDelta8;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 9
        TileQBlkSh tQ9; TileDoBlkSh tDo9;
        TileST tST9, tPT9, tDPT9;
        TileAux tLse9, tDelta9;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 10
        TileQBlkSh tQ10; TileDoBlkSh tDo10;
        TileST tST10, tPT10, tDPT10;
        TileAux tLse10, tDelta10;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 11
        TileQBlkSh tQ11; TileDoBlkSh tDo11;
        TileST tST11, tPT11, tDPT11;
        TileAux tLse11, tDelta11;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 12
        TileQBlkSh tQ12; TileDoBlkSh tDo12;
        TileST tST12, tPT12, tDPT12;
        TileAux tLse12, tDelta12;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 13
        TileQBlkSh tQ13; TileDoBlkSh tDo13;
        TileST tST13, tPT13, tDPT13;
        TileAux tLse13, tDelta13;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 14
        TileQBlkSh tQ14; TileDoBlkSh tDo14;
        TileST tST14, tPT14, tDPT14;
        TileAux tLse14, tDelta14;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 15
        TileQBlkSh tQ15; TileDoBlkSh tDo15;
        TileST tST15, tPT15, tDPT15;
        TileAux tLse15, tDelta15;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 16
        TileQBlkSh tQ16; TileDoBlkSh tDo16;
        TileST tST16, tPT16, tDPT16;
        TileAux tLse16, tDelta16;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 17
        TileQBlkSh tQ17; TileDoBlkSh tDo17;
        TileST tST17, tPT17, tDPT17;
        TileAux tLse17, tDelta17;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 18
        TileQBlkSh tQ18; TileDoBlkSh tDo18;
        TileST tST18, tPT18, tDPT18;
        TileAux tLse18, tDelta18;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 19
        TileQBlkSh tQ19; TileDoBlkSh tDo19;
        TileST tST19, tPT19, tDPT19;
        TileAux tLse19, tDelta19;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 20
        TileQBlkSh tQ20; TileDoBlkSh tDo20;
        TileST tST20, tPT20, tDPT20;
        TileAux tLse20, tDelta20;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 21
        TileQBlkSh tQ21; TileDoBlkSh tDo21;
        TileST tST21, tPT21, tDPT21;
        TileAux tLse21, tDelta21;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 22
        TileQBlkSh tQ22; TileDoBlkSh tDo22;
        TileST tST22, tPT22, tDPT22;
        TileAux tLse22, tDelta22;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 23
        TileQBlkSh tQ23; TileDoBlkSh tDo23;
        TileST tST23, tPT23, tDPT23;
        TileAux tLse23, tDelta23;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 24
        TileQBlkSh tQ24; TileDoBlkSh tDo24;
        TileST tST24, tPT24, tDPT24;
        TileAux tLse24, tDelta24;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 25
        TileQBlkSh tQ25; TileDoBlkSh tDo25;
        TileST tST25, tPT25, tDPT25;
        TileAux tLse25, tDelta25;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 26
        TileQBlkSh tQ26; TileDoBlkSh tDo26;
        TileST tST26, tPT26, tDPT26;
        TileAux tLse26, tDelta26;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 27
        TileQBlkSh tQ27; TileDoBlkSh tDo27;
        TileST tST27, tPT27, tDPT27;
        TileAux tLse27, tDelta27;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 28
        TileQBlkSh tQ28; TileDoBlkSh tDo28;
        TileST tST28, tPT28, tDPT28;
        TileAux tLse28, tDelta28;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 29
        TileQBlkSh tQ29; TileDoBlkSh tDo29;
        TileST tST29, tPT29, tDPT29;
        TileAux tLse29, tDelta29;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 30
        TileQBlkSh tQ30; TileDoBlkSh tDo30;
        TileST tST30, tPT30, tDPT30;
        TileAux tLse30, tDelta30;
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 31
        TileQBlkSh tQ31; TileDoBlkSh tDo31;
        TileST tST31, tPT31, tDPT31;
        TileAux tLse31, tDelta31;
#endif
            { auto g = gQBlk(0, 0); TLOAD<TileQBlk, 1>(tQ0, g); }
            { auto g = gLse(0, 0); TLOAD(tLse0, g); }
            { auto g = gDoBlk(0, 0); TLOAD<TileDoBlk, 1>(tDo0, g); }
            { auto g = gDelta(0, 0); TLOAD(tDelta0, g); }
            { auto g = gQBlk(1, 0); TLOAD<TileQBlk, 1>(tQ1, g); }
            { auto g = gLse(1, 0); TLOAD(tLse1, g); }
            { auto g = gDoBlk(1, 0); TLOAD<TileDoBlk, 1>(tDo1, g); }
            { auto g = gDelta(1, 0); TLOAD(tDelta1, g); }
#if FA_BWD_PRELOAD_QBLOCKS > 0
            TMATMUL(tST0, tKjL, tQ0, sOpt);
            TMULS(tST0, tST0, scale);
            TCOLEXPANDEXPDIF(tPT0, tST0, tLse0);
            TMATMUL(tST1, tKjL, tQ1, sOpt);
            TMATMUL(tDPT0, tVj, tDo0, sOpt);
            TMATMUL(tDPT1, tVj, tDo1, sOpt);
            TMULS(tST1, tST1, scale);
            TCOLEXPANDEXPDIF(tPT1, tST1, tLse1);
        {
            constexpr int qbi = 0;
            auto &tQ = tQ0; auto &tDo = tDo0;
            auto &tPT = tPT0; auto &tDPT = tDPT0;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta0;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 1;
            auto &tQ = tQ1; auto &tDo = tDo1;
            auto &tPT = tPT1; auto &tDPT = tDPT1;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta1;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 3
            { auto g = gQBlk(2, 0); TLOAD<TileQBlk, 1>(tQ2, g); }
            { auto g = gLse(2, 0); TLOAD(tLse2, g); }
            { auto g = gDoBlk(2, 0); TLOAD<TileDoBlk, 1>(tDo2, g); }
            { auto g = gDelta(2, 0); TLOAD(tDelta2, g); }
            { auto g = gQBlk(3, 0); TLOAD<TileQBlk, 1>(tQ3, g); }
            { auto g = gLse(3, 0); TLOAD(tLse3, g); }
            { auto g = gDoBlk(3, 0); TLOAD<TileDoBlk, 1>(tDo3, g); }
            { auto g = gDelta(3, 0); TLOAD(tDelta3, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 2
            TMATMUL(tST2, tKjL, tQ2, sOpt);
            TMULS(tST2, tST2, scale);
            TCOLEXPANDEXPDIF(tPT2, tST2, tLse2);
            TMATMUL(tST3, tKjL, tQ3, sOpt);
            TMATMUL(tDPT2, tVj, tDo2, sOpt);
            TMATMUL(tDPT3, tVj, tDo3, sOpt);
            TMULS(tST3, tST3, scale);
            TCOLEXPANDEXPDIF(tPT3, tST3, tLse3);
        {
            constexpr int qbi = 2;
            auto &tQ = tQ2; auto &tDo = tDo2;
            auto &tPT = tPT2; auto &tDPT = tDPT2;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta2;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 3;
            auto &tQ = tQ3; auto &tDo = tDo3;
            auto &tPT = tPT3; auto &tDPT = tDPT3;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta3;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 5
            { auto g = gQBlk(4, 0); TLOAD<TileQBlk, 1>(tQ4, g); }
            { auto g = gLse(4, 0); TLOAD(tLse4, g); }
            { auto g = gDoBlk(4, 0); TLOAD<TileDoBlk, 1>(tDo4, g); }
            { auto g = gDelta(4, 0); TLOAD(tDelta4, g); }
            { auto g = gQBlk(5, 0); TLOAD<TileQBlk, 1>(tQ5, g); }
            { auto g = gLse(5, 0); TLOAD(tLse5, g); }
            { auto g = gDoBlk(5, 0); TLOAD<TileDoBlk, 1>(tDo5, g); }
            { auto g = gDelta(5, 0); TLOAD(tDelta5, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 4
            TMATMUL(tST4, tKjL, tQ4, sOpt);
            TMULS(tST4, tST4, scale);
            TCOLEXPANDEXPDIF(tPT4, tST4, tLse4);
            TMATMUL(tST5, tKjL, tQ5, sOpt);
            TMATMUL(tDPT4, tVj, tDo4, sOpt);
            TMATMUL(tDPT5, tVj, tDo5, sOpt);
            TMULS(tST5, tST5, scale);
            TCOLEXPANDEXPDIF(tPT5, tST5, tLse5);
        {
            constexpr int qbi = 4;
            auto &tQ = tQ4; auto &tDo = tDo4;
            auto &tPT = tPT4; auto &tDPT = tDPT4;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta4;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 5;
            auto &tQ = tQ5; auto &tDo = tDo5;
            auto &tPT = tPT5; auto &tDPT = tDPT5;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta5;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 7
            { auto g = gQBlk(6, 0); TLOAD<TileQBlk, 1>(tQ6, g); }
            { auto g = gLse(6, 0); TLOAD(tLse6, g); }
            { auto g = gDoBlk(6, 0); TLOAD<TileDoBlk, 1>(tDo6, g); }
            { auto g = gDelta(6, 0); TLOAD(tDelta6, g); }
            { auto g = gQBlk(7, 0); TLOAD<TileQBlk, 1>(tQ7, g); }
            { auto g = gLse(7, 0); TLOAD(tLse7, g); }
            { auto g = gDoBlk(7, 0); TLOAD<TileDoBlk, 1>(tDo7, g); }
            { auto g = gDelta(7, 0); TLOAD(tDelta7, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 6
            TMATMUL(tST6, tKjL, tQ6, sOpt);
            TMULS(tST6, tST6, scale);
            TCOLEXPANDEXPDIF(tPT6, tST6, tLse6);
            TMATMUL(tST7, tKjL, tQ7, sOpt);
            TMATMUL(tDPT6, tVj, tDo6, sOpt);
            TMATMUL(tDPT7, tVj, tDo7, sOpt);
            TMULS(tST7, tST7, scale);
            TCOLEXPANDEXPDIF(tPT7, tST7, tLse7);
        {
            constexpr int qbi = 6;
            auto &tQ = tQ6; auto &tDo = tDo6;
            auto &tPT = tPT6; auto &tDPT = tDPT6;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta6;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 7;
            auto &tQ = tQ7; auto &tDo = tDo7;
            auto &tPT = tPT7; auto &tDPT = tDPT7;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta7;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 9
            { auto g = gQBlk(8, 0); TLOAD<TileQBlk, 1>(tQ8, g); }
            { auto g = gLse(8, 0); TLOAD(tLse8, g); }
            { auto g = gDoBlk(8, 0); TLOAD<TileDoBlk, 1>(tDo8, g); }
            { auto g = gDelta(8, 0); TLOAD(tDelta8, g); }
            { auto g = gQBlk(9, 0); TLOAD<TileQBlk, 1>(tQ9, g); }
            { auto g = gLse(9, 0); TLOAD(tLse9, g); }
            { auto g = gDoBlk(9, 0); TLOAD<TileDoBlk, 1>(tDo9, g); }
            { auto g = gDelta(9, 0); TLOAD(tDelta9, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 8
            TMATMUL(tST8, tKjL, tQ8, sOpt);
            TMULS(tST8, tST8, scale);
            TCOLEXPANDEXPDIF(tPT8, tST8, tLse8);
            TMATMUL(tST9, tKjL, tQ9, sOpt);
            TMATMUL(tDPT8, tVj, tDo8, sOpt);
            TMATMUL(tDPT9, tVj, tDo9, sOpt);
            TMULS(tST9, tST9, scale);
            TCOLEXPANDEXPDIF(tPT9, tST9, tLse9);
        {
            constexpr int qbi = 8;
            auto &tQ = tQ8; auto &tDo = tDo8;
            auto &tPT = tPT8; auto &tDPT = tDPT8;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta8;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 9;
            auto &tQ = tQ9; auto &tDo = tDo9;
            auto &tPT = tPT9; auto &tDPT = tDPT9;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta9;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 11
            { auto g = gQBlk(10, 0); TLOAD<TileQBlk, 1>(tQ10, g); }
            { auto g = gLse(10, 0); TLOAD(tLse10, g); }
            { auto g = gDoBlk(10, 0); TLOAD<TileDoBlk, 1>(tDo10, g); }
            { auto g = gDelta(10, 0); TLOAD(tDelta10, g); }
            { auto g = gQBlk(11, 0); TLOAD<TileQBlk, 1>(tQ11, g); }
            { auto g = gLse(11, 0); TLOAD(tLse11, g); }
            { auto g = gDoBlk(11, 0); TLOAD<TileDoBlk, 1>(tDo11, g); }
            { auto g = gDelta(11, 0); TLOAD(tDelta11, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 10
            TMATMUL(tST10, tKjL, tQ10, sOpt);
            TMULS(tST10, tST10, scale);
            TCOLEXPANDEXPDIF(tPT10, tST10, tLse10);
            TMATMUL(tST11, tKjL, tQ11, sOpt);
            TMATMUL(tDPT10, tVj, tDo10, sOpt);
            TMATMUL(tDPT11, tVj, tDo11, sOpt);
            TMULS(tST11, tST11, scale);
            TCOLEXPANDEXPDIF(tPT11, tST11, tLse11);
        {
            constexpr int qbi = 10;
            auto &tQ = tQ10; auto &tDo = tDo10;
            auto &tPT = tPT10; auto &tDPT = tDPT10;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta10;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 11;
            auto &tQ = tQ11; auto &tDo = tDo11;
            auto &tPT = tPT11; auto &tDPT = tDPT11;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta11;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 13
            { auto g = gQBlk(12, 0); TLOAD<TileQBlk, 1>(tQ12, g); }
            { auto g = gLse(12, 0); TLOAD(tLse12, g); }
            { auto g = gDoBlk(12, 0); TLOAD<TileDoBlk, 1>(tDo12, g); }
            { auto g = gDelta(12, 0); TLOAD(tDelta12, g); }
            { auto g = gQBlk(13, 0); TLOAD<TileQBlk, 1>(tQ13, g); }
            { auto g = gLse(13, 0); TLOAD(tLse13, g); }
            { auto g = gDoBlk(13, 0); TLOAD<TileDoBlk, 1>(tDo13, g); }
            { auto g = gDelta(13, 0); TLOAD(tDelta13, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 12
            TMATMUL(tST12, tKjL, tQ12, sOpt);
            TMULS(tST12, tST12, scale);
            TCOLEXPANDEXPDIF(tPT12, tST12, tLse12);
            TMATMUL(tST13, tKjL, tQ13, sOpt);
            TMATMUL(tDPT12, tVj, tDo12, sOpt);
            TMATMUL(tDPT13, tVj, tDo13, sOpt);
            TMULS(tST13, tST13, scale);
            TCOLEXPANDEXPDIF(tPT13, tST13, tLse13);
        {
            constexpr int qbi = 12;
            auto &tQ = tQ12; auto &tDo = tDo12;
            auto &tPT = tPT12; auto &tDPT = tDPT12;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta12;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 13;
            auto &tQ = tQ13; auto &tDo = tDo13;
            auto &tPT = tPT13; auto &tDPT = tDPT13;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta13;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 15
            { auto g = gQBlk(14, 0); TLOAD<TileQBlk, 1>(tQ14, g); }
            { auto g = gLse(14, 0); TLOAD(tLse14, g); }
            { auto g = gDoBlk(14, 0); TLOAD<TileDoBlk, 1>(tDo14, g); }
            { auto g = gDelta(14, 0); TLOAD(tDelta14, g); }
            { auto g = gQBlk(15, 0); TLOAD<TileQBlk, 1>(tQ15, g); }
            { auto g = gLse(15, 0); TLOAD(tLse15, g); }
            { auto g = gDoBlk(15, 0); TLOAD<TileDoBlk, 1>(tDo15, g); }
            { auto g = gDelta(15, 0); TLOAD(tDelta15, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 14
            TMATMUL(tST14, tKjL, tQ14, sOpt);
            TMULS(tST14, tST14, scale);
            TCOLEXPANDEXPDIF(tPT14, tST14, tLse14);
            TMATMUL(tST15, tKjL, tQ15, sOpt);
            TMATMUL(tDPT14, tVj, tDo14, sOpt);
            TMATMUL(tDPT15, tVj, tDo15, sOpt);
            TMULS(tST15, tST15, scale);
            TCOLEXPANDEXPDIF(tPT15, tST15, tLse15);
        {
            constexpr int qbi = 14;
            auto &tQ = tQ14; auto &tDo = tDo14;
            auto &tPT = tPT14; auto &tDPT = tDPT14;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta14;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 15;
            auto &tQ = tQ15; auto &tDo = tDo15;
            auto &tPT = tPT15; auto &tDPT = tDPT15;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta15;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 17
            { auto g = gQBlk(16, 0); TLOAD<TileQBlk, 1>(tQ16, g); }
            { auto g = gLse(16, 0); TLOAD(tLse16, g); }
            { auto g = gDoBlk(16, 0); TLOAD<TileDoBlk, 1>(tDo16, g); }
            { auto g = gDelta(16, 0); TLOAD(tDelta16, g); }
            { auto g = gQBlk(17, 0); TLOAD<TileQBlk, 1>(tQ17, g); }
            { auto g = gLse(17, 0); TLOAD(tLse17, g); }
            { auto g = gDoBlk(17, 0); TLOAD<TileDoBlk, 1>(tDo17, g); }
            { auto g = gDelta(17, 0); TLOAD(tDelta17, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 16
            TMATMUL(tST16, tKjL, tQ16, sOpt);
            TMULS(tST16, tST16, scale);
            TCOLEXPANDEXPDIF(tPT16, tST16, tLse16);
            TMATMUL(tST17, tKjL, tQ17, sOpt);
            TMATMUL(tDPT16, tVj, tDo16, sOpt);
            TMATMUL(tDPT17, tVj, tDo17, sOpt);
            TMULS(tST17, tST17, scale);
            TCOLEXPANDEXPDIF(tPT17, tST17, tLse17);
        {
            constexpr int qbi = 16;
            auto &tQ = tQ16; auto &tDo = tDo16;
            auto &tPT = tPT16; auto &tDPT = tDPT16;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta16;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 17;
            auto &tQ = tQ17; auto &tDo = tDo17;
            auto &tPT = tPT17; auto &tDPT = tDPT17;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta17;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 19
            { auto g = gQBlk(18, 0); TLOAD<TileQBlk, 1>(tQ18, g); }
            { auto g = gLse(18, 0); TLOAD(tLse18, g); }
            { auto g = gDoBlk(18, 0); TLOAD<TileDoBlk, 1>(tDo18, g); }
            { auto g = gDelta(18, 0); TLOAD(tDelta18, g); }
            { auto g = gQBlk(19, 0); TLOAD<TileQBlk, 1>(tQ19, g); }
            { auto g = gLse(19, 0); TLOAD(tLse19, g); }
            { auto g = gDoBlk(19, 0); TLOAD<TileDoBlk, 1>(tDo19, g); }
            { auto g = gDelta(19, 0); TLOAD(tDelta19, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 18
            TMATMUL(tST18, tKjL, tQ18, sOpt);
            TMULS(tST18, tST18, scale);
            TCOLEXPANDEXPDIF(tPT18, tST18, tLse18);
            TMATMUL(tST19, tKjL, tQ19, sOpt);
            TMATMUL(tDPT18, tVj, tDo18, sOpt);
            TMATMUL(tDPT19, tVj, tDo19, sOpt);
            TMULS(tST19, tST19, scale);
            TCOLEXPANDEXPDIF(tPT19, tST19, tLse19);
        {
            constexpr int qbi = 18;
            auto &tQ = tQ18; auto &tDo = tDo18;
            auto &tPT = tPT18; auto &tDPT = tDPT18;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta18;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 19;
            auto &tQ = tQ19; auto &tDo = tDo19;
            auto &tPT = tPT19; auto &tDPT = tDPT19;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta19;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 21
            { auto g = gQBlk(20, 0); TLOAD<TileQBlk, 1>(tQ20, g); }
            { auto g = gLse(20, 0); TLOAD(tLse20, g); }
            { auto g = gDoBlk(20, 0); TLOAD<TileDoBlk, 1>(tDo20, g); }
            { auto g = gDelta(20, 0); TLOAD(tDelta20, g); }
            { auto g = gQBlk(21, 0); TLOAD<TileQBlk, 1>(tQ21, g); }
            { auto g = gLse(21, 0); TLOAD(tLse21, g); }
            { auto g = gDoBlk(21, 0); TLOAD<TileDoBlk, 1>(tDo21, g); }
            { auto g = gDelta(21, 0); TLOAD(tDelta21, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 20
            TMATMUL(tST20, tKjL, tQ20, sOpt);
            TMULS(tST20, tST20, scale);
            TCOLEXPANDEXPDIF(tPT20, tST20, tLse20);
            TMATMUL(tST21, tKjL, tQ21, sOpt);
            TMATMUL(tDPT20, tVj, tDo20, sOpt);
            TMATMUL(tDPT21, tVj, tDo21, sOpt);
            TMULS(tST21, tST21, scale);
            TCOLEXPANDEXPDIF(tPT21, tST21, tLse21);
        {
            constexpr int qbi = 20;
            auto &tQ = tQ20; auto &tDo = tDo20;
            auto &tPT = tPT20; auto &tDPT = tDPT20;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta20;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 21;
            auto &tQ = tQ21; auto &tDo = tDo21;
            auto &tPT = tPT21; auto &tDPT = tDPT21;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta21;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 23
            { auto g = gQBlk(22, 0); TLOAD<TileQBlk, 1>(tQ22, g); }
            { auto g = gLse(22, 0); TLOAD(tLse22, g); }
            { auto g = gDoBlk(22, 0); TLOAD<TileDoBlk, 1>(tDo22, g); }
            { auto g = gDelta(22, 0); TLOAD(tDelta22, g); }
            { auto g = gQBlk(23, 0); TLOAD<TileQBlk, 1>(tQ23, g); }
            { auto g = gLse(23, 0); TLOAD(tLse23, g); }
            { auto g = gDoBlk(23, 0); TLOAD<TileDoBlk, 1>(tDo23, g); }
            { auto g = gDelta(23, 0); TLOAD(tDelta23, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 22
            TMATMUL(tST22, tKjL, tQ22, sOpt);
            TMULS(tST22, tST22, scale);
            TCOLEXPANDEXPDIF(tPT22, tST22, tLse22);
            TMATMUL(tST23, tKjL, tQ23, sOpt);
            TMATMUL(tDPT22, tVj, tDo22, sOpt);
            TMATMUL(tDPT23, tVj, tDo23, sOpt);
            TMULS(tST23, tST23, scale);
            TCOLEXPANDEXPDIF(tPT23, tST23, tLse23);
        {
            constexpr int qbi = 22;
            auto &tQ = tQ22; auto &tDo = tDo22;
            auto &tPT = tPT22; auto &tDPT = tDPT22;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta22;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 23;
            auto &tQ = tQ23; auto &tDo = tDo23;
            auto &tPT = tPT23; auto &tDPT = tDPT23;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta23;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 25
            { auto g = gQBlk(24, 0); TLOAD<TileQBlk, 1>(tQ24, g); }
            { auto g = gLse(24, 0); TLOAD(tLse24, g); }
            { auto g = gDoBlk(24, 0); TLOAD<TileDoBlk, 1>(tDo24, g); }
            { auto g = gDelta(24, 0); TLOAD(tDelta24, g); }
            { auto g = gQBlk(25, 0); TLOAD<TileQBlk, 1>(tQ25, g); }
            { auto g = gLse(25, 0); TLOAD(tLse25, g); }
            { auto g = gDoBlk(25, 0); TLOAD<TileDoBlk, 1>(tDo25, g); }
            { auto g = gDelta(25, 0); TLOAD(tDelta25, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 24
            TMATMUL(tST24, tKjL, tQ24, sOpt);
            TMULS(tST24, tST24, scale);
            TCOLEXPANDEXPDIF(tPT24, tST24, tLse24);
            TMATMUL(tST25, tKjL, tQ25, sOpt);
            TMATMUL(tDPT24, tVj, tDo24, sOpt);
            TMATMUL(tDPT25, tVj, tDo25, sOpt);
            TMULS(tST25, tST25, scale);
            TCOLEXPANDEXPDIF(tPT25, tST25, tLse25);
        {
            constexpr int qbi = 24;
            auto &tQ = tQ24; auto &tDo = tDo24;
            auto &tPT = tPT24; auto &tDPT = tDPT24;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta24;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 25;
            auto &tQ = tQ25; auto &tDo = tDo25;
            auto &tPT = tPT25; auto &tDPT = tDPT25;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta25;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 27
            { auto g = gQBlk(26, 0); TLOAD<TileQBlk, 1>(tQ26, g); }
            { auto g = gLse(26, 0); TLOAD(tLse26, g); }
            { auto g = gDoBlk(26, 0); TLOAD<TileDoBlk, 1>(tDo26, g); }
            { auto g = gDelta(26, 0); TLOAD(tDelta26, g); }
            { auto g = gQBlk(27, 0); TLOAD<TileQBlk, 1>(tQ27, g); }
            { auto g = gLse(27, 0); TLOAD(tLse27, g); }
            { auto g = gDoBlk(27, 0); TLOAD<TileDoBlk, 1>(tDo27, g); }
            { auto g = gDelta(27, 0); TLOAD(tDelta27, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 26
            TMATMUL(tST26, tKjL, tQ26, sOpt);
            TMULS(tST26, tST26, scale);
            TCOLEXPANDEXPDIF(tPT26, tST26, tLse26);
            TMATMUL(tST27, tKjL, tQ27, sOpt);
            TMATMUL(tDPT26, tVj, tDo26, sOpt);
            TMATMUL(tDPT27, tVj, tDo27, sOpt);
            TMULS(tST27, tST27, scale);
            TCOLEXPANDEXPDIF(tPT27, tST27, tLse27);
        {
            constexpr int qbi = 26;
            auto &tQ = tQ26; auto &tDo = tDo26;
            auto &tPT = tPT26; auto &tDPT = tDPT26;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta26;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 27;
            auto &tQ = tQ27; auto &tDo = tDo27;
            auto &tPT = tPT27; auto &tDPT = tDPT27;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta27;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 29
            { auto g = gQBlk(28, 0); TLOAD<TileQBlk, 1>(tQ28, g); }
            { auto g = gLse(28, 0); TLOAD(tLse28, g); }
            { auto g = gDoBlk(28, 0); TLOAD<TileDoBlk, 1>(tDo28, g); }
            { auto g = gDelta(28, 0); TLOAD(tDelta28, g); }
            { auto g = gQBlk(29, 0); TLOAD<TileQBlk, 1>(tQ29, g); }
            { auto g = gLse(29, 0); TLOAD(tLse29, g); }
            { auto g = gDoBlk(29, 0); TLOAD<TileDoBlk, 1>(tDo29, g); }
            { auto g = gDelta(29, 0); TLOAD(tDelta29, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 28
            TMATMUL(tST28, tKjL, tQ28, sOpt);
            TMULS(tST28, tST28, scale);
            TCOLEXPANDEXPDIF(tPT28, tST28, tLse28);
            TMATMUL(tST29, tKjL, tQ29, sOpt);
            TMATMUL(tDPT28, tVj, tDo28, sOpt);
            TMATMUL(tDPT29, tVj, tDo29, sOpt);
            TMULS(tST29, tST29, scale);
            TCOLEXPANDEXPDIF(tPT29, tST29, tLse29);
        {
            constexpr int qbi = 28;
            auto &tQ = tQ28; auto &tDo = tDo28;
            auto &tPT = tPT28; auto &tDPT = tDPT28;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta28;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 29;
            auto &tQ = tQ29; auto &tDo = tDo29;
            auto &tPT = tPT29; auto &tDPT = tDPT29;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta29;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 31
            { auto g = gQBlk(30, 0); TLOAD<TileQBlk, 1>(tQ30, g); }
            { auto g = gLse(30, 0); TLOAD(tLse30, g); }
            { auto g = gDoBlk(30, 0); TLOAD<TileDoBlk, 1>(tDo30, g); }
            { auto g = gDelta(30, 0); TLOAD(tDelta30, g); }
            { auto g = gQBlk(31, 0); TLOAD<TileQBlk, 1>(tQ31, g); }
            { auto g = gLse(31, 0); TLOAD(tLse31, g); }
            { auto g = gDoBlk(31, 0); TLOAD<TileDoBlk, 1>(tDo31, g); }
            { auto g = gDelta(31, 0); TLOAD(tDelta31, g); }
#endif
#if FA_BWD_PRELOAD_QBLOCKS > 30
            TMATMUL(tST30, tKjL, tQ30, sOpt);
            TMULS(tST30, tST30, scale);
            TCOLEXPANDEXPDIF(tPT30, tST30, tLse30);
            TMATMUL(tST31, tKjL, tQ31, sOpt);
            TMATMUL(tDPT30, tVj, tDo30, sOpt);
            TMATMUL(tDPT31, tVj, tDo31, sOpt);
            TMULS(tST31, tST31, scale);
            TCOLEXPANDEXPDIF(tPT31, tST31, tLse31);
        {
            constexpr int qbi = 30;
            auto &tQ = tQ30; auto &tDo = tDo30;
            auto &tPT = tPT30; auto &tDPT = tDPT30;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta30;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 32
            { auto g = gDoBlk(32, 0); TLOAD<TileDoBlk, 1>(tDo32, g); }
            { auto g = gDelta(32, 0); TLOAD(tDelta32, g); }
#endif
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if FA_BWD_PRELOAD_QBLOCKS > 32
            { auto g = gQBlk(32, 0); TLOAD<TileQBlk, 1>(tQ32, g); }
            { auto g = gLse(32, 0); TLOAD(tLse32, g); }
#endif
#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
        {
            constexpr int qbi = 31;
            auto &tQ = tQ31; auto &tDo = tDo31;
            auto &tPT = tPT31; auto &tDPT = tDPT31;
            // Consume compact delta before allocating the next dV accumulator.
            auto &tDelta = tDelta31;
#if FA_BWD_Q_COMPACT_AUX
            TCOLEXPANDSUB(tDPT, tDPT, tDelta);
#else
            TSUB(tDPT, tDPT, tDelta);
#endif
            TilePTL tPTL;
            TCVT(tPTL, tPT);
            // Keep only dV in internalACC across Q blocks. Explicit C versions
            // carry the chain; the final Q block drains the result to CellReg.
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt.raw_acc(), kGroupM);
            } else if (qbi == kQBlocks - 1) {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt.acc_hint(), kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo,
                            lrOpt.raw_acc().acc_hint(), kGroupM);
            }
            // Final dV is independent of the remaining dS/dK/dQ work.
            // Start its scaled FP32 store as soon as the ACC tail is ready.
            if (qbi == kQBlocks - 1) {
                auto g = gDv(kvi * kPeNum + tid, 0);
                TMULS(tDv, tDv, dvOutScale);
                TSTORE_CUBE(g, tDv);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 33
            { auto g = gDoBlk(33, 0); TLOAD<TileDoBlk, 1>(tDo33, g); }
            { auto g = gDelta(33, 0); TLOAD(tDelta33, g); }
#endif
            TileST tDST;
            TMUL(tDST, tPT, tDPT);
            TMULS(tDST, tDST, scale);

            TileDSTL tDSTL;
            TCVT(tDSTL, tDST);
            TileDstSh tDstSh;
#if FA_BWD_DIRECT_SHARED
            // PTO #399: each PE contributes its explicit M32 writer range.
            // Preserve MatDType; this FP8 FAG transfers 4 KiB/PE, 16 KiB/parent.
            constexpr unsigned kWriterCells = TileDSTL::LogicalTileBytes / 128;
            const uintptr_t writerOffset = static_cast<uintptr_t>(tid) * kWriterCells;
            if (tid == 0) {
                range::Assemble<TileDstSh, TileDstSh::TilesizeCode,
                                true, false, 0, range::AutoRegSrc,
                                TileDSTL::TilesizeCode> dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else if (tid == kPeNum - 1) {
                range::Assemble<TileDstSh, 0, false, true, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            } else {
                range::Assemble<TileDstSh, 0, false, false, 0,
                                range::AutoRegSrc, TileDSTL::TilesizeCode>
                    dstRange(tDstSh, writerOffset);
                TMOV_CUBE_TO_SHARED_ND<15>(dstRange, tDSTL);
            }
#else
            // Original GM staging control; keep dK between store and reload.
            const int bid = kvi * kQBlocks + qbi;
            {
                auto gw = gStageW(bid * kPeNum + tid, 0);
                TSTORE_CUBE(gw, tDSTL);
            }
#endif

            // dK consumes local dS directly while staging prepares dQ.
            if (qbi == 0) {
                TMATMUL(tDk, tDSTL, tQ, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDk, tDk, tDSTL, tQ, lrOpt, kGroupM);
            }
            // The final dK sum has no later consumer; overlap its store with dQ.
            if (qbi == kQBlocks - 1) {
                auto g = gDk(kvi * kPeNum + tid, 0);
                TMULS(tDk, tDk, dkOutScale);
                TSTORE_CUBE(g, tDk);
            }

#if FA_BWD_PRELOAD_QBLOCKS > 33
            { auto g = gQBlk(33, 0); TLOAD<TileQBlk, 1>(tQ33, g); }
            { auto g = gLse(33, 0); TLOAD(tLse33, g); }
#endif
#if !FA_BWD_DIRECT_SHARED
            {
                auto gr = gStageR(bid, 0);
                TLOAD<TileDst, 1>(tDstSh, gr);
            }
#endif
            TileDq tDq;
            TMATMUL(tDq, tDstSh, tKjR, dqOpt);
            {
                auto *base = dqPtr + (qbi * kTn + tid * kPeM) * qD;
                // Single-KV performance control, matching the earlier comparison.
                static_assert(kKVBlocks == 1, "store control requires one KV tile");
                using GmDqSlab = global_tensor<DType, RowMajor<kPeM, qD>>;
                global_iterator<GmDqSlab, TileDq> dqOut(base);
                auto gdq = dqOut(0, 0);
                // Per-tensor output multiplier; keep the stored dtype FP32.
                TMULS(tDq, tDq, dqOutScale);
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif

    }
}

template <typename MatDType, typename DType, int Sq, int Skv, int qD, int vD,
          int kTk, int kTn, int scaleD = qD>
void flash_attention_bwd_q_pto(DType *dqPtr, DType *dkPtr, DType *dvPtr,
                               MatDType *qPtr, MatDType *kPtr, MatDType *vPtr,
                               MatDType *doPtr, DType *lseBPtr,
                               DType *deltaBPtr, MatDType *dsStagePtr,
                                float dqOutScale, float dkOutScale, float dvOutScale) {
    flash_attention_bwd_q_impl<MatDType, DType, Sq, Skv, qD, vD, kTk, kTn,
                               scaleD>(dqPtr, dkPtr, dvPtr, qPtr, kPtr,
                                        vPtr, doPtr, lseBPtr, deltaBPtr,
                                        dsStagePtr, dqOutScale, dkOutScale, dvOutScale);
}
