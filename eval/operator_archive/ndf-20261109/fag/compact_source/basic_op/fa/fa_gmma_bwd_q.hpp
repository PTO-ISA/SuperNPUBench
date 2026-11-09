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
                                MatDType *dsStagePtr) {
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
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt, kGroupM);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 2
            { auto g = gDoBlk(2, 0); TLOAD<TileDoBlk, 1>(tDo2, g); }
            { auto g = gDelta(2, 0); TLOAD(tDelta2, g); }
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

#if FA_BWD_PRELOAD_QBLOCKS > 2
            { auto g = gQBlk(2, 0); TLOAD<TileQBlk, 1>(tQ2, g); }
            { auto g = gLse(2, 0); TLOAD(tLse2, g); }
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
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt, kGroupM);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 3
            { auto g = gDoBlk(3, 0); TLOAD<TileDoBlk, 1>(tDo3, g); }
            { auto g = gDelta(3, 0); TLOAD(tDelta3, g); }
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

#if FA_BWD_PRELOAD_QBLOCKS > 3
            { auto g = gQBlk(3, 0); TLOAD<TileQBlk, 1>(tQ3, g); }
            { auto g = gLse(3, 0); TLOAD(tLse3, g); }
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
                TSTORE_CUBE(gdq, tDq);
            }
        }
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
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt, kGroupM);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 4
            { auto g = gDoBlk(4, 0); TLOAD<TileDoBlk, 1>(tDo4, g); }
            { auto g = gDelta(4, 0); TLOAD(tDelta4, g); }
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

#if FA_BWD_PRELOAD_QBLOCKS > 4
            { auto g = gQBlk(4, 0); TLOAD<TileQBlk, 1>(tQ4, g); }
            { auto g = gLse(4, 0); TLOAD(tLse4, g); }
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
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt, kGroupM);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 5
            { auto g = gDoBlk(5, 0); TLOAD<TileDoBlk, 1>(tDo5, g); }
            { auto g = gDelta(5, 0); TLOAD(tDelta5, g); }
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

#if FA_BWD_PRELOAD_QBLOCKS > 5
            { auto g = gQBlk(5, 0); TLOAD<TileQBlk, 1>(tQ5, g); }
            { auto g = gLse(5, 0); TLOAD(tLse5, g); }
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
                TSTORE_CUBE(gdq, tDq);
            }
        }
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
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt, kGroupM);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 6
            { auto g = gDoBlk(6, 0); TLOAD<TileDoBlk, 1>(tDo6, g); }
            { auto g = gDelta(6, 0); TLOAD(tDelta6, g); }
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

#if FA_BWD_PRELOAD_QBLOCKS > 6
            { auto g = gQBlk(6, 0); TLOAD<TileQBlk, 1>(tQ6, g); }
            { auto g = gLse(6, 0); TLOAD(tLse6, g); }
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
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt, kGroupM);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 7
            { auto g = gDoBlk(7, 0); TLOAD<TileDoBlk, 1>(tDo7, g); }
            { auto g = gDelta(7, 0); TLOAD(tDelta7, g); }
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

#if FA_BWD_PRELOAD_QBLOCKS > 7
            { auto g = gQBlk(7, 0); TLOAD<TileQBlk, 1>(tQ7, g); }
            { auto g = gLse(7, 0); TLOAD(tLse7, g); }
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
                TSTORE_CUBE(gdq, tDq);
            }
        }
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
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt, kGroupM);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 8
            { auto g = gDoBlk(8, 0); TLOAD<TileDoBlk, 1>(tDo8, g); }
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

#if FA_BWD_PRELOAD_QBLOCKS > 8
            { auto g = gQBlk(8, 0); TLOAD<TileQBlk, 1>(tQ8, g); }
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
            if (qbi == 0) {
                TMATMUL(tDv, tPTL, tDo, lrOpt, kGroupM);
            } else {
                TMATMUL_ACC(tDv, tDv, tPTL, tDo, lrOpt, kGroupM);
            }
#if FA_BWD_PRELOAD_QBLOCKS > 9
            { auto g = gDoBlk(9, 0); TLOAD<TileDoBlk, 1>(tDo9, g); }
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

#if FA_BWD_PRELOAD_QBLOCKS > 9
            { auto g = gQBlk(9, 0); TLOAD<TileQBlk, 1>(tQ9, g); }
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
                TSTORE_CUBE(gdq, tDq);
            }
        }
#endif

        {
            auto g = gDv(kvi * kPeNum + tid, 0);
            TSTORE_CUBE(g, tDv);
        }
        {
            auto g = gDk(kvi * kPeNum + tid, 0);
            TSTORE_CUBE(g, tDk);
        }
    }
}

template <typename MatDType, typename DType, int Sq, int Skv, int qD, int vD,
          int kTk, int kTn, int scaleD = qD>
void flash_attention_bwd_q_pto(DType *dqPtr, DType *dkPtr, DType *dvPtr,
                               MatDType *qPtr, MatDType *kPtr, MatDType *vPtr,
                               MatDType *doPtr, DType *lseBPtr,
                               DType *deltaBPtr, MatDType *dsStagePtr) {
    flash_attention_bwd_q_impl<MatDType, DType, Sq, Skv, qD, vD, kTk, kTn,
                               scaleD>(dqPtr, dkPtr, dvPtr, qPtr, kPtr,
                                        vPtr, doPtr, lseBPtr, deltaBPtr,
                                        dsStagePtr);
}
