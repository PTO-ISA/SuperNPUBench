#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

// The current tileop API exposes the right/shared TMATMUL operand as TileRight.
// Keep the local name explicit in this TMATMUL example to show that K/V are loaded
// to shared tile registers instead of PE-private left tiles.
namespace fa_detail {

template <typename LocalTile, bool UseSharedTile>
struct RightTileStorage {
    using type = LocalTile;
};

template <typename LocalTile>
struct RightTileStorage<LocalTile, true> {
    using type = SharedTile<LocalTile>;
};

}  // namespace fa_detail
// 遗留
// 1.高性能上是否存在表达问题？例如，软件pingping流水是否需要暴露(性能)
// 2.layout转换是否需要对程序员可见，数据类型cube-vec之间layout转换

// 4-PE tmatmul FlashAttention programming model.
//
// This kernel models a Blackwell-like execution style where get_thread_idx()
// selects the current PE's Q/O row range. Each PE owns one Q/O row slice of kTm rows,
// while K/V are loaded as full shared tiles and the union TMATMUL
// call consumes the PE-local Q/P tiles with those shared K/V operands.
//
// Mathematical semantics:
//   O = softmax((Q * K^T) / sqrt(scaleD)) * V
//   Q: [Sq, qD], K: [Skv, qD], V: [Skv, vD], O: [Sq, vD]
//
// Big-tile vs small-tile naming:
//   - Big tile is the logical tile visible to the collective tmatmul:
//       Q_big: [4*kTm, qD]
//       K_big: [kTk, qD], consumed by tmatmul as K_big^T [qD, kTk]
//       W_big: [4*kTm, kTk]
//       V_big: [kTk, vD]
//       O_big/PV_big: [4*kTm, vD]
//   - Small tile is the PE-local storage unit:
//       Q_pe: [kTm, qD]
//       W_pe: [kTm, kTk]
//       O_pe/PV_pe: [kTm, vD]
//   - K/V are shared full tiles:
//       K_shared: [kTk, qD]
//       V_shared: [kTk, vD]
//   - Each PE only names its own Q/P/O tiles. Other PE-local tiles are not
//     visible in this SPMD pseudo model. TMATMUL collectively observes the four
//     PE-local lhs tiles plus the shared K/V tile as one logical big tile.
//     Vector tileOPs are still applied PE-locally.
//
// Memory/layout contract:
//   - TLOAD/TSTORE are pure ND DMA copies. They do not transpose, swizzle, or
//     pad data while moving it between global memory and tile storage.
//   - Q, K, V, O global tensors are all RowMajor.
//   - K/V are not split by PE. They are direct row-major shared tiles. TMATMUL
//     consumes K as K^T internally; that interpretation is carried by compute,
//     not by TLOAD.
//   - O is stored as row-major [Sq, vD].
//
// Compute contract:
//   - Each PE independently executes vector tileOPs for its [kTm, *] slice.
//   - Every physical PE-local/shared tile is constrained to at most 8 KiB.
//   - tmatmul is a compiler intrinsic used as a scalar instruction in each PE's
//     program. Each PE passes only its own lhs/acc tile, while K/V are shared
//     staging tiles. The collective execution fuses the PE-local slices into
//     one logical GEMM.
//
// Current simplification:
//   - This TMATMUL example fixes one Q big tile and one K/V big tile per loop
//     step. It intentionally omits the extra array dimensions and merge logic
//     used by multi-block unrolling.

template <bool UseSharedTile, typename dtype, int Sq, int Skv, int qD, int vD,
          int kTm, int kTk, int scaleD = qD>
void flash_attention_2d_unroll_shared_impl(dtype *out_ptr, dtype *q_ptr,
                                            dtype *k_ptr, dtype *v_ptr) {
    const uint32_t tid = get_thread_idx();
    q_ptr += tid * Sq * qD;
    out_ptr += tid * Sq * vD;

    // This function receives the full Q/O base pointer. get_thread_idx() selects
    // the current PE's local Q/O row range before any tile iterator is built.
    //   current PE M slice: kTm
    //   collective big M  : 4 * kTm
    constexpr int kPaddedQ = (qD == 192 ? 256 : qD);
    // FP32 with head dim 128 needs 16-row box alignment, so the minimum
    // legal Q/K/V/O tile is 16 * 128 * 4 = 8 KiB.
    constexpr int kTileByteLimit = 8 * 1024;

    // tile validity check
    static_assert(kTm * kPaddedQ * sizeof(dtype) <= kTileByteLimit,
                  "each PE Q tile must not exceed 8 KiB");
    static_assert(kTm * kTk * sizeof(float) <= kTileByteLimit,
                  "each PE score tile must not exceed 8 KiB");
    static_assert(kTm * vD * sizeof(float) <= kTileByteLimit,
                  "each PE output tile must not exceed 8 KiB");
    static_assert(kTk * kPaddedQ * sizeof(dtype) <= kTileByteLimit,
                  "shared K tile must not exceed 8 KiB");
    static_assert(kTk * vD * sizeof(dtype) <= kTileByteLimit,
                  "shared V tile must not exceed 8 KiB");

    // Global tensor layout. All four tensors are RowMajor so TLOAD/TSTORE can
    // remain pure ND-to-ND copies:
    //   gmQ: [Sq,  qD], row stride qD
    //   gmK: [Skv, qD], row stride qD
    //   gmV: [Skv, vD], row stride vD
    //   gmO: [Sq,  vD], row stride vD
    using gmQ = global_tensor<dtype, RowMajor<Sq, qD>>;
    // Reinterpret row-major K[Skv,qD] as the transposed col-major view
    // K^T[qD,Skv], so the right TMATMUL operand has shape [qD,kTk].
    using gmK = global_tensor<dtype, ColMajor<qD, Skv>>;
    using gmV = global_tensor<dtype, RowMajor<Skv, vD>>;
    using gmO = global_tensor<dtype, RowMajor<Sq, vD>>;

    // Q is PE-local; K/V are full shared tiles loaded by TLOAD:
    //   tileQ: Q_pe, physical [kTm, kPaddedQ], valid [kTm, qD], NZ-layout
    //   tileK: K^T_shared, physical [kPaddedQ, kTk], valid [qD, kTk], Zn-layout
    //   tileV: V_shared, physical/logical [kTk, vD], Zn-layout
    // The tmatmul intrinsic must read the shared K tile as Zn.
    using tileQ = TileLeft<dtype, kTm, kPaddedQ, kTm, qD>;
    using tileKLocal = TileRight<dtype, kPaddedQ, kTk, qD, kTk>;
    using tileVLocal = TileRight<dtype, kTk, vD>;
    using tileK = typename fa_detail::RightTileStorage<
        tileKLocal, UseSharedTile>::type;
    using tileV = typename fa_detail::RightTileStorage<
        tileVLocal, UseSharedTile>::type;

    // QK score tiles:
    //   tmatmul input in each PE:
    //     tQ          -> current PE's Q_pe [kTm, qD]
    //     tK          -> shared K tile [kTk, qD]
    //   logical collective input:
    //     Q_big       -> concat Q_pe from PE0..PE3, shape [4*kTm, qD]
    //   tmatmul output in each PE:
    //     tW          -> current PE's W_pe [kTm, kTk], ordinary Vec tile
    //   logical collective output:
    //     W_big       -> concat W_pe from PE0..PE3, shape [4*kTm, kTk].
    //
    // tileW/tileWCast are PE-local vector tiles used by online softmax.
    using tileW = Tile<Location::Vec, float, kTm, kTk, BLayout::ColMajor>;
    using tileWCast = Tile<Location::Vec, dtype,
                           kTm, kTk, BLayout::ColMajor>;
    // tilePLeft is the PE-local probability tile converted back to a tmatmul lhs:
    //   P_pe: [kTm, kTk]
    using tilePLeft = TileLeft<dtype, kTm, kTk>;

    // PV/output tiles:
    //   TMATMUL(P_big, V_big) -> PV_big [4*kTm, vD]
    //   current PE receives PV_pe/tileO [kTm, vD].
    //   tileO accumulates the online-softmax numerator for this PE row slice.
    //   tileOCast is the dtype tile stored to gmO.
    using tileO = Tile<Location::Vec, float, kTm, vD, BLayout::ColMajor>;
    using tileOCast = Tile<Location::Vec, dtype, kTm, vD, BLayout::ColMajor>;

    // Online softmax row-state tiles. Each PE owns kTm independent query
    // rows, and every row has one scalar max/sum/scale value.
    // Physical cols = 8 only for tile alignment; valid cols = 1.
    //   tileMax/tileSum/tileScale: valid shape [kTm, 1]
    using tileMax = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor,
                         kTm, 1>;
    using tileSum = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor,
                         kTm, 1>;
    using tileScale = Tile<Location::Vec, float, kTm, 8, BLayout::ColMajor,
                           kTm, 1>;

    using itQ = global_iterator<gmQ, tileQ>;
    // global_iterator describes the GM window with the underlying local tile
    // shape. TLOAD may then target either that local tile or its SharedTile
    // wrapper; SharedTile itself is intentionally not an iterator tile type.
    using itK = global_iterator<gmK, tileKLocal>;
    using itV = global_iterator<gmV, tileVLocal>;
    using itO = global_iterator<gmO, tileOCast>;

    itQ gIterQ(q_ptr);
    itK gIterK(k_ptr);
    itV gIterV(v_ptr);
    itO gIterO(out_ptr);

    // Score scaling for softmax(QK / sqrt(scaleD)).
    const float scale = 1.0f / sqrt((float)scaleD);
    // Qb is counted in logical big Q tiles [kTm, qD].
    // Kb is counted in logical big K/V tiles [kTk, qD/vD].
    constexpr int Qb = (Sq + kTm - 1) / kTm;
    constexpr int Kb = (Skv + kTk - 1) / kTk;

    for (int i = 0; i < Qb; ++i) {
        tileQ tQ;

        // Each PE loads only its own row slice. The caller has already passed
        // a PE-local Q pointer/range, so the local iterator uses i directly:
        //   before TLOAD:
        //     Q_pe[i*kTm : (i+1)*kTm, 0:qD]
        //   after TLOAD:
        //     tQ = current PE's Q_pe, valid shape [kTm, qD]
        //
        // Across the four independent PE programs, the four tQ instances
        // logically form Q_big [4*kTm, qD]. No PE can directly see another PE's
        // tQ; the collective tmatmul observes them as a group.
        //
        // TLOAD remains a direct row-major ND copy and does not change layout.
        auto gQ = gIterQ(i, 0);
        // ND->Nz
        // 4 PE 发送4条tload treg指令
        TLOAD(tQ, gQ);

        tileMax tMax;
        tileSum tSum;
        tileO tO, tPV;
        tileScale tScale;

        // Initialize online softmax states for the current PE row slice:
        //   tMax valid shape [kTm, 1] = -inf
        //   tSum valid shape [kTm, 1] = 0
        //   tO is initialized after the first PV block.
        TEXPANDS(tMax, -1e30f);
        TEXPANDS(tSum, 0.0f);

        // tMax/tSum/tO are ordinary Tile values carried between K blocks.
        // The current backend cannot keep ordinary Tile PHIs across a retained
        // loop, so fully expand this compile-time-bounded online-softmax loop.
#pragma clang loop unroll(full)
        for (int j = 0; j < Kb; ++j) {
            tileK tK;

            // K storage is row-major [Skv,qD], exposed through the equivalent
            // col-major K^T [qD,Skv] view and loaded as [qD,kTk].
            //
            // For K block j:
            //   before TLOAD:
            //     K^T[0:qD, j*kTk : (j+1)*kTk]
            //   after TLOAD:
            //     tK = K^T_shared, valid shape [qD, kTk].
            // ND->Zn
            auto gK = gIterK(0, j);
            // map to TMATMUL.LD, load tile to staging B
            // 加载到shared tile reg, 只有tid=0会执行加载指令，tid=1，2，3不执行
            TLOAD(tK, gK);

            tileW tW;

            // QK group GEMM:
            //   Inputs:
            //     tQ            -> current PE's Q_pe [kTm, qD]
            //     tK            -> shared K tile [kTk, qD], consumed as K^T
            //   Logical output:
            //     W_big = Q_big * K_big^T, shape [kTm, kTk]
            //   Physical output:
            //     tW is the current PE's ordinary W_pe [kTm, kTk].
            //
            // Then the current PE scales its own W_pe:
            //   tW = FIXP(Q*K^T) / sqrt(scaleD), shape [kTm, kTk].
            // Each PE program only passes its private tQ/tW. The collective
            // tmatmul observes all PE-local lhs/output tiles plus shared tK as one
            // logical big-tile GEMM.
            // 对应指令gmma，4-PE发送4条gmma指令，
            TMATMUL_FIXP(tW, tQ, tK);
            TMULS(tW, tW, scale);

            tileMax tNewMax;
            tileSum tNewSum;
            tileWCast tExpW;
            tileMax tLocalMax;
            tileSum tLocalSum;
            tileSum tScaledOldSum;

            // Online softmax is PE-local. The current PE computes row
            // reductions over its own [kTm, kTk] score tile.
            //
            //   tW            : current logits, [kTm, kTk]
            //   tLocalMax     : rowmax over kTk, [kTm, 1]
            //   tNewMax       : max(tMax, tLocalMax), [kTm,1]
            //   tScale        : exp(tMax - tNewMax), [kTm,1]
            //   tLocalSum     : rowsum(exp(tW - tNewMax)), [kTm,1]
            //   tNewSum       : updated denominator, [kTm,1]
            TROWMAX(tLocalMax, tW);
            TMAX(tNewMax, tMax, tLocalMax);
            TSUB(tScale, tMax, tNewMax);
            TEXP(tScale, tScale);
            TMUL(tScaledOldSum, tSum, tScale);

            // Convert logits to unnormalized probabilities under tNewMax:
            //   before TROWEXPANDSUB: tW [kTm,kTk]
            //   broadcast source    : tNewMax [kTm,1]
            //   after TEXP          : tW stores p [kTm,kTk]
            //   TROWSUM             : tLocalSum [kTm,1]
            //   TCVT                : tExpW [kTm,kTk]
            TROWEXPANDSUB(tW, tW, tNewMax);
            TEXP(tW, tW);
            TROWSUM(tLocalSum, tW);
            TCVT(tExpW, tW);
            TADD(tNewSum, tScaledOldSum, tLocalSum);

            tileV tV;

            // V is row-major [Skv, vD] and is loaded once as a full shared
            // [kTk, vD] tile:
            //   before TLOAD:
            //     V[j*kTk : (j+1)*kTk, 0:vD]
            //   after TLOAD:
            //     tV = V_shared, valid shape [kTk, vD].
            auto gV = gIterV(j, 0);
            TLOAD(tV, gV);
            // P/probability tile preparation:
            //   tExpW : current PE's Vec probability tile [kTm,kTk]
            //   tPLeft: current PE's Left/tmatmul lhs tile [kTm,kTk]
            //
            // Across the four independent PE programs, all tPLeft instances
            // logically form P_big [kTm,kTk]. No PE-local P tile is represented
            // as an array here.
            tilePLeft tPLeft;

            // PV collective GEMM:
            //   Inputs:
            //     tPLeft        -> current PE's P_pe [kTm, kTk]
            //     tV            -> shared V tile [kTk, vD]
            //   Logical output:
            //     PV_big = P_big * V_big, shape [kTm, vD]
            //   Physical output:
            //     tPV is the current PE's ordinary PV_pe [kTm, vD].
            TCVT(tPLeft, tExpW);
            TMATMUL_FIXP(tPV, tPLeft, tV);

            // Consume the current PV contribution as an ordinary Vec tile.
            //
            // Online output numerator update:
            //   first K/V block:
            //     tO = tPV
            //   later K/V blocks:
            //     tO = tO * exp(m_old - m_new) + tPV
            //
            // TROWEXPANDMUL broadcasts tScale [kTm,1] across vD.
            if (j == 0) {
                tO = tPV;
            } else {
                TROWEXPANDMUL(tO, tO, tScale);
                TADD(tO, tO, tPV);
            }

            tMax = tNewMax;
            tSum = tNewSum;
        }

        tileSum tInvSum;
        tileOCast tOCast;

        // Final normalization and store for the current PE:
        //   tInvSum = 1 / tSum, shape [kTm,1]
        //   tO *= tInvSum with row broadcast, shape [kTm,vD]
        //   tOCast converts float output to dtype, shape [kTm,vD]
        //   TSTORE writes the current PE-local O row slice:
        //     O_pe[i*kTm : (i+1)*kTm, 0:vD]
        //
        // Combining stores from all independent PEs produces O_big [4*kTm,vD].
        TRECIP(tInvSum, tSum);
        TROWEXPANDMUL(tO, tO, tInvSum);
        TCVT(tOCast, tO);
        auto dstO = gIterO(i, 0);
        TSTORE(dstO, tOCast);
    }
}

template <typename dtype, int Sq, int Skv, int qD, int vD, int kTm, int kTk,
          int scaleD = qD>
void flash_attention_2d_unroll_tmatmul_pto(dtype *out_ptr, dtype *q_ptr,
                                           dtype *k_ptr, dtype *v_ptr) {
    flash_attention_2d_unroll_shared_impl<
        false, dtype, Sq, Skv, qD, vD, kTm, kTk, scaleD>(
        out_ptr, q_ptr, k_ptr, v_ptr);
}
