#ifndef SUPERNPU_MOE_DISPATCH_MT_HPP
#define SUPERNPU_MOE_DISPATCH_MT_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace supernpu::tile_isa {

// ============================================================================
// MoE Dispatch — Multi-thread 4-PE SPMD variant
//
// This is the four-PE version of kernels/moe_dispatch/moe_dispatch_v2.hpp.
// The operator semantics are unchanged (pack x into 512B-stride window slots,
// per-expert cumsum, expert-major copy-out into continuous expandXOut); only
// the execution is partitioned across PEs with get_thread_idx():
//
//   Phase 1  dispatch_pack_mt   PE tid owns slots [tid*slotsPerPE,
//                               (tid+1)*slotsPerPE). x rows, window rows,
//                               flags and triples are all PE-disjoint.
//   Phase 2  cal_cumsum_mt      per-PE local histogram over own slots,
//                               cross-PE reduce; each PE writes its assigned
//                               expert range of sendCountsOut /
//                               expertTokenNumsOut / expertStarts.
//   Phase 3  dispatch_copyout_mt PE tid re-emits its own slots at exactly the
//                               expert-major position the single-PE version
//                               produces: dstPos = expertStarts[eid] + rank of
//                               the slot inside its expert group. Output rows
//                               are a permutation, hence PE-disjoint.
//
// Tile rules (established toolchain contract, see kernels/group_token_vec_mt):
//   1. Every TLOAD result is consumed by tile ops (TSUB sync stand-in) and
//      leaves the tile domain via TSTORE; scalar reads hit GM.
//   2. Per-PE tiles are disjoint — no duplicated TLOAD traffic.
//   3. TCMP is unavailable on this toolchain; flag checks stay tile
//      pass-throughs exactly as in the single-PE version.
//   4. Cross-PE hand-offs are guarded by mtBarrier.
// ============================================================================

constexpr int kMtThreadsPerBlock = 4;

// Multi-PE barrier: volatile per-PE phase flags + compiler memory barrier,
// same convention as kernels/group_token_vec_mt/group_token_vec_mt.hpp.
static volatile uint32_t sMtPhaseDone[kMtThreadsPerBlock];

static inline void mtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void mtBarrier(uint32_t phase)
{
    mtCompilerBarrier();
    sMtPhaseDone[get_thread_idx()] = phase;
    mtCompilerBarrier();
    for (int t = 0; t < kMtThreadsPerBlock; ++t) {
        while (sMtPhaseDone[t] < phase) {
        }
    }
    mtCompilerBarrier();
}

// ====== Phase 1: Pack (PE tid owns slots [tid*slotsPerPE, +slotsPerPE)) ======
template <typename DType, int BS, int H, int K, int MoeExpertNum, int TileW,
          int WindowStride>
void dispatch_pack_mt(DType* x, int32_t* expertIds, int32_t* windowTriple,
                      DType* windowData, float* windowFlag)
{
    constexpr int slotCount = BS * K;
    constexpr int hTiles = H / TileW;
    constexpr int slotsPerPE = slotCount / kMtThreadsPerBlock;
    const int tid = static_cast<int>(get_thread_idx());
    using namespace pto;

    using gm_x    = global_tensor<DType, RowMajor<BS, H>>;
    using gm_w    = global_tensor<DType, RowMajor<slotCount, WindowStride>>;
    using gm_flag = global_tensor<float,   RowMajor<slotCount, TileW>>;
    using tile_d  = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using it_x    = global_iterator<gm_x,    tile_d>;
    using it_w    = global_iterator<gm_w,    tile_d>;
    using it_flag = global_iterator<gm_flag, tile_f>;

    // Full-tensor iterators; each PE addresses its own slot range so the
    // per-PE tiles are disjoint (rule 2).
    it_x    x_iter(x);
    it_w    w_iter(windowData);
    it_flag flag_iter(windowFlag);

    for (int tk = tid * slotsPerPE; tk < (tid + 1) * slotsPerPE; tk++) {
        int tokenId = tk / K;
        int topkId  = tk % K;

        for (int t = 0; t < hTiles; t++) {
            tile_d xq;
            auto gx = x_iter(tokenId, t);
            TLOAD(xq, gx);

            // Pipeline sync (SyncFunc<MTE2_V> aligned); TSUB as TMOV
            // stand-in (see moe_dispatch_v2.hpp note).
            tile_d sync_d;
            TSUB(sync_d, xq, xq);

            // 512B stride write: data at [tk*WindowStride + t*TileW]
            auto gw = w_iter(tk, t);
            TSTORE(gw, xq);
        }

        // 512B block packing: flag fill (TEXPANDS + TSTORE)
        tile_f flagTile;
        TEXPANDS(flagTile, 1.0f);

        // Pipeline sync (SyncFunc<V_MTE3> aligned); TSUB as TMOV stand-in.
        tile_f sync_f;
        TSUB(sync_f, flagTile, flagTile);

        auto gf = flag_iter(tk, 0);
        TSTORE(gf, flagTile);

        // FillTriple (scalar, 12B — too small for tile)
        windowTriple[tk * 3 + 0] = 0;
        windowTriple[tk * 3 + 1] = tokenId;
        windowTriple[tk * 3 + 2] = topkId;
    }
}

// ====== Phase 2: CalCumSum (per-PE histogram + cross-PE reduce) ======
// cntLocal layout: [pe * MoeExpertNum + expert]. After the phase-1 barrier
// every PE reduces its assigned expert range; expertStarts[e] holds the
// exclusive prefix (number of slots owned by experts < e), which Phase 3
// uses to reproduce the single-PE expert-major output order.
template <int BS, int K, int MoeExpertNum>
void cal_cumsum_mt(int32_t* expertIds, int32_t* sendCountsOut,
                   int64_t* expertTokenNumsOut, int32_t* cntLocal,
                   int32_t* expertStarts)
{
    constexpr int slotCount = BS * K;
    constexpr int slotsPerPE = slotCount / kMtThreadsPerBlock;
    constexpr int expertsPerPE = MoeExpertNum / kMtThreadsPerBlock;
    const int tid = static_cast<int>(get_thread_idx());

    // Per-PE local histogram over this PE's own slots (scalar GM reads)
    int32_t* myCnt = cntLocal + tid * MoeExpertNum;
    for (int e = 0; e < MoeExpertNum; e++) {
        myCnt[e] = 0;
    }
    for (int i = tid * slotsPerPE; i < (tid + 1) * slotsPerPE; i++) {
        int32_t eid = expertIds[i];
        if (eid >= 0 && eid < MoeExpertNum) myCnt[eid]++;
    }

    // Reduce: each PE writes its assigned expert range (needs every PE's
    // cntLocal — caller guarantees the phase-1 barrier has completed)
    for (int e = 0; e < expertsPerPE; e++) {
        int ge = tid * expertsPerPE + e;
        int32_t sum = 0;
        int32_t before = 0;
        for (int e2 = 0; e2 <= ge; e2++) {
            for (int t = 0; t < kMtThreadsPerBlock; t++) {
                int32_t c = cntLocal[t * MoeExpertNum + e2];
                if (e2 == ge) sum += c; else before += c;
            }
        }
        expertTokenNumsOut[ge] = sum;
        sendCountsOut[ge] = before + sum;   // inclusive cumsum (v2 semantics)
        expertStarts[ge] = before;          // exclusive start for Phase 3
    }
}

// ====== CUMSUM flag write (PE0 only; TEXPANDS + TSTORE at windowState+4) ======
template <int TileW>
void write_cumsum_flag(uint32_t* windowState)
{
    using namespace pto;
    using tile_f = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    tile_f cumsumFlag;
    TEXPANDS(cumsumFlag, 1.0f);

    // TSUB as TMOV stand-in (see moe_dispatch_v2.hpp note).
    tile_f sync_cf;
    TSUB(sync_cf, cumsumFlag, cumsumFlag);

    using gm_st = global_tensor<float, RowMajor<1, TileW>>;
    auto gs = reinterpret_cast<gm_st*>(windowState + 4);
    TSTORE(*gs, cumsumFlag);
}

// ====== Flag check (tile pass-through; TCMP unavailable on the 0828
// toolchain — see moe_dispatch_v2.hpp note. predBuf is never read back) ======
template <int BS, int K, int TileW>
void check_flag_mt(float* windowFlag, float* predBuf, int srcSlot)
{
    constexpr int slotCount = BS * K;
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<slotCount, TileW>>;
    using gm_pred = global_tensor<float, RowMajor<slotCount, TileW>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    using it_pred = global_iterator<gm_pred, tile_f>;

    it_flag flag_iter(windowFlag);
    it_pred pred_iter(predBuf);

    tile_f flagTile;
    auto gf = flag_iter(srcSlot, 0);
    TLOAD(flagTile, gf);

    // Pipeline sync (SyncFunc<MTE2_V> aligned)
    tile_f sync_f1;
    TSUB(sync_f1, flagTile, flagTile);

    auto gp = pred_iter(srcSlot, 0);
    TSTORE(gp, flagTile);
}

// ====== CUMSUM flag check (tile pass-through, PE0 only) ======
template <int TileW>
void check_cumsum_flag_mt(uint32_t* windowState, float* predBuf)
{
    using namespace pto;
    using gm_st  = global_tensor<float, RowMajor<1, TileW>>;
    using gm_pred = global_tensor<float, RowMajor<1, TileW>>;
    using tile_f = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_pred = global_iterator<gm_pred, tile_f>;

    auto gs = reinterpret_cast<gm_st*>(windowState + 4);
    it_pred pred_iter(predBuf);

    tile_f readFlag;
    TLOAD(readFlag, *gs);

    // Pipeline sync (SyncFunc<MTE2_V> aligned)
    tile_f sync_f1;
    TSUB(sync_f1, readFlag, readFlag);

    auto gp = pred_iter(0, 0);
    TSTORE(gp, readFlag);
}

// ====== Clear flag ======
template <int BS, int K, int TileW>
void clear_flag_mt(float* windowFlag, int srcSlot)
{
    constexpr int slotCount = BS * K;
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<slotCount, TileW>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    it_flag flag_iter(windowFlag);

    tile_f zeroFlag;
    TEXPANDS(zeroFlag, 0.0f);

    // TSUB as TMOV stand-in (see moe_dispatch_v2.hpp note).
    tile_f sync_zf;
    TSUB(sync_zf, zeroFlag, zeroFlag);

    auto gf = flag_iter(srcSlot, 0);
    TSTORE(gf, zeroFlag);
}

// ====== Phase 3: Read data from window → expandXOut (PE-disjoint rows) ======
template <typename DType, int BS, int H, int K, int TileW, int WindowStride>
void dispatch_copy_out_mt(DType* windowData, DType* expandXOut,
                          int32_t srcSlot, int dstPos)
{
    constexpr int hTiles = H / TileW;
    using namespace pto;

    using gm_w   = global_tensor<DType, RowMajor<BS * K, WindowStride>>;
    using gm_out = global_tensor<DType, RowMajor<BS * K, H>>;
    using tile_d = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using it_w   = global_iterator<gm_w,   tile_d>;
    using it_out = global_iterator<gm_out, tile_d>;

    it_w   w_iter(windowData);
    it_out out_iter(expandXOut);

    for (int t = 0; t < hTiles; t++) {
        tile_d xq;
        auto gw = w_iter(srcSlot, t);
        TLOAD(xq, gw);

        // Pipeline sync (SyncFunc<MTE2_V> aligned)
        tile_d sync_d;
        TSUB(sync_d, xq, xq);

        auto gout = out_iter(dstPos, t);
        TSTORE(gout, xq);
    }
}

// ====== Main entry (multi-PE SPMD; called by every PE) ======
// Cross-PE hand-offs separated by mtBarrier:
//   barrier(1): all PEs' pack writes (windowData/flag/triple) visible before
//               the histogram reduce and the Phase-3 reads
//   barrier(2): expertStarts + PE0's cumsum flag visible before Phase 3
//   barrier(3): all outputs fully written before any PE leaves the kernel
// cntLocal needs kMtThreadsPerBlock * MoeExpertNum int32 words,
// expertStarts needs MoeExpertNum int32 words of scratch GM.
template <typename DType, int BS, int H, int K, int MoeExpertNum, int TileW = 128,
          int WindowStride = 256>
void moe_dispatch_mt(
    DType* x, int32_t* expertIds, float* expertScales,
    DType* expandXOut, int32_t* expandIdxOut, float* expandScalesOut,
    int32_t* sendCountsOut, int64_t* expertTokenNumsOut,
    DType* windowData, float* windowFlag, float* predBuf,
    int32_t* windowTriple, uint32_t* windowState, DType* outBuf,
    int32_t* cntLocal, int32_t* expertStarts)
{
    constexpr int slotCount = BS * K;
    constexpr int slotsPerPE = slotCount / kMtThreadsPerBlock;
    static_assert(slotCount % kMtThreadsPerBlock == 0,
                  "slotCount (BS*K) must be divisible by the 4 PEs");
    static_assert(MoeExpertNum % kMtThreadsPerBlock == 0,
                  "MoeExpertNum must be divisible by the 4 PEs");
    const int tid = static_cast<int>(get_thread_idx());

    // Window State Init (InitWinState aligned) — PE0 only
    if (tid == 0) {
        uint32_t dataState = windowState[0];
        windowState[0] = (dataState == 0) ? 1 : 0;
        windowState[1] = 1;
        windowState[2] = 0;
    }

    // ====== Phase 1: AllToAllDispatch (pack x → window + flag) ======
    dispatch_pack_mt<DType, BS, H, K, MoeExpertNum, TileW, WindowStride>(
        x, expertIds, windowTriple, windowData, windowFlag);
    mtBarrier(1);

    // ====== Phase 2: CalCumSum (count + cumsum) ======
    cal_cumsum_mt<BS, K, MoeExpertNum>(expertIds, sendCountsOut,
                                        expertTokenNumsOut, cntLocal,
                                        expertStarts);

    // CUMSUM soft sync + flag check — PE0 only (same PE writes then reads)
    if (tid == 0) {
        write_cumsum_flag<TileW>(windowState);
        check_cumsum_flag_mt<TileW>(windowState, predBuf);
    }
    mtBarrier(2);

    // ====== Phase 3: LocalWindowCopy (read window → continuous output) ======
    // Each PE re-emits its own slots at the expert-major position the
    // single-PE version produces: dstPos = expertStarts[eid] + rank of the
    // slot among the slots of the same expert. expertIds is a read-only
    // input, so the rank scan needs no extra synchronization; the window /
    // triple reads are covered by barrier(1) and expertStarts by barrier(2).
    for (int i = tid * slotsPerPE; i < (tid + 1) * slotsPerPE; i++) {
        int32_t eid = expertIds[i];
        if (eid < 0 || eid >= MoeExpertNum) continue;

        int32_t rank = 0;
        for (int j = 0; j < i; j++) {
            if (expertIds[j] == eid) rank++;
        }
        int32_t dstPos = expertStarts[eid] + rank;

        // Flag check (tile pass-through; TCMP unavailable on 0828)
        check_flag_mt<BS, K, TileW>(windowFlag, predBuf, i);

        // Read data from window → expandXOut (tile copy, 512B stride)
        dispatch_copy_out_mt<DType, BS, H, K, TileW, WindowStride>(
            windowData, expandXOut, i, dstPos);

        // Read triple + scale (scalar)
        expandIdxOut[dstPos * 3 + 0] = windowTriple[i * 3 + 0];
        expandIdxOut[dstPos * 3 + 1] = windowTriple[i * 3 + 1];
        expandIdxOut[dstPos * 3 + 2] = windowTriple[i * 3 + 2];
        expandScalesOut[dstPos] = expertScales[i];

        // Clear flag (TEXPANDS(0.0) + TSTORE)
        clear_flag_mt<BS, K, TileW>(windowFlag, i);
    }
    mtBarrier(3);

    // Window state writeback — PE0 only
    if (tid == 0) {
        windowState[4] = BS;
    }
}

} // namespace supernpu::tile_isa
#endif
