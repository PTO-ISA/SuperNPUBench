#ifndef SUPERNPU_MOE_COMBINE_MT_HPP
#define SUPERNPU_MOE_COMBINE_MT_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace supernpu::tile_isa {

// ============================================================================
// MoE Combine — Multi-thread 4-PE SPMD variant
//
// This is the four-PE version of kernels/moe_combine/moe_combine_v2.hpp.
// The operator semantics are unchanged (pack expert outputs into
// per-(token, topk) window slots, then reduce scale-weighted rows into out);
// only the execution is partitioned across PEs with get_thread_idx():
//
//   Phase 1  combine_pack_mt    PE tid owns expanded rows [tid*rowsPerPE,
//                                (tid+1)*rowsPerPE). expandIdx rows are
//                                read PE-disjoint; window/flag writes are
//                                PE-disjoint provided expandIdx maps to
//                                distinct slots (dispatch-stage contract).
//   Phase 2  combine_reduce_mt  PE tid owns tokens [tid*tokensPerPE,
//                                (tid+1)*tokensPerPE). Reads the K window
//                                slots of its own tokens (written by any PE
//                                in Phase 1 — covered by the phase barrier),
//                                accumulates in fp32, writes its own out
//                                rows and clears its own slots' flags.
//
// Tile rules (established toolchain contract, see kernels/group_token_vec_mt
// and kernels/moe_dispatch_mt):
//   1. Every TLOAD result is consumed by tile ops (TSUB sync stand-in,
//      TCVT/TMULS/TADD compute) and leaves the tile domain via TSTORE;
//      scalar reads hit GM.
//   2. Per-PE tiles are disjoint — no duplicated TLOAD traffic.
//   3. TCMP is unavailable on this toolchain; flag checks stay tile
//      pass-throughs exactly as in the single-PE version.
//   4. Cross-PE hand-offs are guarded by combineMtBarrier.
// ============================================================================

constexpr int kCombineMtThreads = 4;

// Multi-PE barrier: volatile per-PE phase flags + compiler memory barrier,
// same convention as kernels/group_token_vec_mt/group_token_vec_mt.hpp.
static volatile uint32_t sCombineMtPhaseDone[kCombineMtThreads];

static inline void combineMtCompilerBarrier()
{
    __asm__ volatile("" : : : "memory");
}

static inline void combineMtBarrier(uint32_t phase)
{
    combineMtCompilerBarrier();
    sCombineMtPhaseDone[get_thread_idx()] = phase;
    combineMtCompilerBarrier();
    for (int t = 0; t < kCombineMtThreads; ++t) {
        while (sCombineMtPhaseDone[t] < phase) {
        }
    }
    combineMtCompilerBarrier();
}

// ====== Phase 1: Pack (PE tid owns expanded rows [tid*rowsPerPE, +rowsPerPE)) ======
template <typename DType, int NumExpanded, int H, int K, int TileW>
void combine_pack_mt(DType* expandX, int32_t* expandIdx,
                     DType* windowData, float* windowFlag)
{
    constexpr int kTiles = H / TileW;
    constexpr int rowsPerPE = NumExpanded / kCombineMtThreads;
    const int tid = static_cast<int>(get_thread_idx());
    using namespace pto;

    using gm_x    = global_tensor<DType, RowMajor<NumExpanded, H>>;
    using gm_win  = global_tensor<DType, RowMajor<NumExpanded, H>>;
    using gm_flag = global_tensor<float,   RowMajor<NumExpanded, TileW>>;
    using tile_d  = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using it_x    = global_iterator<gm_x,    tile_d>;
    using it_win  = global_iterator<gm_win,  tile_d>;
    using it_flag = global_iterator<gm_flag, tile_f>;

    // Full-tensor iterators; each PE addresses its own expanded-row range so
    // the per-PE tiles are disjoint (rule 2).
    it_x    x_iter(expandX);
    it_win  win_iter(windowData);
    it_flag flag_iter(windowFlag);

    for (int tk = tid * rowsPerPE; tk < (tid + 1) * rowsPerPE; tk++) {
        int tokenId = expandIdx[tk * 3 + 1];
        int topkId  = expandIdx[tk * 3 + 2];
        int slot    = tokenId * K + topkId;

        for (int t = 0; t < kTiles; t++) {
            tile_d xq;
            auto gx = x_iter(tk, t);
            TLOAD(xq, gx);

            // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
            // TSUB(dst, src, src) is a compile-time stand-in for TMOV(dst, src):
            // the 0828 toolchain's asm matcher rejects TMOV's 0.58.4 B.DATR
            // syntax ("NORM, DTYPE_NONE, Zero"); TSUB emits no B.DATR and keeps
            // the same src->dst dependency edge (dst = src - src = 0).
            tile_d sync_d;
            TSUB(sync_d, xq, xq);

            auto gw = win_iter(slot, t);
            TSTORE(gw, xq);

            // #1 Flag fill: data ready marker (TEXPANDS + TSTORE)
            tile_f flagTile;
            TEXPANDS(flagTile, 1.0f);

            auto gf = flag_iter(slot, t);
            TSTORE(gf, flagTile);
        }
    }
}

// ====== #4 Flag check (tile pass-through; TCMP's 0.58.4 B.DATR syntax is
// rejected by the 0828 toolchain asm matcher, so the EQ predicate collapses
// to a flag->predBuf copy. predBuf consumers treat non-1.0f as "not ready";
// the flag is 1.0f after pack, so pass-through preserves the wait semantics.
// TSUB stands in for the removed TMOV sync, see combine_pack_mt note) ======
template <int NumExpanded, int TileW>
void combine_check_flag_mt(float* windowFlag, float* predBuf, int slot, int t)
{
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<NumExpanded, TileW>>;
    using gm_pred = global_tensor<float, RowMajor<NumExpanded, TileW>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    using it_pred = global_iterator<gm_pred, tile_f>;

    it_flag flag_iter(windowFlag);
    it_pred pred_iter(predBuf);

    tile_f flagTile;
    auto gf = flag_iter(slot, t);
    TLOAD(flagTile, gf);

    // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
    tile_f sync_f1;
    TSUB(sync_f1, flagTile, flagTile);

    auto gp = pred_iter(slot, t);
    TSTORE(gp, flagTile);
}

// ====== #1 Clear flag ======
template <int NumExpanded, int TileW>
void combine_clear_flag_mt(float* windowFlag, int slot)
{
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<NumExpanded, TileW>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    it_flag flag_iter(windowFlag);

    tile_f zeroFlag;
    TEXPANDS(zeroFlag, 0.0f);

    // TSUB as TMOV stand-in (see combine_pack_mt note).
    tile_f sync_zf;
    TSUB(sync_zf, zeroFlag, zeroFlag);

    auto gf = flag_iter(slot, 0);
    TSTORE(gf, zeroFlag);
}

// ====== Phase 2: Reduce (PE tid owns tokens [tid*tokensPerPE, +tokensPerPE)) ======
// Reads this PE's tokens' K window slots (written by any PE in Phase 1 —
// the caller guarantees the phase-1 barrier has completed), accumulates
// scale-weighted rows in fp32, writes PE-disjoint out rows, clears flags.
template <typename DTypeIn, typename DTypeOut, int BS, int H, int K,
          int NumExpanded, int TileW>
void combine_reduce_mt(float* expertScales, DTypeIn* windowData, float* windowFlag,
                       float* predBuf, DTypeOut* out)
{
    constexpr int kTiles = H / TileW;
    constexpr int tokensPerPE = BS / kCombineMtThreads;
    const int tid = static_cast<int>(get_thread_idx());
    using namespace pto;

    using gm_win  = global_tensor<DTypeIn,  RowMajor<NumExpanded, H>>;
    using gm_out  = global_tensor<DTypeOut, RowMajor<BS, H>>;
    using tile_d  = Tile<Location::Vec, DTypeIn,  1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,    1, TileW, BLayout::RowMajor>;
    using tile_o  = Tile<Location::Vec, DTypeOut, 1, TileW, BLayout::RowMajor>;
    using it_win  = global_iterator<gm_win,  tile_d>;
    using it_out  = global_iterator<gm_out,  tile_o>;

    it_win  win_iter(windowData);
    it_out  out_iter(out);

    for (int n = tid * tokensPerPE; n < (tid + 1) * tokensPerPE; n++) {
        for (int t = 0; t < kTiles; t++) {
            // #4 Flag check (tile pass-through; TCMP unavailable on 0828)
            // Scalar readback skipped — barrier-covered data always ready
            for (int k = 0; k < K; k++) {
                combine_check_flag_mt<NumExpanded, TileW>(windowFlag, predBuf, n * K + k, t);
            }

            // Flag wait (scalar readback of predBuf)
            for (int k = 0; k < K; k++) {
                int slot = n * K + k;
                if (predBuf[slot * TileW] < 0.5f) break;
            }

            tile_f acc;
            TEXPANDS(acc, 0.0f);

            for (int k = 0; k < K; k++) {
                int slot = n * K + k;
                float scale = expertScales[n * K + k];

                tile_d xq;
                auto gw = win_iter(slot, t);
                TLOAD(xq, gw);

                // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
                // TSUB as TMOV stand-in (see combine_pack_mt note).
                tile_d sync_d;
                TSUB(sync_d, xq, xq);

                tile_f xf;
                TCVT(xf, xq);
                TMULS(xf, xf, scale);
                TADD(acc, acc, xf);
            }

            tile_o oq;
            TCVT(oq, acc);
            auto gout = out_iter(n, t);
            TSTORE(gout, oq);
        }

        // #1 Clear flag (TEXPANDS(0.0) + TSTORE)
        for (int k = 0; k < K; k++) {
            combine_clear_flag_mt<NumExpanded, TileW>(windowFlag, n * K + k);
        }
    }
}

// ====== Main entry (multi-PE SPMD; called by every PE) ======
// Cross-PE hand-offs separated by combineMtBarrier:
//   barrier(1): all PEs' pack writes (windowData/windowFlag) visible before
//               any PE's reduce reads another PE's slots
//   barrier(2): all out rows written and flags cleared before any PE leaves
template <typename DTypeIn, typename DTypeOut, int BS, int H, int K,
          int NumExpanded, int TileW = 128>
void moe_combine_mt(DTypeIn* expandX, float* expertScales,
                    int32_t* expandIdx,
                    DTypeIn* windowData, float* windowFlag,
                    uint32_t* windowState, float* predBuf,
                    DTypeOut* out)
{
    static_assert(BS > 0 && H > 0 && K > 0 && NumExpanded > 0, "dim must be positive");
    static_assert(TileW % 8 == 0, "TileW must be multiple of 8");
    static_assert(NumExpanded % kCombineMtThreads == 0,
                  "NumExpanded must be divisible by the 4 PEs");
    static_assert(BS % kCombineMtThreads == 0,
                  "BS must be divisible by the 4 PEs");
    const int tid = static_cast<int>(get_thread_idx());

    // #5 Window State Init (InitWinState aligned) — PE0 only
    if (tid == 0) {
        uint32_t dataState = windowState[0];
        windowState[0] = (dataState == 0) ? 1 : 0;
        windowState[1] = 1;
        windowState[2] = 0;
    }

    // ====== Phase 1: Pack (expandX → window + flag) ======
    combine_pack_mt<DTypeIn, NumExpanded, H, K, TileW>(
        expandX, expandIdx, windowData, windowFlag);
    combineMtBarrier(1);

    // ====== Phase 2: Reduce (window → weighted sum → out) ======
    combine_reduce_mt<DTypeIn, DTypeOut, BS, H, K, NumExpanded, TileW>(
        expertScales, windowData, windowFlag, predBuf, out);
    combineMtBarrier(2);

    // Window state writeback — PE0 only
    if (tid == 0) {
        windowState[4] = BS;
    }
}

} // namespace supernpu::tile_isa
#endif
