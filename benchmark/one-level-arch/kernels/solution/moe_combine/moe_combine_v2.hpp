#ifndef SUPERNPU_MOE_COMBINE_V2_HPP
#define SUPERNPU_MOE_COMBINE_V2_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>

namespace supernpu::tile_isa {

// Combine: expert outputs land in per-(token, topk) window slots
// (slot = tokenId * K + topkId, from expandIdx triple).
// Pack writes data + 1.0f flag per tile row; reduce waits flags,
// accumulates scale-weighted rows into out, clears flag at slot row 0.

// ====== Phase 1: Pack ======
template <typename DType, int NumExpanded, int H, int K, int TileW>
void combine_pack(DType* expandX, int32_t* expandIdx,
                  DType* windowData, float* windowFlag)
{
    constexpr int kTiles = H / TileW;
    using namespace pto;

    using gm_x    = global_tensor<DType, RowMajor<NumExpanded, H>>;
    using gm_win  = global_tensor<DType, RowMajor<NumExpanded, H>>;
    using gm_flag = global_tensor<float,   RowMajor<NumExpanded, TileW>>;
    using tile_d  = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using it_x    = global_iterator<gm_x,    tile_d>;
    using it_win  = global_iterator<gm_win,  tile_d>;
    using it_flag = global_iterator<gm_flag, tile_f>;

    it_x    x_iter(expandX);
    it_win  win_iter(windowData);
    it_flag flag_iter(windowFlag);

    for (int tk = 0; tk < NumExpanded; tk++) {
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
// TSUB stands in for the removed TMOV sync, see combine_pack note) ======
template <int NumExpanded, int TileW>
void check_flag(float* windowFlag, float* predBuf, int slot, int t)
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
void clear_flag(float* windowFlag, int slot)
{
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<NumExpanded, TileW>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    it_flag flag_iter(windowFlag);

    tile_f zeroFlag;
    TEXPANDS(zeroFlag, 0.0f);

    // TSUB as TMOV stand-in (see combine_pack note).
    tile_f sync_zf;
    TSUB(sync_zf, zeroFlag, zeroFlag);

    auto gf = flag_iter(slot, 0);
    TSTORE(gf, zeroFlag);
}

// ====== Phase 2: Reduce (window → scale-weighted sum → out) ======
template <typename DTypeIn, typename DTypeOut, int BS, int H, int K,
          int NumExpanded, int TileW>
void combine_reduce(float* expertScales, DTypeIn* windowData, float* windowFlag,
                    float* predBuf, DTypeOut* out)
{
    constexpr int kTiles = H / TileW;
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

    for (int n = 0; n < BS; n++) {
        for (int t = 0; t < kTiles; t++) {
            // #4 Flag check (tile pass-through; TCMP unavailable on 0828)
            // Scalar readback skipped — self-loopback data always ready
            for (int k = 0; k < K; k++) {
                check_flag<NumExpanded, TileW>(windowFlag, predBuf, n * K + k, t);
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
                // TSUB as TMOV stand-in (see combine_pack note).
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
            clear_flag<NumExpanded, TileW>(windowFlag, n * K + k);
        }
    }
}

// ====== Main entry ======
template <typename DTypeIn, typename DTypeOut, int BS, int H, int K,
          int NumExpanded, int TileW = 128>
void moe_combine_v2(DTypeIn* expandX, float* expertScales,
                    int32_t* expandIdx,
                    DTypeIn* windowData, float* windowFlag,
                    uint32_t* windowState, float* predBuf,
                    DTypeOut* out)
{
    static_assert(BS > 0 && H > 0 && K > 0 && NumExpanded > 0, "dim must be positive");
    static_assert(TileW % 8 == 0, "TileW must be multiple of 8");

    // #5 Window State Init (InitWinState aligned)
    uint32_t dataState = windowState[0];
    windowState[0] = (dataState == 0) ? 1 : 0;
    windowState[1] = 1;
    windowState[2] = 0;

    // ====== Phase 1: Pack (expandX → window + flag) ======
    combine_pack<DTypeIn, NumExpanded, H, K, TileW>(
        expandX, expandIdx, windowData, windowFlag);

    // ====== Phase 2: Reduce (window → weighted sum → out) ======
    combine_reduce<DTypeIn, DTypeOut, BS, H, K, NumExpanded, TileW>(
        expertScales, windowData, windowFlag, predBuf, out);

    // Window state writeback
    windowState[4] = BS;
}

} // namespace supernpu::tile_isa
#endif
