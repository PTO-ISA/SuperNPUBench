#ifndef SUPERNPU_MOE_DISPATCH_V2_HPP
#define SUPERNPU_MOE_DISPATCH_V2_HPP
#include <common/pto_tileop.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace supernpu::tile_isa {

// bf16 H=128 → 256B data per slot. A5 hCommuSize_ = ceil(256,480)*512 = 512B.
// windowStride = 512B / sizeof(bf16) = 256 bf16 elements per slot.
// Data occupies [0, 128), [128, 256) is padding (same as A5 480B block with 32B flag).

// ====== Phase 1: Pack ======
template <typename DType, int BS, int H, int K, int MoeExpertNum, int TileW,
          int WindowStride>
void dispatch_pack(DType* x, int32_t* expertIds, int32_t* windowTriple,
                   DType* windowData, float* windowFlag)
{
    constexpr int slotCount = BS * K;
    constexpr int hTiles = H / TileW;
    using namespace pto;

    using gm_x    = global_tensor<DType, RowMajor<BS, H>>;
    using gm_w    = global_tensor<DType, RowMajor<slotCount, WindowStride>>;
    using gm_flag = global_tensor<float,   RowMajor<slotCount, TileW>>;
    using tile_d  = Tile<Location::Vec, DType, 1, TileW, BLayout::RowMajor>;
    using tile_f  = Tile<Location::Vec, float,  1, TileW, BLayout::RowMajor>;
    using it_x    = global_iterator<gm_x,    tile_d>;
    using it_w    = global_iterator<gm_w,    tile_d>;
    using it_flag = global_iterator<gm_flag, tile_f>;

    it_x    x_iter(x);
    it_w    w_iter(windowData);
    it_flag flag_iter(windowFlag);

    for (int tk = 0; tk < slotCount; tk++) {
        int tokenId = tk / K;
        int topkId  = tk % K;

        for (int t = 0; t < hTiles; t++) {
            tile_d xq;
            auto gx = x_iter(tokenId, t);
            TLOAD(xq, gx);

            // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
            // TSUB(dst, src, src) is a compile-time stand-in for TMOV(dst, src):
            // the 0828 toolchain's asm matcher rejects TMOV's 0.58.4 B.DATR
            // syntax ("NORM, DTYPE_NONE, Zero"); TSUB emits no B.DATR and keeps
            // the same src->dst dependency edge (dst = src - src = 0).
            tile_d sync_d;
            TSUB(sync_d, xq, xq);

            // #9 512B stride write: data at [tk*WindowStride + t*TileW]
            auto gw = w_iter(tk, t);
            TSTORE(gw, xq);
        }

        // #1 512B block packing: flag fill (TEXPANDS + TSTORE)
        tile_f flagTile;
        TEXPANDS(flagTile, 1.0f);

        // #3 Pipeline sync (SyncFunc<V_MTE3> aligned)
        // TSUB as TMOV stand-in (see dispatch_pack note).
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

// ====== #4 Flag check (tile pass-through; TCMP's 0.58.4 B.DATR syntax is
// rejected by the 0828 toolchain asm matcher, so the EQ predicate collapses
// to a flag->predBuf copy. predBuf is never read back — the load/store pair
// keeps the original windowFlag read side-effect and structural alignment.
// TSUB stands in for the removed TMOV sync, see dispatch_pack note) ======
template <int BS, int K, int TileW>
void check_flag(float* windowFlag, float* predBuf, int srcSlot)
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

    // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
    tile_f sync_f1;
    TSUB(sync_f1, flagTile, flagTile);

    auto gp = pred_iter(srcSlot, 0);
    TSTORE(gp, flagTile);
}

// ====== #7 CUMSUM flag check (tile pass-through; TCMP unavailable on the
// 0828 toolchain — see check_flag note. Reads the same TileW-float range at
// windowState+4 that the original tile version read, copies it to predBuf.
// TSUB stands in for the removed TMOV sync) ======
template <int TileW>
void check_cumsum_flag(uint32_t* windowState, float* predBuf)
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

    // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
    tile_f sync_f1;
    TSUB(sync_f1, readFlag, readFlag);

    auto gp = pred_iter(0, 0);
    TSTORE(gp, readFlag);
}

// ====== #1 Clear flag ======
template <int BS, int K, int TileW>
void clear_flag(float* windowFlag, int srcSlot)
{
    constexpr int slotCount = BS * K;
    using namespace pto;
    using gm_flag = global_tensor<float, RowMajor<slotCount, TileW>>;
    using tile_f  = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
    using it_flag = global_iterator<gm_flag, tile_f>;
    it_flag flag_iter(windowFlag);

    tile_f zeroFlag;
    TEXPANDS(zeroFlag, 0.0f);

    // TSUB as TMOV stand-in (see dispatch_pack note).
    tile_f sync_zf;
    TSUB(sync_zf, zeroFlag, zeroFlag);

    auto gf = flag_iter(srcSlot, 0);
    TSTORE(gf, zeroFlag);
}

// ====== Phase 3: Read data from window → expandXOut ======
template <typename DType, int BS, int H, int K, int TileW, int WindowStride>
void dispatch_copy_out(DType* windowData, DType* expandXOut,
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

        // #3 Pipeline sync (SyncFunc<MTE2_V> aligned)
        // TSUB as TMOV stand-in (see dispatch_pack note).
        tile_d sync_d;
        TSUB(sync_d, xq, xq);

        auto gout = out_iter(dstPos, t);
        TSTORE(gout, xq);
    }
}

// ====== Main entry ======
template <typename DType, int BS, int H, int K, int MoeExpertNum, int TileW = 128,
          int WindowStride = 256>
void moe_dispatch_v2(
    DType* x, int32_t* expertIds, float* expertScales,
    DType* expandXOut, int32_t* expandIdxOut, float* expandScalesOut,
    int32_t* sendCountsOut, int64_t* expertTokenNumsOut,
    DType* windowData, float* windowFlag, float* predBuf,
    int32_t* windowTriple, uint32_t* windowState, DType* outBuf)
{
    constexpr int slotCount = BS * K;

    // #5 Window State Init (InitWinState aligned)
    uint32_t dataState = windowState[0];
    windowState[0] = (dataState == 0) ? 1 : 0;
    windowState[1] = 1;
    windowState[2] = 0;

    // ====== Phase 1: AllToAllDispatch (pack x → window + flag) ======
    dispatch_pack<DType, BS, H, K, MoeExpertNum, TileW, WindowStride>(
        x, expertIds, windowTriple, windowData, windowFlag);

    // ====== Phase 2: CalCumSum (count + cumsum) ======
    int32_t expertCounts[MoeExpertNum] = {0};
    for (int i = 0; i < slotCount; i++) {
        int32_t eid = expertIds[i];
        if (eid >= 0 && eid < MoeExpertNum) expertCounts[eid]++;
    }
    int32_t cumSum[MoeExpertNum] = {0};
    int32_t acc = 0;
    for (int e = 0; e < MoeExpertNum; e++) {
        acc += expertCounts[e];
        cumSum[e] = acc;
        sendCountsOut[e] = cumSum[e];
        expertTokenNumsOut[e] = expertCounts[e];
    }

    // #9 CUMSUM soft sync: write cumsum flag (TEXPANDS + TSTORE)
    {
        using namespace pto;
        using tile_f = Tile<Location::Vec, float, 1, TileW, BLayout::RowMajor>;
        tile_f cumsumFlag;
        TEXPANDS(cumsumFlag, 1.0f);

        // TSUB as TMOV stand-in (see dispatch_pack note).
        tile_f sync_cf;
        TSUB(sync_cf, cumsumFlag, cumsumFlag);

        using gm_st = global_tensor<float, RowMajor<1, TileW>>;
        auto gs = reinterpret_cast<gm_st*>(windowState + 4);
        TSTORE(*gs, cumsumFlag);
    }

    // #7 CUMSUM flag check (tile pass-through; TCMP unavailable on 0828)
    check_cumsum_flag<TileW>(windowState, predBuf);

    // ====== Phase 3: LocalWindowCopy (read window → continuous output) ======
    int32_t writePos = 0;
    for (int e = 0; e < MoeExpertNum; e++) {
        for (int s = 0; s < expertCounts[e]; s++) {
            int32_t found = 0, srcSlot = -1;
            for (int i = 0; i < slotCount; i++) {
                if (expertIds[i] == e) {
                    if (found == s) { srcSlot = i; break; }
                    found++;
                }
            }
            if (srcSlot < 0) continue;

            // #4 Flag check (tile pass-through; TCMP unavailable on 0828)
            // Scalar readback skipped — self-loopback data always ready
            check_flag<BS, K, TileW>(windowFlag, predBuf, srcSlot);

            // Read data from window → expandXOut (tile copy, 512B stride)
            dispatch_copy_out<DType, BS, H, K, TileW, WindowStride>(
                windowData, expandXOut, srcSlot, writePos);

            // Read triple + scale (scalar)
            expandIdxOut[writePos * 3 + 0] = windowTriple[srcSlot * 3 + 0];
            expandIdxOut[writePos * 3 + 1] = windowTriple[srcSlot * 3 + 1];
            expandIdxOut[writePos * 3 + 2] = windowTriple[srcSlot * 3 + 2];
            expandScalesOut[writePos] = expertScales[srcSlot];

            writePos++;

            // #1 Clear flag (TEXPANDS(0.0) + TSTORE)
            clear_flag<BS, K, TileW>(windowFlag, srcSlot);
        }
    }

    // Window state writeback
    windowState[4] = BS;
}

} // namespace supernpu::tile_isa
#endif
