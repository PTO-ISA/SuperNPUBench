// ============================================================================
// Broadcast (N,1) -> (N,C) — PTO 一层编程模型
//
// 原始 broadcast_vec_07.hpp 策略:
//   TCOPYIN (kTileRows,1) -> __vec__ 行广播 -> TCOPYOUT (kTileRows,C)
//   __vec__ 块: dst[x + y*RowStride] = src[y*RowStride] (col 0 -> all cols)
//
// PTO 一层策略:
//   TLOAD (kTileRows,1) -> TROWEXPAND (col 0 广播到全部 C 列) -> TSTORE (kTileRows,C)
//   TROWEXPAND 语义: dst[i,j] = src[i,0], 恰好是行广播
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │                    当前编译器不支持 / 不完整的指令汇总                   │
// ├──────────┬──────────────────┬──────────────────────────────────────────┤
// │ Pto ISA  │ 当前编译器状态   │ 说明                                     │
// │ 指令     │                  │                                          │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TLOAD    │ API 有(名不同)， │ PTO ISA 名 TLOAD；                      │
// │          │ 二层实现         │ 当前编译器名 TCOPYIN；                    │
// │          │                  │ jcore/TCopyIn.hpp 用 __vec__ 实现        │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │TROWEXPAND│ API 有(名不同)， │ PTO ISA 名 TROWEXPAND；                  │
// │          │ 二层实现         │ 当前编译器名 TEXPANDROW；                 │
// │          │                  │ jcore/TExpandRow.hpp 用 __vec__ 实现     │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TSTORE   │ API 有(名不同)， │ PTO ISA 名 TSTORE；                      │
// │          │ 二层实现         │ 当前编译器名 TCOPYOUT；                   │
// │          │                  │ jcore/TCopyOut.hpp 用 __vec__ 实现       │
// └──────────┴──────────────────┴──────────────────────────────────────────┘
//
// PTO ISA 文档签名 (Declared in include/pto/pto_instr.hpp):
//
//   TLOAD:
//     template <typename TileData, typename GlobalData, typename... WaitEvents>
//     PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src,
//                                WaitEvents &... events);
//
//   TROWEXPAND:
//     template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
//     PTO_INST RecordEvent TROWEXPAND(TileDataDst &dst, TileDataSrc &src,
//                                     WaitEvents &... events);
//     约束: dst[i,j] = src[i,0]; dst 与 src 必须为 Vec tile, ND layout;
//           A2A3: srcValidRow == dstValidRow; srcValidCol >= 1
//
//   TSTORE:
//     template <typename TileData, typename GlobalData,
//               AtomicType atomicType = AtomicType::AtomicNone,
//               typename... WaitEvents>
//     PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src,
//                                 WaitEvents &... events);
// ============================================================================

#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>
// #include <pto/pto_instr.hpp>            // [!] PTO ISA C++ Intrinsic — 当前编译器未提供

#include <cstdint>
#include <cstdio>

// =====================================================================
// Broadcast (N,1) -> (N,C) via TLOAD + TROWEXPAND + TSTORE
//
// Template params (与原 broadcast_vec_07 一致):
//   dtype     - data type (__half, float, etc.)
//   MAX_DIM   - max dimensions (unused, kept for compat)
//   IN_DIM    - input dim count (unused)
//   OUT_DIM   - output dim count (unused)
//   gIM       - total input elements = N*1 = N   (e.g. 1443)
//   gOM       - total output elements = N*C       (e.g. 1443*129)
//   kTileRows - rows per tile, must be power-of-2 (e.g. 1,2,4,..,64)
// =====================================================================

template<typename dtype, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM,
         size_t gIM, size_t gOM, size_t kTileRows>
void broadcast(dtype *in_ptr, dtype *out_ptr,
               const size_t * /*in_shape*/, const size_t * /*out_shape*/) {
    constexpr size_t kN = gIM;
    constexpr size_t kC = gOM / gIM;
    constexpr size_t tileCols = 256;

    static_assert(gOM % gIM == 0,
                  "gOM must be divisible by gIM for (N,1)->(N,C) broadcast");
    static_assert(tileCols >= kC,
                  "padded tileCols (256) must >= broadcast target columns");
    static_assert((kTileRows & (kTileRows - 1)) == 0,
                  "kTileRows must be power of 2 for 512B tile alignment");

    using tile_in  = Tile<Location::Vec, dtype, kTileRows, tileCols,
                          BLayout::RowMajor, kTileRows, 1>;
    using tile_out = Tile<Location::Vec, dtype, kTileRows, tileCols,
                          BLayout::RowMajor, kTileRows, kC>;
    using gm_in    = global_tensor<dtype, RowMajor<kTileRows, 1>>;
    using gm_out   = global_tensor<dtype, RowMajor<kTileRows, kC>>;

    constexpr size_t Nb  = kN / kTileRows;
    constexpr size_t rmd = kN % kTileRows;

    tile_in inTile;
    tile_out outTile;

    for (size_t i = 0; i < Nb; i++) {
        gm_in gsrc(in_ptr + i * kTileRows);
        gm_out gdst(out_ptr + i * kTileRows * kC);

        // TLOAD: GM -> UB, 加载 (kTileRows, 1) 输入 tile
        // [当前编译器] 名为 TCOPYIN, jcore 为 __vec__
        TLOAD(inTile, gsrc);

        // TROWEXPAND: 将每行 col 0 广播到全部 kC 列 -> (kTileRows, kC)
        // [当前编译器] 名为 TEXPANDROW, jcore 为 __vec__
        TROWEXPAND(outTile, inTile);

        // TSTORE: UB -> GM, 写回 (kTileRows, kC) 输出 tile
        // [当前编译器] 名为 TCOPYOUT, jcore 为 __vec__
        TSTORE(gdst, outTile);
    }

    using tile_in_r  = Tile<Location::Vec, dtype, kTileRows, tileCols,
                            BLayout::RowMajor, rmd, 1>;
    using tile_out_r = Tile<Location::Vec, dtype, kTileRows, tileCols,
                            BLayout::RowMajor, rmd, kC>;
    tile_in_r inTile_rmd;
    tile_out_r outTile_rmd;
    if constexpr (rmd > 0) {
        gm_in gsrc(in_ptr + Nb * kTileRows);
        gm_out gdst(out_ptr + Nb * kTileRows * kC);

        TLOAD(inTile_rmd, gsrc);
        TROWEXPAND(outTile_rmd, inTile_rmd);
        TSTORE(gdst, outTile_rmd);
    }
}
