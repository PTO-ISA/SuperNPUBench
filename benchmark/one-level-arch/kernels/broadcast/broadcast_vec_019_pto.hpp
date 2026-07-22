// ============================================================================
// Broadcast (B,1,K) -> (B,N,K) — PTO 一层编程模型
//
// 原始 broadcast_vec_019.hpp 策略:
//   TLOAD (kTileBatch,K) -> __vec__ 行复制 (K 个元素复制 N 次) -> TSTORE (kTileBatch,N*K)
//   __vec__ 块: dst[batch*RowStride + copy*K + x] = src[batch*RowStride + x]
//
// PTO 一层策略:
//   TLOAD (kTileBatch,K) -> TINSERT × N (将输入 tile 插入输出 tile 的 N 个列偏移) -> TSTORE (kTileBatch,N*K)
//   TINSERT 语义: dst[indexRow+i, indexCol+j] = src[i,j]
//   对 copy c = 0..N-1: TINSERT(outTile, inTile, 0, c*K)
//   N 次插入互不重叠, 合起来恰好填满输出 tile 的 valid 区域
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │                    当前编译器不支持 / 不完整的指令汇总                   │
// ├──────────┬──────────────────┬──────────────────────────────────────────┤
// │ Pto ISA  │ 当前编译器状态   │ 说明                                     │
// │ 指令     │                  │                                          │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TLOAD    │ API 有(名不同)， │ PTO ISA 名 TLOAD；                      │
// │          │ 二层实现         │ 当前编译器名 TLOAD；                    │
// │          │                  │ jcore/TLoad.hpp 用 __vec__ 实现        │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TINSERT  │ 完全缺失         │ pto_tileop.hpp 中无此 API；              │
// │          │                  │ 当前编译器无 TINSERT 实现                │
// │          │                  │ (仅有反向操作 TEXTRACT)                  │
// ├──────────┼──────────────────┼──────────────────────────────────────────┤
// │ TSTORE   │ API 有(名不同)， │ PTO ISA 名 TSTORE；                      │
// │          │ 二层实现         │ 当前编译器名 TSTORE；                   │
// │          │                  │ jcore/TStore.hpp 用 __vec__ 实现       │
// └──────────┴──────────────────┴──────────────────────────────────────────┘
//
// ============================================================================

#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>
// #include <pto/pto_instr.hpp>            // [!] PTO ISA C++ Intrinsic — 当前编译器未提供

#include <cstdint>
#include <cstdio>

// =====================================================================
// Broadcast (B,1,K) -> (B,N,K) via TLOAD + TINSERT×N + TSTORE
//
// Data layout (row-major):
//   Input:  [B][1][K] -> flat = B*K,       batch b data at offset b*K
//   Output: [B][N][K] -> flat = B*N*K,     batch b data at offset b*N*K
//   Broadcast along dim1: 1 -> N, dim0 & dim2 preserved.
//
// Template params (与原 broadcast_vec_019 一致):
//   dtype      - data type (__half, float, etc.)
//   MAX_DIM    - max dimensions (kept for compat)
//   IN_DIM     - input dim count (kept for compat)
//   OUT_DIM    - output dim count (kept for compat)
//   gIM        - total input elements  = B * K
//   gOM        - total output elements = B * N * K
//   kTileBatch - batches per tile, power-of-2
//   kInner     - inner dimension K (e.g. 49, need not be power-of-2)
// =====================================================================

template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM,
         size_t gIM, size_t gOM, size_t kTileBatch, size_t kInner>
void broadcast(dtype *in_ptr, dtype *out_ptr,
               const size_t * /*in_shape*/, const size_t * /*out_shape*/) {
    constexpr size_t kBCast = gOM / gIM;
    constexpr size_t kBatch = gIM / kInner;
    constexpr size_t tileCols = 512;

    static_assert(gOM % gIM == 0,
                  "gOM must be divisible by gIM for (B,1,K)->(B,N,K) broadcast");
    static_assert(gIM % kInner == 0,
                  "gIM must be divisible by kInner (B = gIM/kInner must be integer)");
    static_assert((kTileBatch & (kTileBatch - 1)) == 0,
                  "kTileBatch must be power of 2 for 512B tile alignment");
    static_assert((kBCast & (kBCast - 1)) == 0,
                  "kBCast (N) must be power of 2 for bitwise division in SIMT");
    static_assert(tileCols >= kBCast * kInner,
                  "padded tileCols (512) must >= broadcast target width (N*K)");

    using tile_in  = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                          BLayout::RowMajor, kTileBatch, kInner>;
    using tile_out = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                          BLayout::RowMajor, kTileBatch, kBCast * kInner>;
    using gm_in    = global_tensor<dtype, RowMajor<kTileBatch, kInner>>;
    using gm_out   = global_tensor<dtype, RowMajor<kTileBatch, kBCast * kInner>>;

    constexpr size_t Nb  = kBatch / kTileBatch;
    constexpr size_t rmd = kBatch % kTileBatch;

    tile_in inTile;
    tile_out outTile;

    for (size_t i = 0; i < Nb; i++) {
        gm_in gsrc(in_ptr + i * kTileBatch * kInner);
        gm_out gdst(out_ptr + i * kTileBatch * kBCast * kInner);

        // TLOAD: GM -> UB, 加载 (kTileBatch, kInner) 输入 tile
        // [当前编译器] 名为 TLOAD, jcore 为 __vec__
        TLOAD(inTile, gsrc);

        // TINSERT × kBCast: 将输入 tile 插入输出 tile 的 N 个列偏移
        // 每次 TINSERT 写入 kInner 列, N 次互不重叠, 合起来填满 N*kInner 列
        // [当前编译器] 完全缺失! pto_tileop.hpp 无 TINSERT API
        #pragma clang loop unroll(full)
        for (size_t c = 0; c < kBCast; c++) {
            TINSERT(outTile, inTile, /*indexRow=*/0, /*indexCol=*/(uint16_t)(c * kInner));
        }

        // TSTORE: UB -> GM, 写回 (kTileBatch, kBCast*kInner) 输出 tile
        // [当前编译器] 名为 TSTORE, jcore 为 __vec__
        TSTORE(gdst, outTile);
    }

    using tile_in_r  = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                            BLayout::RowMajor, rmd, kInner>;
    using tile_out_r = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                            BLayout::RowMajor, rmd, kBCast * kInner>;
    tile_in_r inTile_rmd;
    tile_out_r outTile_rmd;
    if constexpr (rmd > 0) {
        gm_in gsrc(in_ptr + Nb * kTileBatch * kInner);
        gm_out gdst(out_ptr + Nb * kTileBatch * kBCast * kInner);

        TLOAD(inTile_rmd, gsrc);

        #pragma clang loop unroll(full)
        for (size_t c = 0; c < kBCast; c++) {
            TINSERT(outTile_rmd, inTile_rmd, 0, (uint16_t)(c * kInner));
        }

        TSTORE(gdst, outTile_rmd);
    }
}
