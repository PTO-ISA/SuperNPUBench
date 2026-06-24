#include <common/pto_tileop.hpp>
#include <benchmark_support/npu/npu_transpose.h>
#include "benchmark.h"
#include "fileop.h"

#include "template_asm.h"

#define D4 1
#define D3 1
#define D2 1
#define D1 1
#define D0 256

#define STRIDE_4 (D3 * D2 * D1 * D0)
#define STRIDE_3 (D2 * D1 * D0)
#define SRC_STRIDE_2 (D1 * D0)
#define DST_STRIDE_2 (D2 * D0)
#define STRIDE_1 D0

#define tD2 D2
#define tD1 D1
#define tD0 D0
#define tD0_ALIGN 256

#define ALIGN_MASK (0xfffffffffffff000ull)
#define ALIGN (4 * 1024)

#ifndef __vbuf__
#define __vbuf__
#endif

int main()
{
    float inp[D4 * D3 * D2 * D1 * D0 + 2 * ALIGN];
    float outp[D4 * D3 * D2 * D1 * D0 + 2 * ALIGN];

    float* in = (float *)(((uint64_t)inp & ALIGN_MASK) + ALIGN);
    float* out = (float *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    #define SRCIN_PATH CHK_DIR "/srcin.bin"
    bool suaccelss = readBinaryFile(SRCIN_PATH, (uint8_t*) in, D4 * D3 * D2 * D1 * D0 * sizeof(float));
    assert(suaccelss);
#endif

    BENCHSTART;

    using TableShape = Shape<1, 1, 1, 1, tD2 * tD1 * tD0_ALIGN>;
    using TableStride = Stride<1, 1, 1, tD2 * tD1 * tD0_ALIGN, 1>;
    using TableGT = GlobalTensor<float, TableShape, TableStride>;

    using TileU16 = Tile<Location::Vec, uint16_t, 1, tD2 * tD1 * tD0_ALIGN, BLayout::RowMajor>;
    TileU16 loadIdxTile16;  // MGATHER FIX: uint16_t offset tile for MGATHER
    TileU16 storeIdxTile16;  // MSCATTER FIX: uint16_t offset tile for MSCATTER
    transpose_src_idx_2d_102<TileU16, tD2, tD1, tD0><<<tD2, tD1, tD0>>>(loadIdxTile16.data());
    transpose_dst_idx_2d_102<TileU16, tD2, tD1, tD0><<<tD2, tD1, tD0>>>(storeIdxTile16.data());

    using TileFP32 = Tile<Location::Vec, float, 1, tD2 * tD1 * tD0_ALIGN, BLayout::RowMajor, 1, tD2 * tD1 * tD0>;
    TileFP32 elemTile;

    for (int d4 = 0; d4 < D4; d4++) {
        for (int d3 = 0; d3 < D3; d3++) {
            // Base address for this (d4,d3) tile
            size_t src_slice_offset = d4 * STRIDE_4 + d3 * STRIDE_3;
            size_t dst_slice_offset = d4 * STRIDE_4 + d3 * STRIDE_3;

            auto gSrc = in + src_slice_offset;
            auto gDst = out + dst_slice_offset;

            TableGT srcGlobal(gSrc);
            TableGT dstGlobal(gDst);

            MGATHER(elemTile, srcGlobal, loadIdxTile16);
            // TLOAD(elemTile, srcGlobal);

            MSCATTER(dstGlobal, elemTile, storeIdxTile16);
        }
    }

    BENCHEND;

#ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*) out, D4 * D3 * D2 * D1 * D0* sizeof(float));
#endif
}
