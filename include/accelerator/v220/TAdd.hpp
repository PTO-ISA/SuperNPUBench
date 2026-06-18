#ifndef TADD_HPP
#define TADD_HPP

#include "type.hpp"

namespace PTO {
    template <typename tile_shape>
    [aicore] void TAdd(Tile::AddrType dstPtr, Tile::AddrType src0Ptr, Tile::AddrType src1Ptr) {
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(tile_tensor::DType);
        unsigned numRepeatPerLine = src0::Cols / elementsPerRepeat;
        unsigned numRemainPerLine = src0::Cols % elementsPerRepeat;
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(tile_tensor::DType);
        constexpr unsigned stride = src0::RowStride;

        if (numRepeatPerLine > 0) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < src0::Rows; i++) {
                if (numLoop) {
                    for (int j = 0; j < numLoop; j++) {
                        vadd(dstPtr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             src0Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX,
                             src1Ptr + i * stride + j * elementsPerRepeat * REPEAT_MAX, REPEAT_MAX, 1, 1, 1, 8, 8, 8);
                    }
                }
                if (remainAfterLoop) {
                    vadd(dstPtr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                         src0Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX,
                         src1Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX, remainAfterLoop, 1, 1, 1, 8,
                         8, 8);
                }
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        src0Ptr += numRepeatPerLine * elementsPerRepeat;
        src1Ptr += numRepeatPerLine * elementsPerRepeat;

        if (numRemainPerLine) {
            unsigned numLoop = src0::Rows / REPEAT_MAX;
            unsigned remainAfterLoop = src0::Rows % REPEAT_MAX;
            constexpr bool strideOverFlag = (src0::RowStride / blockSizeElem > REPEAT_STRIDE_MAX);
            SetContinuousMask(numRemainPerLine);
            if (numLoop) {
                for (int i = 0; i < numLoop; i++) {
                    if constexpr (strideOverFlag) {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                            vadd(dstPtr + i * REPEAT_MAX * src0::RowStride + j * src0::RowStride,
                                 src0Ptr + i * REPEAT_MAX * src0::RowStride + j * src0::RowStride,
                                 src1Ptr + i * REPEAT_MAX * src0::RowStride + j * src0::RowStride, 1, 1, 1, 1, 1, 1, 1);
                        }
                    } else {
                        vadd(dstPtr + i * REPEAT_MAX * src0::RowStride, src0Ptr + i * REPEAT_MAX * src0::RowStride,
                             src1Ptr + i * REPEAT_MAX * src0::RowStride, REPEAT_MAX, 1, 1, 1,
                             src0::RowStride / blockSizeElem, src0::RowStride / blockSizeElem,
                             src0::RowStride / blockSizeElem);
                    }
                }
            }
            if (remainAfterLoop) {
                if constexpr (strideOverFlag) {
                    for (unsigned j = 0; j < remainAfterLoop; j++) {
                        vadd(dstPtr + numLoop * REPEAT_MAX * src0::RowStride + j * src0::RowStride,
                             src0Ptr + numLoop * REPEAT_MAX * src0::RowStride + j * src0::RowStride,
                             src1Ptr + numLoop * REPEAT_MAX * src0::RowStride + j * src0::RowStride,
                             1, 1, 1, 1, 1, 1, 1);
                    }
                } else {
                    vadd(dstPtr + numLoop * REPEAT_MAX * src0::RowStride,
                         src0Ptr + numLoop * REPEAT_MAX * src0::RowStride,
                         src1Ptr + numLoop * REPEAT_MAX * src0::RowStride, remainAfterLoop, 1, 1, 1,
                         src0::RowStride / blockSizeElem, src0::RowStride / blockSizeElem,
                         src0::RowStride / blockSizeElem);
                }
            }
            set_vector_mask(-1, -1);
        }
    }

    template <is_tile_data_v tile_shape>
    [aicore] void TADD(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
        TAdd<tile_shape>(dst.getAddr(), src0.getAddr(), src1.getAddr());
    }
}
#endif