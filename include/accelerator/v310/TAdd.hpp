#ifndef TADD_HPP
#define TADD_HPP

#include "../../common/pto_tile.hpp"
#include "constants.hpp"

namespace PTO {
    template <typename tile_shape>
    [aicore] void TAdd_Vec_RowMajor(typename tile_shape::TileDType &dst, const typename tile_shape::TileDType &src0,
                                    const typename tile_shape::TileDType &src1) {
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(tile_tensor::DType);
        unsigned numRepeatPerLine = src0::active_col / elementsPerRepeat;
        unsigned numRemainPerLine = src0::active_col % elementsPerRepeat;
        constexpr unsigned blockSizeElem = BLOCK_SIZE / sizeof(tile_tensor::DType);
        constexpr unsigned stride = src0::kRowStride;
        tile_shape::DType *src0Ptr = get_tile_data_ptr(src0);
        tile_shape::DType *src1Ptr = get_tile_data_ptr(src1);
        tile_shape::DType *dstPtr = get_tile_data_ptr(dst);

        if (numRepeatPerLine > 0) {
            unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
            unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
            for (int i = 0; i < src0::active_row; i++) {
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
                         src1Ptr + i * stride + numLoop * elementsPerRepeat * REPEAT_MAX, remainAfterLoop, 1, 1, 1, 8, 8, 8);
                }
            }
        }

        dstPtr += numRepeatPerLine * elementsPerRepeat;
        src0Ptr += numRepeatPerLine * elementsPerRepeat;
        src1Ptr += numRepeatPerLine * elementsPerRepeat;

        if (numRemainPerLine) {
            unsigned numLoop = src0::active_row / REPEAT_MAX;
            unsigned remainAfterLoop = src0::active_row % REPEAT_MAX;
            constexpr bool strideOverFlag = (src0::kRowStride / blockSizeElem > REPEAT_STRIDE_MAX);
            SetContinuousMask(numRemainPerLine);
            if (numLoop) {
                for (int i = 0; i < numLoop; i++) {
                    if constexpr (strideOverFlag) {
                        for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                            vadd(dstPtr + i * REPEAT_MAX * src0::kRowStride + j * src0::kRowStride,
                                 src0Ptr + i * REPEAT_MAX * src0::kRowStride + j * src0::kRowStride,
                                 src1Ptr + i * REPEAT_MAX * src0::kRowStride + j * src0::kRowStride, 1, 1, 1, 1, 1, 1, 1);
                        }
                    } else {
                        vadd(dstPtr + i * REPEAT_MAX * src0::kRowStride, src0Ptr + i * REPEAT_MAX * src0::kRowStride,
                             src1 + i * REPEAT_MAX * src0::kRowStride, REPEAT_MAX, 1, 1, 1,
                             src1Ptr::kRowStride / blockSizeElem, src0::kRowStride / blockSizeElem,
                             src0::kRowStride / blockSizeElem);
                    }
                }
            }
            if (remainAfterLoop) {
                if constexpr (strideOverFlag) {
                    for (unsigned j = 0; j < remainAfterLoop; j++) {
                        vadd(dstPtr + numLoop * REPEAT_MAX * src0::kRowStride + j * src0::kRowStride,
                             src0Ptr + numLoop * REPEAT_MAX * src0::kRowStride + j * src0::kRowStride,
                             src1Ptr + numLoop * REPEAT_MAX * src0::kRowStride + j * src0::kRowStride,
                             1, 1, 1, 1, 1, 1, 1);
                    }
                } else {
                    vadd(dstPtr + numLoop * REPEAT_MAX * src0::kRowStride, src0Ptr + numLoop * REPEAT_MAX * src0::kRowStride,
                         src1Ptr + numLoop * REPEAT_MAX * src0::kRowStride, remainAfterLoop, 1, 1, 1,
                         src0::kRowStride / blockSizeElem, src0::kRowStride / blockSizeElem,
                         src0::kRowStride / blockSizeElem);
                }
            }
            set_vector_mask(-1, -1);
        }
    }

    template <is_tile_data tile_shape>
    [aicore] void TADD(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
        TAdd_Vec_RowMajor<tile_shape>(dst.data(), src0.data(), src1.data());
    }
}
#endif