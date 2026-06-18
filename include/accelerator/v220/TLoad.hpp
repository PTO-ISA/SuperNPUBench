#ifndef TLOAD_HPP
#define TLOAD_HPP

#include "../../common/pto_tile.hpp"
#include "type.hpp"

namespace PTO {
    template <is_tile_data tile_shape>
    [aicore] void TLOAD(tile_shape &dst, __gm__ tile_shape::DType *src, unsigned gms) {
        uint16_t nBurst = dst::kRows;
        uint16_t lenBurst = dst::kCols * sizeof(tile_shape::DType);
        uint32_t gmGap = (GMS - dst::kCols) * sizeof(tile_shape::DType);
        constexpr uint32_t blockSize = BLOCK_BYTE_SIZE / sizeof(T);
        constexpr uint32_t ubGap = (dst::kRowStride - dst::kCols) / blockSize;

        if constexpr (sizeof(tile_shape::DType) == 1) {
            copy_gm_to_ubuf_align_b8(
                dst.getAddr(), src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, gmGap, ubGap);
        } else if (sizeof(T) == 2) {
            copy_gm_to_ubuf_align_b16(
            dst.getAddr(), src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, gmGap, ubGap);
        } else {
            copy_gm_to_ubuf_align_b32(
            dst.getAddr(), src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/, gmGap, ubGap);
        }
    }
}
#endif //TLOAD_HPP