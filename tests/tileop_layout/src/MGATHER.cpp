#include <common/pto_tileop.hpp>
#include "template_asm.h"

#ifndef GROW
#define GROW 128
#endif

#ifndef GCOL
#define GCOL 128
#endif

#ifndef TROW
#define TROW 32
#endif

#ifndef TCOL
#define TCOL 32
#endif

template<typename tile_shape>
void __vec__ range_kernel(typename tile_shape::TileDType __out__ tout, const typename tile_shape::DType start){
    size_t j = blkv_get_index_x();
    size_t i = blkv_get_index_y();

    typename tile_shape::DType idx = i * tile_shape::RowStride + j;

    blkv_get_tile_ptr(tout)[idx] = (start + idx * 2) << 2;
}

template <uint16_t grow, uint16_t gcol, uint16_t trow, uint16_t tcol>
void mgather(float *src) {
    using gm_shape = global_tensor<float, RowMajor<grow, gcol>>;
    using tile_shape_dst = Tile<Location::Vec, float, trow, tcol, BLayout::RowMajor>;
    using tile_shape_offset = Tile<Location::Vec, uint16_t, trow, tcol, BLayout::RowMajor>;

    gm_shape gsrc(src);
    tile_shape_dst tdst;
    tile_shape_offset toff;

    #pragma clang loop unroll(full)
    for(int i=0;i<1000;i++){
        range_kernel<tile_shape_offset><<<tile_shape_offset::ValidCol, tile_shape_offset::ValidRow, 1>>>(toff.data(), i);
        MGATHER(tdst, gsrc, toff);
    }
}

int main() {
    const uint16_t grow = GROW;
    const uint16_t gcol = GCOL;
    const uint16_t trow = TROW;
    const uint16_t tcol = TCOL;

    float src[GROW*GCOL];

    mgather<grow, gcol, trow, tcol>(src);

    return 0;
}