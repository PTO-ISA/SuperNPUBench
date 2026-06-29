#include <common/pto_tileop.hpp>

#include "template_asm.h"
#include <cstdint>
#include <cstdio>

template<typename dtype, typename tile_shape>
void __vec__ gen_offset_N_1_to_N_129(
    typename tile_shape::TileDType __out__ out,
    const size_t base
) {
    size_t index = blkv_get_index_x();
    size_t idx = index + base;
    size_t in_offset = (idx / 129) * sizeof(dtype);
    blkv_get_tile_ptr(out)[index] = in_offset;
}

template<typename dtype, typename tile_shapeOffset, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM>
void gen_offset_impl(
    tile_shapeOffset &out,
    const size_t *in_shape,
    const size_t *out_shape,
    const size_t base,
    const size_t total_elements) {
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1, "Only static shape supported");
    if constexpr (MAX_DIM == 2 && IN_DIM == 2 && OUT_DIM == 2) {
        if (in_shape[0] == out_shape[0] && in_shape[1] == 1 && out_shape[1] == 129) {
            gen_offset_N_1_to_N_129<dtype, tile_shapeOffset><<<tile_shapeOffset::ValidCol, 1, 1>>>(
                out.data(), base);
            return;
        }
    }
}

template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM, size_t gIM, size_t gOM, size_t tM>
void broadcast(
    dtype *in_ptr,
    dtype *out_ptr,
    const size_t *in_shape,
    const size_t *out_shape
    ) {
    const size_t Mb = gOM / tM;
    const size_t rmd_M = gOM % tM; 

    using gm_shapeIn = global_tensor<dtype, RowMajor<1, gIM>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gOM>>;
    using tile_shapeData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>; 
    using tile_shapeOffset = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor>;
    using tile_shapeData_rmd = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeOffset_rmd = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor, 1, rmd_M>;

    gm_shapeIn inGm(in_ptr);
    // gm_shapeOut outGm(out_ptr);
    tile_shapeData outTile;
    tile_shapeOffset offsetTile;
    tile_shapeData_rmd outTile_rmd;
    tile_shapeOffset_rmd offsetTile_rmd;
    size_t base = 0;
    size_t all_num = gOM; // 总元素数量
    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;
    itOut gOIter(out_ptr);

    size_t total_elements = tM;
    for (int i = 0; i < Mb; ++i) {
        auto gO = gOIter(0, i);
        gen_offset_impl<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM>(offsetTile, in_shape, out_shape, base, total_elements);
        base += total_elements;
        MGATHER(outTile, inGm, offsetTile);
        TCOPYOUT(gO, outTile);
    }
    if constexpr (rmd_M) {
        auto gO = gOIter(0, Mb);
        total_elements = rmd_M;
        gen_offset_impl<dtype, tile_shapeOffset_rmd, MAX_DIM, IN_DIM, OUT_DIM>(offsetTile_rmd, in_shape, out_shape, base, total_elements);
        base += total_elements;
        MGATHER(outTile_rmd, inGm, offsetTile_rmd);
        TCOPYOUT(gO, outTile_rmd);
    }
}