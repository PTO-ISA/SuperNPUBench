#include <common/pto_tileop.hpp>

#include "template_asm.h"
#include <cstdint>
#include <cstdio>


#define DUMP_TILE(label, TileVar, DumpBuf, Rows, Cols) \
    do { \
        GlobalTensor<typename decltype(TileVar)::DType, \
                     Shape<1,1,1,Rows,Cols>, \
                     Stride<1,1,1,Cols,1>> _g(DumpBuf); \
        TSTORE(_g, TileVar); \
        printf("[DUMP] %s (shape=%dx%d):\n", label, Rows, Cols); \
        for (int ri = 0; ri < Rows; ri++) { \
            printf("  row%2d: ", ri); \
            for (int ci = 0; ci < Cols; ci++) \
                printf("%12lld ", (long long)DumpBuf[ri * Cols + ci]); \
            printf("\n"); \
        } \
        fflush(stdout); \
    } while (0)

// ==============================================
// 维度规则：从后面对齐，前面自动补 1，维度=1 可广播
// ==============================================
template<typename dtype, typename tile_shape, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM>
void __vec__ gen_offset(
    typename tile_shape::TileDType __out__ out,
    #if MAX_DIMs >= 1
    size_t in0, size_t out0,
    #endif
    #if MAX_DIMs >= 2
    size_t in1, size_t out1,
    #endif
    #if MAX_DIMs >= 3
    size_t in2, size_t out2,
    #endif
    #if MAX_DIMs >= 4
    size_t in3, size_t out3,
    #endif
    #if MAX_DIMs >= 5
    size_t in4, size_t out4,
    #endif
    #if MAX_DIMs >= 6
    size_t in5, size_t out5,
    #endif
    #if MAX_DIMs >= 7
    size_t in6, size_t out6,
    #endif
    #if MAX_DIMs >= 8
    size_t in7, size_t out7,
    #endif
    const size_t base,
    const size_t total_elements
) {
    size_t index = blkv_get_index_x();
    // if (index >= total_elements) return;
    size_t idx = index + base;

    size_t out_coord[MAX_DIM] = {0};
    size_t tmp = idx;

    // ====== 输出坐标计算 ======
    for (int d = OUT_DIM - 1; d >= 0; d--) {
        if      (d == 0) { out_coord[d] = tmp % out0; tmp /= out0; }
        #if MAX_DIMs >=2
        else if (d == 1) { out_coord[d] = tmp % out1; tmp /= out1; }
        #endif
        #if MAX_DIMs >=3
        else if (d == 2) { out_coord[d] = tmp % out2; tmp /= out2; }
        #endif
        #if MAX_DIMs >=4
        else if (d == 3) { out_coord[d] = tmp % out3; tmp /= out3; }
        #endif
        #if MAX_DIMs >=5
        else if (d == 4) { out_coord[d] = tmp % out4; tmp /= out4; }
        #endif
        #if MAX_DIMs >=6
        else if (d == 5) { out_coord[d] = tmp % out5; tmp /= out5; }
        #endif
        #if MAX_DIMs >=7
        else if (d == 6) { out_coord[d] = tmp % out6; tmp /= out6; }
        #endif
        #if MAX_DIMs >=8
        else if (d == 7) { out_coord[d] = tmp % out7; tmp /= out7; }
        #endif
    }

    // ====== 输入坐标计算 ======
    size_t in_coord[MAX_DIM] = {0};
    for (int i = OUT_DIM - 1; i >= 0; i--) {
        int o = OUT_DIM - IN_DIM + i;
        size_t in_dim;

        if      (i == 0) in_dim = in0;
        #if MAX_DIMs >=2
        else if (i == 1) in_dim = in1;
        #endif
        #if MAX_DIMs >=3
        else if (i == 2) in_dim = in2;
        #endif
        #if MAX_DIMs >=4
        else if (i == 3) in_dim = in3;
        #endif
        #if MAX_DIMs >=5
        else if (i == 4) in_dim = in4;
        #endif
        #if MAX_DIMs >=6
        else if (i == 5) in_dim = in5;
        #endif
        #if MAX_DIMs >=7
        else if (i == 6) in_dim = in6;
        #endif
        #if MAX_DIMs >=8
        else if (i == 7) in_dim = in7;
        #endif

        if (in_dim == 1)
            in_coord[i] = 0;
        else
            in_coord[i] = out_coord[o];
    }

    // ====== 计算偏移 ======
    size_t in_offset = 0;
    size_t data_width = sizeof(dtype);
    for (int i = 0; i < IN_DIM; i++) {
        if      (i == 0) in_offset = in_offset * in0 + in_coord[0];
        #if MAX_DIMs >=2
        else if (i == 1) in_offset = in_offset * in1 + in_coord[1];
        #endif
        #if MAX_DIMs >=3
        else if (i == 2) in_offset = in_offset * in2 + in_coord[2];
        #endif
        #if MAX_DIMs >=4
        else if (i == 3) in_offset = in_offset * in3 + in_coord[3];
        #endif
        #if MAX_DIMs >=5
        else if (i == 4) in_offset = in_offset * in4 + in_coord[4];
        #endif
        #if MAX_DIMs >=6
        else if (i == 5) in_offset = in_offset * in5 + in_coord[5];
        #endif
        #if MAX_DIMs >=7
        else if (i == 6) in_offset = in_offset * in6 + in_coord[6];
        #endif
        #if MAX_DIMs >=8
        else if (i == 7) in_offset = in_offset * in7 + in_coord[7];
        #endif
    }
    in_offset *= data_width;

    blkv_get_tile_ptr(out)[index] = in_offset;
}
template<typename dtype, typename tile_shapeOffset, size_t MAX_DIM, size_t IN_DIM, size_t OUT_DIM>
void gen_offset_impl(
    tile_shapeOffset &out,
    const size_t *in_shape,
    const size_t *out_shape,
    const size_t base,
    const size_t total_elements) {
    static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                  "Only static shape supported");

    #if MAX_DIMs >= 1
    size_t in_shape0 = in_shape[0];
    size_t out_shape0 = out_shape[0];
    #endif
    #if MAX_DIMs >= 2
    size_t in_shape1 = in_shape[1];
    size_t out_shape1 = out_shape[1];
    #endif
    #if MAX_DIMs >= 3
    size_t in_shape2 = in_shape[2];
    size_t out_shape2 = out_shape[2];
    #endif
    #if MAX_DIMs >= 4
    size_t in_shape3 = in_shape[3];
    size_t out_shape3 = out_shape[3];
    #endif
    #if MAX_DIMs >= 5
    size_t in_shape4 = in_shape[4];
    size_t out_shape4 = out_shape[4];
    #endif
    #if MAX_DIMs >= 6
    size_t in_shape5 = in_shape[5];
    size_t out_shape5 = out_shape[5];
    #endif
    #if MAX_DIMs >= 7
    size_t in_shape6 = in_shape[6];
    size_t out_shape6 = out_shape[6];
    #endif
    #if MAX_DIMs >= 8
    size_t in_shape7 = in_shape[7];
    size_t out_shape7 = out_shape[7];
    #endif

    gen_offset<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM><<<tile_shapeOffset::ValidCol, 1, 1>>>(
        out.data(),
        #if MAX_DIMs >= 1
        in_shape0, out_shape0,
        #endif
        #if MAX_DIMs >= 2
        in_shape1, out_shape1,
        #endif
        #if MAX_DIMs >= 3
        in_shape2, out_shape2,
        #endif
        #if MAX_DIMs >= 4
        in_shape3, out_shape3,
        #endif
        #if MAX_DIMs >= 5
        in_shape4, out_shape4,
        #endif
        #if MAX_DIMs >= 6
        in_shape5, out_shape5,
        #endif
        #if MAX_DIMs >= 7
        in_shape6, out_shape6,
        #endif
        #if MAX_DIMs >= 8
        in_shape7, out_shape7,
        #endif
        base,
        total_elements);
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

    // tile_shapeData inTile;
    // using itIn = global_iterator<gm_shapeIn, tile_shapeData>;

    itOut gOIter(out_ptr);
    // itIn gIIter(in_ptr);
    // for test ///////////////////////////////////////
    // alignas(256) static uint32_t  g_dump[tM];
    // alignas(256) static dtype  g_dump_outTile[tM];
    // alignas(256) static dtype  g_dump_inTile[tM];
    // ///////////////////////////////////////

    size_t total_elements = tM;
    for (int i = 0; i < Mb; ++i) {
        auto gO = gOIter(0, i);
        // auto gI = gIIter(0, i);
        // printf("iter = %d\n", i);
        // printf("base = %d\n", base);
        // printf("total_elements = %d\n", total_elements);
        // printf("in_shape[0] = %d\n", in_shape[0]);
        // printf("inGm = %ld\n", inGm);

        // TLOAD(inTile, gI);
        // DUMP_TILE("inTile", inTile, g_dump_inTile, 1, tM);
        gen_offset_impl<dtype, tile_shapeOffset, MAX_DIM, IN_DIM, OUT_DIM>(offsetTile, in_shape, out_shape, base, total_elements);
        base += total_elements;

        // DUMP_TILE("offsetTile", offsetTile, g_dump, 1, tM);

        MGATHER(outTile, inGm, offsetTile);

        // DUMP_TILE("outTile", outTile, g_dump_outTile, 1, tM);
        TSTORE(gO, outTile);
    }
    if constexpr (rmd_M) {
        // printf("rmd_M = %d\n", rmd_M);
        auto gO = gOIter(0, Mb);
        total_elements = rmd_M;
        gen_offset_impl<dtype, tile_shapeOffset_rmd, MAX_DIM, IN_DIM, OUT_DIM>(offsetTile_rmd, in_shape, out_shape, base, total_elements);
        base += total_elements;
        MGATHER(outTile_rmd, inGm, offsetTile_rmd);
        TSTORE(gO, outTile_rmd);
    }

}



