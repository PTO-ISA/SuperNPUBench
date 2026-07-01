#ifndef CUMSUMCOL_KERNEL_HPP
#define CUMSUMCOL_KERNEL_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void cumsum_col_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    using ScalarTile = Tile<Location::Vec, dtype, 1, 1, BLayout::RowMajor>;
    using gm_shape = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using itType = global_iterator<gm_shape, ScalarTile>;

    itType gIIter(in_ptr);
    itType gOIter(out_ptr);

    // 对每一列独立计算累积和
    for (int j = 0; j < gIN; ++j) {
        ScalarTile running_sum;
        TEXPANDS(running_sum, static_cast<dtype>(0));

        for (int i = 0; i < gIM; ++i) {
            auto gI = gIIter(i, j);
            ScalarTile input_elem;
            TLOAD(input_elem, gI);

            TADD(running_sum, running_sum, input_elem);

            auto gO = gOIter(i, j);
            TSTORE(gO, running_sum);
        }
    }
}

#endif
