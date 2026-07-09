#ifndef CUMSUMROW_KERNEL_HPP
#define CUMSUMROW_KERNEL_HPP

#pragma once
#include <common/pto_tileop.hpp>
#include <pto/pto-inst.hpp>
#include <cstdint>
#include <cstdio>

using namespace pto;

template<typename dtype, const int gIM, const int gIN, const int tM, const int tN>
void cumsum_row_rand(
    dtype *in_ptr,
    dtype *out_ptr
)
{
    using ScalarTile = Tile<Location::Vec, dtype, 1, 1, BLayout::RowMajor>;
    using gm_shape = global_tensor<dtype, RowMajor<gIM, gIN>>;
    using itType = global_iterator<gm_shape, ScalarTile>;

    itType gIIter(in_ptr);
    itType gOIter(out_ptr);

    // 对每一行独立计算累积和
    for (int i = 0; i < gIM; ++i) {
        ScalarTile running_sum;
        TEXPANDS(running_sum, static_cast<dtype>(0));

        for (int j = 0; j < gIN; ++j) {
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
