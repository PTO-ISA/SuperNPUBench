#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

template <int Rows, int Cols>
void vec_multithread(float* out_ptr, float* a_ptr, float* b_ptr) {
    static_assert(Rows * Cols < 8 * 1024,
                  "each PE vector tile must be smaller than 8K elements");

    using tileT = Tile<Location::Vec, float, Rows, Cols, BLayout::RowMajor>;
    using gmIn = global_tensor<float, RowMajor<Rows, Cols>>;
    using gmOut = global_tensor<float, RowMajor<Rows, Cols>>;
    using itIn = global_iterator<gmIn, tileT>;
    using itOut = global_iterator<gmOut, tileT>;
    uint32_t tid = get_thread_idx();
    uint32_t gm_offset = tid * Rows*Cols;

    tileT tA;
    tileT tB;
    tileT tC;

    itIn a_iter(a_ptr+gm_offset);
    itIn b_iter(b_ptr+gm_offset);
    auto src_a = a_iter(0, 0);
    auto src_b = b_iter(0, 0);
    TLOAD(tA, src_a);
    TLOAD(tB, src_b);
    TADD(tC, tA, tB);

    itOut out_iter(out_ptr+gm_offset);
    auto dst = out_iter(0, 0);
    TSTORE(dst, tC);
}
