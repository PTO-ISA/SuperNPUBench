#pragma once

#include <common/pto_tileop.hpp>
#include <cstdint>

using namespace pto;

template <int Rows, int Cols>
void tadd_tile(float* out_ptr, float* a_ptr, float* b_ptr) {
    static_assert(Rows * Cols < 8 * 1024,
                  "each PE vector tile must be smaller than 8K elements");

    using tileT = Tile<Location::Vec, float, Rows, Cols, BLayout::RowMajor>;
    using gmIn = global_tensor<float, RowMajor<Rows, Cols>>;
    using gmOut = global_tensor<float, RowMajor<Rows, Cols>>;
    using itIn = global_iterator<gmIn, tileT>;
    using itOut = global_iterator<gmOut, tileT>;

    tileT tA;
    tileT tB;
    tileT tC;

    itIn a_iter(a_ptr);
    itIn b_iter(b_ptr);
    auto src_a = a_iter(0, 0);
    auto src_b = b_iter(0, 0);
    TLOAD(tA, src_a);
    TLOAD(tB, src_b);
    TADD(tC, tA, tB);

    itOut out_iter(out_ptr);
    auto dst = out_iter(0, 0);
    TSTORE(dst, tC);
}

template <int Rows, int Cols>
void vec_multithread(float* out, float* a, float* b) {
    const auto offset = get_thread_idx() * Rows * Cols;
    tadd_tile<Rows, Cols>(out + offset, a + offset, b + offset);
}
