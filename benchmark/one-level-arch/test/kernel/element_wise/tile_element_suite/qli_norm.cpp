#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include <cstdint>

using namespace pto;

template <uint32_t Capacity, uint32_t Valid>
void qli_norm_stage(const uint32_t *input, uint32_t *gt_count,
                    uint32_t *gt_value, uint32_t *gt_cleared,
                    uint32_t *eq_count, uint32_t *eq_value,
                    uint32_t *eq_cleared, uint32_t *sums,
                    uint32_t cutoff, uint32_t selected) {
    using Row = Tile<Location::Vec, uint32_t, 1, Capacity,
                     BLayout::RowMajor, 1, Valid>;
    using Scalar = Tile<Location::Vec, uint32_t, 32, 1,
                        BLayout::RowMajor, 1, 1>;
    using GlobalRow = global_tensor<uint32_t, RowMajor<1, Valid>>;
    Row key, count, candidate;
    GlobalRow source(const_cast<uint32_t *>(input));
    TLOAD(key, source);
    auto &keys = TPARTELEMENT(key);
    auto &counts = TPARTELEMENT(count);
    auto &values = TPARTELEMENT(candidate);
#pragma pto element for
    for (uint32_t element = 0; element < Valid; ++element) {
        const bool take = keys[element] > cutoff;
        counts[element] = take ? 1u : 0u;
        values[element] = take ? keys[element] : 0u;
    }
    GlobalRow counts_gt(gt_count), values_gt(gt_value), cleared_gt(gt_cleared);
    TSTORE(counts_gt, count);
    TSTORE(values_gt, candidate);
#pragma pto element for
    for (uint32_t element = 0; element < Valid; ++element) {
        if (element == selected)
            values[element] = 0;
    }
    TSTORE(cleared_gt, candidate);
    Scalar total;
    TROWSUM(total, count);
    global_tensor<uint32_t, RowMajor<1, 1>> sum_gt(sums);
    TSTORE(sum_gt, total);

#pragma pto element for
    for (uint32_t element = 0; element < Valid; ++element) {
        const bool take = keys[element] == cutoff;
        counts[element] = take ? 1u : 0u;
        values[element] = take ? keys[element] : 0u;
    }
    GlobalRow counts_eq(eq_count), values_eq(eq_value), cleared_eq(eq_cleared);
    TSTORE(counts_eq, count);
    TSTORE(values_eq, candidate);
#pragma pto element for
    for (uint32_t element = 0; element < Valid; ++element) {
        if (element == selected)
            values[element] = 0;
    }
    TSTORE(cleared_eq, candidate);
    TROWSUM(total, count);
    global_tensor<uint32_t, RowMajor<1, 1>> sum_eq(sums + 1);
    TSTORE(sum_eq, total);
}

#include "qli_norm_data.inc"
int main() {
    BENCHSTART;
#include "qli_norm_calls.inc"
    BENCHEND;
    return 0;
}
