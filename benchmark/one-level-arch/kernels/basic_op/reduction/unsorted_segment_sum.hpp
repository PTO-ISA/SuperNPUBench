#pragma once

#include <common/pto_tileop.hpp>

#include <cstdint>
#include <type_traits>

namespace supernpu::multi_thread {

// A deterministic four-PE baseline. Each PE is the sole writer of its output
// segments, so repeated IDs need neither GM atomics nor a cross-PE barrier.
// IDs outside [0, Segments) are ignored. Empty segments are written as zero.
template <typename DType, int Rows, int Features, int Segments>
void unsorted_segment_sum(DType *data, const std::int32_t *segment_ids,
                          DType *output) {
    static_assert(std::is_same_v<DType, float>, "FP32 baseline only");
    static_assert(Rows > 0 && Segments > 0, "nonempty shapes required");
    static_assert(Features > 0 && Features % 32 == 0,
                  "feature count must be a positive multiple of 32");

    using Input = pto::global_tensor<DType, pto::RowMajor<Rows, Features>>;
    using Output = pto::global_tensor<DType, pto::RowMajor<Segments, Features>>;
    using FeatureTile = pto::Tile<pto::Location::Vec, DType, 1, 32,
                                  pto::BLayout::RowMajor>;
    using InputIterator = global_iterator<Input, FeatureTile>;
    using OutputIterator = global_iterator<Output, FeatureTile>;

    InputIterator input_it(data);
    OutputIterator output_it(output);
    const int tid = static_cast<int>(get_thread_idx());

    for (int segment = tid; segment < Segments; segment += 4) {
        for (int feature_tile = 0; feature_tile < Features / 32;
             ++feature_tile) {
            FeatureTile sum;
            TEXPANDS(sum, 0.0f);

            for (int row = 0; row < Rows; ++row) {
                if (segment_ids[row] != segment) continue;
                FeatureTile value;
                auto src = input_it(row, feature_tile);
                TLOAD(value, src);
                TADD(sum, sum, value);
            }

            auto dst = output_it(segment, feature_tile);
            TSTORE(dst, sum);
        }
    }
}

}  // namespace supernpu::multi_thread
