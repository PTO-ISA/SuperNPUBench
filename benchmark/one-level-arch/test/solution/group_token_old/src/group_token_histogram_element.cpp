#include <common/pto_tileop.hpp>
#include <cstdint>

#include "benchmark.h"
#include "solution/group_token_old/group_token_old.hpp"

namespace {
constexpr uint32_t kCapacity = 129;
constexpr uint32_t kHistogramCapacity = 40;

alignas(4096) uint32_t input[kCapacity];
alignas(4096) uint32_t histogram[kHistogramCapacity];
uint32_t golden[kHistogramCapacity];

int runCase(uint32_t count, uint32_t bins, uint32_t salt)
{
    for (uint32_t i = 0; i < kCapacity; ++i) {
        if (bins == 0u || ((i + salt) % 7u) == 0u) {
            input[i] = bins + 11u + (i & 3u);
        } else {
            input[i] = ((i / 3u) + salt) % bins;
        }
    }
    for (uint32_t i = 0; i < kHistogramCapacity; ++i) {
        histogram[i] = 0xA5A50000u + i;
        golden[i] = i < bins ? 0u : histogram[i];
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (input[i] < bins) {
            ++golden[input[i]];
        }
    }

    calTokenPerExpertCnt_element(input, histogram, bins, count);
    for (uint32_t i = 0; i < kHistogramCapacity; ++i) {
        if (histogram[i] != golden[i]) {
            return 1;
        }
    }
    return 0;
}
} // namespace

int main()
{
    if (get_thread_idx() != 0u) {
        return 0;
    }
    constexpr uint32_t cases[][2] = {
        {0, 0}, {1, 1}, {31, 31}, {32, 32}, {33, 33},
        {63, 17}, {64, 32}, {65, 33}, {127, 31}, {128, 32}, {129, 33},
    };

    BENCHSTART;
    for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (runCase(cases[i][0], cases[i][1], i + 1u) != 0 ||
            runCase(cases[i][0], cases[i][1], i + 19u) != 0) {
            BENCHEND;
            return static_cast<int>(i + 1u);
        }
    }
    BENCHEND;
    return 0;
}
