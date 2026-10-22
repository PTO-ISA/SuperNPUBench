#ifndef GROUP_TOKEN_VEC_MT_DYN_HISTOGRAM_HPP
#define GROUP_TOKEN_VEC_MT_DYN_HISTOGRAM_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

namespace {
constexpr uint32_t kPartElements = 32;

__attribute__((noinline)) void addElementHistogramPart(
    const uint32_t *input, uint32_t valid, uint32_t *histogram,
    uint32_t histogramSize)
{
#pragma pto element for
    for (uint32_t element = 0; element < kPartElements; ++element) {
        if (element < valid) {
            const uint32_t index = input[element];
            if (index < histogramSize) {
                __atomic_fetch_add(histogram + index, 1u,
                                   __ATOMIC_RELAXED);
            }
        }
    }
}
} // namespace

__attribute__((noinline)) static void gt_dyn_clear_element_histogram(
    uint32_t *histogram, uint32_t size)
{
    for (uint32_t begin = 0; begin < size; begin += kPartElements) {
        const uint32_t valid =
            size - begin < kPartElements ? size - begin : kPartElements;
        uint32_t *part = histogram + begin;
#pragma pto element for
        for (uint32_t element = 0; element < kPartElements; ++element) {
            if (element < valid) {
                part[element] = 0u;
            }
        }
    }
}

__attribute__((noinline)) static void gt_dyn_add_element_histogram_range(
    const uint32_t *input, uint32_t count, uint32_t *histogram,
    uint32_t histogramSize)
{
    for (uint32_t begin = 0; begin < count; begin += kPartElements) {
        const uint32_t valid =
            count - begin < kPartElements ? count - begin : kPartElements;
        addElementHistogramPart(input + begin, valid, histogram,
                                histogramSize);
    }
}

#endif
