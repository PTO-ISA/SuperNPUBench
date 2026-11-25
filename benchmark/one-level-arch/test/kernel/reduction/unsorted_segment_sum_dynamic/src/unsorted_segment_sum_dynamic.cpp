#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/reduction/unsorted_segment_sum.hpp"

namespace {
constexpr int kMaxRows = 64;
constexpr int kMaxFeatures = 64;
constexpr int kMaxSegments = 96;

alignas(4096) float data[kMaxRows * kMaxFeatures];
alignas(4096) std::int32_t segment_ids[kMaxRows];
alignas(4096) float output[kMaxSegments * kMaxFeatures];

// Runtime dimensions: tail feature tile (45 = 64 + partial) and two segment
// groups (70 > 64).
volatile int gRows = 37;
volatile int gFeatures = 45;
volatile int gSegments = 70;
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    const int rows = gRows;
    const int features = gFeatures;
    const int segments = gSegments;
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/data.bin", reinterpret_cast<uint8_t *>(data),
                       rows * features * sizeof(float));
        readBinaryFile(CHK_DIR "/segment_ids.bin",
                       reinterpret_cast<uint8_t *>(segment_ids),
                       rows * sizeof(std::int32_t));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif

    BENCHSTART;
    supernpu::multi_thread::unsorted_segment_sum<float>(
        data, segment_ids, output, rows, features, segments);
    BENCHEND;

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<uint8_t *>(output),
                        segments * features * sizeof(float));
    }
#endif
    return 0;
}
