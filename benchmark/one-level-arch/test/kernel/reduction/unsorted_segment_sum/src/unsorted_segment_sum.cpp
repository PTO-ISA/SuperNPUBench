#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/reduction/unsorted_segment_sum.hpp"

namespace {
constexpr int kRows = 64;
constexpr int kFeatures = 64;
constexpr int kSegments = 12;

alignas(4096) float data[kRows * kFeatures];
alignas(4096) std::int32_t segment_ids[kRows];
alignas(4096) float output[kSegments * kFeatures];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/data.bin", reinterpret_cast<uint8_t *>(data),
                       sizeof(data));
        readBinaryFile(CHK_DIR "/segment_ids.bin",
                       reinterpret_cast<uint8_t *>(segment_ids),
                       sizeof(segment_ids));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif

    BENCHSTART;
    supernpu::multi_thread::unsorted_segment_sum<float, kRows, kFeatures,
                                                   kSegments>(
        data, segment_ids, output);
    BENCHEND;

#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<uint8_t *>(output), sizeof(output));
    }
#endif
    return 0;
}
