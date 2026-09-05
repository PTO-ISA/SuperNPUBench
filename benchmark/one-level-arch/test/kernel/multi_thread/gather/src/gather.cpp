#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "multi_thread/gather/gather.hpp"

namespace {
alignas(4096) float table[128 * 64];
alignas(4096) std::int32_t indexes[128];
alignas(4096) float output[128 * 64];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/table.bin", reinterpret_cast<uint8_t *>(table),
                       sizeof(table));
        readBinaryFile(CHK_DIR "/indexes.bin",
                       reinterpret_cast<uint8_t *>(indexes), sizeof(indexes));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    supernpu::multi_thread::gather<
        float, std::int32_t, 128, 128, 64, 32, 64>(
            table, indexes, output);
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
