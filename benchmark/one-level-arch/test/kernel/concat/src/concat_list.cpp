#include <cstdint>
#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "basic_op/concat/concat.hpp"

namespace {
using Data = DTYPE;
using namespace supernpu::multi_thread;
constexpr std::size_t kInputs = 64, kInputSize = 4096, kOutputSize = kInputs * kInputSize;
constexpr std::size_t kGuard = 32;
// Wire format: count, axis, output capacity, flags; then rank, dims[8],
// capacity, pointer-mode per input. Pointer-mode 1=null, 2=alias output.
alignas(4096) std::int64_t metadata[4 + kInputs * 11];
alignas(4096) Data input_data[kInputs][kInputSize];
alignas(4096) Data output_data[kOutputSize + 2 * kGuard];
ConcatTensor<Data> tensors[kInputs];
MultiThreadResCheckSync res_check_sync{};
volatile std::uint32_t inputs_ok;
std::int64_t results[4][10];
}

int main() {
    const auto tid = get_thread_idx();
    if (tid == 0) {
        inputs_ok = readBinaryFile(CHK_DIR "/metadata.bin", reinterpret_cast<uint8_t *>(metadata), sizeof(metadata));
        if (inputs_ok && metadata[0] >= 0 && metadata[0] <= static_cast<int64_t>(kInputs)) {
            inputs_ok = readBinaryFile(CHK_DIR "/inputs.bin", reinterpret_cast<uint8_t *>(input_data),
                                       metadata[0] * kInputSize * sizeof(Data)) &&
                        readBinaryFile(CHK_DIR "/output.bin", reinterpret_cast<uint8_t *>(output_data), sizeof(output_data));
            for (std::int64_t i = 0; i < metadata[0]; ++i) {
                const auto *m = metadata + 4 + i * 11;
                tensors[i].shape.rank = m[0];
                for (int d = 0; d < 8; ++d) tensors[i].shape.dims[d] = m[d + 1];
                tensors[i].capacity = m[9];
                tensors[i].data = m[10] == 1 ? nullptr :
                                  m[10] == 2 ? output_data + kGuard : input_data[i];
            }
        } else {
            inputs_ok = 0;
        }
    }
    res_check_publish_inputs(res_check_sync, tid);
    if (!inputs_ok) return 1;
    ConcatShape inferred{99, {-99, -99, -99, -99, -99, -99, -99, -99}};
    BENCHSTART;
    const auto status = supernpu::multi_thread::concat<Data>(
        metadata[3] & 1 ? nullptr : tensors, metadata[0], metadata[1],
        metadata[3] & 2 ? nullptr : output_data + kGuard, metadata[2], &inferred);
    BENCHEND;
    results[tid][0] = static_cast<std::int64_t>(status);
    results[tid][1] = inferred.rank;
    for (int d = 0; d < 8; ++d) results[tid][d + 2] = inferred.dims[d];
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {
        if (!writeBinaryFile(CHK_DIR "/output.bin", reinterpret_cast<uint8_t *>(output_data), sizeof(output_data)) ||
            !writeBinaryFile(CHK_DIR "/status.bin", reinterpret_cast<uint8_t *>(results), sizeof(results))) return 1;
    }
    // Rejection cases are checked by the external oracle via status.bin.
    return 0;
}
