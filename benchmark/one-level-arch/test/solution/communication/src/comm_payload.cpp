#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "communication/pto_payload.hpp"

#ifndef RES_CHECK
#error "Build with res_check=on and use run_payload_check.py"
#endif

namespace {
constexpr int kSources = 3;
constexpr int kCapacity = 96;
constexpr int kCases = 3;
constexpr int kOperators = 4;
alignas(64) float inputs[kSources][kCapacity];
alignas(64) float weights[kSources];
alignas(64) float outputs[kOperators][kCases][kCapacity];
std::int32_t statuses[kOperators][kCases];
}  // namespace

// Single-PE payload validation only. The cross-process MMAP protocol lives in
// host_mmap_test.cpp; this executable makes no remote synchronization claim.
int main() {
    if (get_thread_idx() != 0) return 0;
    if (!readBinaryFile(CHK_DIR "/inputs.bin", reinterpret_cast<uint8_t*>(inputs), sizeof(inputs)) ||
        !readBinaryFile(CHK_DIR "/weights.bin", reinterpret_cast<uint8_t*>(weights), sizeof(weights)))
        return 1;
    for (int op = 0; op < kOperators; ++op)
        for (int c = 0; c < kCases; ++c)
            for (int i = 0; i < kCapacity; ++i) outputs[op][c][i] = -9999.0f;
    const float* sources[kSources] = {inputs[0], inputs[1], inputs[2]};
    using Ops = supernpu::comm::PtoPayload;
    BENCHSTART;
    for (int c = 0; c < kCases; ++c) {
        const std::size_t elements = (c + 1) * Ops::kWidth;
        statuses[0][c] = static_cast<int>(supernpu::comm::put<Ops>(outputs[0][c], inputs[0], elements));
        statuses[1][c] = static_cast<int>(supernpu::comm::get<Ops>(outputs[1][c], inputs[1], elements));
        statuses[2][c] = static_cast<int>(supernpu::comm::reduce_sum<Ops>(outputs[2][c], sources, kSources, elements));
        statuses[3][c] = static_cast<int>(supernpu::comm::weighted_combine<Ops>(
            outputs[3][c], sources, weights, kSources, elements));
    }
    BENCHEND;
    if (!writeBinaryFile(CHK_DIR "/outputs.bin", reinterpret_cast<const uint8_t*>(outputs), sizeof(outputs)) ||
        !writeBinaryFile(CHK_DIR "/statuses.bin", reinterpret_cast<const uint8_t*>(statuses), sizeof(statuses)))
        return 1;
    for (int op = 0; op < kOperators; ++op)
        for (int c = 0; c < kCases; ++c)
            if (statuses[op][c] != 0) return 1;
    return 0;
}
