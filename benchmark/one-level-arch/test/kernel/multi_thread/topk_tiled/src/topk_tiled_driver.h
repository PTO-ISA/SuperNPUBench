#pragma once

#include <cstdint>

#include "fileop.h"
#ifdef RES_CHECK
#include "multi_thread_res_check.h"
#endif
#include "multi_thread/topk/topk_tiled.hpp"

// Host-owned inputs, same contract as topk_driver.h but for the serial
// tile-op transcription in topk_tiled.hpp.

namespace topk_tiled_test {

using topk_tiled::Scratch;
using topk_tiled::kBatch;
using topk_tiled::kCols;
using topk_tiled::kTopK;
using topk_tiled::kLane;

struct Buffers {
    alignas(4096) float input[kBatch * kCols + kLane];
    alignas(4096) int32_t starts[kBatch];
    alignas(4096) int32_t ends[kBatch];
    alignas(4096) int32_t output[kBatch * kTopK];
    alignas(4096) int32_t errors[kBatch];
    alignas(4096) Scratch scratch[kBatch];
};

#ifdef RES_CHECK

inline void prepare(Buffers &buffers, MultiThreadResCheckSync &sync, uint32_t tid) {
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin",
                       reinterpret_cast<uint8_t *>(buffers.input), sizeof(buffers.input));
        readBinaryFile(CHK_DIR "/starts.bin",
                       reinterpret_cast<uint8_t *>(buffers.starts), sizeof(buffers.starts));
        readBinaryFile(CHK_DIR "/ends.bin",
                       reinterpret_cast<uint8_t *>(buffers.ends), sizeof(buffers.ends));
        for (int i = 0; i < kBatch * kTopK; ++i) buffers.output[i] = -1;
        for (int i = 0; i < kBatch; ++i) buffers.errors[i] = 0;
    }
    res_check_publish_inputs(sync, tid);
}

inline int finish(Buffers &buffers, MultiThreadResCheckSync &sync, uint32_t tid) {
    res_check_wait_for_all(sync, tid);
    if (tid != 0) return 0;
    writeBinaryFile(CHK_DIR "/input_readback.bin",
                    reinterpret_cast<const uint8_t *>(buffers.input), sizeof(buffers.input));
    writeBinaryFile(CHK_DIR "/output.bin",
                    reinterpret_cast<const uint8_t *>(buffers.output), sizeof(buffers.output));
    writeBinaryFile(CHK_DIR "/errors.bin",
                    reinterpret_cast<const uint8_t *>(buffers.errors), sizeof(buffers.errors));
    return 0;
}

#else

struct NoSync {};

inline void prepare(Buffers &, NoSync &, uint32_t) {}
inline int finish(Buffers &, NoSync &, uint32_t) { return 0; }

#endif

}  // namespace topk_tiled_test

#ifndef RES_CHECK
using MultiThreadResCheckSync = topk_tiled_test::NoSync;
#endif
