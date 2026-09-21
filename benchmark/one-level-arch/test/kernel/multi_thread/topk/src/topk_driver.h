#pragma once

#include <cstdint>

#include "fileop.h"
#ifdef RES_CHECK
#include "multi_thread_res_check.h"
#endif
#include "multi_thread/topk/topk.hpp"

// The host owns the input: device-side scalar fill loops were measured not to
// become visible to the buffer the kernel and the syscall path read, silently
// degrading every comparison to all-zero-vs-all-zero. The host writes
// input.bin/ranges, the device exports input_readback.bin, and the host
// asserts the round trip before comparing anything.

namespace topk_test {

using topk_radix::Scratch;
using topk_radix::kBatch;
using topk_radix::kCols;
using topk_radix::kTopK;
using topk_radix::kLane;

struct Buffers {
    alignas(4096) float input[kBatch * kCols + kLane];  // pad: full-tile tail TLOADs
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
        // Sentinel prefill: an output slot the kernel never writes must show
        // up as -1 in the export, not as leftover zero.
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

}  // namespace topk_test

#ifndef RES_CHECK
using MultiThreadResCheckSync = topk_test::NoSync;
#endif
