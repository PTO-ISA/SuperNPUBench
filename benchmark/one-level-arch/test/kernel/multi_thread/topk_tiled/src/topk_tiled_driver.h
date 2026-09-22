#pragma once

#include <cstdint>

#include "fileop.h"
#ifdef RES_CHECK
#include "multi_thread_res_check.h"
#endif
#include "multi_thread/topk/topk_tiled.hpp"

// Host-owned inputs for the tile-op topk.  The outer shape (batch x cols,
// top-k) is runtime (TopkTilingData); the buffers below are sized to the
// compile-time maxima so the same ELF can run different shapes.  The res_check
// case reads its shape from CHK_DIR/shape.bin (two-triple: batch, cols, topk).

namespace topk_tiled_test {

using topk_tiled::Scratch;
using topk_tiled::TopkTilingData;
using topk_tiled::kBatchMax;
using topk_tiled::kColsMax;
using topk_tiled::kTopKMax;
using topk_tiled::kLane;

struct Buffers {
    alignas(4096) float input[kBatchMax * kColsMax + kLane];
    alignas(4096) int32_t starts[kBatchMax];
    alignas(4096) int32_t ends[kBatchMax];
    alignas(4096) int32_t output[kBatchMax * kTopKMax];
    alignas(4096) int32_t errors[kBatchMax];
    alignas(4096) Scratch scratch[kBatchMax];
    alignas(4096) TopkTilingData tiling;
};

#ifdef RES_CHECK

inline void prepare(Buffers &buffers, MultiThreadResCheckSync &sync, uint32_t tid) {
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/shape.bin",
                       reinterpret_cast<uint8_t *>(&buffers.tiling),
                       sizeof(buffers.tiling));
        const int32_t batch = static_cast<int32_t>(buffers.tiling.batch);
        const int32_t cols = static_cast<int32_t>(buffers.tiling.cols);
        const int32_t topk = static_cast<int32_t>(buffers.tiling.topk);
        readBinaryFile(CHK_DIR "/input.bin",
                       reinterpret_cast<uint8_t *>(buffers.input),
                       (batch * cols + kLane) * sizeof(float));
        readBinaryFile(CHK_DIR "/starts.bin",
                       reinterpret_cast<uint8_t *>(buffers.starts),
                       batch * sizeof(int32_t));
        readBinaryFile(CHK_DIR "/ends.bin",
                       reinterpret_cast<uint8_t *>(buffers.ends),
                       batch * sizeof(int32_t));
        for (int32_t i = 0; i < batch * topk; ++i) buffers.output[i] = -1;
        for (int32_t i = 0; i < batch; ++i) buffers.errors[i] = 0;
    }
    res_check_publish_inputs(sync, tid);
}

inline int finish(Buffers &buffers, MultiThreadResCheckSync &sync, uint32_t tid) {
    res_check_wait_for_all(sync, tid);
    if (tid != 0) return 0;
    const int32_t batch = static_cast<int32_t>(buffers.tiling.batch);
    const int32_t cols = static_cast<int32_t>(buffers.tiling.cols);
    const int32_t topk = static_cast<int32_t>(buffers.tiling.topk);
    writeBinaryFile(CHK_DIR "/input_readback.bin",
                    reinterpret_cast<const uint8_t *>(buffers.input),
                    (batch * cols + kLane) * sizeof(float));
    writeBinaryFile(CHK_DIR "/output.bin",
                    reinterpret_cast<const uint8_t *>(buffers.output),
                    batch * topk * sizeof(int32_t));
    writeBinaryFile(CHK_DIR "/errors.bin",
                    reinterpret_cast<const uint8_t *>(buffers.errors),
                    batch * sizeof(int32_t));
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
