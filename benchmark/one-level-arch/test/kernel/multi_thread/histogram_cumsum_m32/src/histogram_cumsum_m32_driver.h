#pragma once

#include <cstdint>

#include "fileop.h"
#ifdef RES_CHECK
#include "multi_thread_res_check.h"
#endif

// Per-PE 384-word scratch: 256 histogram bins followed by the 128-word zero
// pad the grouped TLOAD reads past the sentinel. Each PE suffix-sums its own
// row once; PE0 owns the host I/O. The host writes hist.bin and the device
// exports result.bin, so a broken cumsum cannot pass vacuously on
// device-generated input.

namespace histogram_cumsum_m32_test {

constexpr int kRows = 4;
constexpr int kWords = 384;

struct Buffers {
    alignas(4096) int32_t hist[kRows][kWords];
};

#ifdef RES_CHECK

inline void prepare(Buffers &buffers, MultiThreadResCheckSync &sync, uint32_t tid) {
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/hist.bin",
                       reinterpret_cast<uint8_t *>(buffers.hist), sizeof(buffers.hist));
    }
    res_check_publish_inputs(sync, tid);
}

inline int finish(Buffers &buffers, MultiThreadResCheckSync &sync, uint32_t tid) {
    res_check_wait_for_all(sync, tid);
    if (tid != 0) return 0;
    writeBinaryFile(CHK_DIR "/result.bin",
                    reinterpret_cast<const uint8_t *>(buffers.hist), sizeof(buffers.hist));
    return 0;
}

#else

struct NoSync {};

inline void prepare(Buffers &, NoSync &, uint32_t) {}
inline int finish(Buffers &, NoSync &, uint32_t) { return 0; }

#endif

}  // namespace histogram_cumsum_m32_test

#ifndef RES_CHECK
using MultiThreadResCheckSync = histogram_cumsum_m32_test::NoSync;
#endif
