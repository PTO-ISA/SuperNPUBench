#pragma once

#include <cstdint>

// Hosted RES_CHECK runs the same main() on four PEs.  File I/O must remain on
// PE0, while the input/output buffers live in shared static storage.  This
// small SPMD barrier publishes PE0's input writes and prevents PE0 from
// exporting a partially-computed result before PE1..PE3 have finished.
struct MultiThreadResCheckSync {
    volatile std::uint32_t input_ready;
    volatile std::uint32_t done[4];
};

inline void res_check_publish_inputs(MultiThreadResCheckSync &sync,
                                     std::uint32_t tid) {
    if (tid == 0) {
        __asm__ volatile("" : : : "memory");
        sync.input_ready = 1;
    } else {
        while (sync.input_ready == 0) {
        }
        __asm__ volatile("" : : : "memory");
    }
}

inline void res_check_wait_for_all(MultiThreadResCheckSync &sync,
                                   std::uint32_t tid) {
    __asm__ volatile("" : : : "memory");
    sync.done[tid] = 1;
    if (tid == 0) {
        for (std::uint32_t pe = 1; pe < 4; ++pe) {
            while (sync.done[pe] == 0) {
            }
        }
        __asm__ volatile("" : : : "memory");
    }
}
