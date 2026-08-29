#include "multi_thread/element_wise/tadd_multithread.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"

#ifndef TileRows
#define kTileRows 16
#else
#define kTileRows TileRows
#endif

#ifndef TileCols
#define kTileCols 16
#else
#define kTileCols TileCols
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

int main() {
    static_assert(kTileRows % 4 == 0,
                  "TileRows must be divisible by the four PE threads");

    constexpr uint32_t kIoTid = 0;
    const uint32_t tid = get_thread_idx();
    static float a_buf[kTileRows * kTileCols + 2 * ALIGN];
    static float b_buf[kTileRows * kTileCols + 2 * ALIGN];
    static float out_buf[kTileRows * kTileCols + 2 * ALIGN];
#ifdef RES_CHECK
    static MultiThreadResCheckSync res_check_sync{};
#endif
    float *a = (float *)(((uint64_t)a_buf & ALIGN_MASK) + ALIGN);
    float *b = (float *)(((uint64_t)b_buf & ALIGN_MASK) + ALIGN);
    float *out = (float *)(((uint64_t)out_buf & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC_A_PATH CHK_DIR "/src_a.bin"
#define SRC_B_PATH CHK_DIR "/src_b.bin"
    if (tid == kIoTid) {
        readBinaryFile(SRC_A_PATH, (uint8_t *)a,
                       kTileRows * kTileCols * sizeof(float));
        readBinaryFile(SRC_B_PATH, (uint8_t *)b,
                       kTileRows * kTileCols * sizeof(float));
    }
    res_check_publish_inputs(res_check_sync, tid);
#else
    for (int i = 0; i < kTileRows * kTileCols; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
#endif

    BENCHSTART;
    vec_multithread<kTileRows / 4, kTileCols>(out, a, b);
    BENCHEND;

#ifdef RES_CHECK
#define OUT_PATH CHK_DIR "/vec_out.bin"
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == kIoTid) {
        writeBinaryFile(OUT_PATH, (uint8_t *)out,
                        kTileRows * kTileCols * sizeof(float));
    }
#endif

    return 0;
}
