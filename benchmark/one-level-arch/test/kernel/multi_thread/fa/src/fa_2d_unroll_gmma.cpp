#include "fa/fa_2d_unroll_gmma.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"

#define B 1
#define H 1

#ifndef Tsq
#define globSq 512
#else
#define globSq Tsq
#endif

#ifndef Tskv
#define globSkv 512
#else
#define globSkv Tskv
#endif

#ifndef qD
#define qD 128
#endif

#ifndef vD
#define vD 128
#endif

#ifndef Tm
#define kTm 16
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 16
#else
#define kTk Tk
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

int main() {
    using dtype = float;
    constexpr int kPeNum = 4;

    static_assert(globSq % kPeNum == 0,
                  "global Sq must be divisible by the PE count");

    dtype qp[B * H * globSq * qD + 2 * ALIGN];
    dtype kp[B * H * globSkv * qD + 2 * ALIGN];
    dtype vp[B * H * globSkv * vD + 2 * ALIGN];
    dtype outp[B * H * globSq * vD + 2 * ALIGN];

    dtype *q = (dtype *)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    dtype *k = (dtype *)(((uint64_t)kp & ALIGN_MASK) + ALIGN);
    dtype *v = (dtype *)(((uint64_t)vp & ALIGN_MASK) + ALIGN);
    dtype *out = (dtype *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRCQ_PATH CHK_DIR "/srcq.bin"
#define SRCK_PATH CHK_DIR "/srck.bin"
#define SRCV_PATH CHK_DIR "/srcv.bin"
    readBinaryFile(SRCQ_PATH, (uint8_t *)q, B * H * globSq * qD * sizeof(dtype));
    readBinaryFile(SRCK_PATH, (uint8_t *)k, B * H * globSkv * qD * sizeof(dtype));
    readBinaryFile(SRCV_PATH, (uint8_t *)v, B * H * globSkv * vD * sizeof(dtype));
#endif

    BENCHSTART;
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < H; ++j) {
            // All four PEs load the same full shared Q/K/V tiles (PEMask=1),
            // so the kernel receives the full globSq / globSkv (mirrors
            // matmul_shared passing the full globM).
            flash_attention_2d_unroll_shared_impl<
                dtype, globSq, globSkv, qD, vD, kTm, kTk>(
                out + i * H * globSq * vD + j * globSq * vD,
                q + i * H * globSq * qD + j * globSq * qD,
                k + i * H * globSkv * qD + j * globSkv * qD,
                v + i * H * globSkv * vD + j * globSkv * vD);
        }
    }
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)out,
                    B * H * globSq * vD * sizeof(dtype));
#endif

    return 0;
}
