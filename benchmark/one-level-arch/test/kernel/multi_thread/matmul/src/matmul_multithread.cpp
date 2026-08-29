#include "multi_thread/matmul/matmul_multithread.hpp"

#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef LINX_GROUP_RUNTIME
#include <common/linx_group_runtime.h>
#endif

#ifndef globM
#define globM 16
#endif
#ifndef globN
#define globN 16
#endif
#ifndef globK
#define globK 16
#endif
#ifndef tilM
#define tilM 16
#endif
#ifndef tilN
#define tilN 16
#endif
#ifndef tilK
#define tilK 16
#endif

namespace {
constexpr int kPeCount = 4;
alignas(4096) float a[kPeCount * globM * globK];
alignas(4096) float b[globK * globN];
alignas(4096) float c[kPeCount * globM * globN];

struct Context {
    float *a;
    float *b;
    float *c;
};
}  // namespace

extern "C" int __linx_group_worker_main(uint32_t, void *opaque) {
    auto *context = static_cast<Context *>(opaque);
    BENCHSTART;
    matmul_multithread<float, globM, globN, globK, tilM, tilN, tilK>(
        context->c, context->a, context->b);
    BENCHEND;
    return 0;
}

int main() {
#ifdef RES_CHECK
    constexpr uint32_t kIoTid = 0;
    const uint32_t tid = get_thread_idx();
    static MultiThreadResCheckSync res_check_sync{};
    if (tid == kIoTid) {
        readBinaryFile(CHK_DIR "/src0.bin", reinterpret_cast<uint8_t *>(a),
                       sizeof(a));
        readBinaryFile(CHK_DIR "/src1.bin", reinterpret_cast<uint8_t *>(b),
                       sizeof(b));
    }
#ifndef LINX_GROUP_RUNTIME
    res_check_publish_inputs(res_check_sync, tid);
#endif
#endif
    Context context{a, b, c};
#ifdef LINX_GROUP_RUNTIME
    const int status = linx_group_run(&context);
#else
    const int status = __linx_group_worker_main(0, &context);
#endif
#ifdef RES_CHECK
#ifndef LINX_GROUP_RUNTIME
    res_check_wait_for_all(res_check_sync, tid);
#endif
    if (tid == kIoTid) {
        writeBinaryFile(CHK_DIR "/res.bin", reinterpret_cast<uint8_t *>(c),
                        sizeof(c));
    }
#endif
    return status;
}
