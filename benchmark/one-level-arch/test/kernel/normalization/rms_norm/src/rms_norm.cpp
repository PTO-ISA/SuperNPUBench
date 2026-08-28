#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/rms_norm/rms_norm_pto.hpp"

#ifndef DType
#define DType __half
#endif

#ifndef EPS
#define EPS 1e-6f
#endif

#ifndef PE_NUM
#define PE_NUM 1
#endif

#ifdef RES_CHECK
namespace {
volatile uint32_t input_ready = 0;
volatile uint32_t kernel_done[PE_NUM] = {};
volatile uint32_t output_written = 0;
} // namespace
#endif

int main() {
    using dtype = DType;

    // tiling_info is always the host-visible full shape. PE partitioning is
    // entirely owned by the kernel.
    int64_t tiling_info[4] = {16, 512, 1, -1};

    const int64_t g_a = tiling_info[0];
    const int64_t g_r = tiling_info[1];

    static dtype input_buf[16 * 512];
    static dtype output_buf[16 * 512];
    dtype *input = input_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                       static_cast<size_t>(g_a) * g_r * sizeof(dtype));
        input_ready = 1;
    } else {
        while (input_ready == 0) {
        }
    }
#endif

    rms_norm<dtype>(input, tiling_info, output, EPS);

#ifdef RES_CHECK
    kernel_done[tid] = 1;
    if (tid == 0) {
        for (int pe = 0; pe < PE_NUM; ++pe) {
            while (kernel_done[pe] == 0) {
            }
        }
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output,
                        static_cast<size_t>(g_a) * g_r * sizeof(dtype));
        output_written = 1;
    } else {
        while (output_written == 0) {
        }
    }
#endif
}
