// Fixed-shape 4PE RMSNorm with an outer-R balanced reduction tree.
#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "solution/normalization/rms_norm/rms_norm_simt_static_m_R_tree.hpp"

#ifndef DType
#define DType __half
#endif
#ifndef PE_NUM
#define PE_NUM 4
#endif
#ifndef G_A
#define G_A 512
#endif
#ifndef G_R
#define G_R 8192
#endif

#ifdef RES_CHECK
namespace {
volatile uint32_t input_ready = 0;
volatile uint32_t kernel_done[PE_NUM] = {};
volatile uint32_t output_written = 0;
} // namespace
#endif

int main() {
    static_assert(G_A == 512 && G_R == 8192,
                  "static testcase has a fixed shape");
    using dtype = DType;

    static dtype input_buf[G_A * G_R];
    static dtype gamma_buf[G_R];
    static dtype output_buf[G_A * G_R];

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input_buf,
                       static_cast<size_t>(G_A) * G_R * sizeof(dtype));
        readBinaryFile(CHK_DIR "/gamma.bin", (uint8_t *)gamma_buf,
                       static_cast<size_t>(G_R) * sizeof(dtype));
        input_ready = 1;
    } else {
        while (input_ready == 0) {
        }
    }
#endif

    rms_norm_simt_static_m_R_tree<dtype, PE_NUM>(
        input_buf, gamma_buf, output_buf);

#ifdef RES_CHECK
    kernel_done[tid] = 1;
    if (tid == 0) {
        for (int pe = 0; pe < PE_NUM; ++pe) {
            while (kernel_done[pe] == 0) {
            }
        }
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output_buf,
                        static_cast<size_t>(G_A) * G_R * sizeof(dtype));
        output_written = 1;
    } else {
        while (output_written == 0) {
        }
    }
#endif
}
