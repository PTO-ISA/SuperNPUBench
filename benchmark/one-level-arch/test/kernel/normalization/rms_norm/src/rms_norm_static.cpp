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

// Same as dynamic rms_norm.cpp tiling_info {16,512,2,512}
#ifndef G_A
#define G_A 16
#endif
#ifndef G_R
#define G_R 512
#endif
#ifndef TILE_A
#define TILE_A 2
#endif
#ifndef TILE_R
#define TILE_R 512
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

    static_assert(PE_NUM > 0, "PE_NUM must be positive");
    static_assert(G_A % PE_NUM == 0, "G_A must be divisible by PE_NUM");
    static_assert((G_A / PE_NUM) >= TILE_A, "PE-local G_A must cover one tile_a");

    constexpr int pe_a = G_A / PE_NUM;

    static dtype input_buf[G_A * G_R];
    static dtype output_buf[G_A * G_R];
    dtype *input = input_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                       static_cast<size_t>(G_A) * G_R * sizeof(dtype));
        input_ready = 1;
    } else {
        while (input_ready == 0) {
        }
    }
#endif

    // Full [G_A, G_R] buffers; kernel splits A with get_thread_idx().
    rms_norm<dtype, pe_a, G_A, G_R, TILE_A, TILE_R>(input, output, EPS);

#ifdef RES_CHECK
    kernel_done[tid] = 1;
    if (tid == 0) {
        for (int pe = 0; pe < PE_NUM; ++pe) {
            while (kernel_done[pe] == 0) {
            }
        }
        writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output,
                        static_cast<size_t>(G_A) * G_R * sizeof(dtype));
        output_written = 1;
    } else {
        while (output_written == 0) {
        }
    }
#endif
}
