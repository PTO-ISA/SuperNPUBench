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

int main() {
    using dtype = DType;

    // tiling_info: {g_a, g_r, tile_a, tile_r}
    int64_t tiling_info[4] = {16, 512, 1, -1};

    const int64_t g_a = tiling_info[0];
    const int64_t g_r = tiling_info[1];

    dtype input_buf[16 * 512];
    dtype output_buf[16 * 512];
    dtype *input = input_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                   static_cast<size_t>(g_a) * g_r * sizeof(dtype));
#endif

    rms_norm<dtype>(input, tiling_info, output, EPS);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output,
                    static_cast<size_t>(g_a) * g_r * sizeof(dtype));
#endif
}
