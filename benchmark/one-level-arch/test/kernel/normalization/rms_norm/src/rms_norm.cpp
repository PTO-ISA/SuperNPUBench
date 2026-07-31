#include <common/pto_tileop.hpp>

#include <cstdint>

#include "fileop.h"
#include "normalization/rms_norm_pto.hpp"

#ifndef DType
#define DType __half
#endif

#ifndef EPS
#define EPS 1e-6f
#endif

int main() {
    using dtype = DType;

    // tiling_info: {g_m, g_n, tile_m, tile_n}
    int64_t tiling_info[4] = {16, 512, 1, -1};

    const int64_t g_m = tiling_info[0];
    const int64_t g_n = tiling_info[1];

    dtype input_buf[g_m * g_n];
    dtype output_buf[g_m * g_n];
    dtype *input = input_buf;
    dtype *output = output_buf;

#ifdef RES_CHECK
#ifndef CHK_DIR
#error "CHK_DIR must be set when RES_CHECK is enabled"
#endif
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t *)input,
                   static_cast<size_t>(g_m) * g_n * sizeof(dtype));
#endif

    rms_norm<dtype>(input, tiling_info, output, EPS);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t *)output,
                    static_cast<size_t>(g_m) * g_n * sizeof(dtype));
#endif
}
