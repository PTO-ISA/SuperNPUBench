#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"

#include "template_asm.h"

#define D4 64
#define D3 8
#define D2 8
#define D1 128
#define D0 15

#define STRIDE_4 (D3 * D2 * D1 * D0)
#define STRIDE_3 (D2 * D1 * D0)
#define SRC_STRIDE_2 (D1 * D0)
#define DST_STRIDE_2 (D2 * D0)
#define STRIDE_1 D0

#define tD2 1
#define tD1 1
#define tD0 15
#define tD0_ALIGN 128

#define ALIGN_MASK (0xfffffffffffff000ull)
#define ALIGN (4 * 1024)

int main()
{
    float inp[D4 * D3 * D2 * D1 * D0 + 2 * ALIGN];
    float outp[D4 * D3 * D2 * D1 * D0 + 2 * ALIGN];

    float* in = (float *)(((uint64_t)inp & ALIGN_MASK) + ALIGN);
    float* out = (float *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    #define SRCIN_PATH CHK_DIR "/srcin.bin"
    bool suaccelss = readBinaryFile(SRCIN_PATH, (uint8_t*) in, D4 * D3 * D2 * D1 * D0 * sizeof(float));
    assert(suaccelss);
#endif

    BENCHSTART;

    using gm_shape_in = global_tensor<float, RowMajor<1, tD0_ALIGN>>;
    using gm_shape_out = global_tensor<float, RowMajor<1, tD0_ALIGN>>;
    using tile_shape = Tile<Location::Vec, float, 1, tD0_ALIGN, BLayout::RowMajor, 1, tD0>;

    for (int d4 = 0; d4 < D4; d4++) {
        for (int d3 = 0; d3 < D3; d3++) {
            for (int d2 = 0; d2 < D2; d2++) {
                for (int d1 = 0; d1 < D1; d1++) {
                    size_t src_slice_offset = d4 * STRIDE_4 + d3 * STRIDE_3 + d2 * SRC_STRIDE_2 + d1 * STRIDE_1;
                    size_t dst_slice_offset = d4 * STRIDE_4 + d3 * STRIDE_3 + d1 * DST_STRIDE_2 + d2 * STRIDE_1;
                    gm_shape_in gSrc(in + src_slice_offset);
                    gm_shape_out gDst(out + dst_slice_offset);

                    tile_shape tmp;
                    TLOAD(tmp, gSrc);
                    TSTORE(gDst, tmp);
                }
            }
        }
    }

    BENCHEND;

#ifdef RES_CHECK
    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t*) out, D4 * D3 * D2 * D1 * D0* sizeof(float));
#endif
}
