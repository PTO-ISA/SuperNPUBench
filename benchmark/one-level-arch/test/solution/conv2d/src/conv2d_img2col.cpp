// conv2d_img2col: NCHW in / NHWC out convolution driven by the
// convolution-dedicated GM transports (TIMG2COL + weight-mode TLOAD).
// Golden reference: torch.nn.functional.conv2d (golden_conv2d_img2col.py).
#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "solution/conv2d/conv2d_img2col.hpp"

#include <cstdint>

#ifndef Batch
#define Batch 2
#endif
#ifndef Cin
#define Cin 8
#endif
#ifndef Hin
#define Hin 32
#endif
#ifndef Win
#define Win 32
#endif
#ifndef Cout
#define Cout 32
#endif
#ifndef Kh
#define Kh 2
#endif
#ifndef Kw
#define Kw 2
#endif
#ifndef Sh
#define Sh 2
#endif
#ifndef Sw
#define Sw 2
#endif
#ifndef Pt
#define Pt 0
#endif
#ifndef Pb
#define Pb 0
#endif
#ifndef Pl
#define Pl 0
#endif
#ifndef Pr
#define Pr 0
#endif

namespace {
constexpr int kOutH = (Hin + Pt + Pb - Kh) / Sh + 1;
constexpr int kOutW = (Win + Pl + Pr - Kw) / Sw + 1;

alignas(4096) float input[Batch * Cin * Hin * Win];
alignas(4096) float weight[Cout * Cin * Kh * Kw];
alignas(4096) float output[Batch * kOutH * kOutW * Cout];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin",
                       reinterpret_cast<std::uint8_t *>(input), sizeof(input));
        readBinaryFile(CHK_DIR "/weight.bin",
                       reinterpret_cast<std::uint8_t *>(weight),
                       sizeof(weight));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        supernpu::conv2d_img2col::conv2d_img2col<
            Cin, Hin, Win, Cout, Kh, Kw, Sh, Sw, Pt, Pb, Pl, Pr>(
            output + b * kOutH * kOutW * Cout,
            input + b * Hin * Win * Cin, weight);
    }
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin",
                        reinterpret_cast<std::uint8_t *>(output),
                        sizeof(output));
    }
#endif
    return 0;
}
