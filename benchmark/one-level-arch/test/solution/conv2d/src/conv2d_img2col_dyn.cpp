// conv2d_img2col_dyn: dynamic-shape NCHW in / NHWC out convolution driven
// by the convolution-dedicated GM transports (TIMG2COL + weight-mode
// TLOAD). All shape numbers reach the kernel through the runtime tiling
// descriptor; the only compile-time geometry is the matmul tiling
// (KPhys/GroupM/TileN), derived here from the same macros.
//
// The default shape deliberately differs from the static conv2d_img2col
// test (Cin 16 vs 8, 1x2 vs 2x2 kernel, 32x64 vs 32x32 map, asymmetric
// pads, 48 vs 32 output channels) while sharing the same KPhys = 32:
// one conv2d_img2col_dyn<32> instantiation serves both shape families.
// Golden reference: torch.nn.functional.conv2d
// (golden_conv2d_img2col_dyn.py).
#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "solution/conv2d/conv2d_img2col_dyn.hpp"

#include <cstdint>

#ifndef Batch
#define Batch 2
#endif
#ifndef Cin
#define Cin 16
#endif
#ifndef Hin
#define Hin 32
#endif
#ifndef Win
#define Win 64
#endif
#ifndef Cout
#define Cout 48
#endif
#ifndef Kh
#define Kh 1
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
#define Pt 2
#endif
#ifndef Pb
#define Pb 1
#endif
#ifndef Pl
#define Pl 0
#endif
#ifndef Pr
#define Pr 1
#endif
#ifndef Dh
#define Dh 1
#endif
#ifndef Dw
#define Dw 1
#endif
#ifndef GroupM
#define GroupM 64
#endif
#ifndef TileN
#define TileN 16
#endif

namespace {
constexpr int kC0 = 256 / 32;  // fp32 channel carrier
constexpr int kC1 = (Cin + kC0 - 1) / kC0;
constexpr int kK = Kh * Kw * kC1 * kC0;
constexpr int kEffH = (Kh - 1) * Dh + 1;
constexpr int kEffW = (Kw - 1) * Dw + 1;
constexpr int kOutH = (Hin + Pt + Pb - kEffH) / Sh + 1;
constexpr int kOutW = (Win + Pl + Pr - kEffW) / Sw + 1;
constexpr int kM = kOutH * kOutW;

// Mirror the kernel's runtime contract at compile time so an invalid
// Makefile shape fails the build instead of trapping in the simulator.
static_assert(kK % kC0 == 0 && (kK & (kK - 1)) == 0,
              "K = KH*KW*ceil(Cin/c0)*c0 must be a power of two");
static_assert(kM % GroupM == 0, "Ho*Wo must be a multiple of GroupM");
static_assert(Cout % TileN == 0, "Cout must be a multiple of TileN");
static_assert(kOutH > 0 && kOutW > 0, "empty convolution output");

alignas(4096) float input[Batch * Cin * Hin * Win];
alignas(4096) float weight[Cout * Cin * Kh * Kw];
alignas(4096) float output[Batch * kOutH * kOutW * Cout];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    // Runtime tiling descriptor consumed by the kernel (see the header
    // for the layout); nothing below reaches the kernel as a template
    // parameter.
    std::int64_t tiling_info[14] = {Cin, Hin,   Win,  Cout,     Kh,
                                    Kw,  Sh,    Sw,   Pt,       Pb,
                                    Pl,  Pr,    Dh,   Dw};
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/input.bin",
                       reinterpret_cast<std::uint8_t *>(input), sizeof(input));
        readBinaryFile(CHK_DIR "/weight.bin",
                       reinterpret_cast<std::uint8_t *>(weight), sizeof(weight));
    }
    res_check_publish_inputs(res_check_sync, tid);
#endif
    BENCHSTART;
    for (int b = 0; b < Batch; ++b) {
        supernpu::conv2d_img2col::conv2d_img2col_dyn<kK, GroupM, TileN>(
            output + b * kOutH * kOutW * Cout,
            input + b * Hin * Win * Cin, weight, tiling_info);
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
