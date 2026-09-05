#include <common/pto_tileop.hpp>
#include <cstdint>
#include "fileop.h"
#include "multi_thread/quant/dynamic_mx_quant/dynamic_mx_quant_nontail_ocp_fp4_bigbs.hpp"
using namespace supernpu::tile_isa::mxquant;

// Non-tail OCP-FP4, LARGE-BlockSize branch (方案A: split reduce axis). BlockSize
// >=128 has NO valid TileN for the plain single-load kernel (fp4 alignment lower
// bound 64 > TileSize upper 4096/BS < 64); this branch tiles the reduce axis into
// R_sub=32 sub-chunks so R_sub*TileN=2048<=4096 holds for any BlockSize.
// Axis=128 (=BlockSize, numKb=1), Post=64 (TileN=64=2 MX blocks), R_sub=32
// (numSub=4). x=[128,64], y=[128,32] bytes. scale compact planar uint8 E8M0
// [scaleRows, Post]: scaleRows = evenAlign(Axis/128) = 2 -> [2,64] bytes.
//
// SINGLE-PE: bigbs 场景块行少（numKb=1）、并行度有限，不做 4-PE（与 nontail_cublas_fp8
// bigbs 一致），故无 tid gate / 屏障，单线程直接落盘。
static __bf16 x[128 * 64] __attribute__((aligned(4096))) = {};
static uint8_t y[128 * 32] __attribute__((aligned(4096))) = {};
static uint8_t scale[2 * 64] __attribute__((aligned(4096))) = {};

int main() {
#ifdef RES_CHECK
    readBinaryFile(CHK_DIR "/input.bin", (uint8_t*)x, sizeof(x));
#endif

    dynamic_mx_quant_nontail_ocp_fp4_bigbs<128, 64, 128>(
        x, reinterpret_cast<__fp4_e2m1x2*>(y), scale);

#ifdef RES_CHECK
    writeBinaryFile(CHK_DIR "/output.bin", (uint8_t*)y, sizeof(y));
    writeBinaryFile(CHK_DIR "/scale_output.bin", (uint8_t*)scale, sizeof(scale));
#endif
    return 0;
}
