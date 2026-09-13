#include <common/pto_tileop.hpp>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include "common.h"
#include "benchmark.h"
#include "fileop.h"

#ifndef IN_H
#define IN_H 16
#endif

#ifndef IN_W
#define IN_W 16
#endif

#ifndef IN_C
#define IN_C 16
#endif

#ifndef OUT_C
#define OUT_C 16
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

#include "conv2d/conv2d_rm_mt.hpp"

#ifdef CONV_FP16
using datatype = __half;
#else
using datatype = float;
#endif
using output_type = float;

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main() {
    constexpr int kPeNum = 4;

    // input/output are arrays of 4 PE matrices (batch of 4 inputs, 4
    // outputs); weight is one shared matrix.
    datatype input_buf[kPeNum * IN_C * IN_H * IN_W + 2 * ALIGN];
    datatype weight_buf[OUT_C * IN_C + 2 * ALIGN];
    output_type output_buf[kPeNum * OUT_C * IN_H * IN_W + 2 * ALIGN];

    datatype* input_nchw = (datatype*)(((uint64_t)input_buf & ALIGN_MASK) + ALIGN);
    datatype* weight = (datatype*)(((uint64_t)weight_buf & ALIGN_MASK) + ALIGN);
    output_type* output = (output_type*)(((uint64_t)output_buf & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
    // Per-thread stacks: each PE's kernel writes its own slice into its own
    // output buffer at offset tid*gM*gN. Each PE dumps only its own slice to
    // a distinct res_pe<tid>.bin; the host side assembles the four files.
    // Paths are assembled manually (no snprintf: its varargs path misbehaves
    // under the gfrun SMT4 ecall runtime).
    const uint32_t tid = get_thread_idx();
    char src0_path[256];
    char src1_path[256];
    char res_path[256];
    {
        const char *dir = CHK_DIR;
        int n = 0;
        while (dir[n] != '\0' && n < 240) { src0_path[n] = dir[n]; ++n; }
        for (const char *s = "/src0.bin"; *s; ++s) src0_path[n++] = *s;
        src0_path[n] = '\0';
        int m = 0;
        while (dir[m] != '\0' && m < 240) { src1_path[m] = dir[m]; ++m; }
        for (const char *s = "/src1.bin"; *s; ++s) src1_path[m++] = *s;
        src1_path[m] = '\0';
        int k = 0;
        while (dir[k] != '\0' && k < 240) { res_path[k] = dir[k]; ++k; }
        for (const char *s = "/res_pe"; *s; ++s) res_path[k++] = *s;
        res_path[k++] = static_cast<char>('0' + (tid % 10));
        for (const char *s = ".bin"; *s; ++s) res_path[k++] = *s;
        res_path[k] = '\0';
    }
    readBinaryFile(src0_path, (uint8_t*)input_nchw, kPeNum * IN_C * IN_H * IN_W * sizeof(datatype));
    readBinaryFile(src1_path, (uint8_t*)weight, OUT_C * IN_C * sizeof(datatype));

    for (int i = 0; i < kPeNum * OUT_C * IN_H * IN_W; ++i)
        output[i] = (output_type)0;
#else
#ifdef CONV_FP16
    volatile uint32_t* vi = (volatile uint32_t*)input_nchw;
    for (int i = 0; i < kPeNum * IN_C * IN_H * IN_W / 2; ++i)
        vi[i] = 0x3C003C00u;
    volatile uint32_t* vw = (volatile uint32_t*)weight;
    for (int i = 0; i < OUT_C * IN_C / 2; ++i)
        vw[i] = 0x3C003C00u;
#else
    volatile uint32_t* vi = (volatile uint32_t*)input_nchw;
    for (int i = 0; i < kPeNum * IN_C * IN_H * IN_W; ++i)
        vi[i] = 0x40000000u;
    volatile uint32_t* vw = (volatile uint32_t*)weight;
    for (int i = 0; i < OUT_C * IN_C; ++i)
        vw[i] = 0x3F800000u;
#endif
#endif

    BENCHSTART;
    conv2d_1x1_rm_mt<datatype, IN_C, IN_H, IN_W, OUT_C,
                     tilM, tilN, tilK>(output, input_nchw, weight);
    BENCHEND;

#ifdef RES_CHECK
    // Dump only this PE's slice (the kernel stored it at tid*gM*gN in this
    // PE's private buffer).
    writeBinaryFile(res_path,
                    (uint8_t*)(output + tid * OUT_C * IN_H * IN_W),
                    OUT_C * IN_H * IN_W * sizeof(output_type));
#endif

    return 0;
}
