#include <common/pto_tileop.hpp>
#include <cstring>
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

// quant_batch_matmul_test_mxfp4 — e2m1x2 fp4 MX multi-block matmul 测试入口
//
// NBLOCKS 个 PE 各自独立计算 M 维的一段（blockM = globM / NBLOCKS）。
// A/B 为 e2m1x2 fp4 打包，MX scaling 用 e8m0 / 32-element group。
//
// 精度验证: res_check=on 构建后读取 src0.bin/src1.bin(fp4 打包) +
//           src0_mx.bin/src1_mx.bin(e8m0)，写出 res.bin(fp32)。
// 性能验证: BENCHSTART/BENCHEND 标记算子区间。

#ifndef globM
#define globM 8192
#endif

#ifndef globN
#define globN 4096
#endif

#ifndef globK
#define globK 1600
#endif

#ifndef tilM
#define tilM 32
#endif

#ifndef tilN
#define tilN 32
#endif

#ifndef tilK
#define tilK 32
#endif

#ifndef NBLOCKS
#define NBLOCKS 32
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

#include "solution/quant_batch_matmul/quant_batch_matmul_mxfp4.hpp"

int main()
{
    using fp4_t = __fp4_e2m1x2;
    constexpr int globKv = globK / 2;   // fp4 打包：K 逻辑元素 → carriers
    constexpr int tilKv   = tilK / 2;

    static_assert(globM % NBLOCKS == 0);
    constexpr int blockM = globM / NBLOCKS;
    static_assert(blockM % tilM == 0);
    static_assert(globN % tilN == 0);
    static_assert(globKv % tilKv == 0);

    // 每组 block 拥有自己连续的 A/C + MX 区域
    fp4_t       src0p   [NBLOCKS * blockM * globKv + 2 * ALIGN];
    fp4_t       src1p   [globKv * globN + 2 * ALIGN];
    __fp8_e8m0  src0_mxp[NBLOCKS * blockM * (globKv / 32) + 2 * ALIGN];
    __fp8_e8m0  src1_mxp[(globKv / 32) * globN + 2 * ALIGN];
    float       dstp    [NBLOCKS * blockM * globN + 2 * ALIGN];

    fp4_t      *src0    = (fp4_t      *)(((uint64_t)src0p    & ALIGN_MASK) + ALIGN);
    fp4_t      *src1    = (fp4_t      *)(((uint64_t)src1p    & ALIGN_MASK) + ALIGN);
    __fp8_e8m0 *src0_mx = (__fp8_e8m0 *)(((uint64_t)src0_mxp & ALIGN_MASK) + ALIGN);
    __fp8_e8m0 *src1_mx = (__fp8_e8m0 *)(((uint64_t)src1_mxp & ALIGN_MASK) + ALIGN);
    float      *dst     = (float      *)(((uint64_t)dstp     & ALIGN_MASK) + ALIGN);

#ifdef RES_CHECK
#define SRC0_PATH    CHK_DIR "/src0.bin"
#define SRC1_PATH    CHK_DIR "/src1.bin"
#define SRC0_MX_PATH CHK_DIR "/src0_mx.bin"
#define SRC1_MX_PATH CHK_DIR "/src1_mx.bin"
    readBinaryFile(SRC0_PATH,    (uint8_t *)src0,    NBLOCKS * blockM * globKv * sizeof(fp4_t));
    readBinaryFile(SRC1_PATH,    (uint8_t *)src1,    globKv * globN * sizeof(fp4_t));
    readBinaryFile(SRC0_MX_PATH, (uint8_t *)src0_mx, NBLOCKS * blockM * (globKv / 32) * sizeof(__fp8_e8m0));
    readBinaryFile(SRC1_MX_PATH, (uint8_t *)src1_mx, (globKv / 32) * globN * sizeof(__fp8_e8m0));
#endif

    BENCHSTART;
    quant_batch_matmul_mxfp4<fp4_t, blockM, globN, globKv,
                        tilM, tilN, tilKv, fp4_t, 32>(
        dst, src0, src1, src0_mx, src1_mx);
    BENCHEND;

#ifdef RES_CHECK
#define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)dst, NBLOCKS * blockM * globN * sizeof(float));
#endif

    return 0;
}
