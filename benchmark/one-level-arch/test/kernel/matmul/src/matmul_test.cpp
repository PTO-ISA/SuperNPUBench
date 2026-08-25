// matmul_test.cpp
//
// Test harness for the dynamic-shape matmul_test kernel (matmul_test.hpp).
//   C = A * B,  A:[gM,gK] __half, B:[gK,gN] __half, C:[gM,gN] __half
//   Accumulation in float (TMATMUL/TMATMUL_ACC), output cast via TCVT.
//   gM/gN/gK are runtime variables; only tM/tN/tK are compile-time.
//
// Two verification modes (mutually exclusive):
//
//   1. RES_CHECK  (make ... res_check=on)
//      - Shape:   CHK_DIR/shape.txt  ("M N K\n")
//      - Inputs:  CHK_DIR/src0.bin, CHK_DIR/src1.bin  (__half, row-major)
//      - Output:  CHK_DIR/res.bin    (float32, converted from __half output
//                                     so the host-side gfrun golden compare
//                                     infrastructure works unchanged)
//      - Verify:  host-side (gfrun_matmul.py: torch.matmul golden, np.allclose)
//
//   2. SELF_VERIFY  (make ... self_verify=on)
//      - Shape:   shape.txt in CWD  ("M N K\n")
//      - Inputs:  deterministic all-1 __half (no bin files needed)
//      - Golden:  globK  (scalar — exact in float accumulation)
//      - Verify:  in-kernel, first VERIFY_ELEMS elements (default 1024;
//                 full check via verify_elems=65535)
//      - Return:  0 = pass, nonzero = mismatch count (truncated to 255)
//
//   3. Default (neither flag):
//      - Runs with a small fixed shape, no verification (smoke test).

#include <common/pto_tileop.hpp>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

#ifndef tilM
#define tilM 16
#endif

#ifndef tilN
#define tilN 16
#endif

#ifndef tilK
#define tilK 16
#endif

#ifndef VERIFY_ELEMS
#define VERIFY_ELEMS 1024
#endif

#include "matmul/matmul_test.hpp"

// ---------------------------------------------------------------------------
// shape.txt reader: parses "M N K" (whitespace-separated) via POSIX open/read.
// Works in both res_check (hosted libc) and self_verify (gfrun syscall) paths.
// ---------------------------------------------------------------------------
static bool read_shape_file(const char *path, int *gM, int *gN, int *gK) {
    char buf[128];
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return false;
    buf[n] = '\0';
    return sscanf(buf, "%d %d %d", gM, gN, gK) == 3 &&
           *gM > 0 && *gN > 0 && *gK > 0;
}

// ===========================================================================
// RES_CHECK mode
// ===========================================================================
#ifdef RES_CHECK

int main() {
    int gM, gN, gK;

    #define SHAPE_PATH CHK_DIR "/shape.txt"
    if (!read_shape_file(SHAPE_PATH, &gM, &gN, &gK)) {
        printf("MATMUL_TEST: failed to read shape from %s\n", SHAPE_PATH);
        return 1;
    }
    printf("MATMUL_TEST RES_CHECK: M=%d N=%d K=%d (tM=%d tN=%d tK=%d)\n",
           gM, gN, gK, tilM, tilN, tilK);

    __half *src0 = (__half *)malloc(gM * gK * sizeof(__half));
    __half *src1 = (__half *)malloc(gK * gN * sizeof(__half));
    __half *dst  = (__half *)malloc(gM * gN * sizeof(__half));
    if (!src0 || !src1 || !dst) {
        printf("MATMUL_TEST: malloc failed\n");
        return 1;
    }

    #define SRC0_PATH CHK_DIR "/src0.bin"
    #define SRC1_PATH CHK_DIR "/src1.bin"
    readBinaryFile(SRC0_PATH, (uint8_t *)src0, gM * gK * sizeof(__half));
    readBinaryFile(SRC1_PATH, (uint8_t *)src1, gK * gN * sizeof(__half));
    memset(dst, 0, gM * gN * sizeof(__half));

    BENCHSTART;
    matmul_test<__half, tilM, tilN, tilK>(dst, src0, src1, gM, gN, gK);
    BENCHEND;

    // Convert __half output to float32 so the host-side gfrun compare
    // (which reads res.bin as np.float32) works unchanged.
    float *dst_f32 = (float *)malloc(gM * gN * sizeof(float));
    if (!dst_f32) {
        printf("MATMUL_TEST: malloc failed for dst_f32\n");
        return 1;
    }
    for (int i = 0; i < gM * gN; ++i)
        dst_f32[i] = (float)dst[i];

    #define RES_PATH CHK_DIR "/res.bin"
    writeBinaryFile(RES_PATH, (uint8_t *)dst_f32, gM * gN * sizeof(float));

    free(src0); free(src1); free(dst); free(dst_f32);
    return 0;
}

// ===========================================================================
// SELF_VERIFY mode
// ===========================================================================
#elif defined(SELF_VERIFY)

int main() {
    int gM, gN, gK;

    if (!read_shape_file("shape.txt", &gM, &gN, &gK)) {
        gM = 64; gN = 64; gK = 64;
        printf("MATMUL_TEST SELF_VERIFY: shape.txt not found, "
               "using default M=%d N=%d K=%d\n", gM, gN, gK);
    }
    printf("MATMUL_TEST SELF_VERIFY: M=%d N=%d K=%d (tM=%d tN=%d tK=%d), "
           "verify_elems=%d\n",
           gM, gN, gK, tilM, tilN, tilK, VERIFY_ELEMS);

    __half *src0 = (__half *)malloc(gM * gK * sizeof(__half));
    __half *src1 = (__half *)malloc(gK * gN * sizeof(__half));
    __half *dst  = (__half *)malloc(gM * gN * sizeof(__half));
    if (!src0 || !src1 || !dst) {
        printf("MATMUL_TEST: malloc failed\n");
        return 1;
    }

    // Deterministic all-1 inputs: A[i][k]=1, B[k][j]=1 => C[i][j]=gK.
    // Accumulation in float is exact for integer addends; the only rounding
    // is the final float->half cast, checked with atol/rtol below.
    for (int i = 0; i < gM * gK; ++i) src0[i] = (__half)1.0f;
    for (int i = 0; i < gK * gN; ++i) src1[i] = (__half)1.0f;
    memset(dst, 0, gM * gN * sizeof(__half));

    BENCHSTART;
    matmul_test<__half, tilM, tilN, tilK>(dst, src0, src1, gM, gN, gK);
    BENCHEND;

    const float golden = (float)gK;
    const float atol = 0.5f;
    const float rtol = 5e-3f;
    const float tol = atol + rtol * fabsf(golden);

    int total = gM * gN;
    int check_count = total < VERIFY_ELEMS ? total : VERIFY_ELEMS;
    int mismatches = 0;
    float max_err = 0.0f;

    for (int i = 0; i < check_count; ++i) {
        float v = (float)dst[i];
        float err = fabsf(v - golden);
        if (err > max_err) max_err = err;
        if (err > tol) ++mismatches;
    }

    if (mismatches == 0) {
        printf("MATMUL_TEST PASS: checked %d/%d cells, golden=%g, "
               "max_err=%.6f\n", check_count, total, (double)golden, max_err);
    } else {
        printf("MATMUL_TEST FAIL: %d/%d mismatches, golden=%g, "
               "max_err=%.6f\n", mismatches, check_count,
               (double)golden, max_err);
    }

    free(src0); free(src1); free(dst);
    return mismatches > 255 ? 255 : mismatches;
}

// ===========================================================================
// Default: smoke test (no verification)
// ===========================================================================
#else

int main() {
    int gM = 64, gN = 64, gK = 64;
    printf("MATMUL_TEST: M=%d N=%d K=%d (tM=%d tN=%d tK=%d) [no verify]\n",
           gM, gN, gK, tilM, tilN, tilK);

    __half *src0 = (__half *)malloc(gM * gK * sizeof(__half));
    __half *src1 = (__half *)malloc(gK * gN * sizeof(__half));
    __half *dst  = (__half *)malloc(gM * gN * sizeof(__half));
    if (!src0 || !src1 || !dst) {
        printf("MATMUL_TEST: malloc failed\n");
        return 1;
    }

    for (int i = 0; i < gM * gK; ++i) src0[i] = (__half)1.0f;
    for (int i = 0; i < gK * gN; ++i) src1[i] = (__half)1.0f;
    memset(dst, 0, gM * gN * sizeof(__half));

    BENCHSTART;
    matmul_test<__half, tilM, tilN, tilK>(dst, src0, src1, gM, gN, gK);
    BENCHEND;

    printf("MATMUL_TEST done. dst[0]=%g (expected %d)\n",
           (double)(float)dst[0], gK);

    free(src0); free(src1); free(dst);
    return 0;
}

#endif
