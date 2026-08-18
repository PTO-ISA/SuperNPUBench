// matmul_validate.cpp
//
// Numerical-precision validation harness for the PTO one-level mask matmul.
//
// Unlike matmul.cpp (which reads src0.bin/src1.bin under RES_CHECK), this
// variant fills both input matrices with all-1 arrays in-memory and checks the
// result against the analytic golden value:
//
//     A[i][k] = 1, B[k][j] = 1   =>   C[i][j] = sum_k 1*1 = globK
//
// Because every addend is exactly 1.0f and the partial sums 1..globK are all
// exactly representable in float (globK << 2^24), the golden is exact
// regardless of accumulation order, so the check dst[i*globN+j] == (float)globK
// is bit-exact. Tile-aligned default sizes (multiples of tilM/tilN/tilK) are
// used so no partial-tile masking reasoning is needed.
//
// Build: make TYPE=MATMUL_VALIDATE M=.. N=.. K=.. tM=.. tN=.. tK=.. COMPILER_DIR=<latest> [diss]

#include <common/pto_tileop.hpp>
#include <cstring>
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

#ifndef globM
#define globM 64
#endif

#ifndef globN
#define globN 64
#endif

#ifndef globK
#define globK 64
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

#ifndef Batch
#define Batch 1
#endif

#include "matmul/matmul.hpp"

// dst is a file-scope global so its address is fixed in .bss; this lets the
// result be dumped verbatim via `gfrun --dump-memory` for printf-independent,
// bit-exact verification (the bare-metal printf in this harness mangles %d).
float g_dst[globM * globN];

int main() {
  // Inputs as plain stack arrays (FP32 mask path needs no 4 KiB alignment,
  // same as matmul.cpp's MASK_FP32 branch).
  float src0[globM * globK];
  float src1[globK * globN];
  float *dst = g_dst;

  // All-1 inputs: no file read. dst zeroed so any uncomputed cell is obvious.
  for (int i = 0; i < globM * globK; ++i) src0[i] = 1.0f;
  for (int i = 0; i < globK * globN; ++i) src1[i] = 1.0f;
  for (int i = 0; i < globM * globN; ++i) dst[i] = 0.0f;

  // C = A * B  (mask matmul, any-shape tiled MAC).
  matmul_mask_tileop<float, globM, globN, globK, tilM, tilN, tilK>(dst, src0,
                                                                   src1);

  // Golden: C[i][j] = globK for every (i,j).
  const float golden = (float)globK;
  int mismatches = 0;
  int first_bad_i = -1, first_bad_j = -1;
  float first_bad_val = 0.0f;
  for (int i = 0; i < globM; ++i) {
    for (int j = 0; j < globN; ++j) {
      float v = dst[i * globN + j];
      if (v != golden) {
        if (mismatches == 0) {
          first_bad_i = i;
          first_bad_j = j;
          first_bad_val = v;
        }
        ++mismatches;
      }
    }
  }

  if (mismatches == 0) {
    printf("MATMUL_VALIDATE PASS: M=%d N=%d K=%d (tM=%d tN=%d tK=%d), "
           "all %d cells == %g\n",
           globM, globN, globK, tilM, tilN, tilK, globM * globN,
           (double)golden);
  } else {
    printf("MATMUL_VALIDATE FAIL: %d/%d mismatches; first bad C[%d][%d]=%g "
           "(expected %g)\n",
           mismatches, globM * globN, first_bad_i, first_bad_j,
           (double)first_bad_val, (double)golden);
  }
  return mismatches ? 1 : 0;
}
