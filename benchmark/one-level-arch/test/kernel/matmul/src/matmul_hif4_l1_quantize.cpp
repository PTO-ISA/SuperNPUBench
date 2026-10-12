#include "basic_op/matmul/matmul_hif4_l1_quantize.hpp"

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#ifdef LINX_GROUP_RUNTIME
#include <common/linx_group_runtime.h>
#endif

#include <cstdint>

#ifndef DTYPE
#define DTYPE float
#endif
#ifndef globM
#define globM 128
#endif
#ifndef globN
#define globN 64
#endif
#ifndef globK
#define globK 256
#endif
#ifndef tilM
#define tilM 128
#endif
#ifndef tilN
#define tilN 64
#endif
#ifndef tilK
#define tilK 128
#endif
#ifndef Batch
#define Batch 1
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

constexpr int kHif4Group = 64;
constexpr int kScaleCols = globN / kHif4Group;
using dtype = DTYPE;

struct MatmulHif4L1Context {
  dtype *src0;
  dtype *src1;
  float *c_scratch;
  __fp4_hif4x2 *data;
  uint8_t *scale;
};

extern "C" int __linx_group_worker_main(uint32_t pe_id, void *opaque) {
  (void)pe_id;
  auto *context = static_cast<MatmulHif4L1Context *>(opaque);

  BENCHSTART;
  for (int b = 0; b < Batch; ++b) {
    matmul_hif4_l1_quantize<dtype, globM, globN, globK, tilM, tilN, tilK>(
        reinterpret_cast<__fp4_hif4x2 *>(
            reinterpret_cast<uint8_t *>(context->data) +
            b * globM * globN / 2),
        context->scale + b * globM * kScaleCols,
        context->c_scratch + b * globM * globN,
        context->src0 + b * globM * globK,
        context->src1 + b * globN * globK);
  }
  BENCHEND;
  return 0;
}

int main() {
  constexpr uint32_t kIoTid = 0;
  const uint32_t tid = get_thread_idx();

  static dtype src0_storage[Batch * globM * globK + 2 * ALIGN];
  // B is stored B-major as [Batch, N, K].
  static dtype src1_storage[Batch * globN * globK + 2 * ALIGN];
  static float c_storage[Batch * globM * globN + 2 * ALIGN];
  // TCVT/TSTORE packs two logical HiF4 lanes per byte.
  static uint8_t data_storage[Batch * globM * globN / 2 + 2 * ALIGN];
  static uint8_t scale_storage[Batch * globM * kScaleCols + 2 * ALIGN];

  auto *src0 = reinterpret_cast<dtype *>(
      (reinterpret_cast<uint64_t>(src0_storage) & ALIGN_MASK) + ALIGN);
  auto *src1 = reinterpret_cast<dtype *>(
      (reinterpret_cast<uint64_t>(src1_storage) & ALIGN_MASK) + ALIGN);
  auto *c_scratch = reinterpret_cast<float *>(
      (reinterpret_cast<uint64_t>(c_storage) & ALIGN_MASK) + ALIGN);
  auto *data_bytes = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uint64_t>(data_storage) & ALIGN_MASK) + ALIGN);
  auto *scale = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uint64_t>(scale_storage) & ALIGN_MASK) + ALIGN);
  auto *data = reinterpret_cast<__fp4_hif4x2 *>(data_bytes);

#ifdef RES_CHECK
  static MultiThreadResCheckSync res_check_sync{};
  if (tid == kIoTid) {
    readBinaryFile(CHK_DIR "/src0.bin", reinterpret_cast<uint8_t *>(src0),
                   Batch * globM * globK * sizeof(dtype));
    readBinaryFile(CHK_DIR "/src1.bin", reinterpret_cast<uint8_t *>(src1),
                   Batch * globN * globK * sizeof(dtype));
  }
#ifndef LINX_GROUP_RUNTIME
  res_check_publish_inputs(res_check_sync, tid);
#endif
#endif

  MatmulHif4L1Context context{src0, src1, c_scratch, data, scale};
#ifdef LINX_GROUP_RUNTIME
  const int status = linx_group_run(&context);
#else
  const int status = __linx_group_worker_main(0, &context);
#endif

#ifdef RES_CHECK
#ifndef LINX_GROUP_RUNTIME
  res_check_wait_for_all(res_check_sync, tid);
#endif
  if (tid == kIoTid) {
    writeBinaryFile(CHK_DIR "/data.bin", data_bytes,
                    Batch * globM * globN / 2);
    writeBinaryFile(CHK_DIR "/scale.bin", scale,
                    Batch * globM * kScaleCols);
  }
#endif

  return status;
}
