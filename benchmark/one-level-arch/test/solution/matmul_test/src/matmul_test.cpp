#include <common/pto_tileop.hpp>
#include <cstring>
#include "fileop.h"
#include "common.h"
#include "benchmark.h"

// matmul_test 测试入口:
//   - 动态 shape：gM/gN/gK 以运行时 int 传给内核（内核模板只有 tM/tN/tK）
//   - 输入 fp16 (__half)，累加 fp32（CUBE 累加器），输出 fp16 (__half)
//   - PTO 0.58.3 起 TMATMUL 强制 CUBE CELL layout 且拒绝动态 valid 形状，
//     尾块改用零填充：A [Mpad,Kpad]、B [Kpad,Npad]、C [Mpad,Npad]
//     （Mpad/Kpad/Npad = ceil(dim/tile)*tile），填充区清零，逻辑结果取
//     前 [gM,gN]。
//   - 接口带 leading dimension：lda=Kpad、ldb=Npad、ldc=Npad。
//   - 精度验证: res_check=on 构建后读取 src0.bin/src1.bin(f16, 逻辑 [gM,gK]/
//     [gK,gN]) 并写出 res.bin(f16, 逻辑 [gM,gN])，由 verify_matmul_test.py
//     与 torch/numpy fp32 golden 比对。
//   - 性能验证: BENCHSTART/BENCHEND 标记算子区间（gfsim 周期 / gfrun 工作量）

#ifndef globM
#define globM 256
#endif

#ifndef globN
#define globN 256
#endif

#ifndef globK
#define globK 256
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

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

#include "solution/matmul_test/matmul_test.hpp"

// 确定性 LCG（PCG 风格 64-bit）：host 侧 verify 脚本按相同算法位级复现。
static inline uint64_t lcg_next(uint64_t *s) {
  *s = *s * 6364136223846793005ull + 1442695040888963407ull;
  return *s;
}

// 构造 hard 随机 fp16 位模式：sign 随机 | exp ∈ [8,16)（幅值 [2^-7, 2)）|
// mantissa 全随机。直接按位写入，不经浮点转换（避免 fcvt 舍入语义差异）。
// 只填逻辑区 [rows_logic, cols_logic]；其余（填充区）保持调用方清好的 0。
static void fill_hard_half(__half *buf, int ld, int /*rows_pad*/,
                           int rows_logic, int cols_logic, uint64_t seed) {
  for (int r = 0; r < rows_logic; ++r) {
    for (int c = 0; c < cols_logic; ++c) {
      const uint64_t rv = lcg_next(&seed);
      const uint16_t sign = (uint16_t)(rv & 1ull);
      const uint16_t exp = (uint16_t)(8ull + ((rv >> 1) & 7ull));
      const uint16_t man = (uint16_t)((rv >> 4) & 0x3FFull);
      const uint16_t bits =
          (uint16_t)((sign << 15) | (exp << 10) | man);
      memcpy(&buf[(size_t)r * ld + c], &bits, sizeof(bits));
    }
  }
}

int main() {
  using dtype = __half;

  // 填充后的缓冲尺寸（PTO 0.58.3 CUBE 静态满宽瓦片 + 零填充尾块）。
  const int gMpad = ((globM + tilM - 1) / tilM) * tilM;
  const int gKpad = ((globK + tilK - 1) / tilK) * tilK;
  const int gNpad = ((globN + tilN - 1) / tilN) * tilN;

  dtype *src0;
  dtype *src1;
  dtype *dst;

  dtype src0p[gMpad * gKpad + 2 * ALIGN];
  dtype src1p[gKpad * gNpad + 2 * ALIGN];
  dtype dstp[gMpad * gNpad + 2 * ALIGN];
  // fp32 scratch：CUBE 累加器 -> Vec fp16 的 GM 往返中转（一个 tile 大小）。
  float scratchp[tilM * tilN + 2 * ALIGN];

  src0 = (dtype *)(((uint64_t)src0p & ALIGN_MASK) + ALIGN);
  src1 = (dtype *)(((uint64_t)src1p & ALIGN_MASK) + ALIGN);
  dst = (dtype *)(((uint64_t)dstp & ALIGN_MASK) + ALIGN);
  float *scratch = (float *)(((uint64_t)scratchp & ALIGN_MASK) + ALIGN);

  // 填充区清零（A 的 M/K 尾、B 的 K/N 尾、C 的 M/N 尾）。
  memset(src0, 0, (size_t)gMpad * gKpad * sizeof(dtype));
  memset(src1, 0, (size_t)gKpad * gNpad * sizeof(dtype));
  memset(dst, 0, (size_t)gMpad * gNpad * sizeof(dtype));

#ifdef RES_CHECK
  #define SRC0_PATH CHK_DIR "/src0.bin"
  #define SRC1_PATH CHK_DIR "/src1.bin"
  // 逻辑 [gM,gK]/[gK,gN] 数据展开到带 leading dimension 的填充缓冲。
  {
    dtype sbuf[globM * globK];
    readBinaryFile(SRC0_PATH, (uint8_t *)sbuf, globM * globK * sizeof(dtype));
    for (int r = 0; r < globM; ++r)
      for (int c = 0; c < globK; ++c)
        src0[(size_t)r * gKpad + c] = sbuf[(size_t)r * globK + c];
  }
  {
    dtype sbuf[globK * globN];
    readBinaryFile(SRC1_PATH, (uint8_t *)sbuf, globK * globN * sizeof(dtype));
    for (int r = 0; r < globK; ++r)
      for (int c = 0; c < globN; ++c)
        src1[(size_t)r * gNpad + c] = sbuf[(size_t)r * globN + c];
  }
#else
  // 非 res_check（功能跑通验证）时用确定性 LCG 填充"hard"随机 fp16：
  // 直接构造 fp16 位模式（符号随机、指数域限 [8,16) 即幅值 [2^-7,2)、
  // 尾数域全随机），不经任何浮点转换，host 侧可按相同 LCG 位级复现。
  // 相比旧的 %17/%13 对齐 pattern（所有中间值在 fp32/fp16 均可精确表示，
  // 完全不触发舍入），hard 随机值真正检验 fp32 累加舍入与 fp16 舍入路径。
  // 填充顺序：A 按逻辑 [gM,gK] 行优先、B 按逻辑 [gK,gN] 行优先；填充区恒 0。
  fill_hard_half(src0, gKpad, gMpad, globM, globK, 0x123456789ABCDEF0ull);
  fill_hard_half(src1, gNpad, gKpad, globK, globN, 0x0FEDCBA987654321ull);
#endif

  // 运行期 shape（动态）：内核模板只带 tile size。
  int gM = globM;
  int gN = globN;
  int gK = globK;

  // leading dimension：填充后 A 行距 Kpad、B/C 行距 Npad。
  int lda = gKpad;
  int ldb = gNpad;
  int ldc = gNpad;

  BENCHSTART;
  matmul_test<tilM, tilN, tilK>(dst, src0, src1, scratch, gM, gN, gK, lda, ldb,
                                ldc);
  BENCHEND;

#ifdef RES_CHECK
  // 输出只写逻辑 gM x gN（丢弃填充区）：从带填充 stride 的输出缓冲拷贝到
  // 连续逻辑缓冲，再一次写入 res.bin。
  #define RES_PATH CHK_DIR "/res.bin"
  dtype resbuf[globM * globN];
  for (int r = 0; r < globM; ++r) {
    for (int c = 0; c < globN; ++c) {
      resbuf[(size_t)r * globN + c] = dst[(size_t)r * ldc + c];
    }
  }
  writeBinaryFile(RES_PATH, (uint8_t *)resbuf, globM * globN * sizeof(dtype));
#endif

  return 0;
}
