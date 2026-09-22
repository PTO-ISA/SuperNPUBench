#include "basic_op/fa/fa_gmma_dynamic.hpp"

#include <cstdint>
#include <cmath>
#include <type_traits>
#include "benchmark.h"
#include "fileop.h"

#ifndef MATRIX_DTYPE
#define MATRIX_DTYPE float
#endif
#ifndef VECTOR_DTYPE
#define VECTOR_DTYPE float
#endif
#ifndef PACKED_FACTOR
#define PACKED_FACTOR 1
#endif

#ifndef Tsq
#define Tsq 128
#endif
#ifndef Tskv
#define Tskv 128
#endif
#ifndef FA_MAX_SQ
#define FA_MAX_SQ 256
#endif
#ifndef FA_MAX_SKV
#define FA_MAX_SKV 256
#endif
#ifndef Tm
#define Tm 128
#endif
#ifndef Tk
#define Tk 64
#endif
#ifndef FA_QD
#define FA_QD 128
#endif
#ifndef FA_VD
#define FA_VD 128
#endif

static_assert(Tsq <= FA_MAX_SQ && Tskv <= FA_MAX_SKV);
constexpr int kGroupM = Tm <= 128 ? Tm : 128;
constexpr int kStoredQD = FA_QD / PACKED_FACTOR;
constexpr int kStoredSkv = FA_MAX_SKV / PACKED_FACTOR;
static_assert(Tsq % kGroupM == 0 && FA_MAX_SQ % kGroupM == 0);
static_assert(Tskv % Tk == 0 && FA_MAX_SKV % Tk == 0);
static_assert(PACKED_FACTOR == 1,
              "dynamic FA test supports unpacked FP32/BF16/FP8 inputs");

using matrix_dtype = MATRIX_DTYPE;
using vector_dtype = VECTOR_DTYPE;

// Two different shapes are executed by the same ELF. The buffers for both
// cases are distinct, so all PEs can advance without a per-case barrier.
alignas(4096) static matrix_dtype q[2][FA_MAX_SQ * kStoredQD];
alignas(4096) static matrix_dtype k[2][FA_MAX_SKV * kStoredQD];
alignas(4096) static matrix_dtype v[2][kStoredSkv * FA_VD];
alignas(4096) static vector_dtype out[2][FA_MAX_SQ * FA_VD];
static FaGmmaTilingData shapes[2];
static volatile uint32_t inputs_ready = 0;
static volatile uint32_t shape_invalid = 0;
static volatile uint32_t done[4] = {};
volatile int fa_gmma_dynamic_errors = 0;

#ifdef FA_TRACE_ONLY
constexpr int kCaseCount = 1;
#else
constexpr int kCaseCount = 2;
#endif

int main() {
    const uint32_t tid = get_thread_idx();
    if (tid == 0) {
        shapes[0] = {Tsq, Tskv};
        shapes[1] = {FA_MAX_SQ, FA_MAX_SKV};
#ifdef RES_CHECK
        // shape.bin: two consecutive {int64_t sq, int64_t skv} records.
        // Replacing this file changes the shapes without rebuilding the ELF.
        readBinaryFile(CHK_DIR "/shape.bin", reinterpret_cast<uint8_t *>(shapes),
                       sizeof(shapes));
#endif
        // Check that each runtime shape fits the preallocated buffers and
        // consists of complete cooperative Q/KV tiles.
        for (int c = 0; c < kCaseCount; ++c) {
            if (shapes[c].sq <= 0 || shapes[c].skv <= 0 ||
                shapes[c].sq > FA_MAX_SQ || shapes[c].skv > FA_MAX_SKV ||
                shapes[c].sq % kGroupM != 0 || shapes[c].skv % Tk != 0)
                shape_invalid = 1;
        }
        // Trace builds only need each shared-load region to be materialized;
        // touching one nonzero element per tile avoids a long scalar setup.
        // Numerical builds fill the complete deterministic reference input.
        if (shape_invalid == 0) for (int c = 0; c < kCaseCount; ++c) {
#ifdef FA_TRACE_ONLY
            if constexpr (std::is_same_v<matrix_dtype, __fp8_e4m3>) {
                auto *qBits = reinterpret_cast<uint8_t *>(q[c]);
                auto *kBits = reinterpret_cast<uint8_t *>(k[c]);
                auto *vBits = reinterpret_cast<uint8_t *>(v[c]);
                for (int row = 0; row < shapes[c].sq; row += kGroupM)
                    qBits[row * kStoredQD] = 0x38;
                for (int row = 0; row < shapes[c].skv; row += Tk) {
                    kBits[row * kStoredQD] = 0x38;
                    vBits[row * FA_VD] = 0x38;
                }
            } else {
                for (int row = 0; row < shapes[c].sq; row += kGroupM)
                    q[c][row * kStoredQD] = static_cast<matrix_dtype>(1.0f);
                for (int row = 0; row < shapes[c].skv; row += Tk) {
                    k[c][row * kStoredQD] = static_cast<matrix_dtype>(1.0f);
                    v[c][row * FA_VD] = static_cast<matrix_dtype>(1.0f);
                }
            }
#else
            if constexpr (std::is_same_v<matrix_dtype, __fp8_e4m3>) {
                // Avoid unsupported scalar float->FP8 code generation in the
                // test driver. These are exact positive E4M3 encodings 0..6.
                constexpr uint8_t fp8[] = {
                    0x00, 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c};
                auto *qBits = reinterpret_cast<uint8_t *>(q[c]);
                auto *kBits = reinterpret_cast<uint8_t *>(k[c]);
                auto *vBits = reinterpret_cast<uint8_t *>(v[c]);
                for (int row = 0; row < shapes[c].sq; ++row)
                    qBits[row * kStoredQD] = fp8[1];
                for (int row = 0; row < shapes[c].skv; ++row) {
                    kBits[row * kStoredQD] = fp8[row % 5];
                    for (int col = 0; col < FA_VD; ++col)
                        vBits[row * FA_VD + col] = fp8[row % 7];
                }
            } else {
                for (int row = 0; row < shapes[c].sq; ++row)
                    q[c][row * kStoredQD] = static_cast<matrix_dtype>(1.0f);
                for (int row = 0; row < shapes[c].skv; ++row) {
                    k[c][row * kStoredQD] = static_cast<matrix_dtype>(
                        static_cast<float>(row % 5));
                    for (int col = 0; col < FA_VD; ++col) {
                        v[c][row * FA_VD + col] = static_cast<matrix_dtype>(
                            static_cast<float>(row % 7));
                    }
                }
            }
#endif
        }
        inputs_ready = 1;
    } else {
        while (inputs_ready == 0) {}
    }

    if (shape_invalid != 0) return 1;

    BENCHSTART;
    for (int c = 0; c < kCaseCount; ++c) {
        if (!fa_gmma_dynamic<matrix_dtype, vector_dtype, PACKED_FACTOR,
                             FA_QD, FA_VD, Tm, Tk>(
                out[c], q[c], k[c], v[c], &shapes[c])) {
            fa_gmma_dynamic_errors = fa_gmma_dynamic_errors + 1;
        }
    }
    BENCHEND;

#ifdef FA_TRACE_ONLY
    // The timing model does not make the post-benchmark software barrier
    // progress reliably. Trace builds stop after the measured kernel region.
    return 0;
#endif

    done[tid] = 1;
    if (tid == 0) {
        for (int pe = 0; pe < 4; ++pe) {
            while (done[pe] == 0) {}
        }
        for (int c = 0; c < kCaseCount; ++c) {
            float numerator = 0.0f;
            float denominator = 0.0f;
            const float scale = 1.0f / sqrtf(static_cast<float>(FA_QD));
            for (int row = 0; row < shapes[c].skv; ++row) {
                const float weight = expf(static_cast<float>(row % 5) * scale);
                numerator += weight * static_cast<float>(row % 7);
                denominator += weight;
            }
            const float expected = numerator / denominator;
            for (int row = 0; row < shapes[c].sq; ++row) {
                for (int col = 0; col < FA_VD; ++col) {
                    const float diff =
                        static_cast<float>(out[c][row * FA_VD + col]) - expected;
                    if (diff < -0.02f || diff > 0.02f)
                        fa_gmma_dynamic_errors = fa_gmma_dynamic_errors + 1;
                }
            }
        }
    }
    return fa_gmma_dynamic_errors != 0;
}
