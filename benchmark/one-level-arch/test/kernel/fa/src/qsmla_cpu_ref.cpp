// Stage-0 CPU reference for contiguous FP16 QSMLA SWA.
// Inputs are rounded to IEEE FP16, while dot products, softmax and output use FP32.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#ifndef QB
#define QB 1
#endif
#ifndef QS1
#define QS1 64
#endif
#ifndef QS2
#define QS2 128
#endif
#ifndef QN1
#define QN1 1
#endif
#ifndef QN2
#define QN2 1
#endif
#ifndef QD
#define QD 512
#endif
#ifndef QK
#define QK 128
#endif
#ifndef QTM
#define QTM 32
#endif
#ifndef QTK
#define QTK 32
#endif
#ifndef QTD
#define QTD 64
#endif
#ifndef QWIN_LEFT
#define QWIN_LEFT 1
#endif
#ifndef QWIN_RIGHT
#define QWIN_RIGHT 1
#endif
#ifndef QSOFTMAX_SCALE
#define QSOFTMAX_SCALE 0.125f
#endif
#ifndef QCASE_NAME
#define QCASE_NAME "baseline_swa"
#endif
#ifndef QOUTPUT_ROOT
#define QOUTPUT_ROOT "."
#endif
#ifndef QLAYOUT_BSND
#define QLAYOUT_BSND 1
#endif
#ifndef QKV_LAYOUT_BSND
#define QKV_LAYOUT_BSND 1
#endif

static_assert(QB > 0, "B must be positive");
static_assert(QS1 >= 0 && QS2 >= 0, "S1/S2 must be non-negative");
static_assert(QN1 > 0 && QN2 > 0 && QN1 % QN2 == 0, "N1 must be divisible by N2");
static_assert(QD > 0 && QK >= 0, "D must be positive and K non-negative");
static_assert(QTM > 0 && QTK > 0 && QTD > 0, "tile sizes must be positive");
static_assert(QWIN_LEFT >= -1 && QWIN_RIGHT >= -1, "window bounds must be -1 or non-negative");

static uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x007fffffu;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        mantissa += 0x00000fffu + ((mantissa >> 13) & 1u);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    mantissa += 0x00000fffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x00800000u) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

static float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static void init_deterministic_fp16(std::vector<uint16_t>& storage, std::vector<float>& values, int seed) {
    for (size_t i = 0; i < storage.size(); ++i) {
        const float source = static_cast<float>((static_cast<int64_t>(i) * 31 + seed * 17) % 100) / 100.0f - 0.5f;
        storage[i] = float_to_half(source);
        values[i] = half_to_float(storage[i]);
    }
}

static bool swa_valid(int q_pos, int kv_pos) {
    const int threshold = QS2 - QS1 + q_pos + 1;
    const int lo = QWIN_LEFT == -1 ? 0 : threshold - QWIN_LEFT - 1;
    const int hi = QWIN_RIGHT == -1 ? QS2 : threshold + QWIN_RIGHT;
    return kv_pos >= std::max(0, lo) && kv_pos < std::min(QS2, hi);
}

static size_t q_offset(int b, int head, int token, int dim) {
    return (((static_cast<size_t>(b) * QS1 + token) * QN1 + head) * QD + dim);
}

static size_t kv_offset(int b, int head, int token, int dim) {
    return (((static_cast<size_t>(b) * QS2 + token) * QN2 + head) * QD + dim);
}

static bool write_binary(const std::filesystem::path& path, const void* data, size_t bytes) {
    FILE* file = std::fopen(path.string().c_str(), "wb");
    if (file == nullptr) return false;
    const bool ok = std::fwrite(data, 1, bytes, file) == bytes;
    std::fclose(file);
    return ok;
}

int main() {
    const size_t q_count = static_cast<size_t>(QB) * QN1 * QS1 * QD;
    const size_t kv_count = static_cast<size_t>(QB) * QN2 * QS2 * QD;
    const size_t out_count = q_count;
    std::vector<uint16_t> q_storage(q_count), kv_storage(kv_count);
    std::vector<float> q(q_count), kv(kv_count), out(out_count, 0.0f);
    std::vector<float> scores(QS2);

    init_deterministic_fp16(q_storage, q, 1);
    init_deterministic_fp16(kv_storage, kv, 2);

    constexpr int group_size = QN1 / QN2;
    for (int b = 0; b < QB; ++b) {
        for (int q_head = 0; q_head < QN1; ++q_head) {
            const int kv_head = q_head / group_size;
            for (int q_pos = 0; q_pos < QS1; ++q_pos) {
                float row_max = -std::numeric_limits<float>::infinity();
                for (int kv_pos = 0; kv_pos < QS2; ++kv_pos) {
                    if (!swa_valid(q_pos, kv_pos)) {
                        scores[kv_pos] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < QD; ++d) {
                        dot += q[q_offset(b, q_head, q_pos, d)] * kv[kv_offset(b, kv_head, kv_pos, d)];
                    }
                    scores[kv_pos] = dot * QSOFTMAX_SCALE;
                    row_max = std::max(row_max, scores[kv_pos]);
                }
                if (!std::isfinite(row_max)) continue;

                float denominator = 0.0f;
                for (int kv_pos = 0; kv_pos < QS2; ++kv_pos) {
                    if (std::isfinite(scores[kv_pos])) denominator += std::exp(scores[kv_pos] - row_max);
                }
                for (int d = 0; d < QD; ++d) {
                    float numerator = 0.0f;
                    for (int kv_pos = 0; kv_pos < QS2; ++kv_pos) {
                        if (!std::isfinite(scores[kv_pos])) continue;
                        const float probability = std::exp(scores[kv_pos] - row_max) / denominator;
                        numerator += probability * kv[kv_offset(b, kv_head, kv_pos, d)];
                    }
                    out[q_offset(b, q_head, q_pos, d)] = numerator;
                }
            }
        }
    }

    const std::filesystem::path output_dir = std::filesystem::path(QOUTPUT_ROOT) / QCASE_NAME;
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    if (error) {
        std::fprintf(stderr, "Failed to create %s: %s\n", output_dir.string().c_str(), error.message().c_str());
        return 1;
    }
    if (!write_binary(output_dir / "q.fp16.bin", q_storage.data(), q_storage.size() * sizeof(uint16_t)) ||
        !write_binary(output_dir / "kv.fp16.bin", kv_storage.data(), kv_storage.size() * sizeof(uint16_t)) ||
        !write_binary(output_dir / "out.fp32.bin", out.data(), out.size() * sizeof(float))) {
        std::fprintf(stderr, "Failed to write reference data under %s\n", output_dir.string().c_str());
        return 1;
    }

    FILE* manifest = std::fopen((output_dir / "manifest.txt").string().c_str(), "w");
    if (manifest == nullptr) return 1;
    std::fprintf(manifest,
        "case=%s\nB=%d S1=%d S2=%d N1=%d N2=%d D=%d K=%d\n"
        "Tm=%d Tk=%d Td=%d win_left=%d win_right=%d softmax_scale=%.9g\n"
        "q_layout=%s kv_layout=%s\n"
        "q_dtype=fp16 kv_dtype=fp16 accumulation=fp32 out_dtype=fp32\n",
        QCASE_NAME, QB, QS1, QS2, QN1, QN2, QD, QK,
        QTM, QTK, QTD, QWIN_LEFT, QWIN_RIGHT, static_cast<double>(QSOFTMAX_SCALE),
        "BSND", "BSND");
    std::fclose(manifest);

    std::printf("QSMLA_STAGE0 case=%s output=%s elements=%zu\n",
        QCASE_NAME, output_dir.string().c_str(), out.size());
    return 0;
}
