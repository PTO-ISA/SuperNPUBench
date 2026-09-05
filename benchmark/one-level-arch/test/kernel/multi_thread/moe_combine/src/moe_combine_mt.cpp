#include <common/pto_tileop.hpp>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "benchmark.h"
#include "moe_combine_mt/moe_combine_mt.hpp"

using namespace supernpu::tile_isa;

using dtype = __bf16;

constexpr int BS = 8;
constexpr int H = 128;
constexpr int K = 4;
constexpr int NUM_EXPANDED = BS * K;
constexpr int TILE_W = 128;

static dtype expand_x[NUM_EXPANDED * H] __attribute__((aligned(4096))) = {};
static float expert_scales[NUM_EXPANDED * TILE_W] __attribute__((aligned(4096))) = {};
static int32_t expand_idx[NUM_EXPANDED * 3] __attribute__((aligned(4096))) = {};
static dtype window_data[NUM_EXPANDED * H] __attribute__((aligned(4096))) = {};
static float window_flag[NUM_EXPANDED * TILE_W] __attribute__((aligned(4096))) = {};
static uint32_t window_state[16] __attribute__((aligned(4096))) = {};
static float pred_buf[NUM_EXPANDED * TILE_W] __attribute__((aligned(4096))) = {};
static dtype out_buf[BS * H] __attribute__((aligned(4096))) = {};

// Keeps the non-leader park loop side-effecting so -O2 cannot drop it.
static volatile uint32_t sParkSink = 0;

static inline float bf16ToF32(uint16_t raw)
{
    uint32_t full = static_cast<uint32_t>(raw) << 16;
    float v;
    std::memcpy(&v, &full, 4);
    return v;
}

// Round-to-nearest-even fp32 -> bf16 bits (TCVT rounding contract).
static inline uint16_t f32ToBf16Bits(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

int main() {
    const uint32_t tid = get_thread_idx();

    // Input init runs redundantly on every PE (deterministic, identical
    // writes — same convention as group_token_vec_mt), so no extra barrier
    // is needed before the kernel reads expandX / expandIdx / expertScales.
    for (int i = 0; i < NUM_EXPANDED * H; i++) {
        float fval = static_cast<float>(i) * 0.1f;
        uint32_t bits; std::memcpy(&bits, &fval, 4);
        uint16_t raw = (uint16_t)(bits >> 16);
        std::memcpy(&expand_x[i], &raw, 2);
    }
    for (int tk = 0; tk < NUM_EXPANDED; tk++) {
        expand_idx[tk * 3 + 0] = 0;
        expand_idx[tk * 3 + 1] = tk / K;
        expand_idx[tk * 3 + 2] = tk % K;
    }
    // The kernel reads expertScales densely at [n*K + k]; fill the dense
    // prefix so every slot carries a real weight.
    for (int i = 0; i < NUM_EXPANDED; i++) {
        expert_scales[i] = 0.25f;
    }

    BENCHSTART;

    moe_combine_mt<dtype, dtype, BS, H, K, NUM_EXPANDED, TILE_W>(
        expand_x, expert_scales, expand_idx, window_data, window_flag,
        window_state, pred_buf, out_buf);

    BENCHEND;

    // Verification runs on PE0 only (all outputs are visible after the
    // kernel's final barrier). Non-leader PEs park: a worker reaching _end
    // first raises exit_group and would truncate PE0's verification mid-way
    // (the leader's exit ends the parked workers instead).
    if (tid != 0) {
        for (;;) {
            sParkSink = tid;
        }
    }

    // 1. out rows == fp32 scale-weighted sum of the K window slots,
    //    converted to bf16 (kernel accumulates in fp32 via TCVT/TMULS/TADD,
    //    then TCVT back; allow the last mantissa bit to round either way).
    for (int n = 0; n < BS; n++) {
        for (int j = 0; j < H; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                uint16_t raw;
                std::memcpy(&raw, &expand_x[(n * K + k) * H + j], 2);
                acc += expert_scales[n * K + k] * bf16ToF32(raw);
            }
            uint16_t expBits = f32ToBf16Bits(acc);
            uint16_t actBits;
            std::memcpy(&actBits, &out_buf[n * H + j], 2);
            uint16_t diff = expBits ^ actBits;
            if (diff != 0 && diff != 1) return 1;
        }
    }

    // 2. window flags all cleared by the reduce stage
    for (int i = 0; i < NUM_EXPANDED * TILE_W; i++) {
        if (window_flag[i] != 0.0f) return 2;
    }

    // 3. window state: toggle 0 -> 1, run flags, token-count writeback
    if (window_state[0] != 1u || window_state[1] != 1u ||
        window_state[2] != 0u || window_state[4] != static_cast<uint32_t>(BS)) {
        return 3;
    }

    // 4. predBuf carries the 1.0f flag copies from the last check round
    for (int slot = 0; slot < NUM_EXPANDED; slot++) {
        if (pred_buf[slot * TILE_W] != 1.0f) return 4;
    }

    return 0;
}
