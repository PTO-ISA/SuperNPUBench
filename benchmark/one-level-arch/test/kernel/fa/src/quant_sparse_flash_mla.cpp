#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
// #include "fa/quant_sparse_flash_mla_pto.hpp"
#if defined(QSMLA_USE_TADD) || defined(QSMLA_USE_TADD_4PE)
#include "fa/quant_sparse_flash_mla_tadd_pto.hpp"
#else
#include "fa/quant_sparse_flash_mla_onepass_pto.hpp"
#endif

#ifndef Tbatch
#define B 1
#else
#define B Tbatch
#endif

#ifndef Tn1
#define N1 1
#else
#define N1 Tn1
#endif

#ifndef Tn2
#define N2 1
#else
#define N2 Tn2
#endif

#ifndef Ts1
#define s1 64
#else
#define s1 Ts1
#endif

#ifndef Ts2
#define s2 128
#else
#define s2 Ts2
#endif

#ifndef Td
#define D 512
#else
#define D Td
#endif

#ifndef Tm
#define kTm 32
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 32
#else
#define kTk Tk
#endif

#ifndef Td_block
#define kTd 64
#else
#define kTd Td_block
#endif

#ifndef Tsoftmax_scale
#define softmax_scale_val 0.125f
#else
#define softmax_scale_val Tsoftmax_scale
#endif

#ifndef Twleft
#define win_left 1
#else
#define win_left Twleft
#endif

#ifndef Twright
#define win_right 1
#else
#define win_right Twright
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024
#define MAP_MEM_BASE 0x4000802000ULL

constexpr uint64_t align_up_4k(uint64_t bytes) {
    return (bytes + ALIGN - 1) & ALIGN_MASK;
}

static void init_deterministic(__half* data, int count, int seed) {
    for (int i = 0; i < count; ++i) {
        float val = ((float)((i * 31 + seed * 17) % 100)) / 100.0f - 0.5f;
        data[i] = (__half)val;
    }
}

int main(){
    using qdtype = __half;
    using kvdtype = __half;
    using odttype = __half;
    constexpr int group_size = N1 / N2;
    constexpr int g_slice_max = group_size < 64 ? group_size : 64;
    using Config = QsmlaConfig<
        B, s1, s2, N1, N2, D, 0, kTm, kTk, kTd, g_slice_max>;

    qdtype qp[B*s1*N1*D + 2*ALIGN];
    kvdtype kvp[B*s2*N2*D + 2*ALIGN];

    qdtype* q = (qdtype*)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    kvdtype* kv = (kvdtype*)(((uint64_t)kvp & ALIGN_MASK) + ALIGN);

    odttype* out = (odttype*)MAP_MEM_BASE;

#ifdef QSMLA_USE_TADD_4PE
    // Cooperative scratch must live in shared GM rather than a PE-private
    // function stack.  Each PE computes and receives these identical mapped
    // addresses, matching the shared-workspace contract used by the 4-PE FA.
    constexpr uint64_t out_bytes =
        static_cast<uint64_t>(B) * s1 * N1 * D * sizeof(odttype);
    constexpr uint64_t score_bytes =
        static_cast<uint64_t>(kTm) * kTk * sizeof(float);
    constexpr uint64_t prob_bytes =
        static_cast<uint64_t>(kTm) * kTk * sizeof(qdtype);
    constexpr uint64_t score_addr = MAP_MEM_BASE + align_up_4k(out_bytes);
    constexpr uint64_t prob_addr = score_addr + align_up_4k(score_bytes);
    constexpr uint64_t pv_addr = prob_addr + align_up_4k(prob_bytes);
    float* score_scratch = reinterpret_cast<float*>(score_addr);
    qdtype* prob_scratch = reinterpret_cast<qdtype*>(prob_addr);
    float* pv_scratch = reinterpret_cast<float*>(pv_addr);
#endif

    init_deterministic(q, B*s1*N1*D, 1);
    init_deterministic(kv, B*s2*N2*D, 2);

    BENCHSTART;
#ifdef QSMLA_USE_TADD_4PE
    static_assert(N1 > 1,
                  "tadd_4pe is a BSND G-slice implementation");
    quant_sparse_flash_mla_swa_tadd_4pe_bsnd_pto<
        qdtype, kvdtype, odttype, Config>(
            out, q, kv,
            softmax_scale_val,
            win_left,
            win_right,
            (float*)nullptr,
            (float*)nullptr,
            (int*)nullptr,
            (int*)nullptr,
            (int*)nullptr,
            (int*)nullptr,
            (int*)nullptr,
            (int*)nullptr,
            (float*)nullptr,
            (int*)nullptr,
            (float*)nullptr,
            score_scratch,
            prob_scratch,
            pv_scratch);
#else
    if constexpr (N1 == 1 && N2 == 1) {
#ifdef QSMLA_USE_TADD
        quant_sparse_flash_mla_swa_tadd_config_pto<
#else
        quant_sparse_flash_mla_swa_onepass_config_pto<
#endif
            qdtype, kvdtype, odttype, Config>(
                out, q, kv,
                softmax_scale_val,
                win_left,
                win_right,
                (float*)nullptr,   // q_descale
                (float*)nullptr,   // ori_kv_descale
                (int*)nullptr,     // ori_sparse_indices
                (int*)nullptr,     // ori_block_table
                (int*)nullptr,     // cu_seqlens_q
                (int*)nullptr,     // cu_seqlens_ori_kv
                (int*)nullptr,     // seqused_q
                (int*)nullptr,     // seqused_ori_kv
                (float*)nullptr,   // sinks
                (int*)nullptr,     // metadata
                (float*)nullptr    // softmax_lse
            );
    } else {
#ifdef QSMLA_USE_TADD
        quant_sparse_flash_mla_swa_tadd_bsnd_pto<
#else
        quant_sparse_flash_mla_swa_onepass_bsnd_pto<
#endif
            qdtype, kvdtype, odttype, Config>(
                out, q, kv,
                softmax_scale_val,
                win_left,
                win_right,
                (float*)nullptr,   // q_descale
                (float*)nullptr,   // ori_kv_descale
                (int*)nullptr,     // ori_sparse_indices
                (int*)nullptr,     // ori_block_table
                (int*)nullptr,     // cu_seqlens_q
                (int*)nullptr,     // cu_seqlens_ori_kv
                (int*)nullptr,     // seqused_q
                (int*)nullptr,     // seqused_ori_kv
                (float*)nullptr,   // sinks
                (int*)nullptr,     // metadata
                (float*)nullptr    // softmax_lse
            );
    }
#endif
    BENCHEND;

    return 0;
}
