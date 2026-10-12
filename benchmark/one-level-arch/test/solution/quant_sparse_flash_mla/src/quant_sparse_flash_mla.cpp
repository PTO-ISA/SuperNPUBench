#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#if defined(QSMLA_USE_TADD_4PE) || \
    defined(QSMLA_USE_HCA_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_4PE) || \
    defined(QSMLA_USE_ORI_SPARSE_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE)
#define QSMLA_USE_UNIFIED_TADD_4PE
#include "solution/quant_sparse_flash_mla/quant_sparse_flash_mla.hpp"
#ifdef QSMLA_USE_DYNAMIC_SHAPE
#include "solution/quant_sparse_flash_mla/quant_sparse_flash_mla_v2.hpp"
#endif
#elif defined(QSMLA_USE_CSA_TADD_1PE)
#include "solution/quant_sparse_flash_mla/quant_sparse_flash_mla_csa_1pe.hpp"
#else
#include "solution/quant_sparse_flash_mla/quant_sparse_flash_mla_tadd.hpp"
#ifdef QSMLA_USE_HIF8
#error "Single-PE tadd supports FP16 only; use the unified four-PE kernel for HIF8"
#endif
#endif

#ifndef Tcmp_s2
#define cmp_s2 64
#else
#define cmp_s2 Tcmp_s2
#endif

#ifndef Tori_topk
#define ori_topk 40
#else
#define ori_topk Tori_topk
#endif

#ifndef Tcmp_topk
#define cmp_topk 40
#else
#define cmp_topk Tcmp_topk
#endif

#ifndef Tcmp_ratio
#define cmp_ratio 4
#else
#define cmp_ratio Tcmp_ratio
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

#ifndef Tq_descale
#define q_descale_val 1.0f
#else
#define q_descale_val Tq_descale
#endif

#ifndef Tori_kv_descale
#define ori_kv_descale_val 1.0f
#else
#define ori_kv_descale_val Tori_kv_descale
#endif

#ifndef Tcmp_kv_descale
#define cmp_kv_descale_val 1.0f
#else
#define cmp_kv_descale_val Tcmp_kv_descale
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

#ifdef QSMLA_USE_EMBEDDED_INPUT
extern "C" {
extern const uint8_t _binary_q_fp16_start[];
extern const uint8_t _binary_kv_fp16_start[];
}
#endif

#ifdef QSMLA_USE_SPARSE_EMBEDDED_INPUT
extern "C" {
#ifdef QSMLA_USE_HIF8
extern const uint8_t _binary_q_hif8_start[];
extern const uint8_t _binary_ori_kv_hif8_start[];
#ifdef QSMLA_USE_CSA_TADD_1PE
extern const uint8_t _binary_ori_kv_t_hif8_start[];
#endif
#if defined(QSMLA_USE_HCA_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_1PE)
extern const uint8_t _binary_cmp_kv_hif8_start[];
#endif
#else
extern const uint8_t _binary_q_fp16_start[];
extern const uint8_t _binary_ori_kv_fp16_start[];
#ifdef QSMLA_USE_CSA_TADD_1PE
extern const uint8_t _binary_ori_kv_t_fp16_start[];
#endif
#if defined(QSMLA_USE_HCA_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_1PE)
extern const uint8_t _binary_cmp_kv_fp16_start[];
#endif
#endif
#if defined(QSMLA_USE_CSA_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_1PE)
extern const uint8_t _binary_cmp_sparse_indices_int32_start[];
#endif
#if defined(QSMLA_USE_ORI_SPARSE_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE)
extern const uint8_t _binary_ori_sparse_indices_int32_start[];
extern const uint8_t _binary_ori_topk_length_int32_start[];
#endif
#if defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE)
extern const uint8_t _binary_cmp_topk_length_int32_start[];
#endif
}
#endif

int main(){
#ifdef QSMLA_USE_HIF8
    using qdtype = __hif8;
    using kvdtype = __hif8;
    using odttype = __bf16;
#else
    using qdtype = __half;
    using kvdtype = __half;
    using odttype = __half;
#endif
    constexpr int group_size = N1 / N2;
    constexpr int g_slice_max = group_size < 64 ? group_size : 64;
    using Config = QsmlaConfig<
        B, s1, s2, N1, N2, D, 0, kTm, kTk, kTd, g_slice_max>;

#ifdef QSMLA_USE_TADD_4PE
    using ModeConfig = QsmlaModeConfig<
        Config, QsmlaMode::SWA, 0, 0, 0>;
#elif defined(QSMLA_USE_HCA_TADD_4PE)
    using ModeConfig = QsmlaModeConfig<
        Config, QsmlaMode::HCA, cmp_s2, 0, 0>;
#elif defined(QSMLA_USE_CSA_TADD_4PE)
    using ModeConfig = QsmlaModeConfig<
        Config, QsmlaMode::CSA, cmp_s2, ori_topk, cmp_topk>;
#elif defined(QSMLA_USE_ORI_SPARSE_TADD_4PE)
    using ModeConfig = QsmlaModeConfig<
        Config, QsmlaMode::ORI_SPARSE, cmp_s2, ori_topk, cmp_topk>;
#elif defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE)
    using ModeConfig = QsmlaModeConfig<
        Config, QsmlaMode::ORI_CMP_SPARSE,
        cmp_s2, ori_topk, cmp_topk>;
#endif

#ifdef QSMLA_USE_SPARSE_EMBEDDED_INPUT
#ifdef QSMLA_USE_HIF8
    qdtype* q = reinterpret_cast<qdtype*>(
        const_cast<uint8_t*>(_binary_q_hif8_start));
    kvdtype* kv = reinterpret_cast<kvdtype*>(
        const_cast<uint8_t*>(_binary_ori_kv_hif8_start));
#else
    qdtype* q = reinterpret_cast<qdtype*>(
        const_cast<uint8_t*>(_binary_q_fp16_start));
    kvdtype* kv = reinterpret_cast<kvdtype*>(
        const_cast<uint8_t*>(_binary_ori_kv_fp16_start));
#endif
#ifdef QSMLA_USE_CSA_TADD_1PE
    // qli-style pre-transposed ORI KV [D, S2] per batch (BMM1 K^T source)
#ifdef QSMLA_USE_HIF8
    kvdtype* ori_kv_t = reinterpret_cast<kvdtype*>(
        const_cast<uint8_t*>(_binary_ori_kv_t_hif8_start));
#else
    kvdtype* ori_kv_t = reinterpret_cast<kvdtype*>(
        const_cast<uint8_t*>(_binary_ori_kv_t_fp16_start));
#endif
#else
    kvdtype* ori_kv_t = nullptr;
#endif
#if defined(QSMLA_USE_HCA_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_1PE)
#ifdef QSMLA_USE_HIF8
    kvdtype* cmp_kv = reinterpret_cast<kvdtype*>(
        const_cast<uint8_t*>(_binary_cmp_kv_hif8_start));
#else
    kvdtype* cmp_kv = reinterpret_cast<kvdtype*>(
        const_cast<uint8_t*>(_binary_cmp_kv_fp16_start));
#endif
#else
    kvdtype* cmp_kv = nullptr;
#endif
#if defined(QSMLA_USE_CSA_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE) || \
    defined(QSMLA_USE_CSA_TADD_1PE)
    const int* cmp_indices = reinterpret_cast<const int*>(
        _binary_cmp_sparse_indices_int32_start);
#else
    const int* cmp_indices = nullptr;
#endif
#if defined(QSMLA_USE_ORI_SPARSE_TADD_4PE) || \
    defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE)
    const int* ori_indices = reinterpret_cast<const int*>(
        _binary_ori_sparse_indices_int32_start);
    const int* ori_lengths = reinterpret_cast<const int*>(
        _binary_ori_topk_length_int32_start);
#else
    const int* ori_indices = nullptr;
    const int* ori_lengths = nullptr;
#endif
#if defined(QSMLA_USE_ORI_CMP_SPARSE_TADD_4PE)
    const int* cmp_lengths = reinterpret_cast<const int*>(
        _binary_cmp_topk_length_int32_start);
#else
    const int* cmp_lengths = nullptr;
#endif
#elif defined(QSMLA_USE_EMBEDDED_INPUT)
    // The fixed typical-case inputs are linked into an aligned read-only ELF
    // section.  All four PEs share these addresses and the kernel never writes
    // Q/KV, avoiding simulated scalar initialization before BENCHSTART.
    qdtype* q = reinterpret_cast<qdtype*>(
        const_cast<uint8_t*>(_binary_q_fp16_start));
    kvdtype* kv = reinterpret_cast<kvdtype*>(
        const_cast<uint8_t*>(_binary_kv_fp16_start));
#else
    qdtype qp[B*s1*N1*D + 2*ALIGN];
    kvdtype kvp[B*s2*N2*D + 2*ALIGN];

    qdtype* q = (qdtype*)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    kvdtype* kv = (kvdtype*)(((uint64_t)kvp & ALIGN_MASK) + ALIGN);
#endif

#ifndef QSMLA_USE_SPARSE_EMBEDDED_INPUT
    kvdtype* cmp_kv = nullptr;
    const int* ori_indices = nullptr;
    const int* cmp_indices = nullptr;
    const int* ori_lengths = nullptr;
    const int* cmp_lengths = nullptr;
#endif

    odttype* out = (odttype*)MAP_MEM_BASE;

#if defined(QSMLA_USE_UNIFIED_TADD_4PE)
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

#if !defined(QSMLA_USE_EMBEDDED_INPUT) && \
    !defined(QSMLA_USE_SPARSE_EMBEDDED_INPUT)
    init_deterministic(q, B*s1*N1*D, 1);
    init_deterministic(kv, B*s2*N2*D, 2);
#endif

    BENCHSTART;
#ifdef QSMLA_USE_CSA_TADD_1PE
    // 单 PE CSA：Local CUBE 路径（FP16/HIF8 双态，嵌入输入；
    // FP16 下 descale 宏为 1，HIF8 下为 per-tensor 反量化因子）
    using ModeConfig = QsmlaModeConfig<
        Config, QsmlaMode::CSA, cmp_s2, 0, cmp_topk>;
    quant_sparse_flash_mla_csa_1pe_pto<
        qdtype, kvdtype, odttype, Config, ModeConfig>(
            out, q, kv, ori_kv_t, cmp_kv,
            cmp_indices, nullptr,   // cmp_topk_length（缺省 CmpTopK 全收集）
            softmax_scale_val,
            q_descale_val, ori_kv_descale_val, cmp_kv_descale_val,
            cmp_ratio, win_left, win_right);
#else
#ifdef QSMLA_USE_UNIFIED_TADD_4PE
    static_assert(N1 > 1,
                  "unified tadd 4pe is a BSND G-slice implementation");
#ifdef QSMLA_USE_DYNAMIC_SHAPE
    // v2: dynamic-shape variant (DYNAMIC ValidRow/ValidCol on Vec tiles)
    quant_sparse_flash_mla_tadd_4pe_bsnd_pto_v2<
        qdtype, kvdtype, odttype, ModeConfig>(
            out, q, kv, cmp_kv,
            ori_indices, cmp_indices, ori_lengths, cmp_lengths,
            softmax_scale_val,
            q_descale_val, ori_kv_descale_val, cmp_kv_descale_val,
            cmp_ratio, win_left, win_right,
            score_scratch, prob_scratch, pv_scratch);
#else
    quant_sparse_flash_mla_tadd_4pe_bsnd_pto<
        qdtype, kvdtype, odttype, ModeConfig>(
            out, q, kv, cmp_kv,
            ori_indices, cmp_indices, ori_lengths, cmp_lengths,
            softmax_scale_val,
            q_descale_val, ori_kv_descale_val, cmp_kv_descale_val,
            cmp_ratio, win_left, win_right,
            score_scratch, prob_scratch, pv_scratch);
#endif // QSMLA_USE_DYNAMIC_SHAPE
#else
    if constexpr (N1 == 1 && N2 == 1) {
        quant_sparse_flash_mla_swa_tadd_config_pto<
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
        quant_sparse_flash_mla_swa_tadd_bsnd_pto<
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
#endif
    BENCHEND;

    // gfsim 判读通道（本地性能采集用，不入 PR）: test-finisher 0x10009000=0x5555
    {
        volatile uint32_t* finisher =
            reinterpret_cast<volatile uint32_t*>(0x10009000ULL);
        *finisher = 0x5555;
    }

    return 0;
}
