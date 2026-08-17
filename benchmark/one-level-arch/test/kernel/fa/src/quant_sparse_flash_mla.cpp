#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
// #include "fa/quant_sparse_flash_mla_pto.hpp"
// #include "fa/quant_sparse_flash_mla_tadd_pto.hpp"
#include "fa/quant_sparse_flash_mla_onepass_pto.hpp"

#define B 1
#define H 1

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
    using Config = QsmlaConfig<B, s1, s2, H, H, D, 0, kTm, kTk, kTd>;

    qdtype qp[B*H*s1*D + 2*ALIGN];
    kvdtype kvp[B*H*s2*D + 2*ALIGN];

    qdtype* q = (qdtype*)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    kvdtype* kv = (kvdtype*)(((uint64_t)kvp & ALIGN_MASK) + ALIGN);

    odttype* out = (odttype*)MAP_MEM_BASE;

    init_deterministic(q, B*H*s1*D, 1);
    init_deterministic(kv, B*H*s2*D, 2);

    BENCHSTART;
    for(int i=0;i<B;i++){
        for(int j=0;j<H;j++){
            quant_sparse_flash_mla_swa_onepass_config_pto<
                qdtype, kvdtype, odttype, Config>(
                out + i*H*s1*D + j*s1*D,
                q   + i*H*s1*D + j*s1*D,
                kv  + i*H*s2*D + j*s2*D,
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
    }
    BENCHEND;

    return 0;
}
