#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "fileop.h"
#include "single_thread/fa/sfa.hpp"

#define B 1
#define H 1

#ifndef Tsq
#define Sq 256
#else
#define Sq Tsq
#endif

#ifndef Tskv
#define Skv 512
#else
#define Skv Tskv
#endif

#define qD 128
#define vD 128

#ifndef Tm
#define kTm 8
#else
#define kTm Tm
#endif

#ifndef Tk
#define kTk 16
#else
#define kTk Tk
#endif

#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN 4*1024

int main(){
    using dtype = __half;

    dtype qp[B*H*Sq*qD + 2*ALIGN];
    dtype kp[B*H*Skv*qD + 2*ALIGN];
    dtype vp[B*H*Skv*vD + 2*ALIGN];
    dtype outp[B*H*Sq*vD + 2*ALIGN];

    dtype* q = (dtype *)(((uint64_t)qp & ALIGN_MASK) + ALIGN);
    dtype* k = (dtype *)(((uint64_t)kp & ALIGN_MASK) + ALIGN);
    dtype* v = (dtype *)(((uint64_t)vp & ALIGN_MASK) + ALIGN);
    dtype* out = (dtype *)(((uint64_t)outp & ALIGN_MASK) + ALIGN);

    // ---- 构建块级稀疏模式（CSR）: local-window attention ----
    // 每个 Q 块 i 仅关注 K/V 块 j in [max(0,i-w), min(Kb,i+w+1))。
    const int Qb = (Sq + kTm - 1) / kTm;
    const int Kb = (Skv + kTk - 1) / kTk;
    const int w  = 2;
    // 上界：每块最多 (2w+1) 个活跃块，+1 个哨兵。
    int kv_off[Qb + 1];
    int kv_idx[Qb * (2 * w + 1)];
    int cnt = 0;
    for (int i = 0; i < Qb; ++i) {
        kv_off[i] = cnt;
        int lo = i - w; if (lo < 0) lo = 0;
        int hi = i + w + 1; if (hi > Kb) hi = Kb;
        for (int j = lo; j < hi; ++j) {
            kv_idx[cnt++] = j;
        }
    }
    kv_off[Qb] = cnt;

    BENCHSTART;
    for(int i=0;i<B;i++){
        for(int j=0;j<H;j++){
            sparse_flash_attention_pto<dtype, Sq, Skv, qD, vD, kTm, kTk>(
                out + i*H*Sq*vD + j*Sq*vD,
                q   + i*H*Sq*qD + j*Sq*qD,
                k   + i*H*Skv*qD + j*Skv*qD,
                v   + i*H*Skv*vD + j*Skv*vD,
                kv_idx,
                kv_off
            );
        }
    }
    BENCHEND;

    return 0;
}
