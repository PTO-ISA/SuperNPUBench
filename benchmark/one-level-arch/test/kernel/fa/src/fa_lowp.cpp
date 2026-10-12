#ifdef FA_LOWP_RECIP
#include "basic_op/fa/fa_lowp_recip.hpp"
#else
#include "basic_op/fa/fa_lowp.hpp"
#endif
#include <cstdint>
#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"

#ifndef Tsq
#define Tsq 256
#endif
#ifndef Tskv
#define Tskv 256
#endif
#ifndef FA_QD
#define FA_QD 128
#endif
#ifndef FA_VD
#define FA_VD 128
#endif
#ifndef Tm
#define Tm 128
#endif
#ifndef Tk
#define Tk 128
#endif
#define ALIGN_MASK 0xfffffffffffff000ull
#define ALIGN (4 * 1024)

template <typename T> T *aligned_ptr(T *p) {
    return reinterpret_cast<T *>((reinterpret_cast<uint64_t>(p) & ALIGN_MASK) + ALIGN);
}

int main() {
    const uint32_t tid = get_thread_idx();
    static __fp4_e2m1x2 qb[Tsq * (FA_QD / 2) + 2 * ALIGN];
    static __fp4_e2m1x2 kb[Tskv * (FA_QD / 2) + 2 * ALIGN];
    static __fp4_e2m1x2 vb[(Tskv / 2) * FA_VD + 2 * ALIGN];
    static __fp8_e8m0 qsb[Tsq * (FA_QD / 32) + 2 * ALIGN];
    static __fp8_e8m0 ksb[Tskv * (FA_QD / 32) + 2 * ALIGN];
    static __fp8_e8m0 vsb[(Tskv / 32) * FA_VD + 2 * ALIGN];
    static __bf16 ob[Tsq * FA_VD + 2 * ALIGN];
#ifdef RES_CHECK
    static MultiThreadResCheckSync sync{};
#endif
    auto *q = aligned_ptr(qb);
    auto *k = aligned_ptr(kb);
    auto *v = aligned_ptr(vb);
    auto *qs = aligned_ptr(qsb);
    auto *ks = aligned_ptr(ksb);
    auto *vs = aligned_ptr(vsb);
    auto *out = aligned_ptr(ob);
#ifdef RES_CHECK
    if (tid == 0) {
        readBinaryFile(CHK_DIR "/srcq.bin", reinterpret_cast<uint8_t *>(q), Tsq * (FA_QD / 2));
        readBinaryFile(CHK_DIR "/srck.bin", reinterpret_cast<uint8_t *>(k), Tskv * (FA_QD / 2));
        readBinaryFile(CHK_DIR "/srcv.bin", reinterpret_cast<uint8_t *>(v), (Tskv / 2) * FA_VD);
        readBinaryFile(CHK_DIR "/srcq_scale.bin", reinterpret_cast<uint8_t *>(qs), Tsq * (FA_QD / 32));
        readBinaryFile(CHK_DIR "/srck_scale.bin", reinterpret_cast<uint8_t *>(ks), Tskv * (FA_QD / 32));
        readBinaryFile(CHK_DIR "/srcv_scale.bin", reinterpret_cast<uint8_t *>(vs), (Tskv / 32) * FA_VD);
    }
    res_check_publish_inputs(sync, tid);
#endif
    BENCHSTART;
#ifdef FA_LOWP_RECIP
    fa_lowp_recip::flash_attention_lowp_recip_impl<Tsq, Tskv, FA_QD, FA_VD, Tm, Tk>(
        out, q, k, v, qs, ks, vs);
#else
    fa_lowp::flash_attention_lowp_impl<Tsq, Tskv, FA_QD, FA_VD, Tm, Tk>(
        out, q, k, v, qs, ks, vs);
#endif
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(sync, tid);
    if (tid == 0)
        writeBinaryFile(CHK_DIR "/res.bin", reinterpret_cast<uint8_t *>(out),
                        Tsq * FA_VD * sizeof(__bf16));
#endif
    return 0;
}
