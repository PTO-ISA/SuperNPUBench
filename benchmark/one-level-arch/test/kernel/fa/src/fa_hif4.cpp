#include "basic_op/fa/fa_hif4.hpp"
#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"

#ifndef HIF4_VARIANT
#define HIF4_VARIANT 0
#endif
static_assert(Tm == 128, "HiF4 FA baseline currently uses group_M=128");
int main() {
    constexpr int QD = FA_QD, VD = FA_VD;
    alignas(4096) static __fp4_hif4x2 q[Tsq*QD/2], k[Tskv*QD/2], v[VD*Tskv/2];
    alignas(4096) static uint32_t qs[Tsq*(QD/64)], ks[Tskv*(QD/64)], vs[VD*(Tskv/64)];
    alignas(4096) static __bf16 out[Tsq*VD];
    const int pe = get_thread_idx();
#ifdef RES_CHECK
    static MultiThreadResCheckSync sync{};
    if (pe == 0) {
        readBinaryFile(CHK_DIR "/srcq.bin",reinterpret_cast<uint8_t*>(q),sizeof(q));
        readBinaryFile(CHK_DIR "/srck.bin",reinterpret_cast<uint8_t*>(k),sizeof(k));
        readBinaryFile(CHK_DIR "/srcv.bin",reinterpret_cast<uint8_t*>(v),sizeof(v));
        readBinaryFile(CHK_DIR "/srcq_scale.bin",reinterpret_cast<uint8_t*>(qs),sizeof(qs));
        readBinaryFile(CHK_DIR "/srck_scale.bin",reinterpret_cast<uint8_t*>(ks),sizeof(ks));
        readBinaryFile(CHK_DIR "/srcv_scale.bin",reinterpret_cast<uint8_t*>(vs),sizeof(vs));
    }
    res_check_publish_inputs(sync,pe);
#endif
    BENCHSTART;
    flash_attention_hif4<Tsq,Tskv,QD,VD,Tk,HIF4_VARIANT>(out,q,k,v,qs,ks,vs);
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(sync,pe);
    if (pe == 0) writeBinaryFile(CHK_DIR "/res.bin",reinterpret_cast<uint8_t*>(out),sizeof(out));
#endif
    return 0;
}
