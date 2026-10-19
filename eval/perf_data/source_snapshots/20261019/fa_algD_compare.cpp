#ifdef COMPARE_ALGC
#define FA_DOUBLEQ_ADDR_PRELOAD 1
#include "basic_op/fa/fa_lowp_algC_final_doubleQ.hpp"
#else
#include "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl/perf_handoff/algoD_dblQ_fusion_20261012/cmd_dispatch2_20261013/source/fa_lowp_algD.hpp"
#endif
#include "benchmark.h"
#include <cstdint>
template<class T> T* aligned(T* p) { return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(p)+4095)&~uintptr_t(4095)); }
int main() {
 static __fp4_e2m1x2 qb[Tsq*FA_QD/2+4096], kb[Tskv*FA_QD/2+4096], vb[Tskv*FA_VD/2+4096];
 static __fp8_e8m0 qs[Tsq*FA_QD/32+4096], ks[Tskv*FA_QD/32+4096], vs[Tskv*FA_VD/32+4096];
 static __bf16 out[Tsq*FA_VD+4096];
 BENCHSTART;
#ifdef COMPARE_ALGC
 fa_lowp_algC_final_doubleQ::fa_lowp_algC_final_doubleQ_impl<Tsq,Tskv,FA_QD,FA_VD,Tm,Tk>(aligned(out),aligned(qb),aligned(kb),aligned(vb),aligned(qs),aligned(ks),aligned(vs));
#else
 fa_lowp_logdomain::fa_lowp_algD_impl<Tsq,Tskv,FA_QD,FA_VD,Tm,Tk>(aligned(out),aligned(qb),aligned(kb),aligned(vb),aligned(qs),aligned(ks),aligned(vs));
#endif
 BENCHEND;
 return 0;
}
