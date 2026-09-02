#include "scalar_bench.hpp"
// auto-generated: or (bin) i32 thr
int main() {
    int32_t a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (int32_t)(i * 0.7 + 1); b[i] = (int32_t)(i * 0.3 + 2); }
    volatile int32_t sink = (int32_t)0;
    auto scalar_op = [](auto x,auto y){return (int32_t)((uint32_t)x | (uint32_t)y);};
    BENCHSTART;
    int32_t r = bench_throughput<int32_t>(a, b, scalar_op);
    BENCHEND;
    sink = r;
#ifdef RES_CHECK
    int32_t ref = reference_throughput<int32_t>(a, b, scalar_op);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
