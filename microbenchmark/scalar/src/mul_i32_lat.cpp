#include "scalar_bench.hpp"
// auto-generated: mul (bin) i32 lat
int main() {
    int32_t a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (int32_t)(i * 0.7 + 1); b[i] = (int32_t)(i * 0.3 + 2); }
    volatile int32_t sink = (int32_t)0;
    auto scalar_op = [](auto x,auto y){return x*y;};
    BENCHSTART;
    int32_t r = bench_latency<int32_t>(a, b, scalar_op);
    BENCHEND;
    sink = r;
#ifdef RES_CHECK
    int32_t ref = reference_latency<int32_t>(a, b, scalar_op);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
