#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: mul (bin) i64 thr
int main() {
    int64_t a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (int64_t)(i * 0.7 + 1); b[i] = (int64_t)(i * 0.3 + 2); }
    volatile int64_t sink = (int64_t)0;
    auto scalar_op = [](auto x,auto y){return x*y;};
    BENCHSTART;
    int64_t r = bench_throughput<int64_t>(a, b, scalar_op);
    BENCHEND;
    sink = r;
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_scalar(r);
#endif
#ifdef RES_CHECK
    int64_t ref = reference_throughput<int64_t>(a, b, scalar_op);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
