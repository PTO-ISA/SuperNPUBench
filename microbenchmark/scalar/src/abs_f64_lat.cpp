#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: abs (un) f64 lat
int main() {
    double a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (double)(i * 0.7 + 1); b[i] = (double)(i * 0.3 + 2); }
    volatile double sink = (double)0;
    auto scalar_op = [](auto x,auto y){auto t=x+y; return t<0?-t:t;};
    BENCHSTART;
    double r = bench_latency<double>(a, b, scalar_op);
    BENCHEND;
    sink = r;
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_scalar(r);
#endif
#ifdef RES_CHECK
    double ref = reference_latency<double>(a, b, scalar_op);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
