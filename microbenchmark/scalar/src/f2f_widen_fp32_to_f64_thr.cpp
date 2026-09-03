#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: f2f_widen (cv) fp32->f64 throughput
int main() {
    float b[16];
    for (int i = 0; i < 16; ++i) b[i] = (float)(i * 0.7 + 1);
    volatile double sink = (double)0;
    BENCHSTART;
    double r = bench_cv<float, double>(b);
    BENCHEND;
    sink = r;
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_scalar(r);
#endif
#ifdef RES_CHECK
    double ref = (double)0;
    for (int lane = 0; lane < 8; ++lane)
        ref = (double)(ref + (double)b[(1023 * 8 + lane) & 15]);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
