#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: f2i (cv) fp32->i32 throughput
int main() {
    float b[16];
    for (int i = 0; i < 16; ++i) b[i] = (float)(i * 0.7 + 1);
    volatile int32_t sink = (int32_t)0;
    BENCHSTART;
    int32_t r = bench_cv<float, int32_t>(b);
    BENCHEND;
    sink = r;
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_scalar(r);
#endif
#ifdef RES_CHECK
    int32_t ref = (int32_t)0;
    for (int lane = 0; lane < 8; ++lane)
        ref = (int32_t)(ref + (int32_t)b[(1023 * 8 + lane) & 15]);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
