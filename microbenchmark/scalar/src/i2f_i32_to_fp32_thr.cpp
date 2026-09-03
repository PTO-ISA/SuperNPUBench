#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: i2f (cv) i32->fp32 throughput
int main() {
    int32_t b[16];
    for (int i = 0; i < 16; ++i) b[i] = (int32_t)(i * 0.7 + 1);
    volatile float sink = (float)0;
    BENCHSTART;
    float r = bench_cv<int32_t, float>(b);
    BENCHEND;
    sink = r;
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_scalar(r);
#endif
#ifdef RES_CHECK
    float ref = (float)0;
    for (int lane = 0; lane < 8; ++lane)
        ref = (float)(ref + (float)b[(1023 * 8 + lane) & 15]);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
