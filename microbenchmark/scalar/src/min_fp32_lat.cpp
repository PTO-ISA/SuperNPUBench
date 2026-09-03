#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: min (bin) fp32 lat
int main() {
    float a[16], b[16];
    for (int i = 0; i < 16; ++i) { a[i] = (float)(i * 0.7 + 1); b[i] = (float)(i * 0.3 + 2); }
    volatile float sink = (float)0;
    auto scalar_op = [](auto x,auto y){return x<y?x:y;};
    BENCHSTART;
    float r = bench_latency<float>(a, b, scalar_op);
    BENCHEND;
    sink = r;
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_scalar(r);
#endif
#ifdef RES_CHECK
    float ref = reference_latency<float>(a, b, scalar_op);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
