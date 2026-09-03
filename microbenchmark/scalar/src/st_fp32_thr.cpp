#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: st (st) fp32 throughput
int main() {
    float out[16], val = (float)5;
    for (int i = 0; i < 16; ++i) out[i] = (float)0;
    BENCHSTART;
    bench_store<float>(out, val);
    BENCHEND;
    volatile float sink = out[0];
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_result(out, 16);
#endif
#ifdef RES_CHECK
    for (int i = 0; i < 16; ++i) if (out[i] != val) return 1;
#endif
    return 0;
}
