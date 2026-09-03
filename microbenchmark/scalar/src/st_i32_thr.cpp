#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: st (st) i32 throughput
int main() {
    int32_t out[16], val = (int32_t)5;
    for (int i = 0; i < 16; ++i) out[i] = (int32_t)0;
    BENCHSTART;
    bench_store<int32_t>(out, val);
    BENCHEND;
    volatile int32_t sink = out[0];
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_result(out, 16);
#endif
#ifdef RES_CHECK
    for (int i = 0; i < 16; ++i) if (out[i] != val) return 1;
#endif
    return 0;
}
