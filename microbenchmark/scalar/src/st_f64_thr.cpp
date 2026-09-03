#include "scalar_bench.hpp"
#ifdef CROSS_MODEL_CORPUS
#include "../common/cross_model_result.hpp"
#endif
// auto-generated: st (st) f64 throughput
int main() {
    double out[16], val = (double)5;
    for (int i = 0; i < 16; ++i) out[i] = (double)0;
    BENCHSTART;
    bench_store<double>(out, val);
    BENCHEND;
    volatile double sink = out[0];
#ifdef CROSS_MODEL_CORPUS
    publish_cross_model_result(out, 16);
#endif
#ifdef RES_CHECK
    for (int i = 0; i < 16; ++i) if (out[i] != val) return 1;
#endif
    return 0;
}
