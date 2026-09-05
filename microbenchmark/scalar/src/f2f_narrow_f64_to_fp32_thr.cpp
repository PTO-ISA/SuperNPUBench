#include "scalar_bench.hpp"
// auto-generated: f2f_narrow (cv) f64->fp32 throughput
int main() {
    double b[16];
    for (int i = 0; i < 16; ++i) b[i] = (double)(i * 0.7 + 1);
    volatile float sink = (float)0;
    BENCHSTART;
    float r = bench_cv<double, float>(b);
    BENCHEND;
    sink = r;
#ifdef RES_CHECK
    float ref = (float)0;
    for (int lane = 0; lane < 8; ++lane)
        ref = (float)(ref + (float)b[(1023 * 8 + lane) & 15]);
    return verify_scalar(r, ref) ? 0 : 1;
#else
    return 0;
#endif
}
