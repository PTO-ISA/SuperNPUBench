#include "scalar_bench.hpp"
// auto-generated: st (st) f32 throughput
int main() {
    float out[16], val = (float)5;
    for (int i = 0; i < 16; ++i) out[i] = (float)0;
    BENCHSTART;
    bench_store<float>(out, val);
    BENCHEND;
    volatile float sink = out[0];
    return 0;
}
