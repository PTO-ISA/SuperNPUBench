#include "elem_bench.hpp"
#include "benchmark.h"

// Test configuration
constexpr int M = 16;
constexpr int N = 16;

int main() {
    // Use stack-allocated arrays instead of malloc
    float a[M * N];
    float b[M * N];
    float c[M * N];
    
    // Initialize with simple pattern
    for (int i = 0; i < M * N; i++) {
        a[i] = (float)(i % 10) * 0.1f;
        b[i] = (float)(i % 10) * 0.1f;
        c[i] = 0.0f;
    }
    
    // Run benchmark
    BENCHSTART;
    tadd_fp32_16x16(c, a, b);
    BENCHEND;
    
    return 0;
}
