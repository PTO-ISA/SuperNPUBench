#include "matmul_bench.hpp"
#include "benchmark.h"

// Test configuration
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

int main() {
    // Use stack-allocated arrays instead of malloc
    float a[M * K];
    float b[K * N];
    float c[M * N];
    
    // Initialize with simple pattern
    for (int i = 0; i < M * K; i++) {
        a[i] = (float)(i % 10) * 0.1f;
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = (float)(i % 10) * 0.1f;
    }
    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0f;
    }
    
    // Run benchmark
    BENCHSTART;
    matmul_fp32_16x16(c, a, b);
    BENCHEND;
    
    return 0;
}
