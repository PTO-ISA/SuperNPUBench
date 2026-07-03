#include "matmul_bench.hpp"
#include <cstdio>
#include <cstdlib>
#include "benchmark.h"

// Test configuration
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

int main() {
    printf("Matrix Multiplication Benchmark: FP32 %dx%d\n", M, N);
    
    // Allocate memory with alignment
    float* a = (float*)malloc(M * K * sizeof(float));
    float* b = (float*)malloc(K * N * sizeof(float));
    float* c = (float*)malloc(M * N * sizeof(float));
    
    // Initialize with random data
    for (int i = 0; i < M * K; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0f;
    }
    
    // Run benchmark
    matmul_fp32_16x16(c, a, b);
    
    // Verify result (simple check)
    float sum = 0.0f;
    for (int i = 0; i < M * N; i++) {
        sum += c[i];
    }
    
    printf("Result sum: %f\n", sum);
    printf("Benchmark completed successfully!\n");
    
    // Cleanup
    free(a);
    free(b);
    free(c);
    
    return 0;
}
