#include "elem_bench.hpp"
#include <cstdio>
#include <cstdlib>

// Test configuration
constexpr int M = 16;
constexpr int N = 16;

int main() {
    printf("Vector Element-wise Addition Benchmark: FP32 %dx%d\n", M, N);
    
    // Allocate memory
    float* a = (float*)malloc(M * N * sizeof(float));
    float* b = (float*)malloc(M * N * sizeof(float));
    float* c = (float*)malloc(M * N * sizeof(float));
    
    // Initialize with random data
    for (int i = 0; i < M * N; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
        c[i] = 0.0f;
    }
    
    // Run benchmark
    tadd_fp32_16x16(c, a, b);
    
    // Verify result
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
